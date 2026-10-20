#!/usr/bin/env bash
# Start Strata for local BYOK serving. Defaults target the Qwen3.8-Flash-Next
# unsloth UD-IQ4_XS config on the healthy V100's NUMA node.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STRATA_DIR="$REPO_ROOT/strata"
PYTHON_BIN="$STRATA_DIR/.venv/bin/python"
SERVER_PY="$STRATA_DIR/serve/server.py"

# Default config can be overridden so other serve scripts can reuse this runner.
STRATA_CONFIG="${STRATA_CONFIG:-$REPO_ROOT/local/strata/strata-unsloth-ud-iq4_xs.json}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
NUMA_NODE="${NUMA_NODE:-1}"
OPEN_BROWSER="${OPEN_BROWSER:-1}"

HEALTH_HOST="$HOST"
if [[ "$HEALTH_HOST" == "0.0.0.0" ]]; then
	HEALTH_HOST="127.0.0.1"
fi

[[ -x "$PYTHON_BIN" ]] || { echo "error: python venv not found: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$SERVER_PY" ]] || { echo "error: strata server not found: $SERVER_PY" >&2; exit 1; }
[[ -f "$STRATA_CONFIG" ]] || { echo "error: strata config not found: $STRATA_CONFIG" >&2; exit 1; }

echo "strata config : $STRATA_CONFIG"
echo "listen        : http://$HOST:$PORT"
echo "numa node     : $NUMA_NODE"

# If a healthy Strata is already serving on this port, treat it as success.
if health_json="$(curl -fsS --max-time 1 "http://$HEALTH_HOST:$PORT/health" 2>/dev/null)"; then
	if [[ "$health_json" == *'"status": "ok"'* && "$health_json" == *'"loaded": true'* ]]; then
		echo "strata already running at http://$HEALTH_HOST:$PORT (loaded=true); reusing existing server"
		exit 0
	fi
fi

# Fail fast with context when some other process is holding the port.
if ss -ltn | awk '{print $4}' | grep -qxE ".*:$PORT"; then
	echo "error: port $PORT is already in use and /health is not reporting a loaded Strata server" >&2
	echo "hint: stop the old process or choose another PORT" >&2
	exit 1
fi

cmd=(
	numactl --cpunodebind="$NUMA_NODE"
	"$PYTHON_BIN" "$SERVER_PY"
	--engine strata
	--config "$STRATA_CONFIG"
	--host "$HOST"
	--port "$PORT"
)

if [[ "$OPEN_BROWSER" == "1" ]]; then
	cmd+=(--open)
fi

cd "$STRATA_DIR"
exec "${cmd[@]}"
