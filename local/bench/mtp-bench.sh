#!/usr/bin/env bash
# MTP sweep runner: start the server with a given spec config, run fixed-prompt
# generations via mtp-bench.py, record decode t/s + draft acceptance.
#
# Usage:
#   local/bench/mtp-bench.sh <label> [SPEC_TYPE] [DRAFT_MAX] [REPS]
#
# Examples:
#   local/bench/mtp-bench.sh mtp-off none 0 3
#   local/bench/mtp-bench.sh mtp-n2 draft-mtp 2 3
#
# Env overrides: PORT CTX CACHE_K CACHE_V (passed through to the serve script)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="$REPO_ROOT/local/bench"
RESULTS_DIR="$BENCH_DIR/results"
CSV="$RESULTS_DIR/mtp-runs.csv"
SERVE="$REPO_ROOT/local/serve-qwen3-27b.sh"

LABEL="${1:?usage: mtp-bench.sh <label> [SPEC_TYPE] [DRAFT_MAX] [REPS]}"
SPEC_TYPE="${2:-draft-mtp}"
DRAFT_MAX="${3:-3}"
REPS="${4:-3}"
PORT="${PORT:-8091}"

V100_UUIDS="GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c,GPU-f07f3af3-5f02-f364-0d37-2ab3a4c4afce"
snap() { nvidia-smi -i "$V100_UUIDS" --query-gpu="$1" --format=csv,noheader,nounits | tr '\n' ';' | sed 's/;$//'; }

mkdir -p "$RESULTS_DIR/mtp"
timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
log="$RESULTS_DIR/mtp/$LABEL-$timestamp.log"

echo "label : $LABEL (spec-type=$SPEC_TYPE, draft-max=$DRAFT_MAX, reps=$REPS)"
echo "port  : $PORT"
echo "log   : $log"

# start server in background
SPEC_TYPE="$SPEC_TYPE" DRAFT_MAX="$DRAFT_MAX" PORT="$PORT" \
    "$SERVE" > "$log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

# wait for /health (model load + KV alloc at 128k can take a couple of minutes)
ok=""
for _ in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then ok=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "error: server died during startup, see $log" >&2
        tail -n 30 "$log" >&2
        exit 1
    fi
    sleep 2
done
if [[ -z "$ok" ]]; then
    echo "error: server not healthy after 180s, see $log" >&2
    exit 1
fi

temp_start="$(snap temperature.gpu)"
result="$(python3 "$BENCH_DIR/mtp-bench.py" --url "http://127.0.0.1:$PORT/v1" \
    --label "$LABEL" --reps "$REPS")" || { echo "error: mtp-bench.py failed" >&2; exit 1; }
temp_end="$(snap temperature.gpu)"

echo "$result" | python3 -m json.tool
echo "$result" > "$RESULTS_DIR/mtp/$LABEL-$timestamp.json"

python3 - "$CSV" "$timestamp" "$LABEL" "$commit" "$SPEC_TYPE" "$DRAFT_MAX" \
    "$temp_start" "$temp_end" "$result" <<'PY'
import csv, json, sys
csv_path, ts, label, commit, spec_type, draft_max, t0, t1, result = sys.argv[1:10]
s = json.loads(result)
cols = ["timestamp", "label", "commit", "spec_type", "draft_max", "reps_ok",
        "prompt_tps", "decode_tps", "draft_n", "draft_accepted", "accept_rate",
        "temp_start_c", "temp_end_c"]
row = [ts, label, commit, spec_type, draft_max, s["reps_ok"],
       s["prompt_tps_avg"], s["decode_tps_avg"], s["draft_n_total"],
       s["draft_accepted_total"], s["accept_rate"], t0, t1]
import os
new = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(cols)
    w.writerow(row)
PY
