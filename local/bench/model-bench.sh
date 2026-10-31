#!/usr/bin/env bash
# Single-model benchmark with the quickbench2 methodology, so numbers are
# directly comparable with the recorded baselines (61.9 t/s decode etc.):
#   WARMUP=1 (discarded), REPS=3, 40x ~35-word blocks (~1.4k tokens),
#   max_tokens=96, temperature=0, reasoning_effort=none, port 8092.
#
# Usage: model-bench.sh <config.json> <model_name> <label> [extra_args...]
#   extra_args are appended to the config's engine args (e.g. for A/B runs).
#
# Output: local/bench/results/model-bench-<label>-<TS>/ with prompt.txt,
# server.log, summary.json, summary.csv. The server is started and stopped
# by this script; any other Strata/llama-server process is killed first.
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <config.json> <model_name> <label> [extra_args...]" >&2
    exit 1
fi

CONFIG_FILE="$1"
MODEL_NAME="$2"
LABEL="$3"
shift 3
EXTRA_ARGS=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_UUID="${GPU_UUID:-GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c}"
PORT="${PORT:-8092}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
CHAT_URL="http://127.0.0.1:${PORT}/v1/chat/completions"

PROMPT_REPEATS="${PROMPT_REPEATS:-40}"
WARMUP="${WARMUP:-1}"
REPS="${REPS:-3}"
MAX_TOKENS="${MAX_TOKENS:-96}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$REPO_ROOT/local/bench/results/model-bench-${LABEL}-${TS}"
PROMPT_FILE="$RUN_DIR/prompt.txt"
SUMMARY_CSV="$RUN_DIR/summary.csv"
SERVER_LOG="$RUN_DIR/server.log"

need() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "error: missing dependency: $1" >&2
        exit 1
    }
}

need python3
need curl
need rg
need numactl
need nvidia-smi

[[ -f "$CONFIG_FILE" ]] || { echo "error: no such config: $CONFIG_FILE" >&2; exit 1; }
# Resolve to an absolute path: the server is launched from $REPO_ROOT/strata (below),
# so a relative config path passed by a caller would not be found there.
CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"
mkdir -p "$RUN_DIR"

cleanup() {
    pkill -f 'strata/serve/server.py|engine-cuda12/strata|engine-cuda12/strata-aggr|llama-server' >/dev/null 2>&1 || true
}
trap cleanup EXIT

make_prompt() {
    python3 - "$PROMPT_FILE" "$PROMPT_REPEATS" <<'PY'
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
repeats = int(sys.argv[2])
block = (
    "Optimize a Volta V100 + dual Xeon inference setup for coding-agent throughput. "
    "Discuss prompt ingest, decode tok/s, speculative draft acceptance, NUMA locality, "
    "PCIe traffic, and VRAM safety margins. Provide concise numbered actions."
)
out.write_text("\n\n".join([block for _ in range(repeats)]), encoding="utf-8")
PY
}

gpu_used_mib() {
    nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | awk -F', *' -v id="$GPU_UUID" '$1==id{print int($2)}'
}

stop_and_wait() {
    pkill -f 'strata/serve/server.py|engine-cuda12/strata|engine-cuda12/strata-aggr|llama-server' >/dev/null 2>&1 || true

    local i
    for i in $(seq 1 120); do
        if ! pgrep -f 'strata/serve/server.py|engine-cuda12/strata|engine-cuda12/strata-aggr|llama-server' >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    for i in $(seq 1 120); do
        local used
        used="$(gpu_used_mib || true)"
        if [[ -n "$used" && "$used" -lt 1200 ]]; then
            break
        fi
        sleep 1
    done
}

start_server() {
    stop_and_wait
    (
        cd "$REPO_ROOT/strata"
        exec numactl --cpunodebind=1 "$REPO_ROOT/strata/.venv/bin/python" "$REPO_ROOT/strata/serve/server.py" --engine strata --config "$CONFIG_FILE" --port "$PORT" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    ) >"$SERVER_LOG" 2>&1 &

    local i
    for i in $(seq 1 300); do
        if curl -fsS "$HEALTH_URL" 2>/dev/null | rg -q '"loaded": true'; then
            return 0
        fi
        sleep 1
    done
    echo "error: server did not become healthy; last log lines:" >&2
    tail -20 "$SERVER_LOG" >&2 || true
    return 1
}

bench_requests() {
    local out_json="$1"
    python3 - "$CHAT_URL" "$PROMPT_FILE" "$WARMUP" "$REPS" "$MAX_TOKENS" "$MODEL_NAME" "$LABEL" "$out_json" <<'PY'
import json
import pathlib
import statistics
import sys
import time
import urllib.request

url = sys.argv[1]
prompt = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")
warmup = int(sys.argv[3])
reps = int(sys.argv[4])
max_tokens = int(sys.argv[5])
model = sys.argv[6]
label = sys.argv[7]
out_path = pathlib.Path(sys.argv[8])

def one(marker):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a performance engineer."},
            {"role": "user", "content": prompt + f"\n\nmarker {marker}"},
        ],
        "reasoning_effort": "none",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as resp:
        data = json.loads(resp.read())
    wall = time.time() - t0
    t = data.get("timings") or {}
    return {
        "wall_s": round(wall, 3),
        "prompt_tps": float(t.get("prompt_per_second", 0.0) or 0.0),
        "decode_tps": float(t.get("predicted_per_second", 0.0) or 0.0),
        "draft_n": int(t.get("draft_n", 0) or 0),
        "draft_accepted": int(t.get("draft_n_accepted", 0) or 0),
        "prompt_n": int(t.get("prompt_n", 0) or 0),
        "pred_n": int(t.get("predicted_n", 0) or 0),
    }

for w in range(warmup):
    one(f"w{w+1}")

rows = [one(i + 1) for i in range(reps)]

decode = [r["decode_tps"] for r in rows]
prompt_tps = [r["prompt_tps"] for r in rows]
dn = sum(r["draft_n"] for r in rows)
da = sum(r["draft_accepted"] for r in rows)
summary = {
    "label": label,
    "model": model,
    "reps": reps,
    "warmup": warmup,
    "decode_tps_avg": round(statistics.mean(decode), 3),
    "decode_tps_med": round(statistics.median(decode), 3),
    "prompt_tps_avg": round(statistics.mean(prompt_tps), 3),
    "prompt_tps_med": round(statistics.median(prompt_tps), 3),
    "draft_accept_rate": round(da / dn, 4) if dn else None,
    "draft_n_total": dn,
    "draft_accepted_total": da,
    "rows": rows,
}
out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
}

echo "[model-bench] label=$LABEL model=$MODEL_NAME config=$CONFIG_FILE"
make_prompt
if ! start_server; then
    echo "[model-bench] startup failed: $LABEL" >&2
    exit 1
fi

out_json="$RUN_DIR/summary.json"
if bench_requests "$out_json"; then
    python3 - "$out_json" >> "$SUMMARY_CSV" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    s = json.load(f)
print(f"label,decode_tps_avg,decode_tps_med,prompt_tps_avg,prompt_tps_med,draft_accept_rate,draft_n_total,draft_accepted_total")
print(f"{s['label']},{s['decode_tps_avg']},{s['decode_tps_med']},{s['prompt_tps_avg']},{s['prompt_tps_med']},{s['draft_accept_rate']},{s['draft_n_total']},{s['draft_accepted_total']}")
PY
else
    echo "[model-bench] request failed: $LABEL" >&2
    stop_and_wait
    exit 1
fi

stop_and_wait
echo "[model-bench] done: $RUN_DIR"
cat "$SUMMARY_CSV"
