#!/usr/bin/env bash
# Strata V100 sweep #2: additive optimizations on top of the production config.
#
# Arms (each = production config + ONE change):
#   adapt_overlap    STRATA_ADAPT_OVERLAP=1   (copies land while next window runs; needs adaptive tier on, ADAPT_LAG=1)
#   adapt_lag2       STRATA_ADAPT_LAG=2       (lagged adaptive swaps; mutually exclusive with overlap)
#   pcie_balance     STRATA_PCIE_BALANCE=1    (per-layer PCIe count from measured costs, #1548)
#   route_tail_skip1 STRATA_ROUTE_TAIL_SKIP=1 (production has 0)
#   aux_cpus_auto    --aux-cpus auto          (helper threads to spare CPUs; likely no-op under numactl node1 pin)
#   esp_cvec         experimental-speed-projection control vector (project, per-layer, layers 4-44)
#
# Differences vs quickbench #1: one warmup request (discarded) before the measured reps,
# REPS=3 for stability, port 8092. Within-sweep comparisons are the point; cross-sweep
# absolute numbers differ slightly because of the warmup.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="$REPO_ROOT/local/strata/strata-unsloth-ud-iq4_xs.json"
CONFIG_BAK="$REPO_ROOT/local/strata/strata-unsloth-ud-iq4_xs.json.quickbench2.bak"
ENGINE_BASE="$REPO_ROOT/strata/engine-cuda12/strata"
ESP_GGUF="$REPO_ROOT/strata/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf"

GPU_UUID="${GPU_UUID:-GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c}"
PORT="${PORT:-8092}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
CHAT_URL="http://127.0.0.1:${PORT}/v1/chat/completions"

VRAM_RESERVE_MIB="${VRAM_RESERVE_MIB:-768}"
PROMPT_REPEATS="${PROMPT_REPEATS:-40}"
WARMUP="${WARMUP:-1}"
REPS="${REPS:-3}"
MAX_TOKENS="${MAX_TOKENS:-96}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$REPO_ROOT/local/bench/results/strata-v100-quickbench2-$TS"
PROMPT_FILE="$RUN_DIR/prompt.txt"
SUMMARY_CSV="$RUN_DIR/summary.csv"

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

mkdir -p "$RUN_DIR"

cleanup() {
    pkill -f 'strata/serve/server.py|engine-cuda12/strata|engine-cuda12/strata-aggr|llama-server' >/dev/null 2>&1 || true
    if [[ -f "$CONFIG_BAK" ]]; then
        mv "$CONFIG_BAK" "$CONFIG_FILE"
    fi
}
trap cleanup EXIT

if [[ -f "$CONFIG_BAK" ]]; then
    echo "error: $CONFIG_BAK already exists; restore/remove it first" >&2
    exit 1
fi
cp "$CONFIG_FILE" "$CONFIG_BAK"

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

# write_cfg <exe> <extra_env: K=V;K=V;...> <extra_args: arg1 arg2 ...>
write_cfg() {
    local exe="$1"
    local extra_env="$2"
    local extra_args="$3"

    python3 - "$CONFIG_FILE" "$exe" "$VRAM_RESERVE_MIB" "$extra_env" "$extra_args" <<'PY'
import json
import sys

cfg_path, exe, reserve, extra_env, extra_args = sys.argv[1:]

with open(cfg_path, encoding="utf-8") as f:
    cfg = json.load(f)

cfg["exe"] = exe

args = list(cfg.get("args") or [])
if "--vram-reserve-mib" in args:
    i = args.index("--vram-reserve-mib")
    if i + 1 < len(args):
        args[i + 1] = reserve
    else:
        args.append(reserve)
else:
    args += ["--vram-reserve-mib", reserve]

# remove any extra args from a previous arm, then append this arm's
for token in ("--aux-cpus", "--control-vector-scaled", "--control-vector-layer-range", "--cvec-mode", "--cvec-dir"):
    if token in args:
        i = args.index(token)
        n = {"--aux-cpus": 1, "--control-vector-scaled": 1, "--control-vector-layer-range": 2,
             "--cvec-mode": 1, "--cvec-dir": 1}[token]
        del args[i:i + n + 1]

extra = extra_args.split() if extra_args else []
cfg["args"] = args + extra

env = dict(cfg.get("env") or {})
# reset the swept vars to the production baseline, then apply this arm's overrides.
# (CONFIG_FILE is rewritten in place per arm, so a previous arm's value would leak forward.)
for k in ("STRATA_ADAPT_OVERLAP", "STRATA_ADAPT_LAG", "STRATA_PCIE_BALANCE"):
    env.pop(k, None)
env["STRATA_ROUTE_TAIL_SKIP"] = "0"   # production baseline; the route_tail_skip1 arm overrides it
env["STRATA_ATTN_PRE75"] = "1"        # production baseline; the attn_pre75_off arm overrides it
if extra_env:
    for kv in extra_env.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v
cfg["env"] = env

with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=1)
    f.write("\n")
PY
}

start_server() {
    local log="$1"
    stop_and_wait
    (
        cd "$REPO_ROOT/strata"
        exec numactl --cpunodebind=1 "$REPO_ROOT/strata/.venv/bin/python" "$REPO_ROOT/strata/serve/server.py" --engine strata --config "$CONFIG_FILE" --port "$PORT"
    ) >"$log" 2>&1 &

    local i
    for i in $(seq 1 240); do
        if curl -fsS "$HEALTH_URL" 2>/dev/null | rg -q '"loaded": true'; then
            return 0
        fi
        sleep 1
    done
    return 1
}

bench_requests() {
    local label="$1"
    local out_json="$2"
    python3 - "$CHAT_URL" "$PROMPT_FILE" "$WARMUP" "$REPS" "$MAX_TOKENS" "$label" "$out_json" <<'PY'
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
label = sys.argv[6]
out_path = pathlib.Path(sys.argv[7])

def one(marker):
    body = {
        "model": "qwen3.8-flash-next-unsloth-ud-iq4_xs",
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

printf "label,decode_tps_avg,decode_tps_med,prompt_tps_avg,prompt_tps_med,draft_accept_rate,draft_n_total,draft_accepted_total\n" > "$SUMMARY_CSV"
make_prompt

# arm format: label|extra_env|extra_args   (empty field = none)
declare -a ARMS=()
ARMS+=("baseline||")
ARMS+=("attn_pre75_off|STRATA_ATTN_PRE75=0|")
ARMS+=("adapt_overlap|STRATA_ADAPT_OVERLAP=1|")
ARMS+=("adapt_lag2|STRATA_ADAPT_LAG=2|")
ARMS+=("pcie_balance|STRATA_PCIE_BALANCE=1|")
ARMS+=("route_tail_skip1|STRATA_ROUTE_TAIL_SKIP=1|")
ARMS+=("aux_cpus_auto||--aux-cpus auto")
ARMS+=("esp_cvec||--control-vector-scaled $ESP_GGUF:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer")

for arm in "${ARMS[@]}"; do
    IFS='|' read -r label extra_env extra_args <<< "$arm"
    echo "[quickbench2] arm: $label (env='$extra_env' args='$extra_args')"
    write_cfg "$ENGINE_BASE" "$extra_env" "$extra_args"
    if ! start_server "$RUN_DIR/$label.server.log"; then
        echo "[quickbench2] startup failed: $label" | tee -a "$RUN_DIR/failures.log"
        stop_and_wait
        continue
    fi

    # record whether the arm's feature actually engaged (the engine prints why not)
    grep -hE 'adaptive overlap|STRATA_ADAPT|pcie balance|aux-cpus|control vector|cvec' "$RUN_DIR/$label.server.log" 2>/dev/null | head -6 > "$RUN_DIR/$label.engaged.txt" || true

    out_json="$RUN_DIR/$label.json"
    if bench_requests "$label" "$out_json" >/dev/null 2>>"$RUN_DIR/failures.log"; then
        python3 - "$out_json" >> "$SUMMARY_CSV" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    s = json.load(f)
print(f"{s['label']},{s['decode_tps_avg']},{s['decode_tps_med']},{s['prompt_tps_avg']},{s['prompt_tps_med']},{s['draft_accept_rate']},{s['draft_n_total']},{s['draft_accepted_total']}")
PY
    else
        echo "[quickbench2] request failed: $label" | tee -a "$RUN_DIR/failures.log"
    fi

    stop_and_wait
done

echo "[quickbench2] done: $RUN_DIR"
echo "[quickbench2] summary: $SUMMARY_CSV"
