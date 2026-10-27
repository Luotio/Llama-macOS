#!/usr/bin/env bash
# Reliable Strata V100 A/B benchmark for this machine.
#
# Compares documented Volta switches and optional aggressive build, waiting for
# full GPU memory release between runs to avoid false startup failures.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="$REPO_ROOT/local/strata/strata-unsloth-ud-iq4_xs.json"
CONFIG_BAK="$REPO_ROOT/local/strata/strata-unsloth-ud-iq4_xs.json.quickbench.bak"
ENGINE_BASE="$REPO_ROOT/strata/engine-cuda12/strata"
ENGINE_AGGR="$REPO_ROOT/strata/engine-cuda12/strata-aggr"

GPU_UUID="${GPU_UUID:-GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c}"
PORT="${PORT:-8091}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
CHAT_URL="http://127.0.0.1:${PORT}/v1/chat/completions"

VRAM_RESERVE_MIB="${VRAM_RESERVE_MIB:-768}"
PROMPT_REPEATS="${PROMPT_REPEATS:-40}"
REPS="${REPS:-2}"
MAX_TOKENS="${MAX_TOKENS:-96}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$REPO_ROOT/local/bench/results/strata-v100-quickbench-$TS"
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

write_cfg() {
    local exe="$1"
    local sm70="$2"
    local bf16="$3"
    local prompt_old="$4"
    local attn_pre75="$5"
    local exp_mode="$6"
    local mmvq="$7"
    local gr_fast="$8"

    python3 - "$CONFIG_FILE" "$exe" "$VRAM_RESERVE_MIB" "$sm70" "$bf16" "$prompt_old" "$attn_pre75" "$exp_mode" "$mmvq" "$gr_fast" <<'PY'
import json
import sys

cfg_path, exe, reserve, sm70, bf16, prompt_old, attn_pre75, exp_mode, mmvq, gr_fast = sys.argv[1:]

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
cfg["args"] = args

env = dict(cfg.get("env") or {})
env["STRATA_SM70_TABLE"] = sm70
env["STRATA_BF16_TC"] = bf16
env["STRATA_PROMPT_ATTN_OLD"] = prompt_old
env["STRATA_ATTN_PRE75"] = attn_pre75
env["STRATA_EXP_MODE"] = exp_mode
env["STRATA_MMVQ_IL"] = mmvq
env["STRATA_GR_FAST"] = gr_fast
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
    python3 - "$CHAT_URL" "$PROMPT_FILE" "$REPS" "$MAX_TOKENS" "$label" "$out_json" <<'PY'
import json
import pathlib
import statistics
import sys
import time
import urllib.request

url = sys.argv[1]
prompt = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")
reps = int(sys.argv[3])
max_tokens = int(sys.argv[4])
label = sys.argv[5]
out_path = pathlib.Path(sys.argv[6])

rows = []
for i in range(reps):
    body = {
        "model": "qwen3.8-flash-next-unsloth-ud-iq4_xs",
        "messages": [
            {"role": "system", "content": "You are a performance engineer."},
            {"role": "user", "content": prompt + f"\n\nmarker {i+1}"},
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
    rows.append({
        "wall_s": round(wall, 3),
        "prompt_tps": float(t.get("prompt_per_second", 0.0) or 0.0),
        "decode_tps": float(t.get("predicted_per_second", 0.0) or 0.0),
        "draft_n": int(t.get("draft_n", 0) or 0),
        "draft_accepted": int(t.get("draft_n_accepted", 0) or 0),
        "prompt_n": int(t.get("prompt_n", 0) or 0),
        "pred_n": int(t.get("predicted_n", 0) or 0),
    })

decode = [r["decode_tps"] for r in rows]
prompt_tps = [r["prompt_tps"] for r in rows]
dn = sum(r["draft_n"] for r in rows)
da = sum(r["draft_accepted"] for r in rows)
summary = {
    "label": label,
    "reps": reps,
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

declare -a ARMS=()
ARMS+=("baseline|$ENGINE_BASE|1|1|0|1|1|1|1")
ARMS+=("sm70_table_off|$ENGINE_BASE|0|1|0|1|1|1|1")
ARMS+=("bf16_tc_off|$ENGINE_BASE|1|0|0|1|1|1|1")
ARMS+=("prompt_attn_old_on|$ENGINE_BASE|1|1|1|1|1|1|1")
ARMS+=("attn_pre75_off|$ENGINE_BASE|1|1|0|0|1|1|1")
ARMS+=("exp_mode_off|$ENGINE_BASE|1|1|0|1|0|1|1")
ARMS+=("mmvq_il_off|$ENGINE_BASE|1|1|0|1|1|0|1")
ARMS+=("gr_fast_off|$ENGINE_BASE|1|1|0|1|1|1|0")
if [[ -x "$ENGINE_AGGR" ]]; then
    ARMS+=("aggr_binary|$ENGINE_AGGR|1|1|0|1|1|1|1")
fi

for arm in "${ARMS[@]}"; do
    IFS='|' read -r label exe sm70 bf16 prompt_old attn_pre75 exp_mode mmvq gr_fast <<< "$arm"
    echo "[quickbench] arm: $label"
    write_cfg "$exe" "$sm70" "$bf16" "$prompt_old" "$attn_pre75" "$exp_mode" "$mmvq" "$gr_fast"
    if ! start_server "$RUN_DIR/$label.server.log"; then
        echo "[quickbench] startup failed: $label" | tee -a "$RUN_DIR/failures.log"
        stop_and_wait
        continue
    fi

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
        echo "[quickbench] request failed: $label" | tee -a "$RUN_DIR/failures.log"
    fi

    stop_and_wait
done

echo "[quickbench] done: $RUN_DIR"
echo "[quickbench] summary: $SUMMARY_CSV"
