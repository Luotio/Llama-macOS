#!/usr/bin/env bash
# Strata V100 autotune harness for this host.
#
# Runs controlled A/B trials over documented Volta toggles and writes JSON
# summaries to local/bench/results. It can also benchmark baseline vs an
# aggressive CUDA12 build if both binaries are present.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="$REPO_ROOT/local/bench/results"
CONFIG_FILE="$REPO_ROOT/local/strata/strata-unsloth-ud-iq4_xs.json"
CONFIG_BAK="$REPO_ROOT/local/strata/strata-unsloth-ud-iq4_xs.json.autotune.bak"
ENGINE_BASE="$REPO_ROOT/strata/engine-cuda12/strata"
ENGINE_AGGR="$REPO_ROOT/strata/engine-cuda12/strata-aggr"

BENCH_PORT="${BENCH_PORT:-8091}"
HEALTH_URL="http://127.0.0.1:${BENCH_PORT}/health"
CHAT_URL="http://127.0.0.1:${BENCH_PORT}/v1/chat/completions"
BENCH_VRAM_RESERVE_MIB="${BENCH_VRAM_RESERVE_MIB:-768}"

# Runtime defaults: tuned for repeatability and moderate runtime.
REPS="${REPS:-3}"
MAX_TOKENS="${MAX_TOKENS:-192}"
PROMPT_MULTIPLIER="${PROMPT_MULTIPLIER:-24}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$RESULTS_DIR/strata-v100-autotune-$TS"

cleanup() {
    pkill -f 'strata/serve/server.py|engine-cuda12/strata|engine-cuda12/strata-aggr' >/dev/null 2>&1 || true
    if [[ -f "$CONFIG_BAK" ]]; then
        mv "$CONFIG_BAK" "$CONFIG_FILE"
    fi
}
trap cleanup EXIT

need() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "error: missing dependency: $1" >&2
        exit 1
    }
}

need python3
need curl
need numactl
need rg

mkdir -p "$RUN_DIR"

if [[ -f "$CONFIG_BAK" ]]; then
    echo "error: backup config already exists: $CONFIG_BAK" >&2
    echo "hint: restore or remove it before running autotune again" >&2
    exit 1
fi

cp "$CONFIG_FILE" "$CONFIG_BAK"

make_prompt_file() {
    local path="$1"
    python3 - "$path" "$PROMPT_MULTIPLIER" <<'PY'
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
mult = int(sys.argv[2])
block = (
    "You are optimizing a Volta V100 + dual Xeon local inference stack. "
    "Analyze tradeoffs between expert cache hit rate, PCIe pressure, prompt prefill, "
    "and speculative decode acceptance. Provide a concrete plan with numbered steps, "
    "possible regressions, and validation checks."
)
text = "\n\n".join([block for _ in range(mult)])
out.write_text(text, encoding="utf-8")
PY
}

wait_ready() {
    local i
    for i in $(seq 1 180); do
        if curl -fsS "$HEALTH_URL" | rg -q '"loaded": true'; then
            return 0
        fi
        sleep 1
    done
    echo "error: Strata did not become ready in time" >&2
    return 1
}

start_server() {
    local log="$1"
    pkill -f 'strata/serve/server.py|engine-cuda12/strata|engine-cuda12/strata-aggr' >/dev/null 2>&1 || true
    (
        cd "$REPO_ROOT/strata"
        exec numactl --cpunodebind=1 "$REPO_ROOT/strata/.venv/bin/python" "$REPO_ROOT/strata/serve/server.py" --engine strata --config "$CONFIG_FILE" --port "$BENCH_PORT"
    ) >"$log" 2>&1 &
    wait_ready
}

write_config() {
    local exe="$1"
    local sm70="$2"
    local bf16tc="$3"
    local prompt_old="$4"
    local attn_pre75="$5"
    local exp_mode="$6"
    local mmvq_il="$7"
    local gr_fast="$8"

    python3 - "$CONFIG_FILE" "$exe" "$sm70" "$bf16tc" "$prompt_old" "$attn_pre75" "$exp_mode" "$mmvq_il" "$gr_fast" "$BENCH_VRAM_RESERVE_MIB" <<'PY'
import json
import sys

cfg_path = sys.argv[1]
exe = sys.argv[2]
sm70 = sys.argv[3]
bf16 = sys.argv[4]
prompt_old = sys.argv[5]
attn_pre75 = sys.argv[6]
exp_mode = sys.argv[7]
mmvq = sys.argv[8]
gr_fast = sys.argv[9]
reserve_mib = sys.argv[10]

with open(cfg_path, encoding="utf-8") as f:
    cfg = json.load(f)

cfg["exe"] = exe

args = list(cfg.get("args") or [])
if "--vram-reserve-mib" in args:
    i = args.index("--vram-reserve-mib")
    if i + 1 < len(args):
        args[i + 1] = reserve_mib
    else:
        args.append(reserve_mib)
else:
    args += ["--vram-reserve-mib", reserve_mib]
cfg["args"] = args

env = dict(cfg.get("env") or {})
env["STRATA_SM70_TABLE"] = sm70
env["STRATA_BF16_TC"] = bf16
env["STRATA_PROMPT_ATTN_OLD"] = prompt_old
env["STRATA_ATTN_PRE75"] = attn_pre75
if exp_mode:
    env["STRATA_EXP_MODE"] = exp_mode
else:
    env.pop("STRATA_EXP_MODE", None)   # unset = the card's default (mode 8 on a V100 with STRATA_SM70_TABLE=1).
                                        # NEVER "1": mode 1 is the gfx906 64-lane layout and __trap()s on CUDA
env["STRATA_MMVQ_IL"] = mmvq
env["STRATA_GR_FAST"] = gr_fast
cfg["env"] = env

with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=1)
    f.write("\n")
PY
}

run_requests() {
    local label="$1"
    local prompt_file="$2"
    local out_json="$3"
    python3 - "$CHAT_URL" "$prompt_file" "$REPS" "$MAX_TOKENS" "$out_json" "$label" <<'PY'
import json
import pathlib
import statistics
import sys
import time
import urllib.request
import urllib.error

url = sys.argv[1]
prompt_file = pathlib.Path(sys.argv[2])
reps = int(sys.argv[3])
max_tokens = int(sys.argv[4])
out_json = pathlib.Path(sys.argv[5])
label = sys.argv[6]

prompt_text = prompt_file.read_text(encoding="utf-8")

def req(user_text):
    body = json.dumps({
        "model": "qwen3.8-flash-next-unsloth-ud-iq4_xs",
        "messages": [
            {"role": "system", "content": "You are an expert performance engineer."},
            {"role": "user", "content": user_text},
        ],
        "reasoning_effort": "none",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(request, timeout=1200) as r:
        data = json.loads(r.read())
    wall_s = time.time() - t0
    t = data.get("timings") or {}
    return {
        "wall_s": round(wall_s, 3),
        "prompt_n": t.get("prompt_n", 0),
        "prompt_tps": float(t.get("prompt_per_second", 0.0) or 0.0),
        "pred_n": t.get("predicted_n", 0),
        "decode_tps": float(t.get("predicted_per_second", 0.0) or 0.0),
        "draft_n": int(t.get("draft_n", 0) or 0),
        "draft_acc": int(t.get("draft_n_accepted", 0) or 0),
    }

rows = []
for i in range(reps):
    payload = prompt_text + f"\n\nRun marker {i+1}."
    last_err = None
    for _attempt in range(3):
        try:
            rows.append(req(payload))
            last_err = None
            break
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            last_err = RuntimeError(f"HTTP {e.code}: {body.strip()}")
            time.sleep(2)
        except Exception as e:
            last_err = e
            time.sleep(2)
    if last_err is not None:
        raise last_err

decode = [r["decode_tps"] for r in rows]
prompt = [r["prompt_tps"] for r in rows]
draft_n = sum(r["draft_n"] for r in rows)
draft_a = sum(r["draft_acc"] for r in rows)

summary = {
    "label": label,
    "reps": reps,
    "max_tokens": max_tokens,
    "prompt_multiplier": len(prompt_text),
    "decode_tps_avg": round(statistics.mean(decode), 3),
    "decode_tps_med": round(statistics.median(decode), 3),
    "prompt_tps_avg": round(statistics.mean(prompt), 3),
    "prompt_tps_med": round(statistics.median(prompt), 3),
    "draft_accept_rate": round((draft_a / draft_n), 4) if draft_n else None,
    "draft_n_total": draft_n,
    "draft_accepted_total": draft_a,
    "rows": rows,
}

out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
}

collect_metrics() {
    local out="$1"
    curl -fsS "http://127.0.0.1:${BENCH_PORT}/metrics" > "$out"
}

prompt_file="$RUN_DIR/prompt.txt"
make_prompt_file "$prompt_file"

# Each arm: name|exe|sm70|bf16|prompt_old|attn_pre75|exp_mode|mmvq_il|gr_fast
# exp_mode EMPTY = unset (the card's default: mode 8 on a V100 with STRATA_SM70_TABLE=1).
# Do NOT put "1" in any arm: mode 1 is the gfx906-only 64-lane layout; on CUDA it __trap()s
# (Xid 43 + sticky "unspecified launch failure" = "verify: layer N never rang", every request dies).
declare -a ARMS=()

ARMS+=("baseline|$ENGINE_BASE|1|1|0|1||1|1")
ARMS+=("sm70_table_off|$ENGINE_BASE|0|1|0|1||1|1")
ARMS+=("bf16_tc_off|$ENGINE_BASE|1|0|0|1||1|1")
ARMS+=("prompt_attn_old_on|$ENGINE_BASE|1|1|1|1||1|1")
ARMS+=("attn_pre75_off|$ENGINE_BASE|1|1|0|0||1|1")
ARMS+=("mmvq_il_off|$ENGINE_BASE|1|1|0|1||0|1")
ARMS+=("gr_fast_off|$ENGINE_BASE|1|1|0|1||1|0")

if [[ -x "$ENGINE_AGGR" ]]; then
    ARMS+=("aggr_binary_baseline_env|$ENGINE_AGGR|1|1|0|1|1|1|1")
fi

summary_csv="$RUN_DIR/summary.csv"
printf "label,decode_tps_avg,decode_tps_med,prompt_tps_avg,prompt_tps_med,draft_accept_rate,draft_n_total,draft_accepted_total\n" > "$summary_csv"

for arm in "${ARMS[@]}"; do
    IFS='|' read -r label exe sm70 bf16 prompt_old attn_pre75 exp_mode mmvq gr_fast <<< "$arm"
    echo "[autotune] running arm: $label"
    write_config "$exe" "$sm70" "$bf16" "$prompt_old" "$attn_pre75" "$exp_mode" "$mmvq" "$gr_fast"
    start_server "$RUN_DIR/$label.server.log"
    out_json="$RUN_DIR/$label.json"
    if run_requests "$label" "$prompt_file" "$out_json" >/dev/null; then
        collect_metrics "$RUN_DIR/$label.metrics.json"

        python3 - "$out_json" >> "$summary_csv" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    j = json.load(f)
print(
    f"{j['label']},{j['decode_tps_avg']},{j['decode_tps_med']},{j['prompt_tps_avg']},{j['prompt_tps_med']},"
    f"{j['draft_accept_rate']},{j['draft_n_total']},{j['draft_accepted_total']}"
)
PY
    else
        echo "[autotune] arm failed: $label" | tee -a "$RUN_DIR/failures.log"
        collect_metrics "$RUN_DIR/$label.metrics.json" || true
    fi

    pkill -f 'strata/serve/server.py|engine-cuda12/strata|engine-cuda12/strata-aggr' >/dev/null 2>&1 || true
done

echo "[autotune] complete: $RUN_DIR"
echo "[autotune] summary:  $summary_csv"
