#!/usr/bin/env bash
# Repeated llama-bench runner with environment metadata capture.
#
# Each run appends one row per test (pp###/tg###) to results/bench.csv so
# runs with different labels can be compared with compare.py.
#
# Usage:
#   local/bench/run-bench.sh <label> [extra llama-bench args...]
#
# Env overrides (defaults match the production serve config):
#   MODEL DEPTH PP TG REPS BATCH UBATCH CT_K CT_V NGL TS FA DELAY
#   CUDA_VISIBLE_DEVICES
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="$REPO_ROOT/local/bench"
RESULTS_DIR="$BENCH_DIR/results"
CSV="$RESULTS_DIR/bench.csv"
BIN="$REPO_ROOT/build/bin/llama-bench"

MODEL="${MODEL:-$REPO_ROOT/local/models/Qwen3.8-27B-UD-Q4_K_XL.gguf}"
DEPTH="${DEPTH:-131072}"
PP="${PP:-512}"
TG="${TG:-128}"
REPS="${REPS:-5}"
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-512}"
CT_K="${CT_K:-q8_0}"
CT_V="${CT_V:-q8_0}"
NGL="${NGL:-all}"
TS="${TS:-1,1}"
FA="${FA:-on}"
DELAY="${DELAY:-2}"

# UUIDs: CUDA device indices do NOT match nvidia-smi indices on this box
V100_UUIDS="GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c,GPU-f07f3af3-5f02-f364-0d37-2ab3a4c4afce"
GPUS="${CUDA_VISIBLE_DEVICES:-$V100_UUIDS}"
export CUDA_VISIBLE_DEVICES="$GPUS"

LABEL="${1:?usage: run-bench.sh <label> [extra llama-bench args...]}"
shift

[[ -x "$BIN" ]]   || { echo "error: $BIN not found, run local/build.sh first" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "error: model not found: $MODEL (set MODEL or put it in local/models/)" >&2; exit 1; }

snap() {
    nvidia-smi -i "$GPUS" --query-gpu="$1" --format=csv,noheader,nounits | tr '\n' ';' | sed 's/;$//'
}

timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
temp_start="$(snap temperature.gpu)"
power_limit="$(snap power.limit)"
clocks_start="$(snap clocks.sm)"

echo "label : $LABEL"
echo "model : $MODEL"
echo "gpus  : $CUDA_VISIBLE_DEVICES (split $TS, ngl $NGL)"
echo "tests : pp$PP tg$TG, reps=$REPS, depth=$DEPTH, KV $CT_K/$CT_V"
echo "power : $power_limit W, temp start: $temp_start C"

mkdir -p "$RESULTS_DIR"
jsonl="$RESULTS_DIR/$LABEL-$timestamp.jsonl"

args=(-m "$MODEL" -d "$DEPTH" -p "$PP" -n "$TG" -r "$REPS"
      -b "$BATCH" -ub "$UBATCH" -ctk "$CT_K" -ctv "$CT_V"
      -ts "$TS" -fa "$FA" --delay "$DELAY" -o jsonl)
if [[ "$NGL" != "all" ]]; then
    args+=(-ngl "$NGL")
fi

"$BIN" "${args[@]}" "$@" > "$jsonl"

temp_end="$(snap temperature.gpu)"
clocks_end="$(snap clocks.sm)"

meta=$(printf '{"timestamp":"%s","label":"%s","commit":"%s","power_limit_w":"%s","temp_start_c":"%s","temp_end_c":"%s","clocks_start_mhz":"%s","clocks_end_mhz":"%s"}' \
    "$timestamp" "$LABEL" "$commit" "$power_limit" "$temp_start" "$temp_end" "$clocks_start" "$clocks_end")

python3 - "$jsonl" "$CSV" "$meta" <<'PY'
import csv, json, os, sys

jsonl_path, csv_path, meta_json = sys.argv[1], sys.argv[2], sys.argv[3]
meta = json.loads(meta_json)

rows = []
with open(jsonl_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("test_time"):
            rows.append(row)

cols = ["timestamp", "label", "commit", "test", "n_prompt", "n_gen", "n_depth",
        "avg_ts", "stddev_ts", "type_k", "type_v", "n_batch", "n_ubatch",
        "flash_attn", "tensor_split", "power_limit_w", "temp_start_c",
        "temp_end_c", "clocks_start_mhz", "clocks_end_mhz"]

new_file = not os.path.exists(csv_path)
with open(csv_path, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if new_file:
        w.writerow(cols)
    for r in rows:
        w.writerow([meta["timestamp"], meta["label"], meta["commit"], r["test_time"],
                    r.get("n_prompt"), r.get("n_gen"), r.get("n_depth"),
                    r.get("avg_ts"), r.get("stddev_ts"),
                    r.get("type_k"), r.get("type_v"), r.get("n_batch"), r.get("n_ubatch"),
                    r.get("flash_attn"), r.get("tensor_split"),
                    meta["power_limit_w"], meta["temp_start_c"], meta["temp_end_c"],
                    meta["clocks_start_mhz"], meta["clocks_end_mhz"]])

print(f"recorded {len(rows)} test row(s) to {csv_path}")
PY

echo "raw output: $jsonl"
