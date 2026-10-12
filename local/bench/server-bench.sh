#!/usr/bin/env bash
# Run SPEED-Bench (tools/server/bench/speed-bench) against a running llama-server
# and record a manifest row with environment metadata.
#
# Measures what the client actually sees: prefill/decode throughput, latency
# and MTP draft acceptance rate. Use this tier for A/B testing speculative
# decoding (tier 1, run-bench.sh, cannot exercise MTP).
#
# One-time setup (portage):
#   emerge --ask dev-python/datasets dev-python/tqdm
# First run downloads nvidia/SPEED-Bench from huggingface.co (cached afterwards).
#
# Usage:
#   local/bench/server-bench.sh <label> [extra speed_bench.py args...]
#
# Env overrides: URL SB_BENCH SB_CATEGORY OSL LIMIT CONCURRENCY CUDA_VISIBLE_DEVICES
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="$REPO_ROOT/local/bench"
RESULTS_DIR="$BENCH_DIR/results"
MANIFEST="$RESULTS_DIR/server-runs.csv"
SPEED_BENCH="$REPO_ROOT/tools/server/bench/speed-bench/speed_bench.py"

URL="${URL:-http://127.0.0.1:8080/v1}"
SB_BENCH="${SB_BENCH:-qualitative}"
SB_CATEGORY="${SB_CATEGORY:-coding}"
OSL="${OSL:-512}"
LIMIT="${LIMIT:-8}"
CONCURRENCY="${CONCURRENCY:-1}"

# UUIDs: CUDA device indices do NOT match nvidia-smi indices on this box
V100_UUIDS="GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c,GPU-f07f3af3-5f02-f364-0d37-2ab3a4c4afce"
GPUS="${CUDA_VISIBLE_DEVICES:-$V100_UUIDS}"

LABEL="${1:?usage: server-bench.sh <label> [extra speed_bench.py args...]}"
shift

python3 -c "import datasets, tqdm" 2>/dev/null || {
    echo "error: missing python deps, run: emerge --ask dev-python/datasets dev-python/tqdm" >&2
    exit 1
}

health_url="${URL%/v1}/health"
curl -fsS --max-time 5 "$health_url" >/dev/null || {
    echo "error: server not reachable at $health_url (start it with local/serve-qwen3-27b.sh)" >&2
    exit 1
}

snap() {
    nvidia-smi -i "$GPUS" --query-gpu="$1" --format=csv,noheader,nounits | tr '\n' ';' | sed 's/;$//'
}

timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
temp_start="$(snap temperature.gpu)"
power_limit="$(snap power.limit)"
clocks_start="$(snap clocks.sm)"

echo "label : $LABEL"
echo "url   : $URL"
echo "bench : $SB_BENCH, category=$SB_CATEGORY, osl=$OSL, limit=$LIMIT"
echo "power : $power_limit W, temp start: $temp_start C"

mkdir -p "$RESULTS_DIR"
json_out="$RESULTS_DIR/$LABEL-$timestamp.json"

python3 "$SPEED_BENCH" \
    --url "$URL" \
    --bench "$SB_BENCH" \
    --category "$SB_CATEGORY" \
    --osl "$OSL" \
    --limit "$LIMIT" \
    --concurrency "$CONCURRENCY" \
    --output "$json_out" \
    "$@"

temp_end="$(snap temperature.gpu)"
clocks_end="$(snap clocks.sm)"

meta=$(printf '{"timestamp":"%s","label":"%s","url":"%s","bench":"%s","category":"%s","osl":"%s","limit":"%s","concurrency":"%s","power_limit_w":"%s","temp_start_c":"%s","temp_end_c":"%s","clocks_start_mhz":"%s","clocks_end_mhz":"%s","json_file":"%s"}' \
    "$timestamp" "$LABEL" "$URL" "$SB_BENCH" "$SB_CATEGORY" "$OSL" "$LIMIT" "$CONCURRENCY" \
    "$power_limit" "$temp_start" "$temp_end" "$clocks_start" "$clocks_end" "$json_out")

python3 - "$MANIFEST" "$meta" <<'PY'
import csv, json, os, sys

csv_path, meta_json = sys.argv[1], sys.argv[2]
meta = json.loads(meta_json)

cols = ["timestamp", "label", "url", "bench", "category", "osl", "limit",
        "concurrency", "power_limit_w", "temp_start_c", "temp_end_c",
        "clocks_start_mhz", "clocks_end_mhz", "json_file"]

new_file = not os.path.exists(csv_path)
with open(csv_path, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if new_file:
        w.writerow(cols)
    w.writerow([meta[c] for c in cols])

print(f"recorded run to {csv_path}")
PY

echo "raw output: $json_out"
