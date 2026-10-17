#!/usr/bin/env bash
# Sample V100 temps/power/clocks/VRAM to a CSV log while benchmarks run.
#
# Usage: temp-log.sh <output.csv> [interval_s]
# Stop with Ctrl-C or kill; the header is written once at start.
set -euo pipefail

OUT="${1:?usage: temp-log.sh <output.csv> [interval_s]}"
INTERVAL="${2:-5}"

V100_UUIDS="GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c,GPU-f07f3af3-5f02-f364-0d37-2ab3a4c4afce"

mkdir -p "$(dirname "$OUT")"
echo "timestamp,gpu_index,uuid,temperature_c,power_w,clocks_sm_mhz,mem_used_mb" > "$OUT"

while true; do
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi -i "$V100_UUIDS" \
        --query-gpu=index,uuid,temperature.gpu,power.draw,clocks.sm,memory.used \
        --format=csv,noheader,nounits | while IFS=',' read -r idx uuid temp power clock mem; do
        echo "$ts,$idx,$uuid,$temp,$power,$clock,$mem" >> "$OUT"
    done
    sleep "$INTERVAL"
done
