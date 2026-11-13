#!/usr/bin/env bash
# Experimental two-V100 Coder profile; see local/bench/RESULTS.md.
# This retains aggressive route-tail skipping and is not yet a quality-validated default.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
expected_1="GPU-e487578c-0be1-8c0b-ad8b-0ab0615c4850"
expected_2="GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c"
actual_1="$(nvidia-smi -i "$expected_1" --query-gpu=index --format=csv,noheader,nounits)"
actual_2="$(nvidia-smi -i "$expected_2" --query-gpu=index --format=csv,noheader,nounits)"
if [[ "$actual_1" != 1 || "$actual_2" != 2 ]]; then
    echo "GPU indices changed: $expected_1=$actual_1, $expected_2=$actual_2; update the dual config first." >&2
    exit 1
fi
cd "$repo/strata"
exec numactl --cpunodebind=1 "$repo/strata/.venv/bin/python" "$repo/strata/serve/server.py" \
    --engine strata --config "$repo/local/strata/strata-coder-iq1_m-dual.json" --port "${PORT:-8080}"
