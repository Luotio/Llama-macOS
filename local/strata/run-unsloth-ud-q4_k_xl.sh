#!/bin/sh
# Both V100s are on NUMA node 1; memory can fall back to node 0.
# The split and cache sizes are based on Coder measurements, not Q4 benchmarks.
set -eu
repo="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
config="$repo/local/strata/strata-unsloth-ud-q4_k_xl.json"
python="$repo/strata/.venv/bin/python"
expected_1="GPU-e487578c-0be1-8c0b-ad8b-0ab0615c4850"
expected_2="GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c"

# The config names physical nvidia-smi indices, so check both its order and
# the UUID at each index before the server maps them into CUDA device order.
if ! "$python" -c 'import json, sys; c = json.load(open(sys.argv[1])); sys.exit(0 if c.get("gpu") == [1, 2] and c.get("gpu_order") == "as_given" else 1)' "$config"; then
	echo "Unexpected GPU order in $config; update the launcher guard and config together." >&2
	exit 1
fi
actual_1="$(nvidia-smi -i "$expected_1" --query-gpu=index --format=csv,noheader,nounits)"
actual_2="$(nvidia-smi -i "$expected_2" --query-gpu=index --format=csv,noheader,nounits)"
if [ "$actual_1" != 1 ] || [ "$actual_2" != 2 ]; then
	echo "GPU indices changed: $expected_1=$actual_1, $expected_2=$actual_2; update the launcher guard and config together." >&2
	exit 1
fi

cd "$repo/strata"
exec numactl --cpunodebind=1 "$python" "$repo/strata/serve/server.py" \
	--engine strata --config "$config" --port "${PORT:-8080}" --open
