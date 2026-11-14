#!/usr/bin/env bash
# Qwen3.8-27B, two V100s, llama.cpp. Measurements: local/bench/RESULTS.md.
# The original V100 link reports many correctable PCIe errors under load.
# 130048 prompt tokens plus 22 generated tokens passed a retrieval smoke test;
# this is not a full agent-quality or sustained-reliability validation.
# Local-only by default. Override HOST, PORT, THREADS, SPEC_TYPE, DRAFT_MAX,
# CACHE_RAM_MIB or TENSOR_SPLIT. GPU UUIDs deliberately exclude the GTX 1080.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu1="GPU-e487578c-0be1-8c0b-ad8b-0ab0615c4850"
gpu2="GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c"
nvidia-smi -i "$gpu1" --query-gpu=uuid --format=csv,noheader >/dev/null
nvidia-smi -i "$gpu2" --query-gpu=uuid --format=csv,noheader >/dev/null
export CUDA_VISIBLE_DEVICES="$gpu1,$gpu2"
exec numactl --cpunodebind=1 "$repo/build/bin/llama-server" \
    --model "$repo/local/models/Qwen3.8-27B-UD-Q4_K_XL.gguf" \
    --ctx-size 131072 --n-gpu-layers 999 --split-mode layer \
    --tensor-split "${TENSOR_SPLIT:-60,40}" --batch-size 1024 --ubatch-size 512 \
    --parallel 1 --cont-batching --flash-attn on --jinja --metrics \
    --threads "${THREADS:-8}" --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" \
    --cache-type-k q8_0 --cache-type-v q8_0 --kv-unified --cache-ram "${CACHE_RAM_MIB:-4096}" \
    --load-mode none --temp 0.4 --top-k 40 --top-p 0.9 --min-p 0.05 \
    --repeat-penalty 1.05 --presence-penalty 0.1 \
    --spec-type "${SPEC_TYPE:-draft-mtp}" --spec-draft-n-max "${DRAFT_MAX:-2}" \
    --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 \
    --cache-prompt --cache-reuse 256 --no-cache-idle-slots "$@"
