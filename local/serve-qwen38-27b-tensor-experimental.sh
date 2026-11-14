#!/usr/bin/env bash
# Experimental two-V100 tensor-parallel + MTP configuration.
# NCCL profile passed one 130048-token synthetic retrieval at 131072 context.
# Sustained reliability and real agent quality have not been established.
# Original V100 link logs very large numbers of correctable AER errors and
# the card thermally throttled at its 83 C maximum operating temperature.
# Tensor mode requires unquantized KV and has limited VRAM headroom.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu1="GPU-e487578c-0be1-8c0b-ad8b-0ab0615c4850"
gpu2="GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c"
nvidia-smi -i "$gpu1" --query-gpu=uuid --format=csv,noheader >/dev/null
nvidia-smi -i "$gpu2" --query-gpu=uuid --format=csv,noheader >/dev/null
export CUDA_VISIBLE_DEVICES="$gpu1,$gpu2"
export GGML_CUDA_ALLREDUCE="${GGML_CUDA_ALLREDUCE:-nccl}"
exec numactl --cpunodebind=1 "${SERVER_BIN:-$repo/llama-cpp/build-v100-nccl/bin/llama-server}" \
    --model "$repo/local/models/Qwen3.8-27B-UD-Q4_K_XL.gguf" \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" \
    --ctx-size "${CTX_SIZE:-131072}" --n-gpu-layers 999 --split-mode tensor \
    --tensor-split 1,1 --fit off --flash-attn on --batch-size 1024 --ubatch-size 1024 \
    --cache-type-k f16 --cache-type-v f16 --spec-type draft-mtp \
    --spec-draft-n-max 2 --spec-draft-type-k f16 --spec-draft-type-v f16 \
    --parallel 1 --threads 8 --jinja --kv-unified --cache-ram 4096 --load-mode none "$@"
