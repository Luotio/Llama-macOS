#!/usr/bin/env bash
# Experimental two-V100 tensor-parallel + MTP configuration.
# NCCL profile passed one 130048-token synthetic retrieval at 131072 context.
# Sustained reliability and real agent quality have not been established.
# Original V100 link logs very large numbers of correctable AER errors and
# the card thermally throttled at its 83 C maximum operating temperature.
# Tensor mode requires unquantized KV and has limited VRAM headroom.
# Experimental agent settings: batch 1536, ubatch 1024, 64 checkpoints,
# 2048-token checkpoint spacing, 16 GiB prompt snapshot limit and metrics.
# This is an unvalidated A/B test against the previous agent profile; set
# UBATCH_SIZE=1536 to restore that profile. The earlier measured settings can
# be restored with BATCH_SIZE=1024 UBATCH_SIZE=1024 CTX_CHECKPOINTS=32
# CHECKPOINT_MIN_STEP=8192 CACHE_RAM_MIB=4096.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

gpu1="GPU-e487578c-0be1-8c0b-ad8b-0ab0615c4850"
gpu2="GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c"

# Sanity check
nvidia-smi -i "$gpu1" --query-gpu=uuid --format=csv,noheader >/dev/null
nvidia-smi -i "$gpu2" --query-gpu=uuid --format=csv,noheader >/dev/null

export CUDA_VISIBLE_DEVICES="$gpu1,$gpu2"
export GGML_CUDA_ALLREDUCE="${GGML_CUDA_ALLREDUCE:-nccl}"

exec numactl --cpunodebind=1 --membind=1 \
  "${SERVER_BIN:-$repo/llama-cpp/build-v100-nccl/bin/llama-server}" \
  --model "$repo/local/models/Qwen3.8-27B-UD-Q4_K_XL.gguf" \
  --alias "Michael" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" \
  --ctx-size "${CTX_SIZE:-131072}" \
  --n-gpu-layers 999 \
  --split-mode tensor \
  --tensor-split 1,1 \
  --fit off \
  --flash-attn on \
  --batch-size "${BATCH_SIZE:-2048}" \
  --ubatch-size "${UBATCH_SIZE:-512}" \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-type-k f16 \
  --spec-draft-type-v f16 \
  --parallel 1 \
  --threads 14 \
  --jinja \
  --kv-unified \
  --metrics \
  --ctx-checkpoints "${CTX_CHECKPOINTS:-32}" \
  --checkpoint-min-step "${CHECKPOINT_MIN_STEP:-4096}" \
  --cache-ram "${CACHE_RAM_MIB:-16384}" \
  --load-mode none \
  "$@"