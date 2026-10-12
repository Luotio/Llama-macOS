#!/usr/bin/env bash
# Serve Qwen3.8-27B (UD-Q4_K_XL) on 2x V100-SXM2-16GB for coding-agent workloads.
#
#   - 128k context, all layers on GPU, tensor split across both V100s
#   - MTP speculative decoding (--spec-type draft-mtp, module lives in the GGUF)
#   - quantized KV cache so 128k fits into 32GB total VRAM
#   - flash attention on
#
# Usage:
#   local/serve-qwen3-27b.sh [path/to/model.gguf]
#
# Env overrides: MODEL HOST PORT CTX TENSOR_SPLIT CACHE_K CACHE_V
#                DRAFT_MAX PARALLEL CACHE_REUSE CUDA_VISIBLE_DEVICES
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="$REPO_ROOT/build/bin/llama-server"

MODEL="${1:-${MODEL:-$REPO_ROOT/local/models/Qwen3.8-27B-UD-Q4_K_XL.gguf}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CTX="${CTX:-131072}"
TENSOR_SPLIT="${TENSOR_SPLIT:-1,1}"
CACHE_K="${CACHE_K:-q8_0}"
CACHE_V="${CACHE_V:-q8_0}"
DRAFT_MAX="${DRAFT_MAX:-3}"
PARALLEL="${PARALLEL:-1}"
CACHE_REUSE="${CACHE_REUSE:-256}"

# Address the V100s by UUID: CUDA device indices do NOT match nvidia-smi
# indices on this box (numeric 1,2 would pull in the desktop GTX 1080).
V100_UUIDS="GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c,GPU-f07f3af3-5f02-f364-0d37-2ab3a4c4afce"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$V100_UUIDS}"

[[ -x "$SERVER" ]] || { echo "error: $SERVER not found, run local/build.sh first" >&2; exit 1; }
[[ -f "$MODEL" ]]  || { echo "error: model not found: $MODEL (pass path as \$1 or set MODEL)" >&2; exit 1; }

echo "model  : $MODEL"
echo "gpus   : $CUDA_VISIBLE_DEVICES (tensor split $TENSOR_SPLIT)"
echo "ctx    : $CTX, KV cache K=$CACHE_K V=$CACHE_V"
echo "mtp    : draft-mtp, n_max=$DRAFT_MAX"
echo "listen : http://$HOST:$PORT"

exec "$SERVER" \
    -m "$MODEL" \
    -c "$CTX" \
    -ngl all \
    -ts "$TENSOR_SPLIT" \
    -fa on \
    -ctk "$CACHE_K" \
    -ctv "$CACHE_V" \
    --spec-type draft-mtp \
    --spec-draft-n-max "$DRAFT_MAX" \
    --jinja \
    --cache-reuse "$CACHE_REUSE" \
    --parallel "$PARALLEL" \
    --batch-size 2048 \
    --ubatch-size 512 \
    --host "$HOST" \
    --port "$PORT"
