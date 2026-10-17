#!/usr/bin/env bash
# Serve Qwen3.8-27B (UD-Q4_K_XL) on one V100-SXM2-16GB for coding-agent workloads.
#
#   - 128k context, NGL=45 layers on GPU (server ceiling; 47+ OOMs with the
#     MTP draft context), rest on CPU
#   - MTP speculative decoding (--spec-type draft-mtp, n_max=3: best in bench,
#     see local/bench/RESULTS.md)
#   - q4_0 KV cache so 128k fits into 16GB VRAM
#   - flash attention on (required for quantized KV in this build)
#
# Usage:
#   local/serve-qwen3-27b.sh [path/to/model.gguf]
#
# Env overrides: MODEL HOST PORT CTX NGL TENSOR_SPLIT CACHE_K CACHE_V
#                SPEC_TYPE DRAFT_MAX PARALLEL CACHE_REUSE CUDA_VISIBLE_DEVICES
#   NGL=all (default) puts every layer on GPU; NGL=auto lets llama.cpp fit
#   layers to free VRAM and offload the rest to CPU (single-GPU mode).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="$REPO_ROOT/llama-cpp/build/bin/llama-server"

MODEL="${1:-${MODEL:-$REPO_ROOT/local/models/Qwen3.8-27B-UD-Q4_K_XL.gguf}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CTX="${CTX:-131072}"
NGL="${NGL:-45}"
TENSOR_SPLIT="${TENSOR_SPLIT:-1}"
CACHE_K="${CACHE_K:-q4_0}"
CACHE_V="${CACHE_V:-q4_0}"
SPEC_TYPE="${SPEC_TYPE:-draft-mtp}"
DRAFT_MAX="${DRAFT_MAX:-3}"
PARALLEL="${PARALLEL:-1}"
CACHE_REUSE="${CACHE_REUSE:-256}"

# Address the GPU by UUID: CUDA device indices do NOT match nvidia-smi
# indices on this box (numeric 1,2 would pull in the desktop GTX 1080).
# Default is the single healthy V100; the second card is faulty (RMA pending,
# see local/HARDWARE.md) and must not be used.
V100_UUIDS="GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$V100_UUIDS}"

[[ -x "$SERVER" ]] || { echo "error: $SERVER not found, run local/build.sh first" >&2; exit 1; }
[[ -f "$MODEL" ]]  || { echo "error: model not found: $MODEL (pass path as \$1 or set MODEL)" >&2; exit 1; }

echo "model  : $MODEL"
echo "gpus   : $CUDA_VISIBLE_DEVICES (tensor split $TENSOR_SPLIT, ngl $NGL)"
echo "ctx    : $CTX, KV cache K=$CACHE_K V=$CACHE_V"
echo "spec   : type=$SPEC_TYPE, n_max=$DRAFT_MAX"
echo "listen : http://$HOST:$PORT"

exec "$SERVER" \
    -m "$MODEL" \
    -c "$CTX" \
    -ngl "$NGL" \
    -ts "$TENSOR_SPLIT" \
    -fa on \
    -ctk "$CACHE_K" \
    -ctv "$CACHE_V" \
    --spec-type "$SPEC_TYPE" \
    --spec-draft-n-max "$DRAFT_MAX" \
    --jinja \
    --cache-reuse "$CACHE_REUSE" \
    --parallel "$PARALLEL" \
    --batch-size 2048 \
    --ubatch-size 512 \
    --host "$HOST" \
    --port "$PORT"
