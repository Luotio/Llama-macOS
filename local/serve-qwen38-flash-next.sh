#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next (UD-IQ4_XS, ~94 GB) on one V100 + system RAM.
#
#   - MoE 125B params / 6B active: routed experts (~60 GB) stay in CPU RAM via
#     --cpu-moe; the PLE n-gram table (~25 GB, lazy row reads via mmap) is
#     forced to CPU with a tensor override; everything else (attention, SSM,
#     indexer, HC, shared experts, embeddings, ~4-5 GB) goes to the V100
#   - 128k context, q4_0 KV cache, flash attention on
#   - MTP speculative decoding: the unsloth UD-IQ4_XS main file has no nextn
#     tensors, so a self-contained sidecar (mtp-...-Q8_0.gguf) is loaded via
#     --spec-draft-model; set DRAFT_MODEL="" to disable MTP
#
# Usage:
#   local/serve-qwen38-flash-next.sh [path/to/model.gguf]
#
# Env overrides: MODEL HOST PORT CTX CACHE_K CACHE_V MOE_CACHE SPEC_TYPE
#                DRAFT_MAX DRAFT_MODEL PARALLEL CACHE_REUSE CUDA_VISIBLE_DEVICES
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="$REPO_ROOT/build/bin/llama-server"

# sharded GGUF: pass the first shard, llama.cpp loads the rest automatically
MODEL="${1:-${MODEL:-$REPO_ROOT/local/models/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CTX="${CTX:-131072}"
CACHE_K="${CACHE_K:-q4_0}"
CACHE_V="${CACHE_V:-q4_0}"
MOE_CACHE="${MOE_CACHE:-2048}"
SPEC_TYPE="${SPEC_TYPE:-draft-mtp}"
DRAFT_MAX="${DRAFT_MAX:-3}"
# self-contained head: carries its own token_embd + output (this build's
# qwen4exp MTP graph has no cross-model tensor borrowing, so the shared-
# variants do not load)
DRAFT_MODEL="${DRAFT_MODEL:-$REPO_ROOT/local/models/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}"
PARALLEL="${PARALLEL:-1}"
CACHE_REUSE="${CACHE_REUSE:-256}"

# Address the GPU by UUID: CUDA device indices do NOT match nvidia-smi
# indices on this box (numeric 0 would pull in the desktop GTX 1080).
# Default is the single healthy V100; the second card is faulty (RMA pending,
# see local/HARDWARE.md) and must not be used.
V100_UUIDS="GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$V100_UUIDS}"

[[ -x "$SERVER" ]] || { echo "error: $SERVER not found, run local/build.sh first" >&2; exit 1; }
[[ -f "$MODEL" ]]  || { echo "error: model not found: $MODEL (pass path as \$1 or set MODEL)" >&2; exit 1; }

echo "model     : $MODEL"
echo "gpus      : $CUDA_VISIBLE_DEVICES"
echo "offload   : all layers on GPU, routed experts + PLE table in CPU RAM"
echo "moe cache : ${MOE_CACHE} MiB VRAM for hot experts"
echo "ctx       : $CTX, KV cache K=$CACHE_K V=$CACHE_V"
echo "spec      : type=$SPEC_TYPE, n_max=$DRAFT_MAX${DRAFT_MODEL:+, draft=$DRAFT_MODEL}"
echo "listen    : http://$HOST:$PORT"

args=(
    -m "$MODEL"
    -c "$CTX"
    -ngl 99
    --cpu-moe
    -ot 'per_layer_token_embd=CPU'
    --moe-cache-mib "$MOE_CACHE"
    -fa on
    -ctk "$CACHE_K"
    -ctv "$CACHE_V"
    --spec-type "$SPEC_TYPE"
    --spec-draft-n-max "$DRAFT_MAX"
    --jinja
    --cache-reuse "$CACHE_REUSE"
    --parallel "$PARALLEL"
    --batch-size 2048
    --ubatch-size 512
    --host "$HOST"
    --port "$PORT"
)
if [[ -n "$DRAFT_MODEL" ]]; then
    [[ -f "$DRAFT_MODEL" ]] || { echo "error: draft model not found: $DRAFT_MODEL" >&2; exit 1; }
    args+=(--spec-draft-model "$DRAFT_MODEL" --spec-draft-ngl all)
fi

exec "$SERVER" "${args[@]}"
