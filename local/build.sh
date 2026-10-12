#!/usr/bin/env bash
# Build llama.cpp with the CUDA backend for 2x V100-SXM2-16GB (sm_70).
#
# One-time prerequisite (run with sudo):
#   emerge --ask dev-util/nvidia-cuda-toolkit=12.9.1-r1
#
# Note: CUDA 13.x dropped Volta (sm_70) support, so the 12.9 toolkit is required.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
JOBS="${JOBS:-$(nproc)}"

CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "error: $CUDA_HOME/bin/nvcc not found" >&2
    echo "install the toolkit first: emerge --ask dev-util/nvidia-cuda-toolkit=12.9.1-r1" >&2
    exit 1
fi

# nvcc 12.9 does not accept gcc 15 as host compiler; the ebuild pulls in gcc-14.
# Must be passed to cmake explicitly: env CC/CXX do not reach the CUDA host compiler.
HOST_FLAGS=()
if command -v g++-14 >/dev/null 2>&1; then
    HOST_FLAGS=(-DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14)
fi

cmake -B "$BUILD_DIR" -S "$REPO_ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    "${HOST_FLAGS[@]}" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="70-real" \
    -DGGML_CUDA_GRAPHS=ON

cmake --build "$BUILD_DIR" --config Release -j "$JOBS" \
    --target llama-server llama-cli llama-bench

echo
echo "build complete:"
ls -lh "$BUILD_DIR/bin/llama-server" "$BUILD_DIR/bin/llama-cli" "$BUILD_DIR/bin/llama-bench"
