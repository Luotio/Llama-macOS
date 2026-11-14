#!/usr/bin/env bash
# Isolated Broadwell + V100 llama.cpp build with NVIDIA NCCL 2.27.7.
# Does not replace the existing build/ binaries. Requires uv and CUDA 12.9.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${BUILD_DIR:-$repo/llama-cpp/build-v100-nccl}"
nccl="$build/deps/nvidia/nccl"
cuda="${CUDA_HOME:-/opt/cuda}"

if [[ ! -f "$nccl/include/nccl.h" || ! -f "$nccl/lib/libnccl.so.2" ]]; then
    uv pip install --python /usr/bin/python3 --target "$build/deps" \
        'nvidia-nccl-cu12==2.27.7'
fi
cmake -S "$repo/llama-cpp" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc-14 \
    -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
    -DCMAKE_CUDA_COMPILER="$cuda/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES=70-real \
    -DGGML_NATIVE=ON -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON \
    -DGGML_CUDA_NCCL=ON -DNCCL_INCLUDE_DIR="$nccl/include" \
    -DNCCL_LIBRARY="$nccl/lib/libnccl.so.2"
cmake --build "$build" --config Release -j "${JOBS:-8}" \
    --target llama-server llama-cli llama-bench