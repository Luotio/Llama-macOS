#!/usr/bin/env bash
# Thermal/power tuning for the two V100-SXM2 GPUs, addressed by UUID
# (CUDA/nvidia-smi numeric indices disagree on this box). The GTX 1080
# that drives the desktop is left untouched.
#
# Usage:
#   local/gpu-tune.sh status        # show current power limit, clocks, temps
#   local/gpu-tune.sh apply [W]     # set power limit (default 250W, range 150-300)
#   local/gpu-tune.sh reset         # restore default 300W
#
# Needs root. Reverts automatically on driver reload / reboot.
set -euo pipefail

V100_UUIDS="GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c,GPU-f07f3af3-5f02-f364-0d37-2ab3a4c4afce"
GPUS="${GPUS:-$V100_UUIDS}"
DEFAULT_LIMIT=300

need_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "error: this needs root (sudo local/gpu-tune.sh ...)" >&2
        exit 1
    fi
}

case "${1:-status}" in
    status)
        nvidia-smi -i "$GPUS" \
            --query-gpu=index,name,temperature.gpu,power.limit,power.draw,clocks.sm,utilization.gpu,memory.used \
            --format=csv
        ;;
    apply)
        need_root
        W="${2:-250}"
        nvidia-smi -i "$GPUS" -pl "$W"
        echo "power limit set to ${W}W on GPU $GPUS (default was ${DEFAULT_LIMIT}W)"
        ;;
    reset)
        need_root
        nvidia-smi -i "$GPUS" -pl "$DEFAULT_LIMIT"
        echo "power limit restored to ${DEFAULT_LIMIT}W on GPU $GPUS"
        ;;
    *)
        echo "usage: $0 {status|apply [W]|reset}" >&2
        exit 1
        ;;
esac
