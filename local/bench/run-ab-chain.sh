#!/usr/bin/env bash
# Runs the Q4_K_XL optimization A/B chain sequentially. Each arm is one
# model-bench.sh run (own server start/stop, quickbench2 methodology).
# Order = expected value first, so if the chain is interrupted the most
# important results are already on disk.
set -uo pipefail
cd /mnt/git/local-llama-cpp

BENCH=local/bench/model-bench.sh
MODEL=qwen3.8-flash-next-unsloth-ud-q4_k_xl
CHAIN_LOG=local/bench/results/ab-chain.log

echo "=== A/B chain start $(date -u +%FT%TZ) ===" >> "$CHAIN_LOG"
echo "AER before: $(cat /sys/bus/pci/devices/0000:81:00.0/aer_dev_correctable 2>/dev/null | tr '\n' ' ')" >> "$CHAIN_LOG"

# All four shards must be present and done, or abort.
for i in 1 2 3 4; do
    f=$(printf "local/models/Qwen3.8-Flash-Next-UD-Q4_K_XL-%05d-of-00004.gguf.done" "$i")
    if [[ ! -f "$f" ]]; then
        echo "ABORT: missing $f (download not finished)" >> "$CHAIN_LOG"
        exit 1
    fi
done

run_arm() {
    local cfg="$1" label="$2"; shift 2
    echo "--- arm $label start $(date -u +%FT%TZ) ($cfg $*)" >> "$CHAIN_LOG"
    if "$@" "$BENCH" "local/strata/$cfg.json" "$MODEL" "$label" >> "$CHAIN_LOG" 2>&1; then
        echo "--- arm $label OK $(date -u +%FT%TZ)" >> "$CHAIN_LOG"
    else
        echo "--- arm $label FAILED (exit $?) $(date -u +%FT%TZ)" >> "$CHAIN_LOG"
    fi
}

# 1. BASELINE on the new v0.2.8 engine: production config, no extra env.
#    All other arms compare against THIS (the old 46.0/51.0/53.9 numbers are
#    from the v0.1.45-era binary and not comparable).
run_arm strata-unsloth-ud-q4_k_xl q4kxl-v028base env
# 2. adaptive overlap: off by default, +4.4% P100 / +5.7-8.3% A4000 in docs
run_arm strata-q4kxl-ab-adaptoverlap ab-adaptoverlap env
# 3. PF_WMMA: FP16 tensor-core prompt experts, +10% prompt vs MMQ (answers differ)
run_arm strata-q4kxl-ab-pfwmma ab-pfwmma env
# 4. dequant wide: OFF by default on sm_70, bit-exact, big prompt win in docs
run_arm strata-q4kxl-ab-dequantwide ab-dequantwide env
# 5. v100 small-gain batch: ARGMAX_ROWS_SM70 +0.7-0.8% decode (bitwise),
#    GR_NORM_FAST +0.5-0.6% decode, Q8_GATHER_WIDE=0 +11% on the gather kernel
run_arm strata-q4kxl-ab-v100small ab-v100small env
# 6. GDN_REC_WARP=2: chain-split DeltaNet recurrence, +6-9% (summation order changes)
run_arm strata-q4kxl-ab-gdnrecwarp2 ab-gdnrecwarp2 env
# 7. bit-exact small-gain batch (ATTN_DAILY + B6_MMVQ_ROWS + QSA_ROWS)
run_arm strata-q4kxl-ab-bitexact ab-bitexact env
# 8. adapt-async: expert cache updates on a helper thread (+4-5% decode in docs)
run_arm strata-q4kxl-ab-adaptasync ab-adaptasync env
# 9. scores thread: prompt path, +1..3% on P100-class cards
run_arm strata-q4kxl-ab-scores-thread ab-scores-thread env
# 10. adapt lag: +3.5% P100 PCIe 3.0 x16 in docs
run_arm strata-q4kxl-ab-adaptlag ab-adaptlag env

# 11. KV prefetch needs a long prompt to show (+5.3% at 32K, none at 8K):
#    baseline and arm both at ~30K tokens (PROMPT_REPEATS=600).
run_arm strata-unsloth-ud-q4_k_xl q4kxl-longbase env PROMPT_REPEATS=600
run_arm strata-q4kxl-ab-kvprefetch ab-kvprefetch-long env PROMPT_REPEATS=600

echo "AER after: $(cat /sys/bus/pci/devices/0000:81:00.0/aer_dev_correctable 2>/dev/null | tr '\n' ' ')" >> "$CHAIN_LOG"
echo "=== A/B chain done $(date -u +%FT%TZ) ===" >> "$CHAIN_LOG"
