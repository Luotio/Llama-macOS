#!/usr/bin/env bash
# download-model.sh - codified model downloads for this box.
#
# Resume-safe (wget -c), .done markers per file (existing convention), one log per run.
# Usage:
#   ./download-model.sh q4_k_xl    # Unsloth UD-Q4_K_XL, 4 shards, ~122 GB
#   ./download-model.sh nvfp4      # nvidia ModelOpt NVFP4 checkpoint into ./nvfp4/,
#                                  # 10 shards + config/tokenizer/index metadata, ~79 GB
#                                  # (model-fp8-mtp-ple.safetensors is intentionally NOT
#                                  # downloaded: the PLE table and MTP layer are reused from
#                                  # the existing IQ4_XS shard / mtp-Qwen3.8-Flash-Next-Q8_0.gguf)
#   ./download-model.sh iq4_xs     # re-download of the current production model, ~94 GB
set -euo pipefail
cd "$(dirname "$0")"

HF="https://huggingface.co"
LOG=""
DIR=""
FILES=()
NEED_GIB=0

case "${1:-}" in
  q4_k_xl)
    BASE="$HF/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-Q4_K_XL"
    FILES=(
      "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
      "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf"
      "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf"
      "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf"
    )
    NEED_GIB=125   # 10.9 MB + 49.9 + 49.4 + 12.1 GB, plus headroom
    LOG="download-q4_k_xl.log"
    ;;
  nvfp4)
    # The converter (tools/nvfp4_convert.py -> convert_hf_to_gguf.py) needs a
    # clean checkpoint dir: config.json, tokenizer files, the safetensors index
    # and the shards. Download into its own subdirectory for that reason.
    BASE="$HF/nvidia/Qwen3.8-Flash-Next-NVFP4/resolve/main"
    DIR="nvfp4"
    FILES=(
      "config.json"
      "generation_config.json"
      "hf_quant_config.json"
      "chat_template.jinja"
      "merges.txt"
      "model.safetensors.index.json"
      "preprocessor_config.json"
      "processor_config.json"
      "tokenizer.json"
      "tokenizer_config.json"
      "video_preprocessor_config.json"
      "vocab.json"
      "model-00001-of-00010.safetensors"
      "model-00002-of-00010.safetensors"
      "model-00003-of-00010.safetensors"
      "model-00004-of-00010.safetensors"
      "model-00005-of-00010.safetensors"
      "model-00006-of-00010.safetensors"
      "model-00007-of-00010.safetensors"
      "model-00008-of-00010.safetensors"
      "model-00009-of-00010.safetensors"
      "model-00010-of-00010.safetensors"
    )
    NEED_GIB=85    # 3.12 + 7x10 + 5.68 + 0.13 GB, plus headroom
    LOG="download-nvfp4.log"
    ;;
  iq4_xs)
    BASE="$HF/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-IQ4_XS"
    FILES=(
      "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
      "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf"
      "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf"
    )
    NEED_GIB=100   # 93.7 GB, plus headroom
    LOG="download-iq4xs.log"
    ;;
  *)
    echo "usage: $0 {q4_k_xl|nvfp4|iq4_xs}" >&2
    exit 2
    ;;
esac

# Disk preflight: require the full download to fit with 10 GiB to spare.
AVAIL_GIB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
if (( AVAIL_GIB < NEED_GIB + 10 )); then
  echo "ERROR: only ${AVAIL_GIB} GiB free, need ~$((NEED_GIB + 10)) GiB (download + headroom)." >&2
  echo "       Delete the IQ4_XS shards first if they are not needed anymore:" >&2
  echo "         rm -v Qwen3.8-Flash-Next-UD-IQ4_XS-*.gguf*   # frees ~94 GB" >&2
  exit 1
fi
echo "disk: ${AVAIL_GIB} GiB free (need ~${NEED_GIB} GiB)"

[[ -n "$DIR" ]] && mkdir -p "$DIR"

for f in "${FILES[@]}"; do
  p="${DIR:+$DIR/}$f"   # empty DIR (q4_k_xl, iq4_xs) stays a plain relative name
  if [[ -f "$p.done" ]]; then
    echo "skip  $p (already done)"
    continue
  fi
  echo "fetch $p"
  wget -c --tries=5 --timeout=60 -O "$p" "$BASE/$f" 2>>"$LOG"
  touch "$p.done"
done

echo "all files present:"
if [[ -n "$DIR" ]]; then
  ls -lh "${FILES[@]/#/$DIR/}"
else
  ls -lh "${FILES[@]}"
fi
