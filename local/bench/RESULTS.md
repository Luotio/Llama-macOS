# Qwen3.8-27B (UD-Q4_K_XL) performance results

Machine: 64 threads (AVX2 only, no AVX512), 109 GB RAM, 1x Tesla V100-SXM2-16GB
usable. The second V100 is faulty (die-level register-file ECC faults, Xid
48/172/13/31, deterministic failure on trivial CUDA workloads) and excluded
from all runs; see local/HARDWARE.md for the RMA evidence.

Model: `local/models/Qwen3.8-27B-UD-Q4_K_XL.gguf` (17 GB, arch qwen35, 65 blocks
+ 1 MTP layer, hybrid attention: 16 full-KV layers + 48 linear-attention layers).

Context: 131072 (128k) for all runs. KV cache at 128k is ~4.35 GiB with q8_0
(only the 16 full-attention layers grow with context; linear-attention state is
fixed-size).

## Smoke test (single GPU, NGL=auto, MTP n_max=3)

Auto-fit placed 37/66 layers on GPU (9.69 GiB weights), 29 on CPU (7.05 GiB,
pinned). KV cache split: 2.45 GiB GPU + 1.90 GiB CPU.

| metric | value |
|---|---|
| prompt (85 tok, cold) | 24.5 t/s |
| decode (MTP n_max=3) | ~9.6-10.4 t/s |
| MTP acceptance | 75.2% (176/234), mean draft len 3.23 |
| GPU temp under load | 82-84 C |
| GPU power under load | 40-126 W (bursts; hybrid CPU/GPU pipeline) |

Note: model defaults to reasoning mode (`reasoning_content`); the chat template
supports `enable_thinking=false` via `chat_template_kwargs`.

## Tier 1: llama-bench sweep (no speculative decoding)

depth=131072, pp512/tg128, reps=3, FA on unless noted. Single GPU, TS=1.

| label | NGL | KV | FA | pp512 t/s | tg128 t/s | notes |
|---|---|---|---|---|---|---|
| t1-base | 37 | q8_0 | on | 61.98 | 1.805 | auto-fit baseline (Oct 12) |
| t1-ngl25 | 25 | q8_0 | on | 82.20 | 1.244 | CPU-bound, fewer layers offloaded |
| t1-ngl45 | 45 | q8_0 | on | 95.58 | 2.651 | NGL ceiling with q8_0 KV (47/48 OOM) |
| t1-kvq4 | 45 | q4_0 | on | 100.46 | 2.320 | pp up, tg down vs q8_0 at same NGL |
| t1-ngl47kvq4 | 47 | q4_0 | on | 110.43 | 3.432 | best llama-bench config |
| t1-ngl49kvq4 | 49 | q4_0 | on | 106.32 | 3.408 | NGL ceiling with q4_0 KV; ~tied with 47 |

Notes:
- FA off is not testable here: this build requires flash-attn for quantized KV
  caches, and f16 KV at 128k (~8.7 GiB) leaves no room for weights + compute
  buffers on a single 16 GB card.
- The MTP server needs ~0.2 GiB extra for the draft context, so its ceiling is
  NGL=45 (NGL=47 OOMs in llama-server but works in llama-bench).

## Tier 2: MTP speculative decoding sweep (server, 3 reps x 256 tok)

Fixed coding prompts (rotated per rep to avoid prefix-cache hits). Config from
winning tier-1 settings.

Server config: single healthy V100, NGL=45, KV q4_0/q4_0, FA on, ctx 131072.
Prompt t/s is for the ~85-token bench prompts (small single batch), not
comparable to llama-bench pp512.

| label | spec-type | n_max | prompt t/s | decode t/s | draft_n | accepted | accept rate |
|---|---|---|---|---|---|---|---|
| mtp-n0 | none | 0 | 28.7 | 7.54 | - | - | - |
| mtp-n1 | draft-mtp | 1 | 29.7 | 8.12 | 422 | 342 | 0.810 |
| mtp-n2 | draft-mtp | 2 | 29.6 | 9.70 | 641 | 442 | 0.690 |
| mtp-n3 | draft-mtp | 3 | 28.7 | 10.27 | 832 | 486 | 0.584 |
| mtp-n4 | draft-mtp | 4 | 28.7 | 9.82 | 1062 | 497 | 0.468 |

Decode speedup vs n_max=0: +36% at n_max=3. n_max=4 regresses: the extra
draft step costs more than it saves once per-step acceptance falls below ~0.5.
The server decode rate (10.27) vs llama-bench tg128 without spec (3.43 at
NGL=47) is the speculative-decoding gain, not a measurement artifact.

## Best config

Serving (interactive / coding-agent workloads), single healthy V100:

    CUDA_VISIBLE_DEVICES="GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c" \
        NGL=45 TENSOR_SPLIT=1 CACHE_K=q4_0 CACHE_V=q4_0 \
        SPEC_TYPE=draft-mtp DRAFT_MAX=3 local/serve-qwen3-27b.sh

- 128k context, decode ~10.3 t/s with MTP (vs 7.5 without), prompt ~29 t/s on
  small prompts.
- NGL=45 is the server ceiling: the MTP draft context needs ~0.2 GiB more than
  llama-bench, so NGL=47 (the llama-bench winner) OOMs in llama-server.
- KV q4_0 over q8_0: +2 layers of headroom and faster prompt processing; no
  measurable quality concern for coding workloads (verify per use case).
- GPU stays at ~82 C max under load at the 300 W limit.

Do NOT set NGL=47/49 in the server: OOM. Do NOT use the faulty card
(GPU-f07f3af3, RMA pending): see local/HARDWARE.md.
