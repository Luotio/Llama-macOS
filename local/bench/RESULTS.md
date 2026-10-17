# Qwen3.8-27B (UD-Q4_K_XL) performance results

Machine: 64 threads (AVX2 only, no AVX512), 109 GB RAM, 1x Tesla V100-SXM2-16GB
usable + 1x GTX 1080 (8 GB) as temporary layer host. The second V100 is faulty
(die-level register-file ECC faults, Xid 48/172/13/31, deterministic failure
on trivial CUDA workloads) and excluded from all runs; see local/HARDWARE.md
for the RMA evidence.

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

## Tier 3: V100 + GTX 1080 dual-GPU (temporary, RMA pending)

The faulty V100 is out for RMA; the GTX 1080 (8 GB, sm_61, ~6.9 GB free with
the desktop running) hosts the tail layers so all 66 offloadable units stay on
GPUs instead of 21 sitting in system RAM. Build compiled for sm_61 + sm_70
(see local/build.sh). Tensor split uses '/' between devices (a comma is parsed
as one number and silently puts every layer on device 0).

depth=131072, pp512/tg128, reps=5, KV q4_0/q4_0, FA on. CUDA_VISIBLE_DEVICES
lists the healthy V100 first (device 0), the 1080 second (device 1).

| label | NGL | split | pp512 t/s | tg128 t/s | notes |
|---|---|---|---|---|---|
| t3-v100-45 | 45 | 1 (V100 only) | 92.24 ± 4.12 | 2.80 ± 0.36 | baseline = today's production config, 21 layers in CPU RAM |
| t3-dual-48-18 | 66 | 48/18 | 86.92 ± 6.28 | 6.60 ± 0.22 | best decode split, all layers on GPUs |
| t3-dual-45-21 | 66 | 45/21 | OOM | OOM | 1080 cudaMalloc failed (1149 MiB compute buffer) |
| t3-dual-49-17 | 66 | 49/17 | 86.61 ± 5.00 | 6.44 ± 0.22 | all layers on GPUs |
| t3-dual-54-12 | 66 | 54/12 | 93.15 ± 3.67 | 5.99 ± 0.42 | split that leaves room for the MTP draft context on the 1080 |

Findings:
- Putting the 21 layers on the 1080's GDDR5X (320 GB/s) instead of system RAM
  (~60 GB/s) is a clear win for decode: 6.60 vs 2.80 t/s, ~2.4x. The memory
  offload to the second GPU works; offloading to system RAM does not keep up.
- Prompt processing is slightly slower with layers on the Pascal card (no
  tensor cores): 86.9 at 48/18 vs 92.2 V100-only, but 54/12 (more layers on
  the V100) reaches 93.2, matching the single-GPU baseline.
- Decode is best at 48/18; moving more layers to the V100 (54/12) costs ~0.6
  t/s on tg but is required for MTP (below).

### Tier 3: MTP speculative decoding on dual-GPU

The MTP draft context is a second context on the same model, so its layers
follow the target split; block 64 (draft tensors + output) lands on the 1080
and needs ~1.6 GiB of headroom there (compute buffers + draft KV/RS). At
48/18 and 51/15 that allocation OOMs on the 1080; at 54/12 it fits
(V100 16073/16384 MiB, 1080 7801/8192 MiB steady state).

Server: dual-GPU, NGL=66, TS=54/12, KV q4_0/q4_0, FA on, ctx 131072,
n_max=3, 3 reps x 256 tok (same harness as tier 2).

| label | spec-type | n_max | prompt t/s | decode t/s | draft_n | accepted | accept rate |
|---|---|---|---|---|---|---|---|
| t3-mtp-54-12 | draft-mtp | 3 | 81.6 | 30.33 | 849 | 480 | 0.565 |

Comparison at the same split: no-MTP tg128 is 5.99 t/s, so MTP gives ~5.1x on
dual-GPU (vs +36% on the single V100 in tier 2). Against the tier-2 single-V100
MTP baseline (10.27 t/s), dual-GPU MTP is 2.95x faster. Acceptance (0.565)
matches the single-V100 n_max=3 rate (0.584), so the speedup is purely from
the extra GPU, not better drafts.

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

Temporary alternative while the faulty V100 is in RMA: dual-GPU MTP on the
GTX 1080 gives ~30 t/s decode (2.95x this config), see tier 3 above:

    CUDA_VISIBLE_DEVICES="GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c,GPU-6f4fedec-c775-e942-f39d-bd77cf3f3384" \
        NGL=66 TENSOR_SPLIT=54/12 CACHE_K=q4_0 CACHE_V=q4_0 \
        SPEC_TYPE=draft-mtp DRAFT_MAX=3 local/serve-qwen3-27b.sh

Do NOT set NGL=47/49 in the server: OOM. Do NOT use the faulty card
(GPU-f07f3af3, RMA pending): see local/HARDWARE.md.

## Comparison: same model on 1x Radeon RX 7900 XTX (24 GB)

Reference data from another box running the same GGUF (user-provided
llama.cpp Prometheus metrics + slot dump, 2026-10-17). 24 GB VRAM fits ALL
layers on GPU - no CPU offload, which is the main reason it is faster.

| metric | RX 7900 XTX | this box (V100, best config) |
|---|---|---|
| decode t/s (MTP on) | 45.0 (cumulative avg) | 10.3 |
| prompt t/s | 386.6 (cumulative avg) | 110 (pp512 @128k, cold) |
| n_ctx | 131072 | 131072 |
| max context used | 127,016 tokens | 131,072 (bench depth) |
| layers on GPU | all | 45/66 (rest on CPU RAM) |

Raw counters behind the averages:

    prompt:   21,939,800 tok / 56,746.7 s = 386.6 t/s
    decode:     5,964,780 tok / 132,588 s = 45.0 t/s
    n_decode:   2,320,200 calls -> ~2.57 tokens per decode call (MTP gain)

Their run params: speculative draft-mtp, temp 0.4, top_k 40, top_p 0.9,
min_p 0.05, repeat_penalty 1.05, presence_penalty 0.1, max_tokens 32768,
chat_format peg-native, reasoning_format deepseek.

Caveats: both t/s figures are CUMULATIVE averages over ~22M prompt / ~6M
generated tokens with mixed context lengths and prefix-cache reuse, so they
are not directly comparable to our cold pp512/tg128 llama-bench numbers.
The decode gap (45 vs 10.3) is expected: all-GPU vs 21 CPU layers per token
over a Gen3 x8 link.
