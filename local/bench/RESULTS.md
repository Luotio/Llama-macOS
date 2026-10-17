# Qwen3.8-27B (UD-Q4_K_XL) performance results

Machine: 64 threads (AVX2 only, no AVX512), 109 GB RAM, 1x Tesla V100-SXM2-16GB
usable (PCI 81). The second V100 (PCI 82) is faulty (Xid 13/31, deterministic
illegal memory access on trivial CUDA workloads) and excluded from all runs.

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
| (pending) | | | | | | |

## Tier 2: MTP speculative decoding sweep (server, 3 reps x 256 tok)

Fixed coding prompts (rotated per rep to avoid prefix-cache hits). Config from
winning tier-1 settings.

| label | spec-type | n_max | prompt t/s | decode t/s | draft_n | accepted | accept rate |
|---|---|---|---|---|---|---|---|
| (pending) | | | | | | | |

## Best config

(pending)
