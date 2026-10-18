# Qwen3.8-27B and Qwen3.8-Flash-Next performance results

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

# Qwen3.8-Flash-Next (qwen4exp, 125B MoE / 6B active) on this box

Model: `local/models/Qwen3.8-Flash-Next-UD-IQ4_XS-*.gguf` (93.7 GB, 3 shards)
plus a self-contained MTP sidecar (4.14 GB Q8_0). Architecture: hidden 2560,
vocab 248320, 48 layers of 12 x (3x Gated DeltaNet -> MoE, 1x QSA -> MoE),
MoE = 512 experts / 10 routed + 1 shared, plus a 51 GB n-gram PLE table.

llama.cpp recipe (single healthy V100, everything else in system RAM):
`--n-cpu-moe 999 -ot 'per_layer_token_embd=CPU' --moe-cache-mib 2048`, KV q8_0
except noted. The MoE experts and the PLE table stay on CPU; only attention,
shared experts and embeddings touch the GPU.

## llama-bench sweep (stopped early by user)

Single healthy V100, pp512/tg128, reps=3, FA on. Sweep was killed mid-run at
the user's request ("stop it"); the q4-KV and no-FA variants at 128k were NOT
run.

| label | ctx | KV | pp512 t/s | tg128 t/s | start temp C |
|---|---|---|---|---|---|
| fn-smoke | 4096 | q8_0 | 60.48 (pp256) | 13.41 (tg64) | - |
| fn-32k | 32768 | q8_0 | 81.04 | 13.08 | 54 |
| fn-64k | 65536 | q8_0 | 77.24 | 12.18 | 80 |
| fn-128k | 131072 | q8_0 | 64.17 | 10.87 | 82 |

Notes:
- Prompt rate peaks around 32k and falls at longer context (KV traffic).
- Decode is remarkably stable: 13.4 -> 10.9 t/s from 4k to 128k, because the
  per-token cost is dominated by CPU expert traffic, not KV length.
- GPU stays under the 85 C throttle threshold (max 82 C at start of fn-128k).
- Raw data: local/bench/results/bench.csv + fn-*.jsonl.

MTP server test (serve-qwen38-flash-next.sh, port 8095, ctx 131072, KV q4_0,
spec draft-mtp with the self-contained sidecar) is still pending; see
local/bench/mtp-bench.py --label fn-mtp.

# Strata (Niko1221/Strata v0.1.45) on this box

Strata is a purpose-built engine for Flash-Next class MoE models: dense layers
on the GPU, all 512 experts streamed from system RAM over PCIe per token,
MTP speculative decoding built in. Setup: `strata/setup.sh --family unsloth
--model UD-IQ4_XS --gguf-dir local/models --gpu 2 --context 65536` (reuses our
93.7 GB shards, SHA-256 verified; fetches only the ~5 GB MTP draft layer).

V100 is EXPERIMENTAL in Strata: CUDA 13 dropped sm_70, so setup compiled a
second engine into `strata/engine-cuda12/` with `-DSTRATA_EXPERIMENTAL_SM60=ON`
(auto-selected because the chosen card is below sm_75). Host compiler must be
gcc-14 (`export CC=gcc-14 CXX=g++-14`) because nvcc 12.9 rejects gcc 15.

Config: `--gpu 2` (healthy V100; the default "most VRAM" pick ties between both
V100s and could grab the faulty one), ctx 65536 (KV int8 at 128k would be
~138 GB), KV streaming on (KV lives in RAM, 0.9 GB), `--spec 4 --spec-min-p
0.5`, MTP draft from Strata-data/mtp/rt (experts q2_0).

## Quick test (no long benchmarks, per user request)

Server load: ~55 s to ready, 485 MiB VRAM free with everything loaded.
One small completion + one ~432-token prompt, MTP on.

| metric | value | docs expectation (V100) |
|---|---|---|
| prompt t/s (432 tok) | 136.0 | ~1,100-1,250 |
| decode t/s (MTP spec=4) | 24.3 | 38-51 |
| MTP draft acceptance | 13/13 (long), 6/9 + 3/3 (small) | - |
| model load time | ~55 s | - |

The rates are far below the docs' V100 numbers. Root cause is the PCIe link:
the engine's own probe measured only 5.6 GB/s host->device (pcie_frac 0.16,
default 0.55), and nvidia-smi shows BOTH V100s at Gen3 x8 (8 of 16 lanes).
Strata streams all 512 experts from RAM per token, so it is directly bound by
that link; the docs assume a full-width link. The GTX 1080 also reads Gen1 x16
(59 C at idle) - this box's PCIe slots appear to be the bottleneck for any
RAM-streaming engine.

Comparison on the same card, same model (llama.cpp vs Strata):

| metric | llama.cpp (fn-32k) | Strata (quick test) |
|---|---|---|
| prompt t/s | 81.04 (pp512 @32k) | 136.0 (432 tok) |
| decode t/s | 13.08 (tg128, no MTP) | 24.3 (MTP spec=4) |

Strata is ~1.7x faster on prompt and ~1.9x on decode despite the crippled
link, mostly from its expert-streaming pipeline + built-in MTP. llama.cpp's
decode here has no speculative decoding in the bench numbers.

Start/stop: `cd strata && ./run-unsloth-ud-iq4_xs.sh` (port 8080, OpenAI API).
Stop with `pkill -f "strata/serve/server.py"; pkill -f engine-cuda12`.

## Real agent workload (VS Code Copilot via llama-copilot extension), 2026-10-17

First sustained real-world run: a coding agent driving the local endpoint
(http://127.0.0.1:8080) through the delft-solutions.llama-copilot extension.
Snapshot from GET /metrics after 24 chat requests (~89 min server uptime),
captured mid-generation (live request at 70,069 prompt tokens). Same tuned
config as above (--spec 4 --spec-min-p 0.5 + MTP, prefill auto:32768, ctx
131072 int8 KV, kv-resident 32768, conv-cache 8192 MiB, expert-cache auto,
STRATA_SM70_TABLE=1, numactl node 1).

| metric | value |
|---|---|
| requests since start | 24 (22/24 hit the conversation cache) |
| prompt tokens total | 1,282,271 (91.3% reused from conv-cache) |
| output tokens total | 26,879 |
| decode t/s (mean over all requests) | 28.6 (per-request range 25.3-34.2) |
| MTP draft acceptance | 14,854/23,487 = 63.2% overall (per request 59-84%) |
| conv-cache state | 6.75 GB parked in 4 slots, 5 parks, 1 eviction, 0 restores |
| live session at snapshot | 70,069 prompt tok, 6,846 generated in 257.7 s = 23.3 t/s (2 s window), 27.8 mean |

Cached-prompt prefill: a request with 67,380 prompt tokens (65,475 reused)
took 10.9 s of prefill (~6.2k tok/s over the full context with KV resident)
vs 136 t/s cold at 432 tokens in the quick test. The engine reports spec=6
(mtp_max 4 + lookup 3) in /metrics for the --spec 4 config.

Hardware during load:

| metric | value |
|---|---|
| GPU util / temp | 100% / 82-84 C (no throttle; power 80-157 W of a 300 W limit) |
| VRAM | 15.8/16 GiB (485 MiB free) |
| PCIe rx (expert streaming) | bursts to ~8 GB/s, typical ~2.3 GB/s (Gen3 x8 link) |
| CPU | 52-64% of 32 cores (node 1 bound) |
| RAM | 98.3/118 GiB (model + expert cache 8.5 GiB + conv-cache 8 GiB) |

Takeaways:
- Decode under real agent traffic (28.6 t/s mean) is ~1.2x the quick-test
  number (24.3 t/s); long cached prompts keep prefill cheap and MTP acceptance
  stays healthy at 63%.
- The conversation cache is doing its job: 91% of all prompt tokens were
  reused, so agent turns mostly pay for the delta + attention over resident KV.
- GPU never nears the power limit; the box is PCIe/DRAM-bound, not thermally
  bound (82 C at full load).

## A/B test of three decode optimizations + numad/NUMA findings, 2026-10-17

Probe: deterministic agent-style request (temperature=0), 37,258 prompt tokens
(payment-service prose, ~260 blocks), max_tokens=2000, stream=false, via
/v1/chat/completions. Server-side numbers from /metrics requests[]
(decode_tok_s, prompt_ms, hit_rate, drafts). Caveat: an earlier probe version
omitted temperature (API default 1.0), so those runs generated different
content each time and their decode comparisons were invalid; all numbers below
use temperature=0.

### Phase 1: three-item A/B (numad still active -- later found confounded)

Baseline config: spec-min-p 0.5, expert-cache auto, no pool-workers flag.

| arm | change | decode t/s (cold) | drafts offered / 2000 tok | draft acceptance | expert hit |
|---|---|---|---|---|---|
| baseline | - | 31.2 | 1,634 | 63.5% (1,038) | 69.9% |
| #1 | spec-min-p 0.3 | 25.7 (warm: 26.3) | 2,033 (warm: 1,995) | 47.9% (warm: 49.5%) | 72.4% |
| #3 | pool-workers 16 | 27.2 | - | - | - |

- #1 spec-min-p 0.3: REJECTED. The lower threshold lets the MTP drafter extend
  windows on low-confidence guesses: +25% drafts offered, but acceptance
  collapses 63.5% -> ~48%, tokens per verify step drops 1.22 -> 0.98, decode
  -18%. Reverted to 0.5.
- #2 expert-cache grow (auto -> 16 GiB): INFEASIBLE on this 16 GB card.
  --expert-cache takes a slot count, not MiB, and auto already fills all free
  VRAM (3,689 slots / 8.30 GiB from the profile's 49,152 ranked experts). Only
  485 MiB VRAM remains after load; the engine itself warns that below 512 MiB
  free the driver may page GPU memory to system RAM (#2065).
- #3 pool-workers: inconclusive in phase 1 (see numad finding); re-tested
  cleanly in phase 2.

### Phase 2: numad root cause + clean re-measurement

The numad daemon (sys-process/numad-0.5, enabled at boot) was silently applying
a PID-based policy that relocated the engine's CPU pool to the wrong NUMA node
across restarts -- pool size flipped between 15 and 31 workers and core
selection varied, which is what confounded phase 1. numad 0.5 has no
name-based policy support (PID lists only), so it was disabled instead:
`rc-update del numad default && rc-service numad stop`.

With numad gone, `numactl --cpunodebind=1` (from the run script) sticks from
t=0: 15 pool workers on CPUs 17-31 (all node 1), host thread on CPU 16.

Clean deterministic probe (temperature=0, warm conv-cache):

| config | run 1 | run 2 | decode t/s avg | prefill (cold) |
|---|---|---|---|---|
| node-1 bound, 15 workers | 30.3 | 29.8 | ~30.0 | 40.4 s (922 t/s) |
| both nodes, 31 workers | 30.5 | 33.3 | ~31.9 | 38.7 s (964 t/s) |

Run-to-run variance is +/-5% even within one config (greedy output is not
bit-reproducible: variable draft-acceptance batch shapes feed back into MoE
routing), so the two configs are statistically indistinguishable. #3 verdict:
worker count / NUMA binding is NOT a decode bottleneck; keep the default pool.

Prefill after numad removal: 38.4-40.4 s cold (922-971 t/s) vs 61 s (611 t/s)
in the numad era -- ~1.6x faster and consistent across restarts. Decode sits in
the same ~30-33 t/s range as the original baseline, so the earlier "decode
regression" was entirely a temperature-1 sampling artifact.

### NUMA weight placement (accepted as-is)

The 55.43 GiB expert arena lands mostly on node 0 even with the process bound
to node 1; the split varies per load:

| run | node 0 | node 1 |
|---|---|---|
| node-1 bound (earlier) | 45.2 GiB | 13.5 GiB |
| both nodes | 44.6 GiB | 14.0 GiB |
| node-1 bound (latest) | 41.3 GiB | 17.7 GiB |

- `numactl --preferred=1` is ignored on this kernel (numa_balancing=1 +
  first-touch locality; verified with a standalone 6 GiB touch test that gave
  the same ~40/60 split under plain, --preferred=1, and
  --cpunodebind=1 --preferred=1).
- Root cause is capacity, not policy: node 1 has only ~3 GB free at load time,
  so under the engine's 3.49 GiB/s 6-thread arena load, per-fault reclaim
  cannot keep up and faults fall back to node 0. `migratepages` moves only
  ~150 MB (nowhere on node 1 to accept pages).
- Measured impact on decode: none (node-1-bound vs both-nodes within noise).
  Consistent with pcie_share at 2-4% of decode time and cross-socket UPI
  (~25-35 GB/s) being far faster than the Gen3 x8 PCIe link (5.6 GB/s) that
  actually caps expert streaming. Accepted as-is; only worth revisiting if
  node 1's free RAM grows at load time.

### Final state

- Config unchanged: local/strata/strata-unsloth-ud-iq4_xs.json is the committed
  original (spec-min-p 0.5, expert-cache auto, no pool-workers flag).
- numad disabled (`rc-update del numad default`); the run script's
  `numactl --cpunodebind=1` now works as documented.
- Decode ~30-33 t/s, prefill ~920-970 t/s cold, conv-cache reuse working; the
  box remains PCIe/DRAM-bound per the real-workload section above.
