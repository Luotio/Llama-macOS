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

# PCIe Gen3 restoration + env-var sweep, 2026-10-28

## Engine-death root cause (resolved)

Every "verify: layer N never rang" death was caused by `STRATA_EXP_MODE=1` in
the production config: mode 1 is the gfx906-only 64-lane wavefront layout, and
its `row_dot64()` is a `__trap()` on CUDA -> Xid 43 + sticky "unspecified
launch failure" for the rest of the process. Fix: remove the variable. With
`STRATA_SM70_TABLE=1` the default mode on this V100 is 8, the verified fast
path. Zero new NVRM Xids since the fix (last one: Oct 28 21:55, an intentional
bisection test).

## BIOS round 2: Gen3 x8 restored

BIOS changes: Gen3 Phase3 = Hardware Adaptive, Gen3 EQ mode = Auto,
DeEmphasis -3.5 dB, Link Speed forced to Gen3, Gen3 Phase2 = Hardware
Adaptive. Result: V100 back at **Gen3 x8** (was Gen1 x8).

- Idle AER: ~24 correctable/min (GPU TX Timeout ~8/min + root-port RX
  BadDLLP/BadTLP/Timeout ~16/min). ~10x better than the original storm, not
  zero.
- 46k-token stress request at Gen3: HTTP 200 in **39.9 s** vs 127.2 s at
  Gen1 x8 (3.2x faster), AER deltas during the request: GPU Timeout +571,
  root-port BadTLP +40632 / BadDLLP +831 / Timeout +62. The storm returns
  under sustained load but is harmless: all errors are correctable and
  retried, no data corruption, engine survives. This empirically confirmed
  the AER storm was never the cause of the engine deaths (that was the
  `__trap()` above).
- Keep this BIOS config. The residual fix for the load-time storm is
  hardware: replace the SFF-8654 8i cables first, then reseat/reflow the
  SXM2 module. Fallback if Gen3 ever misbehaves (new Xids or corruption):
  force Gen2 in BIOS.

## Sweep 1: six env vars A/B'd (local/bench/strata-v100-quickbench.sh)

Port 8091, 40x ~35-word prompt blocks (~1.4k tokens), max_tokens=96,
temperature=0, REPS=2, no warmup. Sweep baseline = production minus
STRATA_PROMPT_ATTN_OLD (i.e. sm70=1 bf16=1 prompt_old=0 pre75=1 mmvq=1
gr_fast=1); each arm flips exactly one variable.

| arm | change vs sweep baseline | decode avg t/s | rep 1 (cold) | rep 2 (warm) |
|---|---|---|---|---|
| baseline | - | 30.75 | 26.3 | 35.2 |
| sm70_table_off | STRATA_SM70_TABLE=0 | 28.95 | 26.0 | 31.9 |
| bf16_tc_off | STRATA_BF16_TC=0 | 32.45 | 29.9 | 35.0 |
| prompt_attn_old_on | STRATA_PROMPT_ATTN_OLD=1 | 31.65 | 27.3 | 36.0 |
| attn_pre75_off | STRATA_ATTN_PRE75=0 | 32.70 | 28.0 | 37.4 |
| mmvq_il_off | STRATA_MMVQ_IL=0 | 30.60 | 26.9 | 34.3 |
| gr_fast_off | STRATA_GR_FAST=0 | 31.50 | 28.2 | 34.8 |

Prompt rate is flat across arms (~198-201 t/s); draft acceptance 0.62-0.65,
no arm moves it meaningfully.

Key finding: rep 1 is ALWAYS the slow one (26-30 t/s in every arm) - a
cold-start penalty (expert cache, conv-cache, GPU state), not run noise. The
2-rep averages are dominated by that cold rep, so the warm rep is the
comparable number:

- STRATA_SM70_TABLE=1: worth +3.3 t/s warm (+10%). Keep.
- STRATA_ATTN_PRE75=1: costs ~-2.2 t/s warm (-6%) per this single warm rep.
  Candidate for removal; confirmation run in sweep 2 (with proper warmup).
- STRATA_PROMPT_ATTN_OLD=1: +0.8 t/s warm (+2%). Keep (production already has it).
- STRATA_BF16_TC=1, STRATA_GR_FAST=1: neutral on warm reps (the bf16_tc_off
  "win" in the averages was a cold-rep artifact). Keep as-is.
- STRATA_MMVQ_IL=1: +0.9 t/s warm (+2.6%). Keep.

## Sweep 2: additive optimizations (local/bench/strata-v100-quickbench2.sh)

Port 8092, WARMUP=1 (discarded), REPS=3. Arms: baseline (production as-is),
attn_pre75_off (confirmation of sweep 1), adapt_overlap (STRATA_ADAPT_OVERLAP=1),
adapt_lag2 (STRATA_ADAPT_LAG=2), pcie_balance (STRATA_PCIE_BALANCE=1),
route_tail_skip1 (STRATA_ROUTE_TAIL_SKIP=1), aux_cpus_auto (--aux-cpus auto),
esp_cvec (experimental-speed-projection control vector, project mode,
per-layer, layers 4-44).

| arm | rep1 | rep2 | rep3 | decode avg | decode med | prompt avg | draft acc |
|---|---|---|---|---|---|---|---|
| baseline | 35.1 | 35.3 | 34.3 | 34.9 | 35.1 | 196.3 | 0.615 |
| attn_pre75_off | 35.9 | 27.1 | 36.8 | 33.3 | 35.9 | 197.4 | 0.626 |
| adapt_overlap | 25.7 | 40.0 | 27.2 | 31.0 | 27.2 | 196.7 | 0.619 |
| adapt_lag2 | 26.8 | 42.0 | 40.3 | 36.4 | 40.3 | 196.2 | 0.616 |
| pcie_balance | 33.1 | 37.0 | 37.3 | 35.8 | 37.0 | 199.7 | 0.598 |
| route_tail_skip1 | 61.8 | 60.8 | 63.1 | 61.9 | 61.8 | 201.3 | 0.706 |
| aux_cpus_auto | 33.1 | 35.8 | 35.3 | 34.7 | 35.3 | 199.2 | 0.610 |
| esp_cvec | 34.5 | 33.3 | 36.7 | 34.8 | 34.5 | 199.7 | 0.670 |

With WARMUP=1 the baseline reps are all warm (35.1/35.3/34.3), confirming
sweep 1's cold-rep finding: rep-1 slowness is a warmup artifact, not noise.

Findings:

- STRATA_ROUTE_TAIL_SKIP=1: the winner. [61.8, 60.8, 63.1] = +78% vs baseline
  median (61.8 vs 35.1), consistent across all three reps, and draft
  acceptance improves 0.615 -> 0.706. Semantics (expert_source.cpp): for each
  missed expert in a verify window, if no token ranks it above tail_rank the
  expert is neither copied over PCIe nor computed; its contribution is zeroed
  without renormalization. R=7 (the upstream CUDA default since v0.1.42,
  documented +10-20% / KL ~0.003-0.006) skips only experts ranked below 7;
  R=1 skips any expert no token ranks top-1 - far more aggressive. Engagement
  verified via server logs: expert-cache hit 88-96% under R=1 vs 56-74% at
  baseline, and no "+N% over PCIe" suffix on the cache line.
- STRATA_ADAPT_LAG=2: warm reps [42.0, 40.3] vs baseline [35.3, 34.3] (+15%),
  possibly real but engagement not verified; not enabled.
- STRATA_ADAPT_OVERLAP=1: did not engage cleanly ([25.7, 40.0, 27.2]). Not
  enabled.
- STRATA_PCIE_BALANCE=1: warm reps +4-5%, within noise; draft acceptance down
  (0.598). Not enabled.
- STRATA_ATTN_PRE75=0: sweep 1's -6% did not reproduce ([35.9, 27.1, 36.8],
  median 35.9 ~ baseline; the 27.1 rep looks like an AER-retry stall). Keep
  STRATA_ATTN_PRE75=1.
- --aux-cpus auto: no-op as predicted (34.7 vs 34.9).
- ESP (experimental-speed-projection, cvec arm): no speedup (34.8 vs 34.9),
  slightly better draft acceptance (0.670). Do not enable in production.

## Quality probe for STRATA_ROUTE_TAIL_SKIP (local/bench/quality-probe.py)

Deterministic coding task (temperature=0, max_tokens=1200): implement a
`split_ranges` parser. The response's python block is exec'd with stdout
captured and checked on 4 value cases + 4 ValueError cases + the printed demo
output = 9 checks.

| setting | decode t/s (probe) | quality |
|---|---|---|
| R=0 | 37.5 | 9/9 |
| R=7 | 48.0 | 9/9 |
| R=1 | 76.8 | 9/9 (draft acceptance 408/529 = 0.77) |

An early probe run scored R=0 at 8/9; that was a probe bug (max_tokens=600
truncated the response mid-demo-line, and the demo check missed stdout
printed by the exec'd block). After fixing both, the saved R=0 response
re-judges to 9/9. The probe therefore cannot distinguish the three settings on
this task - all pass.

The case for shipping R=1: it passes identically, gives +78% decode in the
benchmark (+105% on this probe), improves draft acceptance, and upstream
already ships a less aggressive form of the same skip (R=7) as the CUDA
default with documented acceptable KL. Caveat: one easy task is not proof of
parity on harder workloads, and R=1 zeros more expert contributions than the
vendor default. Rollback if quality regresses in real use: set
STRATA_ROUTE_TAIL_SKIP=0 (or 7) in local/strata/strata-unsloth-ud-iq4_xs.json
and restart via local/strata/run-unsloth-ud-iq4_xs.sh.

## Final production settings (committed)

local/strata/strata-unsloth-ud-iq4_xs.json env (8 vars): STRATA_HC_Q8=1,
STRATA_ROUTE_TAIL_SKIP=1, STRATA_SM70_TABLE=1, STRATA_BF16_TC=1,
STRATA_PROMPT_ATTN_OLD=1, STRATA_ATTN_PRE75=1, STRATA_MMVQ_IL=1,
STRATA_GR_FAST=1. No STRATA_EXP_MODE (see root-cause section above).

Validated via the launch script: first request after a fresh start 68.6 t/s,
expert cache 88.8% hit, zero new NVRM Xids (last one still the intentional
bisection at Oct 28 21:55).

## Model comparison: UD-Q4_K_XL vs production UD-IQ4_XS (2026-10-29)

local/bench/model-bench.sh, quickbench2 methodology (WARMUP=1 discarded,
REPS=3, same 40-block prompt = 2026 tokens, max_tokens=96, temperature=0,
port 8092). Q4_K_XL pack: tools/iq_pack.py --gguf <shard 1> --out
Strata-data/packs/unsloth-ud-q4_k_xl --compat-bf16 (1079 tensors, 303 native,
arena 1.38 GiB, v4 native_experts.txt: layer 11 gate/up/down split across
shards). Same 8 production env vars as IQ4_XS, incl. STRATA_ROUTE_TAIL_SKIP=1.

| metric | UD-IQ4_XS (production) | UD-Q4_K_XL | delta |
|---|---|---|---|
| decode avg t/s | 61.9 | 52.2 | -16% |
| decode reps | [61.8, 60.8, 63.1] | [50.6, 53.4, 52.7] | |
| prompt avg t/s | 201.3 | 151.0 | -25% |
| draft acceptance | 0.706 | 0.704 | ~equal |
| experts in RAM | ~63 GiB | 71.73 GiB | +9.7 GiB |
| quality top-1 / KL | 88.91% / 0.101 | 93.09% / 0.056 | better |

Expert cache hit 88.4-94.6% under R=1 (same engagement as the IQ4_XS
baseline, no "+N% over PCIe" suffix), so the gap is not a config artifact:
Q4_K_XL expert blobs are larger per miss, and every miss costs more on the
PCIe Gen3 x8 link. Prefill activates many experts per batch, which is why
prompt ingest loses proportionally more (-25% vs -16%).

Conclusion for this box: IQ4_XS stays the speed pick; Q4_K_XL buys ~4.2 top-1
points and half the KL at a 16-25% speed cost. NVFP4 (next section) is the
third data point.

## Model comparison: NVFP4 (GPTQ) vs production UD-IQ4_XS (2026-10-29)

Pipeline: strata/tools/nvfp4_convert.py --model local/models/nvfp4-conv
--outfile local/models/Qwen3.8-Flash-Next-NVFP4.gguf (in-process llama.cpp
converter, 444 s) -> 73.9 GB GGUF, 1511 tensors: 144 NVFP4 expert weights
(blk.N.ffn_{gate,up,down}_exps.weight [2560,640,512]) + 144 F32 .scale
tensors (per-expert weight_scale_2 tails), 302 Q8_0, 484 BF16, 580 F32.
PLE is not in the GGUF by design: the engine reads it from --ple-gguf, which
points at the kept IQ4_XS shard 2 (IQ4_NL, whitelisted). The converter's
index_tensors() opens every part listed in model.safetensors.index.json, so
the skipped 53.7 GB model-fp8-mtp-ple.safetensors had to be removed from a
sanitized copy of the index (local/models/nvfp4-conv/: symlinks + index with
3201 PLE entries dropped); MTP head tensors are dropped by the base converter
and the engine uses Strata-data/mtp/rt.

Pack: tools/iq_pack.py --gguf <NVFP4.gguf> --out Strata-data/packs/nvfp4 ->
index.txt 1367 tensors, 302 native, arena 1.43 GiB; experts.bin 63.28 GiB
(48 layers x 512 experts, uniform 2764816 B = 2.76 MB blobs, v3
native_experts.txt, type 40 NVFP4). Config strata-nvfp4.json = production env
minus STRATA_ROUTE_TAIL_SKIP (R=7 costs ~3.5 quality points on NVFP4).

| metric | UD-IQ4_XS (production) | UD-Q4_K_XL | NVFP4 (GPTQ) |
|---|---|---|---|
| decode avg t/s | 61.9 | 52.2 (-16%) | 30.6 (-51%) |
| decode reps | [61.8, 60.8, 63.1] | [50.6, 53.4, 52.7] | [28.9, 33.2, 29.8] |
| prompt avg t/s | 201.3 | 151.0 (-25%) | 175.1 (-13%) |
| draft acceptance | 0.706 | 0.704 | 0.561 |
| experts in RAM | ~63 GiB | 71.73 GiB | 63.28 GiB |
| GPU expert cache | 8 GiB budget | 2744 experts / 8 GiB | 3107 experts / 8 GiB |
| cache hit (R=1) | ~88-95% | 88.4-94.6% | 50.1-68.9% |
| PCIe share of routed experts | ~0 | ~0 | +1.2-1.3% |
| quality top-1 / KL | 88.91% / 0.101 | 93.09% / 0.056 | 92.09% / 0.070 |

Diagnosis (server.log): experts loaded 63.28 GiB at 2.16 GiB/s; the GPU
expert cache holds 3107 NVFP4 experts in the same 8.00 GiB VRAM budget that
holds only 2744 Q4_K_XL experts, yet hit rate is just 50-69% (NVFP4 routing
is more diverse vs the calibrated profile). PCIe is NOT the bottleneck: only
+1.2-1.3% of routed experts go over the link, so even a 100% cache hit would
gain ~1%. The decode loss is compute-bound: the binary's NVFP4 mmq configs
target ampere and pascal-dp4a; Volta (sm_70) sits between them, so expert
matmuls run slow. Prompt ingest (175 t/s) beats Q4_K_XL (151) but still
trails IQ4_XS (201). Draft acceptance drops to 0.561: NVFP4 quantization
noise lowers draft/target agreement, so speculative decoding accepts fewer
tokens per step.

Conclusion: on this V100 box NVFP4 is the worst decode option by far (-51%
vs production) despite the smallest RAM footprint; its only win is prompt
ingest vs Q4_K_XL. IQ4_XS remains the production pick for speed; Q4_K_XL is
the quality pick if a 16-25% speed cost is acceptable.

Disk: after the bench, local/models/nvfp4/ and nvfp4-conv/ (79 GB of
safetensors) were deleted; the 73.9 GB GGUF + 64 GiB experts.bin pack stay on
disk for re-runs.

# Strata v0.2.8 rebuild + V100 optimization audit, 2026-10-31

## Rebuild (v0.1.45 -> v0.2.8)

Fork updated to upstream v0.2.8 and rebuilt for sm_70. Build recipe that works
on this box: CC=gcc-14 CXX=g++-14 plus -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14
(system gcc 15 breaks nvcc host detection). New binary + libstrata_kq.so swapped
into strata/engine-cuda12/ (old ones kept as *-v0145-backup). Pack and expert
profile formats are compatible, no regeneration needed. Load speed improved to
2.92 GiB/s (was 2.55); warm gen 52.5 t/s at 85.6% cache hit in the smoke test.
The three re-ported llama-server endpoints (/apply-template, /tokenize, /infill)
verified live on the new engine.

## Baseline on v0.2.8 (production env, no extra knobs)

quickbench2 methodology: 1 warmup discarded, 3 reps x 40 blocks of ~35 words
(~2026 prompt tokens), max_tokens=96, temperature=0.

| metric | v0.2.8 baseline | old v0.1.45 numbers (NOT comparable) |
|---|---|---|
| decode avg t/s | 40.6 (reps 38.7/37.1/46.1) | 46.0/51.0/53.9 |
| prompt avg t/s | 151.2 | 151.0 |
| draft acceptance | 0.6047 | 0.704 |

The decode baseline dropped vs the old binary's recorded reps; the old numbers
came from a different engine build and are not a valid comparison. All deltas
below are against the v0.2.8 baseline measured the same day.

## Single-arm A/B (one experimental knob at a time)

| arm | knob(s) | decode t/s | delta | prompt t/s |
|---|---|---|---|---|
| adaptlag | STRATA_ADAPT_LAG=2 | 60.5 (reps 65.6/58.6/57.4) | +49.0% | 152.2 (+0.7%) |
| scores-thread | STRATA_SCORES_THREAD=1 | 54.9 (reps 51.8/49.4/63.5) | +35.1% | 152.3 (+0.7%) |
| adaptasync | --adapt-async 1 | 46.8 | +15.1% | 152.2 (+0.7%) |
| dequantwide | STRATA_DEQUANT_WIDE=1 | 43.1 | +6.2% | 151.2 (flat) |
| adaptoverlap | STRATA_ADAPT_OVERLAP=1 | 42.4 | +4.4% | 150.9 (-0.2%) |
| bitexact | ATTN_DAILY+B6_MMVQ_ROWS+QSA_ROWS | 41.2 | +1.3% | 152.4 (+0.8%) |
| pfwmma | STRATA_PF_WMMA=1 | 41.2 | +1.5% | 151.1 (flat) |
| v100small | ARGMAX_ROWS_SM70+GR_NORM_FAST+Q8_GATHER_WIDE=0 | 40.9 | +0.6% | 151.1 (flat) |
| gdnrecwarp2 | STRATA_GDN_REC_WARP=2 | 40.3 | -0.9% | 152.2 (+0.7%) |

Long prompt (~30K tokens, PROMPT_REPEATS=600): baseline decode 48.5 / prompt
96.6; STRATA_KV_PREFETCH=1 gives decode 51.2 (+5.6%) / prompt 97.9 (+1.3%).

The top three knobs are all expert-cache scheduling options: they change when
and how experts are prefetched from the 71.7 GiB RAM pool over PCIe Gen3 x8.
On this box (x8 link, ~2.5-2.9 GiB/s effective) that pipeline is exactly where
decode stalls, which explains why the gains far exceed the P100 numbers in the
upstream docs (+3.5% for adapt-lag there).

## Combos: the wins do not stack

| combo | knobs | decode t/s | vs best single arm |
|---|---|---|---|
| combo-a | LAG=2 + SCORES_THREAD=1 + --adapt-async 1 | 45.8 (reps 42.6/48.3/46.5) | -24% vs adaptlag alone |
| combo-b | combo-a + ADAPT_OVERLAP=1 + DEQUANT_WIDE=1 | 49.8 | -18% vs adaptlag alone |

Running two or more cache-scheduling strategies at once makes them fight over
the same prefetch pipeline and adds jitter; the result is worse than any single
arm. Pick exactly one scheduling strategy, and on this box it is ADAPT_LAG=2.

## Confirmation re-runs

- adaptlag re-run (short prompt): decode 55.4 avg / 60.6 median -- reproduces
  the win (+36% avg vs baseline avg 40.6, +57% median vs baseline median
  38.7). Its worst rep still beats the baseline's best rep (57.4 vs 46.1), so
  this is not an outlier.
- adaptlag + KV_PREFETCH (long ~30K): decode 49.9 / prompt 95.5, i.e. +2.9%
  over the long baseline and within noise of KV_PREFETCH alone (51.2). The two
  do not conflict at long context.

## Persisted production settings

local/strata/strata-unsloth-ud-q4_k_xl.json env now carries, on top of the
existing eight documented Volta vars:

- STRATA_ADAPT_LAG=2 -- +36..49% decode (the big win)
- STRATA_KV_PREFETCH=1 -- +2.9..5.6% at long context, neutral at short

Expected production decode: ~55-60 t/s short context (was 40.6), ~50 t/s at
~30K context (was 48.5). All other knobs tested were left OFF: they are either
negligible, negative, or conflict with ADAPT_LAG.

## Health notes

AER during the 32-minute chain: +10264 correctable Timeout errors, zero
uncorrectable -- the known harmless PCIe storm under sustained load. V100
selected by UUID throughout; no engine crashes, all 12 arms + 5 follow-up runs
completed cleanly.
