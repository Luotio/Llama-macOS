# AGENTS.md — local LLM inference box (llama.cpp + Strata)

Repo root: `/mnt/git/local-llama-cpp`. Goal: fast local inference for coding workloads.

## Repo layout

| Path | What it is |
|---|---|
| `llama-cpp/` | llama.cpp source, **committed as a subtree** of this repo |
| `build` | symlink → `llama-cpp/build` (prebuilt binaries live there) |
| `local/models/` | GGUF model files (large; `.done` markers + download logs are gitignored) |
| `local/bench/` | benchmark scripts + `RESULTS.md` (all recorded numbers live here) |
| `local/strata/` | Strata server config (`*.json`) + run script — tracked, edit here |
| `local/serve-qwen38-flash-next.sh` | llama.cpp server launcher (port 8095, MTP sidecar) |
| `strata/` | **tracked private fork** of Niko1221/Strata (fork base v0.1.45, upstream commit 477ac13c). Prebuilt engine at `strata/engine-cuda12/strata` (gitignored; rebuild via setup.sh). Edits here are normal commits to this repo; build artifacts stay out via `strata/.gitignore`. Local server config/scripts live in `local/strata/` |
| `Strata-data/` | untracked runtime data: `packs/` (model pack + tokenizer), `mtp/rt` (MTP draft) |

## Machine facts (read before touching anything performance-related)

- Gentoo Linux (no apt/dnf/pacman). 2x Xeon E5-2697A v4 @ 2.60 GHz, **Broadwell: AVX2 only, no AVX-512**. 32 cores / 64 threads, ~109 GB DDR4, NVMe.
- **NUMA**: node0 = CPUs 0-15,32-47 (64 GB); node1 = CPUs 16-31,48-63 (48 GB). Distance 10/21.
- CUDA 12.9 at `/opt/cuda`. **nvcc 12.9 requires gcc-14**: `export CC=gcc-14 CXX=g++-14` before any CUDA build (gcc 15 is rejected).

### GPUs — ALWAYS select by UUID, never by index habit

| UUID | Card | nvidia-smi index | NUMA node | Status |
|---|---|---|---|---|
| `GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c` | V100 16 GB (sm_70) | 2 | **node 1** | **HEALTHY — the one to use** |
| `GPU-f07f3af3-5f02-f364-0d37-2ab3a4c4afce` | V100 16 GB | 1 | node 1 | **FAULTY (RMA) — NEVER use** |
| `GPU-6f4fedec-c775-e942-f39d-bd77cf3f3384` | GTX 1080 8 GB (sm_61) | 0 | node 0 | works, small |

Driver 580.126.18. **PCIe link limit**: the healthy V100 is wired at Gen3 x8 (~5.6 GB/s measured host→device). This caps expert streaming — Strata rates on this box will be below what the upstream docs assume for full-width links; that is hardware, not a bug.

### NUMA rule of thumb

The healthy V100 is on **node 1**. Anything CPU-side that feeds the GPU (host thread, CPU pool workers, first-touch page cache) should run on node 1: `numactl --cpunodebind=1`. The Strata engine's CPU pool honors the process affinity mask (one worker per physical core). Overflow beyond node 1's 48 GB falls back to node 0 automatically.

## Models

- **Qwen3.8-Flash-Next** (`qwen4exp`): 125B MoE / ~6B active + 51B PLE table, UD-IQ4_XS quant, 93.7 GB, 3 shards in `local/models/`. MTP draft at `Strata-data/mtp/rt`.
- Qwen3.8-27B UD-Q4_K_XL: tier-3 benchmark baselines (see `local/bench/RESULTS.md`).

## Running Strata (primary server)

```sh
cd /mnt/git/local-llama-cpp && local/strata/run-unsloth-ud-iq4_xs.sh
# or detached: nohup local/strata/run-unsloth-ud-iq4_xs.sh >/dev/null 2>&1 &
curl -s http://127.0.0.1:8080/health   # poll until "loaded":true (~55 s)
```

- OpenAI-compatible API at `http://127.0.0.1:8080/v1`.
- Config: `local/strata/strata-unsloth-ud-iq4_xs.json` (args, env, gpu index 2). Log: `local/strata/strata-unsloth-ud-iq4_xs.log`.
- Stop: `pkill -f "strata/serve/server.py"; pkill -f engine-cuda12` — the Python wrapper can survive an engine crash, so kill both.
- Key env in config: `STRATA_SM70_TABLE=1` (opt-in Volta decode kernels, bitwise-identical output, ~+10-15% decode), `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=0`.
- Key args: `--prefill auto:32768`, `--max-context 131072` (KV streaming, int8, kv-resident 32768; 131k costs only ~0.8 GiB extra RAM), `--conversation-cache-mib 8192` (prefix reuse across agent turns), `--spec 4 --spec-min-p 0.5 --mtp .../Strata-data/mtp/rt`, `--expert-cache auto`.
- Engine build: `strata/engine-cuda12/`, Release/-O3, CUDA arch 70, `STRATA_EXPERIMENTAL_SM60=ON`. Already optimally configured; don't rebuild casually.

### VS Code Copilot integration (delft-solutions.llama-copilot)

The extension speaks llama-server's NATIVE protocol, not OpenAI paths. The fork's serve layer adds the endpoints it calls on every request: `POST /apply-template` (render the chat template only), `POST /tokenize`, and `POST /infill` (answered with empty content on purpose - a reasoning model is no good at inline ghost text). Chat itself goes to `/v1/chat/completions`. Settings: endpoint `local` = `http://127.0.0.1:8080`; the model appears as `qwen3.8-flash-next-unsloth-ud-iq4_xs@local`. Note: the server injects a "Reasoning effort is set to ..." system message (upstream EFFORT_TEXT feature; per-request `reasoning_effort` or config-level default).

## Running llama.cpp (comparison / fallback)

```sh
local/bench/run-bench.sh                          # benchmark runner
PORT=8095 local/serve-qwen38-flash-next.sh        # server, ctx 131072, KV q4_0, MTP sidecar
local/bench/mtp-bench.py --url http://127.0.0.1:8095/v1 --label fn-mtp --reps 3
```

Flash-Next recipe on this box: `--n-cpu-moe 999 -ot 'per_layer_token_embd=CPU' --moe-cache-mib 2048` (MoE experts on CPU, PLE table on GPU). Build archs: `61-real;70-real`.

## Results and conventions

- All benchmark numbers go in `local/bench/RESULTS.md` — append new sections there, keep the existing style (plain ASCII, no em dashes).
- **Never commit or push without explicit user approval.**
- The run script binds to NUMA node 1 via `numactl`; if you change CPU/GPU placement, re-check `nvidia-smi topo -m` and `/sys/class/pci_bus/*/device/.../numa_node` first.
