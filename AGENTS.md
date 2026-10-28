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
| `GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c` | V100 16 GB (sm_70) | 1 | **node 1** | **HEALTHY — the one to use** |
| `GPU-f07f3af3-5f02-f364-0d37-2ab3a4c4afce` | V100 16 GB | 2 | node 1 | **FAULTY (RMA) — NEVER use** |
| `GPU-6f4fedec-c775-e942-f39d-bd77cf3f3384` | GTX 1080 8 GB (sm_61) | 0 | node 0 | works, small |

(Indices re-verified 2026-10-18 after the two V100s were swapped between slots; UUIDs follow the physical modules.)

Driver 580.126.18. **PCIe link limit**: the healthy V100 is wired at Gen3 x8 (~5.6 GB/s measured host→device). This caps expert streaming — Strata rates on this box will be below what the upstream docs assume for full-width links; that is hardware, not a bug.

**PCIe AER instability (2026-10-27)**: under sustained heavy transfer (long prefills, expert streaming) the V100's Gen3 x8 link throws correctable AER errors — DLL Timeout on the GPU TX + BadDLLP on root port `0000:80:02.0` RX, 1095+ suppressed in seconds. A DMA stalled mid-verify then kills the engine with `verify: layer N never rang (unspecified launch failure)`. 10 such deaths in one boot **with** `--vram-reserve-mib 768` already set, so VRAM pressure is not the cause; the link is. **Physical path**: motherboard PCIe slot → passive adapter card → two SFF-8654 8i cables → riser board → V100 SXM2 socket (passive = invisible to `lspci`; `Retimer-` only means no *active* retimer). The 8i cables/connectors are the prime suspect. **The link is broken at idle, not just under load**: `aer_dev_correctable` after ~80 min uptime showed 179k correctable errors on the root port (171k BadTLP) and the GPU's TX Timeout counter climbing ~1.2/s with no workload — constant NAK/retransmit, overwhelmed under sustained transfer. ASPM is a dead end (`LnkCap: ASPM not supported` on this port; the runtime sysfs policy write is also blocked by the kernel's compile-time lockdown-integrity, EPERM even as root). No retimers in the path. Mitigations, in order: (1) **force Gen2** — `sudo setpci -s 80:02.0 CAP_EXP+0x30.b=0x02` and same on `81:00.0` (Link Control 2, low nibble = target speed; restore with `=0x03`), verify via `nvidia-smi --query-gpu=pcie.link.gen.current`; costs ~half the PCIe bandwidth but 4× the eye area; (2) reseat/reflow the SXM2 module or RMA — the other V100 on this box was already RMA'd, so suspect the baseboard. If Gen2 still throws AER errors at idle, it's hardware and only (2) fixes it. Do not run long-prompt benchmark sweeps until the link is stable; check `cat /sys/bus/pci/devices/0000:81:00.0/aer_dev_correctable` before trusting any benchmark. **setpci needs sudo**: as non-root it fails with a misleading `pcilib: Cannot open .../config` + `Instance #0 of Capability 0010 not found` (permission denial masquerading as missing capability).

**Physical part numbers (2026-10-28)**: riser board = **RBS162545-V53**, SXM2 adapter = **RTE16SXM2B-S**. Web research: RBS162545 has zero public documentation anywhere (not a Supermicro part — their risers use the `RSC-*` prefix; likely an OEM/internal number). RTE16SXM2B is a **third-party ADT-Link SXM2→PCIe x16 adapter** sold on AliExpress/eBay/Amazon/Newegg (~€60–300, variants -V direct-slot and -S cable/breakout; some with active cooling + fan control). Known-issue thread: Level1Techs #252310 — the -V direct-slot variant hangs POST at Q-code 8D on an ASUS X99-E/WS (PLX PEX8747 switch timing theory); our -S cable variant boots fine, so that failure mode doesn't apply here, but it confirms these adapters are cheap no-support hardware with a track record of PCIe link problems. Practical consequence: the whole passive chain (adapter + 2× SFF-8654 8i cables + riser) is replaceable off AliExpress for well under €100 — after the Gen2 discriminator test, replacement of the 8i cables first is the cheapest real fix.

**BIOS Gen3 EQ tuning (2026-10-28)**: user set Link Speed=Gen3, PCI-E ASPM=Disable, Gen3 Eq Mode=Advanced, Phase3=Mid @ Max Boost (SW Ovrd), Phase2=Hardware Adaptive, DeEmphasis=-6.0 dB. Result: link now trains **Gen1 x8** (was Gen3 x8) — the SW-override preset and/or -6.0 dB de-emphasis break Gen3 EQ convergence, so the link falls back; AER counters zero at 3 min uptime. Recall the error direction: root-port RX BadTLP = GPU→host = the **Phase 2** receiver (GPU TX) is the corrupt one; Phase 3 tunes the opposite direction, so a Phase 3 SW override targets the wrong side. Next BIOS tries, one at a time with reboot + Gen check + idle counters + 46k stress: (a) Phase3 back to Hardware Adaptive/Auto; (b) DeEmphasis -3.5 dB (short channel); (c) Eq Mode back to Auto. Note: the BIOS **Link Speed** option can force Gen2 directly if needed — cleaner than setpci (no sudo, survives reboot).

**BIOS round 2 (2026-10-28): GEN3 RESTORED.** User set Link Speed=Gen3 (forced), Gen3 Eq Mode=Auto, Phase3=Hardware Adaptive, Phase2=Hardware Adaptive, DeEmphasis=-3.5 dB. Result: link trains **Gen3 x8** (was Gen1 x8 after round 1). AER idle rate ~24 correctable/min (GPU TX Timeout ~8/min + RP RX BadDLLP/BadTLP/Timeout ~16/min) — ~10x better than the original storm (~2200+/min) but not zero; the passive chain still degrades the eye. Under sustained load (46k-token prefill, ~40 s): the AER storm returns (RP RX BadTLP +40632, BadDLLP +831; GPU TX Timeout +571 in one request) — **but the engine survives**: HTTP 200 in 39.9 s vs 127.2 s at Gen1 x8 (3.2x faster on that request), zero Xid, no data corruption (correctable errors are recovered by link retry). This empirically confirms the corrected diagnosis: the AER storm is a real but *harmless* signal-integrity issue; the earlier deaths were purely the `__trap()`. **Keep this BIOS config.** The residual storm's real fix remains hardware (replace the 8i cables first); until then, check AER deltas after long runs but do not treat correctable errors as engine failures.

**CORRECTED DIAGNOSIS (2026-10-28): the engine deaths are NOT caused by the PCIe link.** At Gen1 x8 (after the BIOS change) a 46k-token prefill died with the identical `verify: layer N never rang (unspecified launch failure)` error but **zero AER errors**; bisection showed 32k, 16k and even **500-token** prompts all die identically — not length- or load-dependent. Every death logs **Xid 43** ("GPU stopped processing") in `/var/log/messages`; history runs Oct 20 (intermittent) → Oct 27 (burst) → now (every request). ECC is clean, no retired pages. Isolation test: vanilla `llama-bench` on the same V100 (5× pp16384 + tg256 with Qwen3-4B-Q4_K_M) ran spotless — zero AER, no Xid, ~2965 t/s prefill. The GPU/driver/PCIe are healthy under sustained load; the Gen3 AER storm was a real but **separate** signal-integrity issue on the passive chain, not the cause of the deaths. Mechanism: `unspecified launch failure` is a *sticky* CUDA error — some kernel did an illegal memory access and poisoned the context; "layer N never rang" is just where the host happened to be waiting (verify.cpp spins on the seq flag, polls `cudaStreamQuery` every 2 ms). The dying layer varies (1 or 2) → data-dependent OOB. Breadcrumb trace (`STRATA_VERIFY_TRACE=1`) shows the GPU completes all of layer 0 including the MoE expert-cache stages and dies in layer 1's **"VRAM hits"** stage — the expert gather over VRAM-cached experts. Prime suspect: commit `ec8181f673` (Oct 27 21:50) added six env vars to the config (`STRATA_BF16_TC`, `STRATA_PROMPT_ATTN_OLD`, `STRATA_ATTN_PRE75`, `STRATA_EXP_MODE`, `STRATA_MMVQ_IL`, `STRATA_GR_FAST`); the failure burst started 6 minutes later. Data files (expert profile, pack) unchanged since Oct 17. Bisection tooling: `local/bench/strata-bisect-config.py [vars...]` writes `local/strata/strata-bisect.json` (config `env` overrides shell env in server.py's `child_env`, so a config copy is required); test with `python3 local/bench/pcie-stress-prompt.py http://127.0.0.1:8080/v1/chat/completions 500` (HTTP 400 + "never rang" in ~10 s = death; HTTP 200 = clean).

**ROOT CAUSE FOUND AND FIXED (2026-10-28): `STRATA_EXP_MODE=1`.** Bisection: baseline (no new vars) OK, first half (`BF16_TC`+`PROMPT_ATTN_OLD`+`ATTN_PRE75`) OK, second half (`EXP_MODE`+`MMVQ_IL`+`GR_FAST`) dies, **`EXP_MODE=1` alone dies**. Mechanism: `STRATA_EXP_MODE` selects the native-expert layout mode; on this V100 the default (with `STRATA_SM70_TABLE=1`) is already **mode 8** (SwiGLU+q8_1 fused in shared memory, V100-verified 227→159 us). Mode 1 is the **gfx906-only 64-lane wavefront layout**: its kernels (`native_gu_amd_kernel<T,1>`/`native_down_amd_kernel<T,1>`, 256-thread blocks) call `row_dot64()` in `strata/src/kernels/cuda/iq_kernels.cu`, which is `__trap()` on every non-HIP build — halting the GPU = Xid 43 + sticky `unspecified launch failure`. How it got there: the autotune harness's own "baseline" arm contained `exp_mode=1` and `write_config` writes each arm into the production config, so running the autotune poisoned it. **Fix**: `STRATA_EXP_MODE` removed from `local/strata/strata-unsloth-ud-iq4_xs.json` (unset = mode 8 on sm70); both bench scripts patched to treat empty exp_mode as "remove the var" and all arms now use unset. Validated: 500 / 16k / 46k-token requests all HTTP 200, zero AER, no new Xid. (The Oct 20 intermittent Xid 43s were a separate issue — `--vram-reserve-mib` had been removed; re-adding it on Oct 27 fixed that part.)

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
- Config: `local/strata/strata-unsloth-ud-iq4_xs.json` (args, env, gpu index 1). Log: `local/strata/strata-unsloth-ud-iq4_xs.log`.
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
