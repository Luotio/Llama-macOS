# Local setup: Qwen3.8-27B on 2x V100-SXM2-16GB

Machine-specific scripts and notes, tracked in this fork. Nothing here
modifies llama.cpp source files; it only wraps the built binaries.
Model weights and benchmark results are gitignored (see local/.gitignore).

## Machine layout

| nvidia-smi index | Card            | Role                                    |
|------------------|-----------------|-----------------------------------------|
| 0                | GTX 1080        | desktop display (X/KDE), never use     |
| 1, 2             | V100-SXM2-16GB  | inference (one of them is faulty, see below) |

WARNING: CUDA device indices do NOT match nvidia-smi indices on this box.
CUDA sees the V100s as devices 0 and 1 and the GTX 1080 as device 2, so a
naive `CUDA_VISIBLE_DEVICES=1,2` would pull the desktop GPU into the tensor
split. All scripts therefore address the V100s by UUID, which both CUDA and
nvidia-smi accept:

    GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c   (V100, HEALTHY, in use)
    GPU-f07f3af3-5f02-f364-0d37-2ab3a4c4afce   (V100, FAULTY, RMA pending - do not use)

PCI bus ID strings (full and short form) were tried for
CUDA_VISIBLE_DEVICES and yielded no devices; UUIDs are the reliable form.
PCI addresses change when a card is reseated, so they are not listed here;
see HARDWARE.md for the physical layout, the faulty-card evidence, and
diagnostic commands.

## Step 1: install CUDA toolkit (user runs, needs sudo)

    emerge --ask dev-util/nvidia-cuda-toolkit=12.9.1-r1

Why this exact version:

- CUDA 13.x removed Volta (sm_70) support entirely, so the bentoo binary
  overlay's 13.4.1 cannot target V100 at all.
- 12.9.1 is the last 12.x in the gentoo repo and still ships sm_70 SASS.
- The ebuild is unslotted, so pin the full version as above.
- It needs a host compiler older than gcc 15; portage will pull in gcc-14
  (source build, CPU only, one time).
- cuBLAS/cuDNN come inside the toolkit runfile, no extra packages needed.

## Step 2: build

    local/build.sh

Builds `llama-server`, `llama-cli` and `llama-bench` into `build/bin/` with
`CMAKE_CUDA_ARCHITECTURES=70-real` (V100-only SASS, no PTX fallback bloat).
Uses `-j$(nproc)`; this is a normal build load, not a GPU load.

## Step 3: get the model

No GGUF was found on disk yet. Put the Qwen3.8-27B UD-Q4_K_XL file at
`local/models/Qwen3.8-27B-UD-Q4_K_XL.gguf`, or pass any path as the first
argument to the serve script. The GGUF must contain the MTP module for
speculative decoding to work (the UD Q4_K_XL build does).

## Step 4: run

    local/serve-qwen3-27b.sh [path/to/model.gguf]

Defaults: 128k context, all layers on GPU, tensor split 1,1 across the two
V100s, flash attention on, KV cache q8_0/q8_0, MTP with n_max=3, port 8080.
Everything is overridable via env vars, see the script header.

## VRAM budget (32GB total)

- Weights: ~17GB (Q4_K_XL) + a bit for the MTP module
- KV cache at 128k tokens is the big variable. The server prints
  "KV self size" at startup; check it against the free VRAM per GPU.
- If it does not fit, in order of preference:
  1. `CACHE_K=q4_0 CACHE_V=q4_0 local/serve-qwen3-27b.sh` (halves KV again)
  2. `CTX=65536 local/serve-qwen3-27b.sh`
  3. drop `-ngl all` to e.g. `-ngl 40` so the tail layers + part of the KV
     cache fall back to system RAM (~99GB free)

Note: `--parallel N` multiplies the KV reservation by N (per-slot context),
so keep it at 1 while targeting 128k.

## Cooling

GPU 1 measured ~67C at idle before any load. To be gentle on the SXM2
cooling:

    sudo local/gpu-tune.sh apply 250     # 250W power limit (default 300W)
    local/gpu-tune.sh status             # temps, power, clocks
    sudo local/gpu-tune.sh reset         # back to 300W

The limit is per-GPU via nvidia-smi and does not survive driver reload.
If temps climb above ~85C under load, drop further (`apply 200`) or reduce
`--batch-size`/`--ubatch-size` in the serve script.

## Smoke test (light only)

Once built, before a long session:

    build/bin/llama-server --version
    CTX=4096 local/serve-qwen3-27b.sh     # short startup check, then Ctrl-C

No benchmarking is run by any of these scripts.
