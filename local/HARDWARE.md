# Hardware setup and GPU diagnostics (taivasloimu)

Machine-specific physical layout and how to diagnose the GPUs. Written so it
survives loss of session history. Verified 2026-10-17.

## Physical layout

- CPU board: Intel Xeon E7 v4 generation, Gentoo Linux.
- Two Tesla V100-SXM2-16GB on a SXM2 riser/adapter card.
- The riser connects to ONE physical motherboard PCIe x16 slot with TWO
  passive SlimSAS cables (part: SFF-8654, "Mini-SAS HD"). One cable per GPU,
  each carrying 8 lanes. Passive = wires only, no electronics, so lspci sees
  no bridge or switch in the path.
- BIOS bifurcation of that x16 port is set to 8x8: one physical slot becomes
  two logical x8 root ports.

Resulting PCIe tree (verified with lspci -t):

    0000:80:02.0  Intel Xeon E7 v4 Root Port 2 (rev 01) -> V100 at PCI 81:00.0
    0000:80:02.2  Intel Xeon E7 v4 Root Port 2 (rev 01) -> V100 at PCI 82:00.0

- Expected healthy link state per card: Gen3 (8.0 GT/s) x8, about 16 GB/s.
  x8 is BY DESIGN (bifurcation), not degradation. A link stuck at Gen1
  (5.0 GT/s) IS abnormal.
- Storage is unrelated to the GPUs: Samsung NVMe PM9A1 direct at PCI
  02:00.0. There is no SAS/RAID HBA anywhere on this box.

## GPU identity (IMPORTANT: addresses change on reseat)

The two V100s were swapped between slots on 2026-10-17, so PCI addresses and
nvidia-smi indices moved. NEVER address the cards by index or PCI BDF in
scripts; always use the UUID (works in both CUDA_VISIBLE_DEVICES and
nvidia-smi -i):

| UUID                                   | Status                | PCI after 2026-10-17 swap |
|----------------------------------------|-----------------------|----------------------------|
| GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c | HEALTHY, in use      | 82:00.0                    |
| GPU-f07f3af3-5f02-f364-0d37-2ab3a4c4afce | FAULTY, RMA pending  | 81:00.0                    |

GTX 1080 (GPU-6f4fedec-c775-e942-f39d-bd77cf3f3384) at PCI 04:00.0 drives the
desktop display; never use it for inference.

CUDA device indices do not match nvidia-smi indices on this box. Verify any
selection with:

    CUDA_VISIBLE_DEVICES="GPU-..." build/bin/llama-server --list-devices

## The faulty card (GPU-f07f3af3) - RMA evidence

Fails a trivial 144 MiB cudaMalloc+cudaMemset+sync deterministically, in both
slots, on two dates. Kernel log shows die-level faults:

    Xid 48: uncorrectable ECC error in the SM at gpc 0, tpc 0
    Xid 172: LRF (local register file), Uncorrectable SRAM error in GPC 0 TPC 0
    Xid 13: Graphics SM Warp Exception on (GPC 0, TPC 0, SM 0)
    Xid 31: MMU Fault FAULT_PDE ACCESS_TYPE_VIRT_READ

Lifetime ECC counters (nvidia-smi -q): 2184 single-bit + 8 double-bit errors,
ALL in the SM register file, zero in HBM. This is die degradation, not a
cable or slot problem: a cable can corrupt data in transit but cannot flip
bits in on-die SRAM. The card also latches into a dead state after the first
failure (VRAM reads 0 GiB until power cycle).

Repro tool: local/scratch/cuda-malloc-test.cu, build with

    /opt/cuda/bin/nvcc -arch=sm_70 -Xcompiler -fPIC \
        local/scratch/cuda-malloc-test.cu -o local/scratch/cuda-malloc-test \
        -ccbin g++-14

Run per card:

    CUDA_VISIBLE_DEVICES="GPU-f07f3af3-5f02-f364-0d37-2ab3a4c4afce" \
        local/scratch/cuda-malloc-test

## Diagnostics cheat sheet

Current link state (want 8.0 GT/s and width 8 on both):

    cat /sys/bus/pci/devices/0000:81:00.0/current_link_speed
    cat /sys/bus/pci/devices/0000:81:00.0/current_link_width
    cat /sys/bus/pci/devices/0000:82:00.0/current_link_speed
    cat /sys/bus/pci/devices/0000:82:00.0/current_link_width

PCIe (AER) errors - they are logged against the ROOT PORTS 80:02.0 / 80:02.2,
not the GPU BDFs. Filter out boot-time "AER: enabled" noise:

    grep -ihE "aer.*(error message received|PCIe Bus Error)" /var/log/messages | tail

GPU faults (Xid) - filter by the card's current PCI address:

    grep -iE "xid" /var/log/messages | grep "PCI:0000:81:00" | sort -u

ECC counters for a card (nvidia-smi index, check with --query-gpu first):

    nvidia-smi -q -i <idx> | sed -n '/ECC Errors/,/Retired/p'

Temps/power while benchmarking: local/bench/temp-log.sh (5 s interval, logs
both V100 UUIDs to a file).

## What is NOT the problem (ruled out)

- Overheating: faults reproduce at ~26 C idle; no thermal Xids.
- SlimSAS cables as cause of the die fault: see above. (A bad cable would show
  as AER/Physical Layer errors or a failed/narrow link, not SM register-file
  ECC.)
- Slot problem: the fault followed the card across both slots.

## Benchmarks

See local/bench/RESULTS.md for numbers and local/bench/README or the scripts
for how to run them. Single-GPU runs use:

    CUDA_VISIBLE_DEVICES="GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c" TS=1 ...
