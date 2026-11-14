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

| UUID                                   | Status                | PCI slot |
|----------------------------------------|-----------------------|----------|
| GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c | in use (AER storm on its link, see 2026-11-14 section below) | 82:00.0 |
| GPU-e487578c-0be1-8c0b-ad8b-0ab0615c4850 | in use, RMA replacement (2026-11-13), link clean | 81:00.0 |

The faulty card GPU-f07f3af3-5f02-f364-0d37-2ab3a4c4afce (described below)
lived in the 81:00.0 slot; it was shipped for RMA on 2026-11-13 and replaced
by GPU-e487578c.

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

## PCIe AER link isolation (2026-11-14, live counters)

Question: only one of the two bifurcated links shows AER errors - is that
real, or does Linux fail to separate the bifurcated x16 slot? Answer: it is
real. The bifurcation produces TWO independent root ports (80:02.0 ->
81:00.0, 80:02.2 -> 82:00.0), each with its own link training, its own AER
capability and its own sysfs counters; the SXM2 adapter runs ONE SFF-8654
8i cable per GPU socket. Nothing is shared between the two links, so a
per-link counter cannot be polluted by the other side.

Counters read 2026-11-14 (sysfs aer_dev_correctable, both ends of each link;
per-boot counters, so absolute values are for that uptime):

    link 81:00.0 (GPU e487578c, the RMA replacement):
        root port 80:02.0 RX:  all zero            GPU TX:  all zero, Gen3 x8
    link 82:00.0 (GPU 40faaa73):
        root port 80:02.2 RX:  BadTLP 2507464  Timeout 300033  BadDLLP 9829
                               TOTAL_ERR_COR 2809647
        GPU TX:                Timeout 3008067, Gen3 x8

Interpretation:

- The GPU-side "TX Timeout" and the root-port-side "RX BadTLP/Timeout"
  count the SAME physical event from both ends: GPU->host transfers that
  arrive corrupted or never get ACKed. That is the Phase 2 (GPU-TX)
  direction of the link eye, matching the 2026-10 diagnosis.
- The 81:00.0 path (cable + connectors + riser socket + new GPU) is now
  empirically proven clean: the faulty f07f3af3 card lived on that side,
  and its replacement shows zero errors on both ends. The 82:00.0 side has
  been the storming one since the October observations.
- Suspect list on the 82:00.0 path, in order: (1) the 8i cable (prime
  suspect), (2) the riser SXM2 socket / connector on that side, (3) the
  GPU 40faaa73's own TX - least likely: a vanilla llama-bench on it ran
  spotless, and the errors appear on both ends of the link (a channel
  problem, not a sender problem).
- Rates under load are far above the October idle baseline (~24/min): the
  27B server is tensor-split across BOTH V100s, so the bad link is under
  sustained transfer (observed ~100k/min on the 82:00.0 side). Still
  correctable-only: link retries recover the errors, no Xids. The llama-mon
  dashboard (local/monitor, port 8085) shows this live as the per-GPU
  "PCIe AER (rate /min + cumulative)" sparkline.

Planned discriminator test - swap the 8i cables (or, equivalently, swap the
two GPUs between the riser sockets; both give the same electrical mapping).
GPU 40faaa73 has never been tested on the clean cable, so this is the test
that separates "bad cable" from "bad GPU TX". When doing it:

- After the swap the BDFs follow the root ports, the UUIDs follow the cards:
  40faaa73 will appear at 81:00.0, e487578c at 82:00.0.
- Re-check any config that selects a card by nvidia-smi index (Strata
  local/strata/*.json "gpu" field, llama-server tensor split) against the
  UUIDs before starting inference, per the AGENTS.md rule.
- Snapshot the counters before and after (and watch the monitor in between):

      for bdf in 80:02.0 80:02.2 81:00.0 82:00.0; do
        echo "== $bdf"; cat /sys/bus/pci/devices/0000:$bdf/aer_dev_correctable | tr '\n' ' '; echo;
      done

- Reading the result: if the storm FOLLOWS GPU 40faaa73 (now shows up on
  root port 80:02.0 RX + the GPU's TX at 81:00.0), the module's TX is the
  suspect and the card goes back for RMA. If the storm STAYS on root port
  80:02.2 (now serving e487578c), the fault is in that side's cable /
  connector / riser socket and the 8i cable is replaced first (cheapest
  real fix, per the October analysis).

Side finding (2026-11-14): the GTX 1080 (native slot 00:03.0, PCI 04:00.0)
retrained Gen3 x16 -> Gen1 x16 -> Gen3 x16 within a couple of minutes, with
zero AER on both ends of its link. A soft "could not converge at Gen3 that
one time" event, not data corruption; a separate, minor issue from the
82:00.0 storm.

## Benchmarks

See local/bench/RESULTS.md for numbers and local/bench/README or the scripts
for how to run them. Single-GPU runs use:

    CUDA_VISIBLE_DEVICES="GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c" TS=1 ...
