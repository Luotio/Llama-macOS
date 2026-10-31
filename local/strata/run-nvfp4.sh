#!/bin/sh
# NUMA: the healthy V100 (UUID GPU-40faaa73-..., nvidia-smi index 1 as of 2026-10-18) sits on node 1 (CPUs 16-31,48-63).
# Bind the whole server to node 1 so the host thread, pool workers and first-touch
# page cache are local to the GPU's PCIe root; overflow beyond node 1's 48 GB
# falls back to node 0 automatically. The engine's CPU pool honors this mask.
cd "/mnt/git/local-llama-cpp/strata"
exec numactl --cpunodebind=1 "/mnt/git/local-llama-cpp/strata/.venv/bin/python" "/mnt/git/local-llama-cpp/strata/serve/server.py" "--engine" "strata" "--config" "/mnt/git/local-llama-cpp/local/strata/strata-nvfp4.json" "--port" "8080" "--open"
