# Benchmarking

Two tiers, both designed to be re-run after any change (build flags, KV cache
type, power limit, MTP settings) and compared against earlier runs.

Results accumulate in `results/`:

- `bench.csv` - one row per test (pp###/tg###) per run, with environment metadata
- `server-runs.csv` - one row per server-bench run, pointing at the raw JSON
- `<label>-<timestamp>.jsonl` / `.json` - raw tool output

Only the latest run per label is shown by compare.py; older runs stay in the
CSVs and can be inspected directly.

## Tier 1: llama-bench (raw pp/tg speed)

    local/bench/run-bench.sh baseline

Runs `llama-bench` with the same settings as the production serve config
(depth 131072, KV q8_0/q8_0, tensor split 1,1, flash attn on, batch 2048/512)
and records avg t/s per test plus power limit, start/end temps and clocks.

Defaults: pp512/tg128, 5 repetitions (each preceded by a warmup run).
Everything is overridable via env vars, see the script header, e.g.:

    DEPTH=32768 PP=4096 TG=256 REPS=3 local/bench/run-bench.sh fa-off -fa off

Sweep multiple values in one process (faster, no reload between):

    local/bench/run-bench.sh sweep -p 512,2048 -n 128,256

Note: tier 1 cannot exercise MTP/speculative decoding; for that use tier 2.
These runs DO load the GPUs (that is the point) - keep the power limit in
place and watch temps with `local/gpu-tune.sh status` from another shell.

## Tier 2: SPEED-Bench via the server (end-to-end + MTP acceptance)

Measures what a client actually sees: prefill/decode throughput, latency and
MTP draft acceptance rate, using the repo's own speed-bench client against a
running llama-server.

One-time setup (portage):

    emerge --ask dev-python/datasets dev-python/tqdm

The first run downloads nvidia/SPEED-Bench from huggingface.co (cached after).

Workflow:

    local/serve-qwen3-27b.sh &            # start server (spec-type draft-mtp)
    local/bench/server-bench.sh mtp-on
    # stop server, restart with --spec-type none (or edit the serve script), then:
    local/serve-qwen3-27b.sh &
    local/bench/server-bench.sh mtp-off

Defaults are deliberately light for this machine's cooling: category=coding,
osl=512, limit=8 samples. Raise `SB_CATEGORY`, `OSL`, `LIMIT` for a heavier
measurement; keep them identical between the runs you want to compare.

## Comparing

    local/bench/compare.py bench  --baseline baseline
    local/bench/compare.py server --baseline mtp-off

Stdlib only, no dependencies. The server table adds pred-speedup and
latency-speedup ratios against the baseline label.

## Keeping runs comparable

- Same model file (the CSV does not record a hash; note it in the label).
- Same power limit - it is recorded per run, so a 250W vs 300W comparison is
  visible instead of silently mixed.
- No other GPU work running (desktop rendering on GPU 0 does not matter,
  nothing else should touch GPUs 1/2).
- Same label = new timestamped run; compare.py shows the newest per label.
