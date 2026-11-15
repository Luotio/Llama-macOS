# llama-mon: llama.cpp + GPU + host monitor with spot-price GPU throttling

Single-file Python 3 (stdlib only, nothing written to disk) dashboard at
`http://127.0.0.1:8085/` with in-memory sparklines plus a Nord Pool (FI
SDAC) day-ahead spot-price GPU power throttle.

Polls every `interval` seconds: llama-server/Strata metrics, NVIDIA GPUs
(nvidia-smi, one call per cycle), AMD GPUs (sysfs, works without ROCm;
augmented by `amd-smi` when present), PCIe AER correctable-error counters
(GPU side and upstream root port), and /proc host stats.

Every `price.fetch_interval` seconds the in-memory price table is made sure
to cover now .. now+24h for the configured zone, fetching missing days from
free mirrors of the Nord Pool / SDAC day-ahead clearing prices: Spotwatts
primary (`https://spotwatts.com/api/v1/{zone}/{YYYY-MM-DD}.json`, EUR/kWh,
attribution required if republished publicly), Volton FI fallback
(`https://public-data.volton.energy/...`, CC-BY-4.0). Nord Pool's own API
needs paid credentials; either mirror serves the identical auction results.

When `throttle.enabled`: above the threshold -> each configured GPU gets its
minimum power limit; at/below -> restored to `restore_watts`. Writes happen
only when current != target, so the loop is quiet most cycles and repairs
caps changed elsewhere. Setting caps needs root: run the service as root or
install the sudoers drop-in (see below).

## Running

    python3 local/monitor/monitor.py [--config PATH]

Config file search: `--config`, `$MON_CONFIG`, `/etc/llama-mon/config.json`.
Precedence: built-in defaults < config file < `MON_*` env vars (so the
OpenRC conf.d keeps working). `MON_HOST`, `MON_PORT`, `MON_URLS`,
`MON_INTERVAL`, `MON_WINDOW`, `MON_GPUS`.

Endpoints: `/` dashboard, `/api/stats` JSON (series + price curve + throttle
status), `/healthz` liveness.

## Config schema (local/monitor/config.example.json)

    host, port            bind address / port
    urls                  llama-server or Strata base URLs, one card each
                          (Strata is auto-detected via /health)
    interval, window      sample period s, history kept in RAM s
    gpus                  visible cards; [] = all. Identifiers: NVIDIA UUID
                          (full or substring), nvidia index ("1", "nvidia:1"),
                          amd:cardN, amd:BDF, bare BDF
    price.enabled         show the spot-price chart
    price.zone            Nord Pool zone, lowercase (fi, se-land, ...)
    price.timezone        local day boundary for the auction (Europe/Helsinki)
    price.sources         fetch order; "spotwatts" and/or "volton"
    price.include_vat     use incl-VAT prices where available (throttle and
                          chart follow this flag)
    price.fallback_vat_percent  used when a source lacks incl-VAT numbers
    price.revalidate_hours      how often cached days are re-fetched (prices
                          get revised; tomorrow appears after ~14:00 EET)
    price.fetch_interval  coverage check cadence, s
    price.chart_back_hours / chart_fwd_hours   chart window around now
    throttle.enabled      master switch (needs price.enabled too)
    throttle.threshold_c_per_kwh  above -> minimum caps
    throttle.interval     decision period, s (min 10)
    throttle.devices[]    one entry per GPU to throttle:
      gpu                 identifier, same forms as gpus[]
      restore_watts       number, or "default" = driver factory default
      low_watts           explicit cheap-hours floor (default: driver min)
      fallback_low_watts  used only if the driver cannot report its minimum

## Dashboard

One card per server / GPU, plus a price card and a host card. GPU cards have
a "Power limit (cap)" sparkline and show `capped` / `cap error` badges while
the throttle is engaged. The price card draws the day-ahead curve (step
line, dashed threshold, red above-threshold slices, "now" marker, THROTTLED /
OK badge, last throttle events).

## Privileges for cap writes

The systemd unit runs as root by default. For an unprivileged service user,
install `local/systemd/llama-mon.sudoers` (installer flag
`--with-sudoers=USER`) so the monitor can `sudo -n` just the cap-writing
commands (`nvidia-smi -pl`, `tee` of AMD hwmon `power1_cap`). Without either,
throttle checks log a permission error per device and caps stay untouched.

## AMD caveats

No AMD card exists on this box, so the AMD path was written blind against
sysfs ABI docs: stats come from `gpu_busy_percent`, `mem_info_vram_*`,
hwmon `power1_average`/`temp1_input`, PCI link attributes; cap writes go
hwmon `power1_cap` -> `sudo -n tee` -> `pp_power_limit` -> `amd-smi set`.
Verify on real AMD hardware before trusting it.

## Install as a service

systemd (recommended; copies the code to /opt/llama-mon, service never runs
out of the git checkout):

    sudo local/systemd/install.sh [--restart|--no-start|--force-conf|--uninstall]
    LLAMA_MON_PREFIX=/other/pfx sudo local/systemd/install.sh   # custom prefix
    DESTDIR=/tmp/stage local/systemd/install.sh                 # dry run
    sudo local/systemd/install.sh --with-sudoers=llama-mon      # unprivileged variant
    systemctl status llama-mon; journalctl -u llama-mon -f
    curl -fsS http://127.0.0.1:8085/healthz

The installer seeds `/etc/llama-mon/config.json` once from the repo template
(never clobbers it; `--force-conf` overwrites), refuses to co-install with a
running OpenRC llama-mon (same port), and removes unit + copied code again
with `--uninstall` (the /etc config is kept).

OpenRC (this box's init system; runs from the checkout, `/etc/conf.d/llama-mon
holds only optional MON_* overrides + MON_REPO/MON_USER, the JSON file holds
everything else):

    sudo local/openrc/llama-mon-install.sh [--enable|--restart|--force-conf|--uninstall]
    rc-service llama-mon start|stop|restart|status|health
    # /etc/conf.d/llama-mon: MON_USER (default root - needed for cap writes),
    # MON_REPO; uncomment MON_HOST/MON_PORT/MON_URLS/MON_INTERVAL/MON_WINDOW/
    # MON_GPUS/MON_CONFIG only to force values past the JSON config. The
    # installer seeds /etc/llama-mon/config.json once from the repo template
    # like the systemd one does. Log: /var/log/llama-mon/monitor.log
