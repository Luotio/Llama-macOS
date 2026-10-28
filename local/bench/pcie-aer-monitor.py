#!/usr/bin/env python3
"""Live PCIe AER correctable-error monitor for the V100 link.

Watches aer_dev_correctable on the GPU (81:00.0) and its root port
(80:02.0), showing 1s / 3s / 10s average rates per error type.
Start it, wiggle the cables/connectors while it runs, and watch which
column spikes.

Usage:  python3 pcie-aer-monitor.py
Keys:   q = quit

Notes:
- 'reset' in a column means the counter wrapped or the link retrained
  (counters zero out) inside that window.
- Safest to wiggle with the Strata engine stopped; a hard disconnect may
  make the driver reset the GPU (Xid in dmesg), which is informative, not
  harmful.
"""

import curses
import subprocess
import time

GPU = ("GPU 81:00.0", "/sys/bus/pci/devices/0000:81:00.0/aer_dev_correctable")
RP = ("RootPort 80:02.0", "/sys/bus/pci/devices/0000:80:02.0/aer_dev_correctable")
FIELDS = ["BadTLP", "BadDLLP", "Timeout", "RxErr"]
WINDOWS = (1, 3, 10)          # seconds
SAMPLE_S = 0.2                # sampling interval
SPARK_N = 150                 # sparkline length in samples (30 s)
SPARK_CHARS = " .:-=+*#%@"


def read_counters(path):
    try:
        with open(path) as f:
            out = {}
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[1].isdigit():
                    out[parts[0]] = int(parts[1])
            return out or None
    except OSError:
        return None


def link_state():
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=pcie.link.gen.current,pcie.link.width.current",
             "--format=csv,noheader,nounits", "-i", "1"],
            capture_output=True, text=True, timeout=5)
        gen, width = out.stdout.strip().split(",")
        return f"Gen{gen.strip()} x{width.strip()}"
    except Exception:
        return "?"


def window_rate(samples, field, window):
    """Average rate of `field` over the last `window` seconds.

    Returns a float (errors/s), None if there is not enough history yet,
    or 'reset' if the counter went backwards inside the window.
    """
    if len(samples) < 2:
        return None
    t1 = samples[-1][0]
    target = t1 - window
    i = 0
    while i + 1 < len(samples) and samples[i + 1][0] <= target:
        i += 1
    t0, d0 = samples[i]
    _, d1 = samples[-1]
    dt = t1 - t0
    if dt < 0.25:
        return None
    dv = d1.get(field, 0) - d0.get(field, 0)
    if dv < 0:
        return "reset"
    return dv / dt


def fmt_rate(r):
    if r is None:
        return "-"
    if r == "reset":
        return "reset"
    return f"{r:7.1f}"


def sparkline(values):
    if not values:
        return ""
    mx = max(values)
    if mx <= 0:
        return SPARK_CHARS[0] * len(values)
    top = len(SPARK_CHARS) - 1
    return "".join(SPARK_CHARS[min(top, int(v / mx * top))] for v in values)


def put(stdscr, y, x, s):
    try:
        stdscr.addstr(y, x, s)
    except curses.error:
        pass


def device_block(stdscr, y, name, samples):
    total = samples[-1][1].get("TOTAL_ERR_COR", 0) if samples else 0
    put(stdscr, y, 0, f"{name:<16}total {total:>10}")
    row = "".join(f"{fmt_rate(window_rate(samples, 'TOTAL_ERR_COR', w)):>12}"
                  for w in WINDOWS)
    put(stdscr, y + 1, 0, f"{'rate/s':<16}{row}")
    for k, field in enumerate(FIELDS):
        row = "".join(f"{fmt_rate(window_rate(samples, field, w)):>12}"
                      for w in WINDOWS)
        put(stdscr, y + 2 + k, 0, f"{field:<16}{row}")


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    gpu_samples = []
    rp_samples = []
    spark = []
    link = "?"
    last_link_check = 0.0

    while True:
        c = stdscr.getch()
        if c in (ord("q"), ord("Q")):
            break

        t = time.monotonic()
        g = read_counters(GPU[1])
        r = read_counters(RP[1])
        if g is not None:
            gpu_samples.append((t, g))
        if r is not None:
            rp_samples.append((t, r))

        # sparkline: per-sample combined total delta, scaled to errors/s
        if len(gpu_samples) >= 2 and len(rp_samples) >= 2:
            now_tot = (gpu_samples[-1][1].get("TOTAL_ERR_COR", 0)
                       + rp_samples[-1][1].get("TOTAL_ERR_COR", 0))
            prev_tot = (gpu_samples[-2][1].get("TOTAL_ERR_COR", 0)
                        + rp_samples[-2][1].get("TOTAL_ERR_COR", 0))
            spark.append(max(0, now_tot - prev_tot) / SAMPLE_S)
        spark = spark[-SPARK_N:]

        # trim history a bit beyond the largest window
        cutoff = t - (max(WINDOWS) + 2)
        while gpu_samples and gpu_samples[0][0] < cutoff:
            gpu_samples.pop(0)
        while rp_samples and rp_samples[0][0] < cutoff:
            rp_samples.pop(0)

        if t - last_link_check > 5:
            link = link_state()
            last_link_check = t

        stdscr.erase()
        put(stdscr, 0, 0,
            f"PCIe AER monitor   {GPU[0]} <-> {RP[0]}   link: {link}   (q=quit)")
        hdr = (f"{'':<16}{'total':>16}"
               + "".join(f"{'last ' + str(w) + 's':>12}" for w in WINDOWS))
        put(stdscr, 2, 0, hdr)
        y = 3
        device_block(stdscr, y, GPU[0], gpu_samples)
        y += 7
        device_block(stdscr, y, RP[0], rp_samples)
        y += 7
        put(stdscr, y, 0, f"combined err/s (last {SPARK_N * SAMPLE_S:.0f} s):")
        put(stdscr, y + 1, 0, sparkline(spark))
        put(stdscr, y + 3, 0,
            "wiggle a connector and watch which column spikes; "
            "'reset' = counter wrapped / link retrained")
        stdscr.refresh()

        time.sleep(SAMPLE_S)


if __name__ == "__main__":
    curses.wrapper(main)
