#!/usr/bin/env python3
"""Compare recorded benchmark runs from local/bench/results/.

Usage:
  compare.py bench  [--baseline LABEL]
  compare.py server [--baseline LABEL]

Only the most recent run per label is shown; older runs stay in the CSVs.
Stdlib only, no dependencies.
"""

import argparse
import csv
import json
import os
import statistics
import sys

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def latest_bench_rows():
    """Group bench.csv by label, keep all rows of the label's latest timestamp."""
    rows = read_rows(os.path.join(RESULTS, "bench.csv"))
    stamps = {}
    for r in rows:
        stamps.setdefault(r["label"], set()).add(r["timestamp"])
    out = []
    for label in sorted(stamps):
        newest = max(stamps[label])
        out.extend(r for r in rows if r["label"] == label and r["timestamp"] == newest)
    return out


def latest_server_rows():
    rows = read_rows(os.path.join(RESULTS, "server-runs.csv"))
    best = {}
    for r in rows:
        if r["label"] not in best or r["timestamp"] > best[r["label"]]["timestamp"]:
            best[r["label"]] = r
    return [best[k] for k in sorted(best)]


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def pct(cur, base):
    c, b = fnum(cur), fnum(base)
    if c is None or not b:
        return "n/a"
    return "%+.1f%%" % ((c - b) / b * 100.0)


def ratio(cur, base):
    c, b = fnum(cur), fnum(base)
    if c is None or not b:
        return "n/a"
    return "%.2fx" % (c / b)


def print_table(headers, rows):
    cells = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))


def cmd_bench(args):
    rows = latest_bench_rows()
    if not rows:
        sys.exit("no runs recorded yet (run local/bench/run-bench.sh first)")
    labels = sorted({r["label"] for r in rows})
    tests = sorted({r["test"] for r in rows if r.get("test")})
    base_rows = {}
    if args.baseline:
        base_rows = {r["test"]: r for r in rows if r["label"] == args.baseline}
        if not base_rows:
            sys.exit("baseline %r not found; labels: %s" % (args.baseline, ", ".join(labels)))

    headers = ["label", "commit", "power W", "temp C start", *tests]
    if base_rows:
        headers += [t + " vs base" for t in tests]
    table = []
    for label in labels:
        lrows = {r["test"]: r for r in rows if r["label"] == label}
        first = next(iter(lrows.values()))
        row = [label, first.get("commit", ""), first.get("power_limit_w", ""),
               first.get("temp_start_c", "")]
        for t in tests:
            row.append(lrows[t]["avg_ts"] if t in lrows else "n/a")
        if base_rows:
            for t in tests:
                if t in lrows and t in base_rows:
                    row.append(pct(lrows[t]["avg_ts"], base_rows[t]["avg_ts"]))
                else:
                    row.append("n/a")
        table.append(row)
    print_table(headers, table)


def overall_summary(path):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload.get("summary", [])
    for row in summary:
        if row.get("category") == "overall":
            return row
    if not summary:
        return None

    def avg(key):
        vals = [row[key] for row in summary if row.get(key) is not None]
        return statistics.mean(vals) if vals else None

    draft_n = sum(row.get("draft_n") or 0 for row in summary)
    accepted = sum(row.get("accepted") or 0 for row in summary)
    return {
        "avg_prompt_t_s": avg("avg_prompt_t_s"),
        "avg_pred_t_s": avg("avg_pred_t_s"),
        "avg_latency": avg("avg_latency"),
        "accept_rate": accepted / draft_n if draft_n else None,
    }


def cmd_server(args):
    rows = latest_server_rows()
    if not rows:
        sys.exit("no runs recorded yet (run local/bench/server-bench.sh first)")
    base_row = None
    if args.baseline:
        base_row = next((r for r in rows if r["label"] == args.baseline), None)
        if base_row is None:
            sys.exit("baseline %r not found; labels: %s"
                     % (args.baseline, ", ".join(r["label"] for r in rows)))

    data = []
    for r in rows:
        s = overall_summary(r["json_file"]) if os.path.exists(r.get("json_file", "")) else None
        data.append((r, s))

    headers = ["label", "prompt t/s", "pred t/s", "latency s", "accept",
               "power W", "temp C start"]
    if base_row:
        headers += ["pred speedup", "latency speedup"]
    table = []
    for r, s in data:
        row = [r["label"]]
        if s:
            row.append("%.2f" % s["avg_prompt_t_s"] if s.get("avg_prompt_t_s") else "n/a")
            row.append("%.2f" % s["avg_pred_t_s"] if s.get("avg_pred_t_s") else "n/a")
            row.append("%.3f" % s["avg_latency"] if s.get("avg_latency") else "n/a")
            row.append("%.3f" % s["accept_rate"] if s.get("accept_rate") is not None else "n/a")
        else:
            row += ["n/a"] * 4
        row += [r.get("power_limit_w", ""), r.get("temp_start_c", "")]
        if base_row:
            bs = overall_summary(base_row["json_file"]) if os.path.exists(base_row.get("json_file", "")) else None
            cur_pred = s.get("avg_pred_t_s") if s else None
            cur_lat = s.get("avg_latency") if s else None
            base_pred = bs.get("avg_pred_t_s") if bs else None
            base_lat = bs.get("avg_latency") if bs else None
            row.append(ratio(cur_pred, base_pred))
            row.append(ratio(base_lat, cur_lat))
        table.append(row)
    print_table(headers, table)


def main():
    p = argparse.ArgumentParser(description="Compare recorded benchmark runs.")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("bench", "server"):
        sp = sub.add_parser(name)
        sp.add_argument("--baseline", default=None, help="label to compare against")
        sp.set_defaults(func=cmd_bench if name == "bench" else cmd_server)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
