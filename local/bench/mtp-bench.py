#!/usr/bin/env python3
"""Measure MTP speculative decoding speedup against a running llama-server.

Sends fixed coding-prompt chat completions and reads the server's own
timings (prompt_per_second, predicted_per_second, draft_n,
draft_n_accepted) from the API response. Stdlib only.

Usage:
  mtp-bench.py --url http://127.0.0.1:8080/v1 [--label X] [--reps 3]
               [--max-tokens 256] [--prompt-file F]

Prints one JSON object with per-request detail and an aggregate summary.
"""
import argparse
import json
import sys
import time
import urllib.request

SYSTEM = "You are an expert Python programmer."
# Rotate prompts per rep: identical prompts hit the server's slot prefix
# cache (LCP reuse), which would skew prompt_tps on later reps.
USERS = [
    ("Write a Python function that implements a thread-safe LRU cache with a "
     "configurable max size, using only the standard library. Include type "
     "hints, a docstring, and a __main__ block demonstrating usage."),
    ("Write a Python class that parses an INI file into nested dicts, "
     "supporting sections, comments, and default values. Just the code."),
    ("Implement a Python generator that yields the prime factors of n in "
     "ascending order, with multiplicity. Include a brief docstring."),
]


def one_request(url: str, max_tokens: int, system: str, user: str) -> dict:
    body = json.dumps({
        "model": "qwen",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        url + "/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    wall_s = time.time() - t0
    t = data.get("timings", {})
    return {
        "wall_s": round(wall_s, 2),
        "prompt_n": t.get("prompt_n"),
        "prompt_tps": round(t.get("prompt_per_second", 0.0), 1),
        "pred_n": t.get("predicted_n"),
        "decode_tps": round(t.get("predicted_per_second", 0.0), 2),
        "draft_n": t.get("draft_n"),
        "draft_accepted": t.get("draft_n_accepted"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="base URL with /v1 suffix")
    ap.add_argument("--label", default="")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt-file", default="",
                    help="optional file with the user prompt (overrides built-in)")
    args = ap.parse_args()

    prompts = USERS
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompts = [f.read().strip()]

    results = []
    for i in range(args.reps):
        try:
            r = one_request(args.url, max_tokens, SYSTEM, prompts[i % len(prompts)])
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"rep {i}: FAILED: {e}", file=sys.stderr)
            break
        results.append(r)
        print(f"rep {i}: decode {r['decode_tps']} t/s, "
              f"draft {r['draft_n']}, accepted {r['draft_accepted']}",
              file=sys.stderr)

    if not results:
        print("no successful requests", file=sys.stderr)
        return 1

    n_draft = sum(r["draft_n"] or 0 for r in results)
    n_acc = sum(r["draft_accepted"] or 0 for r in results)
    summary = {
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reps_ok": len(results),
        "prompt_tps_avg": round(sum(r["prompt_tps"] for r in results) / len(results), 1),
        "decode_tps_avg": round(sum(r["decode_tps"] for r in results) / len(results), 2),
        "draft_n_total": n_draft,
        "draft_accepted_total": n_acc,
        "accept_rate": round(n_acc / n_draft, 4) if n_draft else None,
        "requests": results,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
