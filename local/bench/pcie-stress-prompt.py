#!/usr/bin/env python3
"""One-shot PCIe stress test: send a ~46k-token prompt to the Strata server.

Records AER counters before/after so the error burst from the prefill can be
attributed precisely. Usage: python3 pcie-stress-prompt.py [url] [tokens]
"""
import json
import sys
import time
import urllib.request

GPU = "/sys/bus/pci/devices/0000:81:00.0/aer_dev_correctable"
RP = "/sys/bus/pci/devices/0000:80:02.0/aer_dev_correctable"


def read_aer(path):
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                d[parts[0]] = int(parts[1])
    return d


def build_prompt(target_tokens: int) -> str:
    # ~4 chars/token for this prose style; pad to target with repeated blocks.
    block = (
        "The quick brown fox jumps over the lazy dog while the data center "
        "hummed quietly in the background. Engineers watched the telemetry "
        "dashboards as packet loss crept along the fiber backbone, each "
        "retransmission a small tax on latency that no amount of caching "
        "could fully hide. The model itself was a patchwork of quantized "
        "experts, most of them asleep, a few of them racing to finish their "
        "partial attention before the next token arrived and reshuffled the "
        "priorities. Memory bandwidth was the true bottleneck, not compute: "
        "the GPUs had headroom but the PCIe lanes were already saturated, "
        "every kilobyte of expert weights a negotiation between the "
        "scheduler and the fabric. Somewhere in the rack a cable connector "
        "sat loose by a fraction of a millimeter, and at eight gigatransfers "
        "per second that fraction was enough to corrupt a TLP header every "
        "few hundred milliseconds, a slow leak that only showed up as "
        "correctable errors in the logs until one day it did not."
    )
    n = max(1, int(target_tokens * 4 / len(block)))
    text = (block + " ") * n
    return text[: target_tokens * 4]


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080/v1/chat/completions"
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 46000

    pre_gpu = read_aer(GPU)
    pre_rp = read_aer(RP)

    prompt = build_prompt(target)
    body = json.dumps({
        "model": "qwen3.8-flash-next-unsloth-ud-iq4_xs",
        "messages": [{"role": "user", "content": prompt + "\n\nReply with the single word: done"}],
        "max_tokens": 16,
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            data = json.loads(r.read())
            dt = time.time() - t0
            usage = data.get("usage") or {}
            print(f"HTTP 200 in {dt:.1f}s  prompt_tokens={usage.get('prompt_tokens')} "
                  f"completion_tokens={usage.get('completion_tokens')}")
            choices = data.get("choices") or []
            if choices:
                content = (choices[0].get("message") or {}).get("content", "")
                print(f"reply: {str(content)[:200]!r}")
            else:
                print(f"body (no choices): {json.dumps(data)[:300]}")
    except urllib.error.HTTPError as e:
        dt = time.time() - t0
        err = e.read().decode(errors="replace")
        print(f"HTTP {e.code} in {dt:.1f}s  body: {err[:500]}")
    except Exception as e:
        dt = time.time() - t0
        print(f"ERROR after {dt:.1f}s: {type(e).__name__}: {e}")

    post_gpu = read_aer(GPU)
    post_rp = read_aer(RP)
    print("\nAER delta during test:")
    for name, pre, post in (("GPU 81:00.0", pre_gpu, post_gpu), ("RootPort 80:02.0", pre_rp, post_rp)):
        deltas = {k: post.get(k, 0) - pre.get(k, 0) for k in pre}
        deltas = {k: v for k, v in deltas.items() if v != 0}
        print(f"  {name}: {deltas or 'no change'}")


if __name__ == "__main__":
    main()
