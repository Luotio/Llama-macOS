#!/usr/bin/env python3
"""Measure prompt reuse while alternating two artificial conversations."""
import argparse
import json
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    results = []
    for key in ("A", "B", "A"):
        text = (f"Conversation {key}: The project has a file called {key.lower()}_module.py. " * 500) + "Summarize in one sentence."
        body = {"model": "qwen", "messages": [{"role": "user", "content": text}], "temperature": 0, "max_tokens": 16, "stream": False}
        request = urllib.request.Request(args.url + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.load(response)
        result = {"conversation": key, "wall_seconds": round(time.monotonic() - start, 2), "usage": data.get("usage"), "timings": data.get("timings")}
        results.append(result)
        print(json.dumps(result), flush=True)
    print(json.dumps({"label": args.label, "results": len(results)}), flush=True)


if __name__ == "__main__":
    main()
