#!/usr/bin/env python3
"""Exercise a real near-limit prompt; this is a small retrieval smoke test.

The server tokenizer constructs an exact-size token array. Three distinct facts
are placed near the beginning, middle and end of repeated source-code filler.
This verifies prompt consumption and retrieval, not general agent quality.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FACTS = (
    ("cedar", "violet-4827"),
    ("maple", "amber-6193"),
    ("willow", "silver-7351"),
)


def post(base, endpoint, payload, timeout=600):
    request = urllib.request.Request(
        base + endpoint, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def tokens(base, text):
    return post(base, "/tokenize", {"content": text, "parse_special": True})["tokens"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8094")
    parser.add_argument("--label", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=130048)
    parser.add_argument("--predict", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.prompt_tokens < 512 or args.predict < 1:
        parser.error("prompt-tokens must be >= 512 and predict must be positive")

    placeholder = "UNIQUE_CONTEXT_PLACEHOLDER_59b0d7"
    user = (
        "Read the following repository excerpts and deployment notes. "
        "Find the three exact deployment codes for cedar, maple and willow. "
        "After the excerpts, answer with only the three codes.\n"
        + placeholder
        + "\nReturn only the codes for cedar, maple and willow, in that order."
    )
    rendered = post(args.url, "/apply-template", {"messages": [
        {"role": "system", "content": "You are a precise repository assistant."},
        {"role": "user", "content": user},
    ]})["prompt"]
    if rendered.count(placeholder) != 1:
        raise ValueError("unexpected chat template placeholder count")
    start, end = rendered.split(placeholder)
    # Explicitly close the thinking section so a short deterministic answer
    # tests retrieval rather than exhausting the output cap on reasoning.
    end += "\n<think>\n\n</think>\n\n"
    prefix, suffix = tokens(args.url, start), tokens(args.url, end)
    source = (ROOT / "llama-cpp/src/llama-context.cpp").read_text(encoding="utf-8")
    filler = tokens(args.url, "\nRepository excerpt:\n" + source)
    if not filler:
        raise ValueError("empty filler")
    facts = [tokens(args.url, f"\nDeployment note: {k} code is {v}.\n")
             for k, v in FACTS]
    space = args.prompt_tokens - len(prefix) - len(suffix) - sum(map(len, facts))
    if space < 0:
        raise ValueError("prompt too small for template and facts")
    # Exact positions are reported in token units; filler is cyclic and
    # intentionally synthetic, so do not treat this as a real coding task.
    chunks = [space // 20, space * 9 // 20, space * 9 // 20]
    chunks.append(space - sum(chunks))
    prompt = list(prefix)
    positions = []
    for i, size in enumerate(chunks):
        prompt.extend((filler * (size // len(filler) + 1))[:size])
        if i < len(facts):
            positions.append(len(prompt))
            prompt.extend(facts[i])
    prompt.extend(suffix)
    assert len(prompt) == args.prompt_tokens

    started = time.monotonic()
    data = post(args.url, "/completion", {
        "prompt": prompt, "n_predict": args.predict, "temperature": 0,
        "cache_prompt": False, "return_tokens": True,
        "response_fields": ["content", "tokens_evaluated", "tokens_predicted",
                            "truncated", "stop_type", "tokens_cached", "timings"],
    }, timeout=args.timeout)
    text = data.get("content", "")
    found = {key: value.lower() in text.lower() for key, value in FACTS}
    consumed = data.get("tokens_evaluated") == len(prompt)
    not_truncated = data.get("truncated") is False
    result = {
        "label": args.label, "prompt_tokens_requested": len(prompt),
        "fact_positions": positions, "consumed_exactly": consumed,
        "not_truncated": not_truncated, "retrieved": found,
        "wall_seconds": round(time.monotonic() - started, 2), "response": data,
    }
    print(json.dumps(result, indent=2), flush=True)
    return 0 if consumed and not_truncated and all(found.values()) else 1


if __name__ == "__main__":
    sys.exit(main())