#!/usr/bin/env python3
"""Objective quality probe for Strata decode settings.

Sends one deterministic coding task (temperature=0) to the server, saves the
full response text, extracts the requested function and runs it against a set
of objectively checkable test cases. Usage:

    python3 local/bench/quality-probe.py <chat-url> <out-prefix>

Writes <out-prefix>.txt (raw response) and prints per-case PASS/FAIL plus a
score line like "QUALITY 5/6".
"""
import json
import re
import sys
import textwrap
import urllib.request

url = sys.argv[1]
out_prefix = sys.argv[2]

PROMPT = textwrap.dedent("""\
    You are a careful Python engineer. Write a function split_ranges(s) that
    parses a compact integer range string into a sorted list of unique ints.
    Rules: items are comma-separated; each item is either a single integer or
    "a-b" with a <= b; duplicates and overlapping ranges merge into one; any
    invalid input (bad characters, empty items, a > b) raises ValueError.
    Show the function in a python code block, then show the exact output of
    split_ranges('2-5,3,8-9') on its own line prefixed with "OUTPUT:".
""").strip()

body = {
    "model": "qwen3.8-flash-next-unsloth-ud-iq4_xs",
    "messages": [
        {"role": "system", "content": "You are a precise Python engineer."},
        {"role": "user", "content": PROMPT},
    ],
    "reasoning_effort": "none",
    "max_tokens": 1200,
    "temperature": 0.0,
    "stream": False,
}
req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as resp:
    data = json.loads(resp.read())

text = data["choices"][0]["message"]["content"]
with open(out_prefix + ".txt", "w", encoding="utf-8") as f:
    f.write(text)
t = data.get("timings") or {}
print(f"decode_tps={t.get('predicted_per_second')} draft_accept="
      f"{(t.get('draft_n_accepted') or 0)}/{(t.get('draft_n') or 0)}")

# --- extract the function and run objective test cases ---
m = re.search(r"```python\s*(.*?)```", text, re.S)
src = m.group(1) if m else text
import contextlib
import io

ns = {}
printed = ""
try:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, "<probe>", "exec"), ns)
    printed = buf.getvalue()
except Exception as e:
    print(f"EXEC-FAIL: {e!r}")
    print("QUALITY 0/9")
    sys.exit(0)

fn = ns.get("split_ranges")
cases = [
    ("2-5,3,8-9", [2, 3, 4, 5, 8, 9]),
    ("1,1,2", [1, 2]),
    ("10-12,11", [10, 11, 12]),
    ("5", [5]),
]
error_cases = ["5-1", "a-b", "1,,2", "-3-2"]

score = 0
total = len(cases) + len(error_cases)
for s, want in cases:
    got = "<exception>"
    try:
        got = fn(s)
        ok = list(got) == want
    except Exception:
        ok = False
    score += ok
    print(f"{'PASS' if ok else 'FAIL'} split_ranges({s!r}) -> "
          f"{('ok' if ok else repr(got))} (want {want})")
for s in error_cases:
    try:
        fn(s)
        ok = False
    except ValueError:
        ok = True
    except Exception:
        ok = False   # wrong exception type
    score += ok
    print(f"{'PASS' if ok else 'FAIL'} split_ranges({s!r}) raises ValueError")

m2 = re.search(r"OUTPUT:\s*([\[\d, ]+)", text)
shown = m2.group(1).strip() if m2 else None
ok = shown in ("[2, 3, 4, 5, 8, 9]", "[2,3,4,5,8,9]") or \
    "[2, 3, 4, 5, 8, 9]" in printed or "[2,3,4,5,8,9]" in printed
score += ok
print(f"{'PASS' if ok else 'FAIL'} demo output correct "
      f"(text={shown!r}, printed={'yes' if '[2, 3, 4, 5, 8, 9]' in printed or '[2,3,4,5,8,9]' in printed else 'no'})")
print(f"QUALITY {score}/{total + 1}")
