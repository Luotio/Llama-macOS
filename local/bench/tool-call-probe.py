#!/usr/bin/env python3
"""Small, non-executing OpenAI tool-call smoke test; not an agent benchmark."""
import argparse
import json
import time
import urllib.error
import urllib.request

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a workspace file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "grep", "description": "Search workspace file contents.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern", "path"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "run", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace exact text in a workspace file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["path", "old", "new"], "additionalProperties": False}}},
]
CASES = [
    ("read", "Read the file src/config.py using the read_file tool. Do not guess its contents.", "read_file", {"path": "src/config.py"}),
    ("search", "Search for the exact text TODO in the src directory using grep.", "grep", {"pattern": "TODO", "path": "src"}),
    ("run", "Run the command pytest -q tests/test_parser.py using the run tool.", "run", {"command": "pytest -q tests/test_parser.py"}),
    ("edit", "Use edit_file to replace exactly 'timeout = 5' with 'timeout = 10' in app/settings.py.", "edit_file", {"path": "app/settings.py", "old": "timeout = 5", "new": "timeout = 10"}),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="base URL ending in /v1")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    results = []
    for name, prompt, expected_tool, expected_args in CASES:
        body = {"model": "qwen", "messages": [{"role": "system", "content": "You are a coding assistant. Use the supplied tools when requested. Never claim to have used a tool without calling it."}, {"role": "user", "content": prompt}], "tools": TOOLS, "tool_choice": "auto", "temperature": 0, "max_tokens": 1024, "stream": False}
        request = urllib.request.Request(args.url + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                data = json.load(response)
            choice = data["choices"][0]
            calls = choice["message"].get("tool_calls") or []
            matches = []
            for call in calls:
                function = call.get("function", {})
                try:
                    arguments = json.loads(function.get("arguments", ""))
                except (TypeError, ValueError):
                    arguments = None
                matches.append({"name": function.get("name"), "arguments": arguments})
            passed = len(matches) == 1 and matches[0] == {"name": expected_tool, "arguments": expected_args} and choice.get("finish_reason") == "tool_calls"
            result = {"case": name, "passed": passed, "finish_reason": choice.get("finish_reason"), "calls": matches, "content": choice["message"].get("content"), "usage": data.get("usage"), "timings": data.get("timings"), "wall_seconds": round(time.monotonic() - start, 2)}
        except (urllib.error.URLError, ValueError, KeyError, TypeError) as error:
            result = {"case": name, "passed": False, "error": str(error)}
        results.append(result)
        print(json.dumps(result), flush=True)
    print(json.dumps({"label": args.label, "passed": sum(r["passed"] for r in results), "total": len(results)}), flush=True)


if __name__ == "__main__":
    main()
