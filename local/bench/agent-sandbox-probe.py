#!/usr/bin/env python3
"""One simulated multi-turn coding task; no model-proposed command is executed."""
import argparse
import json
import time
import urllib.error
import urllib.request

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a workspace file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace one exact old string with new text in a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["path", "old", "new"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "run", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"], "additionalProperties": False}}},
]
INITIAL = "def clamp(value, lower, upper):\n    return min(value, lower)\n"
EXPECTED = "def clamp(value, lower, upper):\n    return max(lower, min(value, upper))\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    messages = [
        {"role": "system", "content": "You are a coding agent. Inspect the file before editing, make the smallest correct change, run the specified test command, then report the outcome. Use the supplied tools."},
        {"role": "user", "content": "Fix clamp(value, lower, upper) in src/clamp.py so it correctly clamps the value inclusively to the given lower and upper bounds. After editing, run pytest -q tests/test_clamp.py."},
    ]
    file_content = INITIAL
    reads = edits = tests = passed_tests = 0
    errors = []
    started = time.monotonic()
    finish = None
    for turn in range(8):
        body = {"model": "qwen", "messages": messages, "tools": TOOLS, "tool_choice": "auto", "temperature": 0, "max_tokens": 2048, "stream": False}
        request = urllib.request.Request(args.url + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                data = json.load(response)
            choice = data["choices"][0]
            message = choice["message"]
            finish = choice.get("finish_reason")
            calls = message.get("tool_calls") or []
            print(json.dumps({"turn": turn + 1, "finish_reason": finish, "calls": calls, "content": message.get("content"), "usage": data.get("usage"), "timings": data.get("timings")}), flush=True)
            messages.append(message)
            if not calls:
                break
            for call in calls:
                fn = call["function"]
                name = fn["name"]
                arguments = json.loads(fn["arguments"])
                result = "ERROR: unsupported tool or invalid arguments"
                if name == "read_file" and arguments == {"path": "src/clamp.py"}:
                    reads += 1
                    result = file_content
                elif name == "read_file" and arguments == {"path": "tests/test_clamp.py"}:
                    result = "from src.clamp import clamp\n\ndef test_bounds():\n    assert clamp(-1, 0, 10) == 0\n    assert clamp(11, 0, 10) == 10\n    assert clamp(5, 0, 10) == 5\n    assert clamp(0, 0, 10) == 0\n"
                elif name == "edit_file" and set(arguments) == {"path", "old", "new"} and arguments["path"] == "src/clamp.py" and isinstance(arguments["old"], str) and isinstance(arguments["new"], str) and arguments["old"] and file_content.count(arguments["old"]) == 1:
                    edits += 1
                    file_content = file_content.replace(arguments["old"], arguments["new"], 1)
                    result = "File updated."
                elif name == "run" and arguments == {"command": "pytest -q tests/test_clamp.py"}:
                    tests += 1
                    passed_tests += int(file_content == EXPECTED)
                    result = "4 passed" if file_content == EXPECTED else "FAILED: clamp returned an incorrect result"
                elif name == "run" and arguments.get("command") in ("ls", "ls -R . 2>/dev/null | head -50") and set(arguments) == {"command"}:
                    result = "src\ntests\n"
                else:
                    errors.append({"tool": name, "arguments": arguments})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        except (urllib.error.URLError, ValueError, KeyError, TypeError) as error:
            errors.append(str(error))
            break
    passed = reads > 0 and edits > 0 and passed_tests > 0 and file_content == EXPECTED and not errors and finish == "stop"
    print(json.dumps({"label": args.label, "passed": passed, "read_calls": reads, "edit_calls": edits, "test_calls": tests, "passing_test_calls": passed_tests, "file_correct_exact": file_content == EXPECTED, "finish_reason": finish, "errors": errors, "wall_seconds": round(time.monotonic() - started, 2)}), flush=True)


if __name__ == "__main__":
    main()
