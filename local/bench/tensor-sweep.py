#!/usr/bin/env python3
"""Run short 128K-allocation tensor A/B tests; never claims long-context quality.

Each case starts its own server, runs a cold 8K retrieval and three capped
coding speed probes, then stops only that server. Raw logs and measurements
are stored under local/bench/results/. AER counts are recorded, not reset.
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GPUS = ("GPU-e487578c-0be1-8c0b-ad8b-0ab0615c4850",
        "GPU-40faaa73-acef-a2eb-1f38-7b96df067d5c")
CASES = {
    "baseline": ([], {}),
    "mtp1": (["--spec-draft-n-max", "1"], {}),
    "mtp3": (["--spec-draft-n-max", "3"], {}),
    "mtp4": (["--spec-draft-n-max", "4"], {}),
    "threads4": (["--threads", "4"], {}),
    "threads16": (["--threads", "16"], {}),
    "ubatch256": (["--ubatch-size", "256"], {}),
    "ubatch1024": (["--ubatch-size", "1024"], {}),
    "queue4x": ([], {"CUDA_SCALE_LAUNCH_QUEUES": "4x"}),
}
AER = ("/sys/bus/pci/devices/0000:80:02.2/aer_dev_correctable",
       "/sys/bus/pci/devices/0000:82:00.0/aer_dev_correctable")


def telemetry():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,temperature.gpu,memory.used,"
        "ecc.errors.uncorrected.volatile.total,clocks.current.sm,power.draw,"
        "clocks_event_reasons.active", "--format=csv,noheader,nounits"
    ], text=True, timeout=10)
    rows = {}
    for line in output.splitlines():
        uuid, temp, mem, ecc, clock, power, reasons = (x.strip() for x in line.split(","))
        if uuid in GPUS:
            rows[uuid] = {"temperature_c": int(temp), "memory_mib": int(mem),
                          "uncorrected_ecc": int(ecc), "sm_clock_mhz": int(clock),
                          "power_w": float(power), "clock_event_reasons": reasons}
    if set(rows) != set(GPUS):
        raise RuntimeError("could not read both selected V100s")
    return rows


def aer():
    result = {}
    for path in AER:
        result[path] = dict(line.split() for line in Path(path).read_text().splitlines())
    return result


def terminate(proc):
    if proc and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)


def run_probe(command, timeout, server, stop):
    proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    deadline = time.monotonic() + timeout
    try:
        while True:
            if stop.is_set() or server.poll() is not None:
                raise RuntimeError("server stopped or hardware guard triggered")
            if time.monotonic() >= deadline:
                raise TimeoutError("probe timed out")
            try:
                stdout, stderr = proc.communicate(timeout=1)
                return {"command": command, "exit_code": proc.returncode,
                        "stdout": stdout, "stderr": stderr}
            except subprocess.TimeoutExpired:
                pass
    finally:
        terminate(proc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASES,
                        help="repeat to select cases; defaults to all")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--max-temp", type=int, default=82)
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--probe-timeout", type=int, default=180)
    args = parser.parse_args()
    if not 512 <= args.prompt_tokens <= 130048 or args.probe_timeout < 1:
        parser.error("prompt-tokens must be 512..130048 and probe-timeout positive")
    if not 1 <= args.max_temp <= 82:
        parser.error("max-temp must be between 1 and 82 C")
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            parser.error("test port is already in use")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / "local/bench/results" / f"tensor-sweep-{stamp}-{os.getpid()}"
    out.mkdir(parents=True)
    print(f"results: {out}", flush=True)
    baseline_ecc = {k: v["uncorrected_ecc"] for k, v in telemetry().items()}
    if any(baseline_ecc.values()):
        raise RuntimeError("uncorrectable volatile ECC is nonzero before testing")

    for index, name in enumerate(args.case or CASES, 1):
        stem = f"{index:02d}-{name}"
        case_flags, overrides = CASES[name]
        flags = ["--ubatch-size", "512", *case_flags]
        env = os.environ.copy()
        # Do not silently inherit non-default settings into nominal comparisons.
        for key in list(env):
            if (key.startswith(("GGML_CUDA_", "NCCL_", "LLAMA_ARG_")) or
                    key in ("SERVER_BIN", "CUDA_SCALE_LAUNCH_QUEUES", "CTX_SIZE",
                            "HOST", "PORT")):
                env.pop(key)
        env.update(overrides)
        env["PORT"] = str(args.port)
        env["HOST"] = "127.0.0.1"
        state = {"case": name, "flags": flags, "env_overrides": overrides,
             "prompt_tokens": args.prompt_tokens,
                 "before": telemetry(), "aer_before": aer(), "samples": [],
                 "probes": [], "error": None}
        probe_results = []
        stop = threading.Event()
        guard_errors = []
        server = None
        watcher = None

        def watch():
            while not stop.wait(3):
                try:
                    sample = telemetry()
                    state["samples"].append(sample)
                    if any(v["temperature_c"] >= args.max_temp or
                           v["uncorrected_ecc"] > baseline_ecc[k] or
                           int(v["clock_event_reasons"], 16) & 0x60
                           for k, v in sample.items()):
                        guard_errors.append("temperature, thermal slowdown or uncorrectable ECC guard")
                        stop.set()
                except Exception as exc:
                    guard_errors.append(f"telemetry failed: {exc}")
                    stop.set()

        try:
            if any(v["temperature_c"] >= args.max_temp or
                   int(v["clock_event_reasons"], 16) & 0x60
                   for v in state["before"].values()):
                raise RuntimeError("thermal guard before starting")
            with (out / f"{stem}.server.log").open("w") as log:
                server = subprocess.Popen([str(ROOT / "local/serve-qwen38-27b-tensor-experimental.sh"),
                                           *flags], cwd=ROOT, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                watcher = threading.Thread(target=watch, daemon=True)
                watcher.start()
                deadline = time.monotonic() + 240
                while True:
                    if stop.is_set() or server.poll() is not None:
                        raise RuntimeError("server exited or hardware guard triggered during load")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("server readiness timed out")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=3) as r:
                            if json.load(r).get("status") == "ok":
                                break
                    except (OSError, ValueError):
                        pass
                    stop.wait(1)
                base = f"http://127.0.0.1:{args.port}"
                commands = [
                    [sys.executable, "local/bench/long-context-probe.py", "--url", base,
                     "--label", name, "--prompt-tokens", str(args.prompt_tokens),
                     "--predict", "96", "--timeout", str(args.probe_timeout)],
                    [sys.executable, "local/bench/mtp-bench.py", "--url", base + "/v1",
                     "--label", name, "--reps", "3", "--max-tokens", "256"],
                ]
                for command in commands:
                    result = run_probe(command, args.probe_timeout, server, stop)
                    state["probes"].append(result)
                    if result["exit_code"]:
                        raise RuntimeError("probe failed")
                    data = json.loads(result["stdout"])
                    probe_results.append(data)
                    if "reps_ok" in data and data["reps_ok"] != 3:
                        raise RuntimeError("not all coding speed probes completed")
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            state["error"] = str(exc)
        finally:
            stop.set()
            terminate(server)
            if watcher:
                watcher.join(timeout=15)
            state["guard_errors"] = guard_errors
            state["after"] = telemetry()
            state["aer_after"] = aer()
            (out / f"{stem}.json").write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps({"case": name, "error": state["error"],
                          "guard_errors": guard_errors,
                          "probe_results": probe_results}), flush=True)
        if state["error"] or guard_errors:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())