"""Shared plumbing for bench runners (protocol: see crates/grace-bench/src/runner.rs).

Each runner loads its model once, prints a `ready` line, then answers one JSON request per
stdin line. Anything a library prints goes to stderr so stdout stays a clean protocol stream.
"""

import json
import sys
import time
import traceback

# Windows defaults pipes to the ANSI code page, which mangles æøå. The protocol is UTF-8.
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
PROTOCOL_OUT = sys.stdout
# Libraries love printing to stdout; send all of it to stderr instead.
sys.stdout = sys.stderr


def send(obj: dict) -> None:
    PROTOCOL_OUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    PROTOCOL_OUT.flush()


def serve(load, handle) -> None:
    """`load()` returns (state, info dict); `handle(state, request)` returns the reply fields."""
    started = time.perf_counter()
    state, info = load()
    send({"ready": True, "load_ms": (time.perf_counter() - started) * 1000, **info})
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        try:
            reply = handle(state, request)
        except Exception as exc:  # report and keep serving: one bad clip shouldn't end a run
            traceback.print_exc()
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        send({"id": request["id"], **reply})


def timer():
    started = time.perf_counter()
    return lambda: (time.perf_counter() - started) * 1000
