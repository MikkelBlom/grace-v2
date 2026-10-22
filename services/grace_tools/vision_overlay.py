"""Helpers for grace-overlay (crates/grace-overlay): the orange edge glow and focus boxes.

The overlay listens on 127.0.0.1:7815 for newline JSON. Every helper returns False and does
nothing else when the overlay isn't running, so a vision tool never fails because of it.
Box coordinates are absolute virtual-desktop pixels (physical), as vision_screen returns them.
"""

import json
import os
import socket

PORT = int(os.environ.get("GRACE_OVERLAY_PORT", "7815"))
TIMEOUT_S = 0.3  # local and tiny; never hold up a turn waiting for the overlay


def send(cmd, port=None):
    """Sends one command; True when the overlay answered ok."""
    try:
        with socket.create_connection(("127.0.0.1", port or PORT), timeout=TIMEOUT_S) as s:
            s.sendall((json.dumps(cmd) + "\n").encode())
            reply = s.makefile("r", encoding="utf-8").readline()
        return bool(json.loads(reply).get("ok"))
    except (OSError, ValueError):
        return False


def running():
    return send({"ping": True})


def glow(on=True):
    return send({"glow": bool(on)})


def box(x, y, w, h, label=None, ms=4000):
    spec = {"x": int(x), "y": int(y), "w": max(1, int(w)), "h": max(1, int(h)), "ms": int(ms)}
    if label:
        spec["label"] = str(label)[:80]
    return send({"box": spec})


def clear():
    return send({"clear": True})
