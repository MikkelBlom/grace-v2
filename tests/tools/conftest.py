"""Deterministic tests for Grace's tool modules (services/grace_tools, ADR 0021).

  bench/.venv-openvino/Scripts/python.exe -m pytest tests/tools -q

No test touches the internet, the real user folders, real windows, settings or processes:
the network is blocked (only loopback is allowed, and nothing listens there in tests), every
file lives under tmp_path, and OS actions go through each module's thin OS layer, which tests
replace. Recorded web responses live in fixtures/tools/recorded/.
"""

import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "services"))

from grace_tools import registry  # noqa: E402

RECORDED = ROOT / "fixtures/tools/recorded"


@pytest.fixture(autouse=True)
def no_internet(monkeypatch):
    real = socket.socket.connect

    def connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise OSError(f"tests must not reach the internet ({address})")
        return real(self, address)
    monkeypatch.setattr(socket.socket, "connect", connect)


class Ctx:
    """What a tool sees of the server (services/tools_server.py Tools)."""

    def __init__(self, tmp, dry=False):
        self.dry = dry
        self.checker = None
        self.files = None
        self.apps = None
        self.confirmations = registry.Confirmations()
        self.data_dir = tmp / "grace-data"
        self.scratch_dir = self.data_dir / "scratch"
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.gemma_replies = []  # tests queue Gemma's answers here
        self.gemma_prompts = []

    def gemma(self, prompt, max_tokens=40):
        self.gemma_prompts.append(prompt)
        if not self.gemma_replies:
            raise AssertionError("unexpected Gemma call: " + prompt[:200])
        return self.gemma_replies.pop(0)


@pytest.fixture
def ctx(tmp_path):
    return Ctx(tmp_path)


@pytest.fixture
def dry_ctx(tmp_path):
    return Ctx(tmp_path, dry=True)


def call(ctx, name, alive=lambda: True, confirmed=False, **args):
    """Runs a registered tool as the server would (risk rules included)."""
    import grace_tools
    grace_tools.load()
    out = registry.run(ctx, name, args, alive, confirmed)
    assert out is not None, f"no tool {name}"
    assert isinstance(out.get("result"), str), out
    return out
