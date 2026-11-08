"""services/grace_tools/files_explorer.py: Explorer navigation and the file index's extra folders.
No real Explorer window is touched (shell_explorer and open_selected are replaced)."""

import json
from pathlib import Path

import pytest

import grace_tools
from grace_tools import files as F
from grace_tools import files_explorer as X
from grace_tools import registry


def call(ctx, name, **args):
    out = registry.run(ctx, name, args, lambda: True)
    assert out is not None and isinstance(out.get("result"), str), out
    return out


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    grace_tools.load()
    h = tmp_path / "home"
    for r in F.ROOT_NAMES:
        (h / r).mkdir(parents=True)
    monkeypatch.setattr(F, "HOME", h)
    # tmp_path sits in AppData: judge folders from tmp_path down, as if it were a drive.
    monkeypatch.setattr(X, "path_parts", lambda rp: rp.relative_to(tmp_path.resolve()).parts
                        if tmp_path.resolve() in rp.parents else rp.parts[1:])
    return h


@pytest.fixture
def shell(monkeypatch):
    """Records Explorer steps; `replies` is what the fake window answers."""
    calls, replies = [], []

    def fake(op, path=""):
        calls.append((op, str(path)))
        return replies.pop(0) if replies else "none"
    monkeypatch.setattr(X, "shell_explorer", fake)
    opened = []
    monkeypatch.setattr(X, "open_selected", lambda p: opened.append(str(p)))
    return calls, replies, opened


# ---- explorer ---------------------------------------------------------------------------------

def test_up_reports_the_new_folder(ctx, home, shell):
    calls, replies, _ = shell
    replies.append(f"at={home / 'Documents'}")
    out = call(ctx, "explorer", action="up")
    assert out["ok"] and calls == [("up", "")]
    assert "Documents" in out["result"] and out["risk"] == "visible"


def test_up_without_a_window_says_so(ctx, shell):
    out = call(ctx, "explorer", action="up")
    assert not out["ok"] and "No File Explorer window" in out["result"]


def test_back_with_no_history(ctx, shell):
    shell[1].append("nohistory")
    out = call(ctx, "explorer", action="back")
    assert not out["ok"] and "nothing to go back" in out["result"]


def test_up_at_the_top(ctx, shell):
    shell[1].append("top")
    assert "already at the top" in call(ctx, "explorer", action="up")["result"]


def test_select_a_file_by_danish_folder_name(ctx, home, shell):
    f = home / "Documents" / "rapport.docx"
    f.write_bytes(b"x")
    calls, replies, _ = shell
    replies.append(f"at={home / 'Documents'}")
    out = call(ctx, "explorer", action="select", path="dokumenter/rapport.docx")
    assert out["ok"] and calls == [("select", str(f.resolve()))]
    assert "Selected rapport.docx" in out["result"]


def test_select_opens_a_window_when_none_is_open(ctx, home, shell):
    f = home / "Desktop" / "cv.pdf"
    f.write_bytes(b"x")
    out = call(ctx, "explorer", action="select", path=str(f))
    assert out["ok"] and shell[2] == [str(f.resolve())] and "with cv.pdf selected" in out["result"]


def test_select_a_missing_file(ctx, shell):
    out = call(ctx, "explorer", action="select", path="dokumenter/findes-ikke.pdf")
    assert not out["ok"] and "couldn't find" in out["result"] and shell[0] == []


def test_select_needs_a_path(ctx, shell):
    out = call(ctx, "explorer", action="select")
    assert not out["ok"] and "Which file" in out["result"]


def test_go_refuses_folders_outside_home(ctx, tmp_path, shell):
    other = tmp_path / "elsewhere"
    other.mkdir()
    out = call(ctx, "explorer", action="go", path=str(other))
    assert not out["ok"] and "outside" in out["result"] and shell[0] == []


def test_dry_run_touches_no_window(dry_ctx, home, shell):
    out = call(dry_ctx, "explorer", action="up")
    assert out["ok"] and "Dry run" in out["result"] and shell[0] == []


# ---- index_folders ----------------------------------------------------------------------------

class FakeFiles:
    def __init__(self):
        self.reindexed = 0

    def reindex(self):
        self.reindexed += 1

    def candidates(self, *a, **k):
        return []


def test_add_list_remove(ctx, tmp_path):
    ctx.files = FakeFiles()
    proj = tmp_path / "D" / "Projekter"
    proj.mkdir(parents=True)
    out = call(ctx, "index_folders", action="add", folder=str(proj))
    assert out["ok"] and "Added" in out["result"] and ctx.files.reindexed == 1
    assert X.extra_roots(ctx.data_dir) == [proj.resolve()]
    assert json.loads((ctx.data_dir / X.INDEX_FILE).read_text("utf-8")) == [str(proj.resolve())]
    assert str(proj.resolve()) in call(ctx, "index_folders", action="list")["result"]
    assert "already searched" in call(ctx, "index_folders", action="add", folder=str(proj))["result"]
    out = call(ctx, "index_folders", action="remove", folder=str(proj))
    assert out["ok"] and X.extra_roots(ctx.data_dir) == [] and ctx.files.reindexed == 2


def test_list_when_empty(ctx):
    assert "no extra folders" in call(ctx, "index_folders", action="list")["result"]


def test_user_folders_are_already_searched(ctx, home):
    out = call(ctx, "index_folders", action="add", folder="dokumenter")
    assert not out["ok"] and "already searched" in out["result"]


def test_system_and_hidden_folders_are_refused(ctx, tmp_path):
    for name in ("Program Files", "AppData", ".ssh"):
        p = tmp_path / "x" / name
        p.mkdir(parents=True)
        out = call(ctx, "index_folders", action="add", folder=str(p))
        assert not out["ok"] and "don't index" in out["result"], name
    assert X.extra_roots(ctx.data_dir) == []


def test_missing_and_relative_folders(ctx, tmp_path):
    assert "no folder" in call(ctx, "index_folders", action="add", folder=str(tmp_path / "nope"))["result"]
    assert "full path" in call(ctx, "index_folders", action="add", folder="projekter")["result"]


def test_remove_unknown(ctx, tmp_path):
    out = call(ctx, "index_folders", action="remove", folder=str(tmp_path))
    assert not out["ok"] and "isn't one of" in out["result"]


def test_dry_run_saves_nothing(dry_ctx, tmp_path):
    p = tmp_path / "Projekter"
    p.mkdir()
    out = call(dry_ctx, "index_folders", action="add", folder=str(p))
    assert out["ok"] and "Dry run" in out["result"] and X.extra_roots(dry_ctx.data_dir) == []


def test_extra_roots_skips_vanished_folders(ctx, tmp_path):
    keep = tmp_path / "keep"
    keep.mkdir()
    (ctx.data_dir / X.INDEX_FILE).write_text(json.dumps([str(keep), str(tmp_path / "gone"), 3]), "utf-8")
    assert X.extra_roots(ctx.data_dir) == [keep]


@pytest.mark.parametrize("said,want", [
    ("d drevet projekter", "D:\\projekter"),
    ("D-drev/Projekter/Skole", "D:\\Projekter\\Skole"),
    ("E:\\Billeder", "E:\\Billeder"),
    ("dokumenter", "dokumenter"),
])
def test_spoken_drive_names(said, want):
    assert X.spoken_path(said) == want
