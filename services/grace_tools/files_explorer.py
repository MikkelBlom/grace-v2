"""Files: File Explorer navigation and the file index's extra folders (TOOLS-ROADMAP P1).

`explorer` moves the open File Explorer window ("gå en mappe op", "tilbage", "marker rapporten")
instead of opening new windows (live run 9). `index_folders` lets Mikkel add folders outside his
user folders (a D: drive, a projects folder) to what find_files searches; tools_server.Files reads
the same list (extra_roots) and re-indexes when it changes.

The OS layer is `shell_explorer` and `open_selected`; tests replace them.
"""

import json
import os
import re
import subprocess
from pathlib import Path

from . import files as F
from .registry import s, tool

INDEX_FILE = "index-folders.json"
MAX_EXTRA = 20
# Never indexed: Windows itself, installed programs, other users' and system data.
NO_INDEX = ("windows", "program files", "program files (x86)", "programdata", "$recycle.bin",
            "system volume information", "appdata", "recovery", "perflogs")


# ---- OS layer (tests replace these) -----------------------------------------------------------

EXPLORER_PS = r"""
$w = @((New-Object -ComObject Shell.Application).Windows()) | Where-Object { $_.FullName -like '*explorer.exe' } | Select-Object -Last 1
if (-not $w) { 'none'; exit }
try {
  switch ($env:GRACE_OP) {
    'up'       { $p = Split-Path $w.Document.Folder.Self.Path -Parent; if (-not $p) { 'top'; exit }; $w.Navigate2($p) }
    'back'     { $w.GoBack() }
    'forward'  { $w.GoForward() }
    'navigate' { $w.Navigate2($env:GRACE_PATH) }
    'select'   {
      $w.Navigate2((Split-Path $env:GRACE_PATH -Parent)); Start-Sleep -Milliseconds 500
      $item = $w.Document.Folder.ParseName((Split-Path $env:GRACE_PATH -Leaf))
      if (-not $item) { 'missing'; exit }
      $w.Document.SelectItem($item, 29)
    }
  }
} catch { 'nohistory'; exit }
Start-Sleep -Milliseconds 300
$w.Visible = $true
'at=' + $w.Document.Folder.Self.Path
"""


def shell_explorer(op, path=""):
    """Runs one step in the most recent Explorer window. Returns 'at=<folder>', 'none' (no window
    open), 'top' (already at the top), 'nohistory' or 'missing'."""
    out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                          "[Console]::OutputEncoding = [Text.Encoding]::UTF8; " + EXPLORER_PS],
                         capture_output=True, text=True, encoding="utf-8", timeout=15,
                         env={**os.environ, "GRACE_OP": op, "GRACE_PATH": str(path)})
    lines = [x.strip() for x in out.stdout.splitlines() if x.strip()]
    return lines[-1] if lines else "none"


def path_parts(rp):
    """The folder names of an absolute path below its drive (tests anchor them at tmp_path)."""
    return rp.parts[1:]


def open_selected(path):
    """A new Explorer window with the file selected (when none is open)."""
    subprocess.Popen(["explorer.exe", f"/select,{path}"])


# ---- explorer ---------------------------------------------------------------------------------

def shown(at):
    return F.rel(at[3:]) if at.startswith("at=") else ""


@tool("explorer", "Move around in the open File Explorer window instead of opening a new one: go up a folder "
      "(\"gå en mappe op\"), back or forward, go to a folder, or select (highlight) a file in its folder "
      "(\"marker rapporten\"). where says which folder it shows.",
      {"action": s("What to do.", ["where", "up", "back", "forward", "go", "select"]),
       "path": s("For go: the folder; for select: the file (a path from an earlier result or its name).")},
      ["action"], risk="visible")
@F.guard
def t_explorer(ctx, a, alive):
    action = (a.get("action") or "").strip().lower()
    target = None
    if action in ("go", "select"):
        if not (a.get("path") or "").strip():
            return {"ok": False, "result": f"Which {'folder' if action == 'go' else 'file'}? Ask him."}
        target = F.find_existing(ctx, a["path"], kind="folder" if action == "go" else "file")
    elif action not in ("where", "up", "back", "forward"):
        return {"ok": False, "result": f"Unknown explorer action {action!r}."}
    if ctx.dry:
        what = {"where": "say which folder File Explorer shows", "up": "go up one folder in File Explorer",
                "back": "go back in File Explorer", "forward": "go forward in File Explorer",
                "go": f"show {F.rel(target) if target else ''} in File Explorer",
                "select": f"select {target.name if target else ''} in File Explorer"}[action]
        return {"ok": True, "detail": {"dry_run": True}, "result": f"Dry run: would {what}."}
    if action == "where":
        at = shell_explorer("where")
        if at == "none":
            return {"ok": True, "result": "No File Explorer window is open."}
        return {"ok": True, "detail": {"folder": at[3:]}, "result": f"File Explorer shows {shown(at)}."}
    if action == "select":
        at = shell_explorer("select", target)
        if at == "none":
            open_selected(target)
            return {"ok": True, "detail": {"path": str(target)},
                    "result": f"Opened File Explorer at {F.rel(target.parent)} with {target.name} selected."}
        if at == "missing":
            return {"ok": False, "result": f"{target.name} wasn't in the folder Explorer opened. Say so."}
        return {"ok": True, "detail": {"path": str(target)},
                "result": f"Selected {target.name} in {shown(at) or F.rel(target.parent)}."}
    if action == "go":
        at = shell_explorer("navigate", target)
        if at == "none":
            os.startfile(str(target))
            return {"ok": True, "detail": {"folder": str(target)}, "result": f"Opened {F.rel(target)} in a new window."}
        return {"ok": True, "detail": {"folder": at[3:]}, "result": f"File Explorer shows {shown(at)} now."}
    at = shell_explorer(action)
    if at == "none":
        return {"ok": False, "result": "No File Explorer window is open, so there's nothing to go "
                                       f"{'up' if action == 'up' else action} in. Say so."}
    if at == "top":
        return {"ok": False, "result": "File Explorer is already at the top (This PC). Say so."}
    if at == "nohistory":
        return {"ok": False, "result": f"There is nothing to go {action} to in that window. Say so."}
    return {"ok": True, "detail": {"folder": at[3:]}, "result": f"File Explorer shows {shown(at)} now."}


# ---- the file index's extra folders -----------------------------------------------------------

def index_path(data_dir):
    return Path(data_dir) / INDEX_FILE


def extra_roots(data_dir):
    """The folders he added to the file index, as Paths that exist (tools_server.Files)."""
    try:
        names = json.loads(index_path(data_dir).read_text("utf-8"))
    except (OSError, ValueError):
        return []
    return [Path(p) for p in names if isinstance(p, str) and Path(p).is_dir()]


def spoken_path(text):
    """'d drevet projekter' / 'D-drev/Projekter' -> 'D:\\Projekter'; anything else unchanged."""
    t = (text or "").strip().strip("\"'")
    m = re.match(r"(?i)^([a-z])[ -]?drev(?:et)?[\s\\/]*(.*)$", t)
    if m:
        rest = [r.strip() for r in re.split(r"[\\/]+", m.group(2)) if r.strip()]
        return str(Path(f"{m.group(1).upper()}:\\").joinpath(*rest))
    return t


def indexable(p):
    """Why a folder can't be added, or None."""
    if not p.is_absolute():
        return "That isn't a full path; ask him for the drive and folder (e.g. D:\\Projekter)."
    if not p.is_dir():
        return f"There is no folder {p}."
    rp = p.resolve()
    home = F.HOME.resolve()
    if rp.parent == rp and rp.drive.upper() == os.environ.get("SystemDrive", "C:").upper():
        return "The whole system drive is too big to index; ask for a folder on it."
    parts = [x.lower() for x in path_parts(rp)]
    if any(x in NO_INDEX for x in parts) or any(x.startswith(".") for x in parts):
        return f"{rp} is a system, program or hidden folder, which I don't index."
    if rp == home or any(rp == r.resolve() or r.resolve() in rp.parents for r in F.roots()):
        return f"{F.rel(rp)} is already searched (his user folders always are)."
    if home in rp.parents:
        return None
    if any(x in ("users",) for x in parts[:1]):
        return "That is another user's folder, which I don't index."
    return None


def save_roots(ctx, roots):
    p = index_path(ctx.data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([str(r) for r in roots], ensure_ascii=False, indent=1), "utf-8")
    reindex = getattr(getattr(ctx, "files", None), "reindex", None)
    if reindex:
        reindex()


@tool("index_folders", "Choose which extra folders find_files searches besides his user folders (Desktop, "
      "Documents, Downloads...): add a folder (\"søg også i D-drevet projekter\"), remove one, or list them.",
      {"action": s("What to do.", ["list", "add", "remove"]),
       "folder": s("For add/remove: the folder's full path, e.g. D:\\Projekter.")},
      ["action"], risk="visible")
def t_index_folders(ctx, a, alive):
    action = (a.get("action") or "").strip().lower()
    roots = extra_roots(ctx.data_dir)
    if action == "list":
        if not roots:
            return {"ok": True, "result": "find_files searches only his user folders (Desktop, Documents, "
                                          "Downloads, Pictures, Videos, Music); no extra folders are added."}
        return {"ok": True, "folders": [str(r) for r in roots],
                "result": f"Besides his user folders, find_files also searches: {', '.join(map(str, roots))}."}
    if action not in ("add", "remove"):
        return {"ok": False, "result": f"Unknown index_folders action {action!r}."}
    raw = (a.get("folder") or "").strip()
    if not raw:
        return {"ok": False, "result": "Which folder? Ask him."}
    p = F.expand(spoken_path(raw)) or Path(spoken_path(raw))
    if action == "remove":
        keep = [r for r in roots if os.path.normcase(str(r)) != os.path.normcase(str(p))]
        if len(keep) == len(roots):
            return {"ok": False, "result": f"{p} isn't one of the added folders. Say so."}
        if not ctx.dry:
            save_roots(ctx, keep)
        return {"ok": True, "detail": {"dry_run": ctx.dry},
                "result": f"{'Dry run: would stop' if ctx.dry else 'Stopped'} searching {p}."}
    why = indexable(p)
    if why:
        return {"ok": False, "result": why + " Nothing was changed. Say so."}
    p = p.resolve()
    if any(os.path.normcase(str(r)) == os.path.normcase(str(p)) for r in roots):
        return {"ok": True, "result": f"{p} is already searched."}
    if len(roots) >= MAX_EXTRA:
        return {"ok": False, "result": f"There are already {MAX_EXTRA} extra folders; remove one first."}
    if ctx.dry:
        return {"ok": True, "detail": {"dry_run": True}, "result": f"Dry run: would add {p} to the file search."}
    save_roots(ctx, roots + [p])
    return {"ok": True, "detail": {"folder": str(p)},
            "result": f"Added {p} to the file search. Its files can be found in about a minute, once it is indexed."}
