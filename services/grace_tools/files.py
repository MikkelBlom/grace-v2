"""Files: writing, editing, moving, copying, deleting (to the Recycle Bin), zipping, and undoing
Grace's own last change (ADR 0021, docs/tools/files.md). Reading, creating, comparing and
converting documents live in files_docs.py; both share the path rules here.

Paths: a full path from an earlier result, ~/..., a Danish folder name ("dokumenter/rapport.docx",
"skrivebordet") or a bare file name (looked up in the usual folders, then the server's file index).
Nothing outside his home folder is touched, nor hidden or system folders inside it (AppData, .ssh).
New files go to ~/Documents/Grace/ unless he said where.

Every change Grace makes is logged in ctx.data_dir/files-undo.jsonl (with a backup copy of what
an edit overwrote) so undo_last can take it back.
"""

import functools
import json
import os
import re
import shutil
import time
import zipfile
from pathlib import Path

from .registry import b, s, tool

HOME = Path.home()  # tests replace this with a tmp folder
ROOT_NAMES = ("Desktop", "Documents", "Downloads", "Pictures", "Videos", "Music")
# Copied from tools_server.FOLDER_ALIASES (importing it would load OpenVINO), plus a few more.
ALIAS_NAMES = {
    "skrivebord": "Desktop", "skrivebordet": "Desktop", "desktop": "Desktop",
    "dokumenter": "Documents", "mine dokumenter": "Documents", "documents": "Documents",
    "overførsler": "Downloads", "downloads": "Downloads", "hentede filer": "Downloads",
    "billeder": "Pictures", "pictures": "Pictures", "videoer": "Videos", "videos": "Videos",
    "musik": "Music", "music": "Music", "grace": "Documents/Grace",
}
SKIP_DIRS = {"node_modules", ".git", ".venv", "venv", "__pycache__", "target", ".runtime", "dist",
             "build", ".cache", "site-packages", "$recycle.bin", "appdata", ".idea", ".vs", "obj", "bin"}
# Inside home but never Grace's business: settings, keys, caches, the registry hive.
SYSTEM_PARTS = {"appdata", "application data", "local settings", "$recycle.bin", "ntuser.dat"}
NO_CREATE = {".exe", ".bat", ".cmd", ".ps1", ".vbs", ".js", ".msi", ".com", ".scr", ".lnk", ".reg",
             ".jar", ".py", ".psm1", ".hta", ".wsf", ".cpl", ".dll", ".sys"}  # tools_server.NO_OPEN
TEXT_EXT = {".txt", ".md", ".csv", ".json", ".html", ".htm", ".xml", ".yaml", ".yml", ".ini", ".log",
            ".tex", ".tsv", ".srt"}
MAX_EDIT = 5_000_000  # bytes; bigger text files aren't something to edit by voice
MAX_UNZIP = 2_000_000_000  # bytes unpacked (a zip bomb stops here)
UNDO_KEEP = 30
RESULT_MAX = 1500


class PathError(Exception):
    """A path Grace may not use, or can't find. The message is for Gemma."""


def guard(fn):
    """Path and file errors become a short result for Gemma instead of a crash."""
    @functools.wraps(fn)
    def run(ctx, a, alive):
        try:
            out = fn(ctx, a, alive)
        except PathError as e:
            return {"ok": False, "result": f"{e} Nothing was changed. Say so."}
        except (OSError, zipfile.BadZipFile, ValueError) as e:
            return {"ok": False, "result": f"That didn't work: {e}. Say it failed."}
        if len(out.get("result", "")) > RESULT_MAX:
            out["result"] = out["result"][:RESULT_MAX - 3] + "..."
        return out
    return run


# ---- paths ------------------------------------------------------------------------------------

def aliases():
    return {k: HOME.joinpath(*v.split("/")) for k, v in ALIAS_NAMES.items()}


def default_dir():
    return HOME / "Documents" / "Grace"


def roots():
    return [HOME / r for r in ROOT_NAMES]


def rel(p):
    """How a path is shown to Gemma: relative to home, e.g. Documents\\Grace\\x.docx."""
    try:
        return str(Path(p).resolve().relative_to(HOME.resolve()))
    except ValueError:
        return str(p)


def expand(text):
    """A spoken or earlier-result path as a Path, or None when it is a bare relative name."""
    t = (text or "").strip().strip("\"'").strip()
    if not t:
        raise PathError("No file or folder was named.")
    t = os.path.expandvars(t)
    if t.startswith("~"):
        t = str(HOME) + t[1:]
    parts = [x for x in re.split(r"[\\/]+", t) if x]
    al = aliases()
    if parts and parts[0].lower() in al:
        return al[parts[0].lower()].joinpath(*parts[1:])
    if t.lower() in al:
        return al[t.lower()]
    p = Path(t)
    return p if p.is_absolute() else None


def check(p, write=False):
    """Raises PathError unless p is inside his home folder and outside system or hidden folders."""
    try:
        rp = Path(p).resolve()
    except OSError:
        raise PathError(f"{p} isn't a usable path.")
    home = HOME.resolve()
    if rp != home and home not in rp.parents:
        raise PathError("That is outside Mikkel's own folders, which I never touch.")
    for part in rp.relative_to(home).parts:
        lp = part.lower()
        if lp.startswith(".") or lp in SYSTEM_PARTS or lp.startswith("ntuser"):
            raise PathError(f"{part} is a system or hidden folder, which I don't touch.")
    if write and (rp == home or rp in [r.resolve() for r in roots()]):
        raise PathError(f"{rp.name} is one of his main folders, which I don't move, rename or delete.")
    return rp


def find_existing(ctx, text, kind="any"):
    """An existing file or folder he named, checked. Bare names are looked for in Grace's folder,
    Documents, Desktop and Downloads, then in the server's file index."""
    p = expand(text)
    if p is None:
        name = Path(text.strip().strip("\"'")).as_posix()
        for base in [default_dir()] + [HOME / r for r in ("Documents", "Desktop", "Downloads")]:
            if (base / name).exists():
                p = base / name
                break
        if p is None and getattr(ctx, "files", None) is not None:
            hits = [c for c in ctx.files.candidates(text, 5) if kind == "any" or c[2] == (kind == "folder")]
            if hits:
                p = Path(hits[0][1])
    if p is None or not p.exists():
        raise PathError(f"I couldn't find {text!r} in his folders (use find_files to look for it).")
    if kind == "file" and p.is_dir():
        raise PathError(f"{p.name} is a folder, not a file.")
    if kind == "folder" and not p.is_dir():
        raise PathError(f"{p.name} is a file, not a folder.")
    return check(p)


def find_folder(ctx, text):
    """An existing folder he named as a destination ("skrivebordet", "Dokumenter/Projekter")."""
    p = expand(text)
    if p is not None:
        if not p.is_dir():
            raise PathError(f"There is no folder {rel(p)}. Make it first with make_folder.")
        return check(p)
    return find_existing(ctx, text, kind="folder")


def new_path(name, folder=None, ctx=None, ext=None):
    """Where a new file goes: a full path, a name inside `folder`, or a name in ~/Documents/Grace."""
    name = (name or "").strip().strip("\"'")
    p = expand(name) if name else None
    if p is None:
        base = find_folder(ctx, folder) if folder else default_dir()
        clean = re.sub(r'[<>:"|?*\x00-\x1f]', "", name).strip(" .") or "Grace"
        p = base / clean
    if ext and p.suffix.lower() != ext:
        p = p.with_name(p.name + ext) if not p.suffix or len(p.suffix) > 6 else p.with_suffix(ext)
    if p.suffix.lower() in NO_CREATE:
        raise PathError(f"I don't create {p.suffix} files (programs and scripts).")
    check(p.parent if p.parent.exists() else nearest(p))
    return check(p, write=True)


def nearest(p):
    """The closest existing parent (to check a path whose folders don't exist yet)."""
    q = Path(p)
    while not q.exists() and q.parent != q:
        q = q.parent
    return q


def free(p):
    """p, or 'name (2).ext' if p is taken: generated files never overwrite."""
    p = Path(p)
    i = 2
    while p.exists():
        p = p.with_name(f"{re.sub(r' [(]\d+[)]$', '', p.stem)} ({i}){p.suffix}")
        i += 1
    return p


def names(items, limit=4):
    items = list(items)
    shown = ", ".join(items[:limit])
    return shown + (f" and {len(items) - limit} more" if len(items) > limit else "")


def as_list(v):
    """Gemma sends lists as lists, JSON strings or comma/line separated text."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [x for x in v if x not in (None, "")]
    t = str(v).strip()
    if t.startswith("["):
        try:
            return [x for x in json.loads(t) if x not in (None, "")]
        except ValueError:
            pass
    return [x.strip() for x in re.split(r"\n|;", t) if x.strip()]


# ---- the OS layer (tests replace these) -------------------------------------------------------

def send_to_trash(path):
    """Moves to the Recycle Bin; never a hard delete."""
    from send2trash import send2trash
    send2trash(str(path))


# ---- undo -------------------------------------------------------------------------------------

def stamp(p):
    try:
        st = Path(p).stat()
        return [round(st.st_mtime, 3), st.st_size]
    except OSError:
        return None


def backup(ctx, p):
    """A copy of a file about to be changed, for undo."""
    d = Path(ctx.data_dir) / "files-undo"
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f"{int(time.time() * 1000)}-{Path(p).name}"
    shutil.copy2(p, dst)
    return str(dst)


def log_undo(ctx, what, steps):
    """steps: [{"op": "created", "path"}, {"op": "moved", "src", "dst"},
    {"op": "changed", "path", "backup"}, {"op": "trashed", "path"}], undone in reverse order."""
    for st in steps:
        if st["op"] in ("created", "changed"):
            st["stamp"] = stamp(st["path"])
        elif st["op"] == "moved":
            st["stamp"] = stamp(st["dst"])
    f = Path(ctx.data_dir) / "files-undo.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    lines = f.read_text("utf-8").splitlines() if f.exists() else []
    lines = (lines + [json.dumps({"what": what, "time": time.time(), "steps": steps})])[-UNDO_KEEP:]
    f.write_text("\n".join(lines) + "\n", "utf-8")
    # Backups older than the log can't be undone any more.
    kept = {st.get("backup") for line in lines for st in json.loads(line)["steps"]}
    d = f.parent / "files-undo"
    for old in (d.iterdir() if d.is_dir() else []):
        if str(old) not in kept:
            old.unlink(missing_ok=True)


def unchanged(st, path):
    return Path(path).is_dir() or st.get("stamp") is None or stamp(path) == st["stamp"]


@tool("undo_last", "Undo the last file change Grace made (a write, edit, move, rename, copy, new folder, "
      "zip or document she created), when Mikkel says fortryd or undo or that it was wrong.",
      {}, risk="visible")
@guard
def t_undo_last(ctx, a, alive):
    f = Path(ctx.data_dir) / "files-undo.jsonl"
    lines = f.read_text("utf-8").splitlines() if f.exists() else []
    if not lines:
        return {"ok": False, "result": "There is nothing of mine to undo. Say so."}
    entry = json.loads(lines[-1])
    steps = list(reversed(entry["steps"]))
    # He may have worked on the file since; then undoing would throw his work away.
    for st in steps:
        path = st.get("dst") or st.get("path")
        if st["op"] != "trashed" and Path(path).exists() and not unchanged(st, path):
            return {"ok": False, "result": f"{Path(path).name} was changed after I {entry['what']}, so I won't "
                                           "undo it automatically. Say so."}
        if st["op"] == "moved" and Path(st["src"]).exists():
            return {"ok": False, "result": f"Something new is now at {rel(st['src'])}, so I can't move it back. Say so."}
    if ctx.dry:
        return {"ok": True, "detail": {"dry_run": True, "undo": entry}, "result": f"Would undo: {entry['what']}."}
    notes = []
    for st in steps:
        op = st["op"]
        if op == "created" and Path(st["path"]).exists():
            send_to_trash(st["path"])
        elif op == "moved":
            Path(st["src"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(st["dst"], st["src"])
        elif op == "changed":
            shutil.copy2(st["backup"], st["path"])
        elif op == "trashed":
            notes.append(f"{Path(st['path']).name} is in the Recycle Bin; he can restore it from there")
    f.write_text("\n".join(lines[:-1]) + ("\n" if len(lines) > 1 else ""), "utf-8")
    return {"ok": True, "detail": {"undone": entry["what"]},
            "result": f"Undone: {entry['what']}." + (" " + "; ".join(notes) + "." if notes else "")}


# ---- writing ----------------------------------------------------------------------------------

def read_text(p):
    """(text, encoding) of a text file: UTF-8, else Windows' own code page."""
    raw = Path(p).read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", "replace"), "utf-8"


def looks_markdown(text):
    return bool(re.search(r"(?m)^(#{1,6} |[-*] |\d+\. )", text))


@tool("write_file", "Write a new text or Markdown file (notes, a list, a draft) with the content Mikkel "
      "dictates or asks for. Never replaces an existing file. For Word, PDF, Excel or PowerPoint use "
      "create_document.",
      {"name": s("File name, e.g. 'indkøbsliste.md', or a full path. Without .txt/.md it picks one."),
       "content": s("The full text to write. Markdown is fine."),
       "folder": s("Folder if he named one (skrivebordet, Dokumenter, a path); default Documents/Grace.")},
      ["name", "content"], risk="visible", core=True)
@guard
def t_write_file(ctx, a, alive):
    content = str(a.get("content") or "")
    name = (a.get("name") or "").strip() or "Note"
    suffix = Path(name).suffix
    has_ext = bool(suffix) and len(suffix) <= 6 and " " not in suffix
    ext = None if has_ext else (".md" if looks_markdown(content) else ".txt")
    p = new_path(name, a.get("folder"), ctx, ext)
    if p.suffix.lower() not in TEXT_EXT:
        raise PathError(f"write_file writes text files; for {p.suffix} use create_document or convert_file.")
    if p.exists():
        raise PathError(f"{p.name} already exists in {rel(p.parent)}. Use append_to_file or edit_file, or pick a new name.")
    if ctx.dry:
        return {"ok": True, "detail": {"path": str(p), "dry_run": True},
                "result": f"Would write {p.name} ({len(content)} characters) in {rel(p.parent)}. path: {p}"}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, "utf-8", newline="")
    log_undo(ctx, f"wrote {p.name}", [{"op": "created", "path": str(p)}])
    return {"ok": True, "detail": {"path": str(p)},
            "result": f"Wrote {p.name} in {rel(p.parent)} ({len(content.splitlines())} lines). path: {p}"}


@tool("append_to_file", "Add text to the end of an existing text, Markdown or Word file (add an item to "
      "a list, a line to notes).",
      {"path": s("The file: a path from an earlier result or its name."),
       "text": s("The text to add. In a Word file each line becomes a paragraph; '- ' lines become bullets.")},
      ["path", "text"], risk="visible")
@guard
def t_append_to_file(ctx, a, alive):
    p = find_existing(ctx, a.get("path"), kind="file")
    text = str(a.get("text") or "")
    if not text.strip():
        raise PathError("There was no text to add.")
    ext = p.suffix.lower()
    if ext not in TEXT_EXT and ext != ".docx":
        raise PathError(f"I can only add text to text, Markdown and Word files, not {ext} files.")
    if ctx.dry:
        return {"ok": True, "detail": {"path": str(p), "dry_run": True},
                "result": f"Would add {len(text.splitlines())} lines to {p.name}."}
    bak = backup(ctx, p)
    if ext == ".docx":
        import docx
        d = docx.Document(str(p))
        for line in text.splitlines():
            if line.strip():
                bullet = re.match(r"\s*[-*•]\s+(.*)", line)
                d.add_paragraph(bullet.group(1) if bullet else line.strip(),
                                style="List Bullet" if bullet else None)
        d.save(str(p))
    else:
        old, enc = read_text(p)
        sep = "" if not old or old.endswith("\n") else "\n"
        with open(p, "a", encoding="utf-8" if enc == "utf-8-sig" else enc, newline="") as f:
            f.write(sep + text + ("" if text.endswith("\n") else "\n"))
    log_undo(ctx, f"added text to {p.name}", [{"op": "changed", "path": str(p), "backup": bak}])
    return {"ok": True, "detail": {"path": str(p)}, "result": f"Added {len(text.splitlines())} lines to {p.name}. path: {p}"}


def edit_summary(a):
    name = Path(str(a.get("path") or "the file")).name
    if a.get("new_content") is not None and not a.get("find"):
        return f"replace all the text in {name}"
    find, rep = str(a.get("find") or "")[:40], str(a.get("replace") or "")[:40]
    return f"change {name}: replace '{find}' with '{rep}'" if rep else f"remove '{find}' from {name}"


def replace_once(text, find, rep, every):
    """(new text, count). Exact first; if that isn't there, a case-insensitive match (STT casing)."""
    n = text.count(find)
    if n:
        return (text.replace(find, rep) if every else text.replace(find, rep, 1)), (n if every else 1)
    pat = re.compile(re.escape(find), re.I)
    hits = len(pat.findall(text))
    if not hits:
        return text, 0
    return pat.sub(lambda m: rep, text, count=0 if every else 1), (hits if every else 1)


def edit_docx(p, find, rep, every):
    import docx
    d = docx.Document(str(p))
    paras = list(d.paragraphs) + [q for t in d.tables for row in t.rows for c in row.cells for q in c.paragraphs]
    count = 0
    for para in paras:
        if every is False and count:
            break
        if find.lower() not in para.text.lower():
            continue
        run = next((r for r in para.runs if find in r.text), None)
        if run is not None:  # inside one run: the formatting survives
            run.text, k = replace_once(run.text, find, rep, every)
        else:  # across runs: the paragraph keeps its first run's formatting
            new, k = replace_once(para.text, find, rep, every)
            for r in para.runs[1:]:
                r.text = ""
            if para.runs:
                para.runs[0].text = new
        count += k
    if count:
        d.save(str(p))
    return count


@tool("edit_file", "Change text in an existing text, Markdown or Word file: replace or remove a word or "
      "sentence, or replace all the text of a text file.",
      {"path": s("The file: a path from an earlier result or its name."),
       "find": s("The exact text to change."),
       "replace": s("What it becomes; empty to remove it."),
       "new_content": s("Only to replace the whole text of a .txt/.md file: the new full text."),
       "all": b("Replace every match (default true); false for only the first.")},
      ["path"], risk="destructive", summary=edit_summary)
@guard
def t_edit_file(ctx, a, alive):
    p = find_existing(ctx, a.get("path"), kind="file")
    ext = p.suffix.lower()
    if ext not in TEXT_EXT and ext != ".docx":
        raise PathError(f"I can edit text, Markdown and Word files, not {ext} files.")
    if p.stat().st_size > MAX_EDIT:
        raise PathError(f"{p.name} is too big to edit this way.")
    find, rep, every = str(a.get("find") or ""), str(a.get("replace") or ""), a.get("all", True) is not False
    whole = a.get("new_content")
    if not find and whole is None:
        raise PathError("Say what text to change (find) or give the new full text (new_content).")
    if whole is not None and not find and ext == ".docx":
        raise PathError("I can't replace a whole Word file's text; make a new document instead.")
    if ext == ".docx":
        if ctx.dry:
            return {"ok": True, "detail": {"dry_run": True}, "result": f"Would {edit_summary(a)}."}
        bak = backup(ctx, p)
        count = edit_docx(p, find, rep, every)
    else:
        old, enc = read_text(p)
        new, count = (str(whole), 1) if whole is not None and not find else replace_once(old, find, rep, every)
        if ctx.dry:
            return {"ok": True, "detail": {"dry_run": True, "matches": count},
                    "result": f"Would {edit_summary(a)} ({count} places)." if count else f"'{find}' isn't in {p.name}."}
        bak = backup(ctx, p) if count else None
        if count:
            p.write_text(new, "utf-8" if enc == "utf-8-sig" else enc, newline="")
    if not count:
        return {"ok": False, "result": f"'{find}' isn't in {p.name}, so nothing was changed. Say so."}
    log_undo(ctx, f"edited {p.name}", [{"op": "changed", "path": str(p), "backup": bak}])
    return {"ok": True, "detail": {"path": str(p), "changes": count},
            "result": f"Changed {p.name} in {count} place{'s' if count != 1 else ''}. path: {p}"}


# ---- moving, copying, folders ----------------------------------------------------------------

def target_of(ctx, src, to):
    """Where src goes for move/copy: into an existing folder, or to a new name or full path."""
    to = (to or "").strip()
    if not to:
        raise PathError("Say where it should go.")
    p = expand(to)
    if p is not None:
        if not p.exists() and not p.suffix and src.is_file():
            raise PathError(f"There is no folder {rel(p)}. Make it first with make_folder.")
        dst = p / src.name if p.is_dir() else p
    elif re.search(r"[\\/]", to) or not Path(to).suffix:
        dst = find_folder(ctx, to) / src.name
    else:
        dst = src.parent / to
    if src.is_file() and not dst.suffix and src.suffix:
        dst = dst.with_suffix(src.suffix)
    if dst.suffix.lower() in NO_CREATE and dst.suffix.lower() != src.suffix.lower():
        raise PathError(f"I don't make {dst.suffix} files (programs and scripts).")
    check(nearest(dst.parent))
    return check(dst, write=True)


def taken(dst):
    return PathError(f"{dst.name} already exists in {rel(dst.parent)}. To replace it use replace_file "
                     "(which asks him first), or choose another name.")


def move(ctx, a, verb):
    src = check(find_existing(ctx, a.get("path")), write=True)
    if verb == "renamed":
        name = re.sub(r'[<>:"|?*\x00-\x1f]', "", str(a.get("new_name") or "")).strip(" .")
        if not name or re.search(r"[\\/]", name):
            raise PathError("Give just the new name (to move it elsewhere use move_file).")
        dst = src.parent / name
        if src.is_file() and not dst.suffix:
            dst = dst.with_name(name + src.suffix)
        if dst.suffix.lower() in NO_CREATE and dst.suffix.lower() != src.suffix.lower():
            raise PathError(f"I don't make {dst.suffix} files (programs and scripts).")
        dst = check(dst, write=True)
    else:
        dst = target_of(ctx, src, a.get("to"))
    if dst == src:
        raise PathError("It already has that name and place.")
    if dst.exists() and dst.resolve() != src:  # a case-only rename is the same file on Windows
        raise taken(dst)
    if src in dst.parents:
        raise PathError("A folder can't go inside itself.")
    if ctx.dry:
        return {"ok": True, "detail": {"src": str(src), "dst": str(dst), "dry_run": True},
                "result": f"Would {verb[:-1]} {src.name} to {rel(dst)}."}
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    log_undo(ctx, f"{verb} {src.name} to {dst.name}" if verb == "renamed" else f"{verb} {src.name} to {rel(dst.parent)}",
             [{"op": "moved", "src": str(src), "dst": str(dst)}])
    return {"ok": True, "detail": {"path": str(dst)}, "result": f"{verb.capitalize()} {src.name} to {rel(dst)}. path: {dst}"}


@tool("rename_file", "Rename a file or folder (it stays in the same folder).",
      {"path": s("The file or folder: a path from an earlier result or its name."),
       "new_name": s("The new name; the extension is kept if left out.")},
      ["path", "new_name"], risk="visible")
@guard
def t_rename_file(ctx, a, alive):
    return move(ctx, a, "renamed")


@tool("move_file", "Move a file or folder into another folder (skrivebordet, Dokumenter, a path). Won't "
      "replace a file of the same name there.",
      {"path": s("The file or folder: a path from an earlier result or its name."),
       "to": s("The folder it goes into (Danish names work), or a full new path.")},
      ["path", "to"], risk="visible")
@guard
def t_move_file(ctx, a, alive):
    return move(ctx, a, "moved")


def copy_any(src, dst):
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


@tool("copy_file", "Copy a file or folder, into another folder or as a new name next to it. Won't "
      "replace an existing file.",
      {"path": s("The file or folder: a path from an earlier result or its name."),
       "to": s("The folder it is copied into, or the copy's new name or path.")},
      ["path", "to"], risk="visible")
@guard
def t_copy_file(ctx, a, alive):
    src = find_existing(ctx, a.get("path"))
    dst = target_of(ctx, src, a.get("to"))
    if dst.exists():
        raise taken(dst)
    if ctx.dry:
        return {"ok": True, "detail": {"dst": str(dst), "dry_run": True}, "result": f"Would copy {src.name} to {rel(dst)}."}
    dst.parent.mkdir(parents=True, exist_ok=True)
    copy_any(src, dst)
    log_undo(ctx, f"copied {src.name} to {rel(dst.parent)}", [{"op": "created", "path": str(dst)}])
    return {"ok": True, "detail": {"path": str(dst)}, "result": f"Copied {src.name} to {rel(dst)}. path: {dst}"}


def replace_summary(a):
    verb = "copy" if a.get("keep_source") else "move"
    return (f"{verb} {Path(str(a.get('path') or '?')).name} over the existing "
            f"{Path(str(a.get('target') or '?')).name}, replacing it")


@tool("replace_file", "Move or copy a file onto an existing file, replacing it (when move_file or "
      "copy_file said the name is taken and he wants it replaced).",
      {"path": s("The file to move or copy."),
       "target": s("The existing file it replaces (a path from an earlier result)."),
       "keep_source": b("true to copy (keep the original), false to move (default).")},
      ["path", "target"], risk="destructive", summary=replace_summary)
@guard
def t_replace_file(ctx, a, alive):
    src = find_existing(ctx, a.get("path"), kind="file")
    dst = check(find_existing(ctx, a.get("target"), kind="file"), write=True)
    keep = bool(a.get("keep_source"))
    if src == dst:
        raise PathError("That is the same file.")
    if not keep:
        check(src, write=True)
    if ctx.dry:
        return {"ok": True, "detail": {"dry_run": True}, "result": f"Would {replace_summary(a)}."}
    bak = backup(ctx, dst)
    steps = [{"op": "changed", "path": str(dst), "backup": bak}]
    if keep:
        shutil.copy2(src, dst)
    else:
        os.replace(src, dst)
        # Undo runs in reverse: move it back to where it came from, then restore the old target.
        steps.append({"op": "moved", "src": str(src), "dst": str(dst)})
    log_undo(ctx, f"replaced {dst.name} with {src.name}", steps)
    return {"ok": True, "detail": {"path": str(dst)},
            "result": f"Replaced {dst.name} in {rel(dst.parent)} with {src.name}"
                      f"{' (copied)' if keep else ''}. The old version can be restored with undo_last. path: {dst}"}


@tool("make_folder", "Make a new folder (in Documents/Grace unless he says where).",
      {"name": s("The folder's name, or a full path."),
       "folder": s("Where to make it if he said (skrivebordet, Dokumenter, a path).")},
      ["name"], risk="visible")
@guard
def t_make_folder(ctx, a, alive):
    p = new_path(a.get("name"), a.get("folder"), ctx)
    if p.exists():
        return {"ok": True, "detail": {"path": str(p)}, "result": f"The folder {p.name} already exists in {rel(p.parent)}. path: {p}"}
    if ctx.dry:
        return {"ok": True, "detail": {"path": str(p), "dry_run": True}, "result": f"Would make the folder {rel(p)}."}
    made = nearest(p)
    p.mkdir(parents=True)
    # Undo removes the topmost folder this call created.
    top = p
    while top.parent != made and top.parent.exists():
        top = top.parent
    log_undo(ctx, f"made the folder {p.name}", [{"op": "created", "path": str(top)}])
    return {"ok": True, "detail": {"path": str(p)}, "result": f"Made the folder {p.name} in {rel(p.parent)}. path: {p}"}


def delete_summary(a):
    items = [Path(str(x)).name for x in as_list(a.get("paths"))]
    if not items:
        return "delete nothing (no files were named)"
    what = items[0] if len(items) == 1 else f"{len(items)} items ({names(items, 3)})"
    return f"move {what} to the Recycle Bin"


@tool("delete_files", "Delete files or folders by moving them to the Recycle Bin (he can restore them "
      "from there).",
      {"paths": {"type": "array", "items": {"type": "string"},
                 "description": "The files or folders: full paths from earlier results, or names."}},
      ["paths"], risk="destructive", summary=delete_summary)
@guard
def t_delete_files(ctx, a, alive):
    items = [check(find_existing(ctx, str(x)), write=True) for x in as_list(a.get("paths"))]
    if not items:
        raise PathError("No files were named.")
    if ctx.dry:
        return {"ok": True, "detail": {"dry_run": True, "paths": [str(p) for p in items]},
                "result": f"Would move {len(items)} item(s) to the Recycle Bin: {names([p.name for p in items])}."}
    done, failed = [], []
    for p in items:
        try:
            send_to_trash(p)
            done.append(p)
        except OSError as e:
            failed.append(f"{p.name} ({e})")
    if done:
        log_undo(ctx, f"deleted {names([p.name for p in done], 2)}", [{"op": "trashed", "path": str(p)} for p in done])
    text = f"Moved {len(done)} item(s) to the Recycle Bin: {names([p.name for p in done])}." if done else ""
    if failed:
        text += f" Could not delete: {'; '.join(failed)}."
    return {"ok": bool(done), "detail": {"deleted": [str(p) for p in done]}, "result": text.strip()}


# ---- zip --------------------------------------------------------------------------------------

@tool("zip_files", "Pack files or folders into a .zip file (to send or store them).",
      {"paths": {"type": "array", "items": {"type": "string"},
                 "description": "The files or folders to pack: paths from earlier results, or names."},
       "name": s("The zip file's name (optional); by default after the first item."),
       "folder": s("Where to put the zip (optional); by default next to the first item.")},
      ["paths"], risk="visible", slow=True)
@guard
def t_zip_files(ctx, a, alive):
    items = [find_existing(ctx, str(x)) for x in as_list(a.get("paths"))]
    if not items:
        raise PathError("No files were named.")
    name = a.get("name") or items[0].stem
    folder = a.get("folder")
    if folder or expand(name) is not None:
        dst = new_path(name, folder, ctx, ".zip")
    else:
        dst = check(items[0].parent / (re.sub(r'[<>:"|?*\\/]', "", Path(name).stem) + ".zip"), write=True)
    dst = free(dst)
    if ctx.dry:
        return {"ok": True, "detail": {"path": str(dst), "dry_run": True}, "result": f"Would pack {len(items)} item(s) into {rel(dst)}."}
    count = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for p in items:
            files = [p] if p.is_file() else [q for q in p.rglob("*") if q.is_file()]
            for q in files:
                if not alive():
                    break
                if q.resolve() == dst.resolve():
                    continue
                z.write(q, q.relative_to(p.parent))
                count += 1
    log_undo(ctx, f"made {dst.name}", [{"op": "created", "path": str(dst)}])
    return {"ok": True, "detail": {"path": str(dst), "files": count},
            "result": f"Packed {count} file(s) into {dst.name} in {rel(dst.parent)} ({dst.stat().st_size // 1024} KB). path: {dst}"}


@tool("unzip_file", "Unpack a .zip file into a new folder next to it (or where he says).",
      {"path": s("The zip file: a path from an earlier result or its name."),
       "to": s("Folder to unpack into (optional); by default a new folder named after the zip.")},
      ["path"], risk="visible", slow=True)
@guard
def t_unzip_file(ctx, a, alive):
    src = find_existing(ctx, a.get("path"), kind="file")
    if a.get("to"):
        dst = expand(a["to"])
        if dst is None:
            dst = src.parent / a["to"].strip()
    else:
        dst = src.parent / src.stem
    dst = check(free(dst) if dst.exists() and any(dst.iterdir()) else dst, write=True)
    with zipfile.ZipFile(src) as z:
        infos = z.infolist()
        if sum(i.file_size for i in infos) > MAX_UNZIP:
            raise PathError(f"{src.name} would unpack to more than 2 GB.")
        for i in infos:  # zip slip: no member may land outside the folder
            out = (dst / i.filename).resolve()
            if dst.resolve() != out and dst.resolve() not in out.parents:
                raise PathError(f"{src.name} has a file that would land outside its folder; I won't unpack it.")
            check(out)
        files = [i for i in infos if not i.is_dir()]
        if ctx.dry:
            return {"ok": True, "detail": {"path": str(dst), "dry_run": True},
                    "result": f"Would unpack {len(files)} file(s) from {src.name} into {rel(dst)}."}
        existed = dst.exists()
        dst.mkdir(parents=True, exist_ok=True)
        z.extractall(dst)
    if not existed:
        log_undo(ctx, f"unpacked {src.name}", [{"op": "created", "path": str(dst)}])
    top = sorted({Path(i.filename).parts[0] for i in infos if Path(i.filename).parts})
    return {"ok": True, "detail": {"path": str(dst), "files": len(files)},
            "result": f"Unpacked {len(files)} file(s) from {src.name} into {rel(dst)}. Top level: {names(top, 6)}. path: {dst}"}
