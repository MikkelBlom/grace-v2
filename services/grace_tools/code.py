"""The code group, part 1: running code in the grace-sandbox container (docs/tools/code.md).

Every job is a fresh `docker run --rm` of docker/sandbox/Dockerfile with no network, a read-only
root filesystem, 2 GB / 2 CPUs / 256 processes, and one scratch folder of its own mounted at
/work; nothing else of the PC is visible inside. If Docker is down the tools say so and stop:
code is never run on the host instead. Git and GitHub are in code_git.py.
"""

import hashlib
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from .registry import b, n, s, tool

ROOT = Path(__file__).resolve().parents[2]
SANDBOX_DIR = ROOT / "docker/sandbox"
IMAGE = "grace-sandbox"
HOME = Path.home()
TIMEOUT_S, MAX_TIMEOUT_S = 60, 300
MAX_INPUT_BYTES = 500 * 1024 * 1024
JOB_KEEP_S = 24 * 3600  # old job folders are cleared on the next run
SCRIPT = {"python": ("_job.py", ["python", "/work/_job.py"]),
          "node": ("_job.js", ["node", "/work/_job.js"]),
          "shell": ("_job.sh", ["bash", "/work/_job.sh"])}
ALIASES = {"py": "python", "python3": "python", "js": "node", "javascript": "node", "nodejs": "node",
           "bash": "shell", "sh": "shell"}
SKIP_DIRS = {"node_modules", ".local", "__pycache__", ".npm", ".cache"}
NO_DOCKER = ("Docker isn't running, so I can't run code. Tell him to start Docker Desktop. "
             "(Code is never run directly on his PC.)")
build = {"proc": None, "log": None, "started": 0.0}  # the one background image build


# --- OS layer (tests replace these) -------------------------------------------------------------

def docker(args, timeout=30, alive=lambda: True):
    """Runs the docker CLI: (returncode, stdout, stderr). returncode is None when it timed out
    or the engine gave up on the call; -1 when docker isn't installed."""
    try:
        p = subprocess.Popen(["docker", *args], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except FileNotFoundError:
        return -1, "", "docker is not installed"
    end = time.time() + timeout
    while True:
        try:
            out, err = p.communicate(timeout=0.5)
            return p.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
        except subprocess.TimeoutExpired:
            if time.time() > end or not alive():
                p.kill()
                out, err = p.communicate()
                return None, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def start_build(args, log_path):
    """Starts `docker build` in the background (it takes minutes; a tool call gets seconds)."""
    log = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(["docker", *args], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


# --- helpers --------------------------------------------------------------------------------

def dockerfile_hash():
    return hashlib.sha256((SANDBOX_DIR / "Dockerfile").read_bytes()).hexdigest()[:12]


def docker_up():
    rc, out, err = docker(["info", "--format", "{{.ServerVersion}}"], timeout=15)
    return (rc == 0 and out.strip()) or None


def image_state():
    """'missing', 'old' (built from an older Dockerfile) or 'ok'."""
    rc, out, _ = docker(["image", "inspect", IMAGE, "--format", '{{index .Config.Labels "grace.dockerfile"}}'],
                        timeout=15)
    if rc != 0:
        return "missing"
    return "ok" if out.strip() == dockerfile_hash() else "old"


def building():
    p = build["proc"]
    return p is not None and p.poll() is None


def tilde(p):
    return str(p).replace(str(HOME), "~")


def expand(path):
    return Path(os.path.expandvars(str(path).strip().strip('"').replace("~", str(HOME), 1)))


def trim(text, limit):
    text = text.strip()
    if len(text) <= limit:
        return text
    return "...(start cut)\n" + text[-limit:]  # the end holds the result and the error


def size(nbytes):
    return f"{nbytes / 1024 / 1024:.1f} MB" if nbytes >= 1024 * 1024 else f"{max(1, nbytes // 1024)} KB"


def unique(folder, name):
    """A path in folder that doesn't exist yet: never overwrite (BUILDING-TOOLS.md)."""
    p = folder / name
    stem, suf, i = p.stem, p.suffix, 2
    while p.exists():
        p, i = folder / f"{stem} ({i}){suf}", i + 1
    return p


def clear_old_jobs(scratch):
    now = time.time()
    for d in scratch.glob("job-*"):
        try:
            if d.is_dir() and now - d.stat().st_mtime > JOB_KEEP_S:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def run_args(job, name, cmd, network):
    """The docker run command line: the sandbox's whole security boundary is here."""
    return ["run", "--rm", "--name", name,
            "--network", "bridge" if network else "none",
            "--memory", "2g", "--memory-swap", "2g", "--cpus", "2", "--pids-limit", "256",
            "--read-only", "--tmpfs", "/tmp:rw,exec,nosuid,size=512m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "1000:1000",
            "--mount", f"type=bind,source={job},target=/work",
            "--workdir", "/work", IMAGE, *cmd]


def made_files(job, before):
    out = []
    for dirpath, dirnames, filenames in os.walk(job):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            p = Path(dirpath) / f
            rel = p.relative_to(job).as_posix()
            if rel.startswith("_job.") or before.get(rel) == p.stat().st_mtime:
                continue
            out.append((rel, p))
    return sorted(out)


# --- tools ----------------------------------------------------------------------------------

@tool("run_code",
      "Run Python, Node or shell code in an isolated sandbox (no network, no access to his files except "
      "the ones you pass in). Use it for any non-trivial calculation, any question about data in a file "
      "(CSV, Excel), charts, and to check that code you write actually works before you show it to him. "
      "Python has numpy, pandas, matplotlib, scipy, sympy and openpyxl. Print the answer; files the "
      "code writes in its working folder are listed, and kept only with save_outputs.",
      {"language": s("The language.", ["python", "node", "shell"]),
       "code": s("The code. Input files are in the working folder under their own names."),
       "files": {"type": "array", "items": {"type": "string"},
                 "description": "Paths of his files to copy in (read-only copies), e.g. a CSV he named."},
       "save_outputs": b("Copy the files the code makes to ~/Documents/Grace (or save_to). Only when he asked to keep them."),
       "save_to": s("A folder to save the outputs in, if he named one."),
       "timeout_s": n("Seconds before the job is killed (default 60, max 300)."),
       "network": b("Give the job internet, only to install a package inside it (pip install --user, npm install). "
                    "Off by default; needs a reason."),
       "reason": s("Why the job needs the network (required with network).")},
      ["language", "code"], risk="visible", slow=True)
def run_code(ctx, a, alive):
    language = (a.get("language") or "python").strip().lower()
    language = ALIASES.get(language, language)
    if language not in SCRIPT:
        return {"ok": False, "result": f"Unknown language {language!r}: use python, node or shell."}
    code = a.get("code") or ""
    if not code.strip():
        return {"ok": False, "result": "No code was given."}
    network = bool(a.get("network"))
    if network and not (a.get("reason") or "").strip():
        return {"ok": False, "result": "The network is only for installing a package inside the job: "
                                       "say why in reason, or run without network."}
    try:
        timeout = min(MAX_TIMEOUT_S, max(1, int(float(a.get("timeout_s") or TIMEOUT_S))))
    except (TypeError, ValueError):
        timeout = TIMEOUT_S

    inputs, problems, total = [], [], 0
    for f in a.get("files") or []:
        p = expand(f)
        if not p.is_file():
            problems.append(f"{f} (not found)")
            continue
        total += p.stat().st_size
        if total > MAX_INPUT_BYTES:
            problems.append(f"{p.name} (over the 500 MB limit)")
            continue
        inputs.append(p)
    if problems and not inputs and a.get("files"):
        return {"ok": False, "result": "Nothing ran: couldn't find " + ", ".join(problems) +
                                       ". Find the file with find_files first."}

    script, cmd = SCRIPT[language]
    # Dry run still runs offline jobs: the sandbox changes nothing he has (only its own scratch
    # folder), and the benches need real answers. It never saves outputs or opens the network.
    if ctx.dry and network:
        return {"ok": True, "result": f"Dry run: would run {len(code.splitlines())} lines of {language} in the "
                                      f"sandbox with network ({timeout} s limit)."}
    if not docker_up():
        return {"ok": False, "result": NO_DOCKER}
    state = image_state()
    if state == "missing":
        return {"ok": False, "result": "The sandbox image isn't built yet. "
                + ("It is being built now; try again in a few minutes." if building()
                   else "Build it with build_sandbox (takes a few minutes), then run this again.")}

    clear_old_jobs(ctx.scratch_dir)
    job_id = uuid.uuid4().hex[:8]
    job = ctx.scratch_dir / f"job-{job_id}"
    job.mkdir(parents=True)
    names = set()
    for p in inputs:
        dest = unique(job, p.name) if p.name in names or p.name == script else job / p.name
        names.add(dest.name)
        shutil.copy2(p, dest)
    (job / script).write_text(code, encoding="utf-8", newline="\n")
    before = {p.relative_to(job).as_posix(): p.stat().st_mtime for p in job.rglob("*") if p.is_file()}

    name = f"grace-job-{job_id}"
    t0 = time.time()
    rc, out, err = docker(run_args(job, name, cmd, network), timeout=timeout + 15,
                          alive=lambda: alive() and time.time() - t0 < timeout)
    secs = time.time() - t0
    if rc is None:
        docker(["rm", "-f", name], timeout=30)  # the CLI going away doesn't stop the container; rm -f kills and removes it at once
    made = made_files(job, before)

    if rc is None:
        why = "it was cut off" if secs < timeout else f"it ran past the {timeout} s limit"
        head = f"The code was stopped after {secs:.0f} s: {why}."
    elif rc == 125 or "docker: Error" in err:
        return {"ok": False, "job": str(job), "result": "The sandbox didn't start: " + trim(err, 400)}
    else:
        head = f"Ran {language} in the sandbox ({'network on' if network else 'no network'}) in {secs:.1f} s, exit code {rc}."
    if rc == 137:
        head += " It was killed, probably for using more than 2 GB of memory."
    parts = [head]
    if out.strip():
        parts.append("Output:\n" + trim(out, 900))
    elif rc == 0:
        parts.append("It printed nothing.")
    if err.strip():
        parts.append("Errors:\n" + trim(err, 500 if rc else 250))

    saved = []
    if made:
        listed = ", ".join(f"{rel} ({size(p.stat().st_size)})" for rel, p in made[:8])
        more = f" and {len(made) - 8} more" if len(made) > 8 else ""
        if a.get("save_outputs"):
            dest = expand(a["save_to"]) if a.get("save_to") else HOME / "Documents/Grace"
            if ctx.dry:
                return {"ok": rc == 0, "exit_code": rc, "job": str(job), "saved": [],
                        "result": "\n".join(parts + [f"Files it made: {listed}{more}. Dry run: would save them to {tilde(dest)}."])}
            dest.mkdir(parents=True, exist_ok=True)
            for rel, p in made:
                q = unique(dest, Path(rel).name)
                shutil.copy2(p, q)
                saved.append(tilde(q))
            parts.append(f"Files it made: {listed}{more}. Saved to {tilde(dest)}: " + ", ".join(Path(x).name for x in saved) + ".")
        else:
            parts.append(f"Files it made: {listed}{more}. Not saved; run it again with save_outputs if he wants them.")
    if problems:
        parts.append("Not copied in: " + ", ".join(problems) + ".")
    return {"ok": rc == 0, "exit_code": rc, "job": str(job), "saved": saved, "result": "\n".join(parts)}


@tool("sandbox_status",
      "Check whether the code sandbox works: is Docker running, is the grace-sandbox image built and current.",
      {})
def sandbox_status(ctx, a, alive):
    version = docker_up()
    if not version:
        return {"ok": False, "docker": False, "image": None, "result": NO_DOCKER}
    state = image_state()
    if building():
        note = "The image is being built right now."
    elif state == "ok":
        note = "The sandbox image is built and current: run_code works."
    elif state == "old":
        note = "The sandbox image is from an older Dockerfile; it still works, and build_sandbox updates it."
    else:
        note = "The sandbox image isn't built; build_sandbox builds it (takes a few minutes)."
    if build["log"] and not building() and build["proc"] is not None and build["proc"].returncode:
        note += " The last build failed: " + trim(Path(build["log"]).read_text("utf-8", "replace"), 300)
    return {"ok": True, "docker": True, "image": state, "building": building(),
            "result": f"Docker {version} is running. {note}"}


@tool("build_sandbox",
      "Build (or update) the grace-sandbox Docker image that run_code needs. Slow: it downloads Python and "
      "packages and takes a few minutes; it runs in the background, and sandbox_status says when it is done.",
      {"rebuild": b("Build again even if the image is current.")},
      risk="visible", slow=True)
def build_sandbox(ctx, a, alive):
    if not docker_up():
        return {"ok": False, "result": NO_DOCKER}
    if building():
        return {"ok": True, "result": "The sandbox image is already being built; sandbox_status says when it is done."}
    state = image_state()
    if state == "ok" and not a.get("rebuild"):
        return {"ok": True, "result": "The sandbox image is already built and current."}
    if ctx.dry:
        return {"ok": True, "result": f"Dry run: would build the {IMAGE} image from docker/sandbox (a few minutes)."}
    log = ctx.data_dir / "sandbox-build.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    build.update(proc=start_build(["build", "-t", IMAGE, "--label", f"grace.dockerfile={dockerfile_hash()}",
                                   str(SANDBOX_DIR)], log), log=log, started=time.time())
    return {"ok": True, "result": "Started building the sandbox image. It takes a few minutes; "
                                  "sandbox_status says when it is done."}
