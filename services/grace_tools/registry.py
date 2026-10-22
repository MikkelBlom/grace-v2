"""The tool registry (ADR 0021): every tool module in this package registers its tools with
`@tool(...)`; services/tools_server.py serves them next to its own.

A tool is `fn(ctx, args, alive) -> dict` returning at least {"ok": bool, "result": str}, where
`result` is the text Gemma reads. `ctx` is the server's Tools object (ctx.dry, ctx.checker,
ctx.files, ctx.apps, ctx.data_dir, ctx.scratch_dir, ctx.gemma(...)); `alive()` turns False once
the engine has given up on the call, so long loops should check it.

Risk, enforced here and not by the model ("the model can't approve itself", CLAUDE.md):
  read         looks only (search, list, read a file)            -> runs
  visible      changes something he sees and can undo at once     -> runs, says what it did
               (open a window, set the volume, write a new file)
  destructive  deletes, overwrites, sends, installs, kills,       -> needs his spoken yes:
               changes settings that are hard to undo                the first call only returns
                                                                     a summary and a token; the
                                                                     engine runs it once he says yes
Things Grace must never do (spend money, enter passwords or card details) are not tools at all.

core=True puts a tool in every request; the rest are found with find_tools and run with use_tool,
which keeps the prompt small and stable (prompt cache, live run 8).
"""

import re
import secrets
import threading
import time

RISKS = ("read", "visible", "destructive")
TOOLS = {}  # name -> Tool
CONFIRM_S = 300  # a confirmation token lives this long


class Tool:
    def __init__(self, name, description, params, required, risk, slow, core, summary, fn, group):
        self.name, self.description, self.params, self.required = name, description, params, required
        self.risk, self.slow, self.core, self.summary, self.fn, self.group = risk, slow, core, summary, fn, group

    def definition(self):
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": self.params, "required": self.required}}}


def tool(name, description, params, required=(), risk="read", slow=False, core=False, summary=None):
    """Registers a tool. `summary(args) -> str` is what Grace says before a destructive action
    ("delete the 3 files in Downloads"); required for risk="destructive"."""
    assert risk in RISKS, risk
    assert risk != "destructive" or summary, f"{name}: a destructive tool needs a summary"
    assert re.fullmatch(r"[a-z][a-z0-9_]*", name), name

    def wrap(fn):
        assert name not in TOOLS, f"tool {name} registered twice"
        TOOLS[name] = Tool(name, description, params, list(required), risk, slow, core, summary, fn,
                           fn.__module__.rsplit(".", 1)[-1])
        return fn
    return wrap


def s(description, enum=None):
    """A string parameter."""
    return {"type": "string", "description": description, **({"enum": enum} if enum else {})}


def n(description):
    return {"type": "number", "description": description}


def b(description):
    return {"type": "boolean", "description": description}


class Confirmations:
    """Destructive calls waiting for his yes: token -> (name, args, summary, time)."""

    def __init__(self):
        self.pending, self.lock = {}, threading.Lock()

    def ask(self, t, args):
        token = secrets.token_hex(4)
        summary = t.summary(args)
        with self.lock:
            now = time.time()
            self.pending = {k: v for k, v in self.pending.items() if now - v[3] < CONFIRM_S}
            self.pending[token] = (t.name, args, summary, now)
        return {"ok": False, "needs_confirm": True, "confirm": {"token": token, "summary": summary},
                "result": f"Not done yet: this needs Mikkel's yes first. Tell him in one short sentence that "
                          f"you will {summary}, and ask if that's okay. Don't say it is done."}

    def take(self, token):
        with self.lock:
            v = self.pending.pop(token, None)
        if v is None or time.time() - v[3] > CONFIRM_S:
            return None
        return v


def run(ctx, name, args, alive, confirmed=False):
    t = TOOLS.get(name)
    if t is None:
        return None
    if t.risk == "destructive" and not confirmed:
        return ctx.confirmations.ask(t, args)
    out = t.fn(ctx, args, alive)
    out.setdefault("ok", True)
    out.setdefault("result", "")
    out["risk"] = t.risk
    return out


def words(text):
    return set(re.findall(r"[a-zæøå0-9]+", text.lower()))


def find(need, limit=5, exclude=()):
    """The tools that best match a need, by words in their name, description and parameters."""
    want = words(need)
    scored = []
    for t in TOOLS.values():
        if t.core or t.name in exclude:
            continue
        text = f"{t.name.replace('_', ' ')} {t.group} {t.description} " + " ".join(
            p.get("description", "") for p in t.params.values())
        have = words(text)
        score = len(want & have) + 2 * len(want & words(t.name.replace("_", " ")))
        if score:
            scored.append((score, t.name))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [TOOLS[name] for _, name in scored[:limit]]
