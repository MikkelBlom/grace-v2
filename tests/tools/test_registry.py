import grace_tools
from grace_tools import registry

from conftest import call


def test_every_tool_is_well_formed():
    tools = grace_tools.load()
    for t in tools.values():
        assert t.description and t.description[0].isupper(), t.name
        assert t.risk in registry.RISKS, t.name
        assert set(t.required) <= set(t.params), t.name
        for p in t.params.values():
            assert p.get("description") or p.get("type") == "object", t.name
        assert t.risk != "destructive" or t.summary, t.name


def test_destructive_needs_a_yes(ctx):
    calls = []

    @registry.tool("test_wipe", "Wipes a test thing.", {"what": registry.s("What.")}, ["what"],
                   risk="destructive", summary=lambda a: f"wipe {a['what']}")
    def wipe(ctx, a, alive):
        calls.append(a)
        return {"result": "Wiped."}

    try:
        out = call(ctx, "test_wipe", what="x")
        assert out["needs_confirm"] and not calls
        assert "wipe x" in out["result"]
        token = out["confirm"]["token"]
        name, args, summary, _ = ctx.confirmations.take(token)
        assert (name, args, summary) == ("test_wipe", {"what": "x"}, "wipe x")
        assert ctx.confirmations.take(token) is None  # one yes, one run
        out = call(ctx, "test_wipe", confirmed=True, what="x")
        assert calls == [{"what": "x"}] and out["result"] == "Wiped."
    finally:
        registry.TOOLS.pop("test_wipe")


def test_find_matches_words():
    @registry.tool("test_move_files", "Move files or folders to another folder.", {"src": registry.s("From.")})
    def move(ctx, a, alive):
        return {"result": ""}

    try:
        assert registry.find("move some files into a folder")[0].name == "test_move_files"
    finally:
        registry.TOOLS.pop("test_move_files")
