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
    @registry.tool("test_frobnicate", "Frobnicate widgets into a gadget.", {"src": registry.s("From.")})
    def move(ctx, a, alive):
        return {"result": ""}

    try:
        assert registry.find("frobnicate some widgets")[0].name == "test_frobnicate"
    finally:
        registry.TOOLS.pop("test_frobnicate")


def test_bad_arguments_are_described_not_run(ctx):
    calls = []

    @registry.tool("test_paint", "Paints a wall.", {"wall": registry.s("Which wall."),
                                                    "colour": registry.s("Colour.", ["red", "blue"])}, ["wall"])
    def paint(ctx, a, alive):
        calls.append(a)
        return {"result": "Painted."}

    try:
        out = registry.run(ctx, "test_paint", {"side": "north"}, lambda: True)
        assert not out["ok"] and out["invalid_args"] and not calls
        assert "missing wall" in out["result"] and "side is not among" in out["result"]
        assert "wall: Which wall." in out["result"] and "north" not in out["result"]
        out = registry.run(ctx, "test_paint", {"wall": "north", "extra": 1}, lambda: True)
        assert out["result"] == "Painted."
    finally:
        registry.TOOLS.pop("test_paint")


# ---- use_tool unpacking (tools sim 2026-11-08: file names in use_tool's `name`) ---------------

KNOWN = {"create_document", "make_chart", "write_file", "media"}


def test_unpack_plain_call():
    assert registry.unpack_use_tool({"name": "write_file", "args": {"name": "liste.md", "content": "x"}}, KNOWN) == \
        ("write_file", {"name": "liste.md", "content": "x"}, None)


def test_unpack_file_name_and_tool_swapped():
    call = {"name": "Budget.csv", "args": {"name": "create_document", "format": "csv", "title": "Budget"}}
    assert registry.unpack_use_tool(call, KNOWN) == \
        ("create_document", {"format": "csv", "title": "Budget", "name": "Budget.csv"}, None)


def test_unpack_moves_stray_keys_inside():
    call = {"name": "make_chart", "title": "Weight", "args": {"values": [1, 2]}}
    assert registry.unpack_use_tool(call, KNOWN) == ("make_chart", {"title": "Weight", "values": [1, 2]}, None)


def test_unpack_accepts_a_tool_key():
    assert registry.unpack_use_tool({"tool": "media", "args": {"action": "next"}}, KNOWN)[:2] == \
        ("media", {"action": "next"})


def test_unpack_without_a_tool_explains_the_shape():
    tool, args, why = registry.unpack_use_tool({"args": {"format": "csv"}, "title": "Budget.csv"}, KNOWN)
    assert tool is None and "a file or folder name goes inside args" in why
    tool, _, why = registry.unpack_use_tool({"name": "Why_Eat_Vegetables.pptx", "args": {"format": "pptx"}}, KNOWN)
    assert tool is None and why.startswith("There is no tool called Why_Eat_Vegetables.pptx.")
    assert registry.unpack_use_tool("nonsense", KNOWN)[0] is None
