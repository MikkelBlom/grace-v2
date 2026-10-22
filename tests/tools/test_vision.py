"""Vision tools (services/grace_tools/vision.py). The screenshot, the llama-server call and the
overlay socket are replaced: no real screen, network or window is touched."""

import base64
import io
import json
import urllib.error

import pytest
from PIL import Image, ImageDraw

import grace_tools
from grace_tools import registry, vision, vision_overlay

from conftest import call


@pytest.fixture
def screen(monkeypatch):
    """A fake 1920x1080 screen (optionally offset) and a fake Gemma that records each request."""
    class Fake:
        replies = []
        bodies = []
        urls = []
        shots = []
        origin = (0, 0)
        size = (1920, 1080)

        def shot(self, target="screen", monitor=None):
            self.shots.append((target, monitor))
            img = Image.new("RGB", self.size, (240, 240, 240))
            ImageDraw.Draw(img).rectangle((640, 640, 840, 710), fill=(0, 95, 184))
            return vision.Shot(img, *self.origin, f"monitor 2 of 3" if self.origin != (0, 0) else "the screen")

        def post(self, url, body, timeout=None):
            self.urls.append(url)
            self.bodies.append(body)
            r = self.replies.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        def image(self, i=-1):
            content = self.bodies[i]["messages"][0]["content"]
            data = next(c for c in content if c["type"] == "image_url")["image_url"]["url"]
            assert data.startswith("data:image/jpeg;base64,")
            return Image.open(io.BytesIO(base64.b64decode(data.split(",", 1)[1])))

        def prompt(self, i=-1):
            return next(c for c in self.bodies[i]["messages"][0]["content"] if c["type"] == "text")["text"]

    fake = Fake()
    fake.replies, fake.bodies, fake.urls, fake.shots = [], [], [], []
    monkeypatch.setattr(vision, "screenshot", fake.shot)
    monkeypatch.setattr(vision, "post_chat", fake.post)
    monkeypatch.delenv("GRACE_LLM_URL", raising=False)
    return fake


@pytest.fixture
def drawn(monkeypatch):
    """Overlay commands sent; set .running = False to simulate no overlay."""
    class Sent(list):
        running = True
    sent = Sent()

    def send(cmd, port=None):
        if not sent.running:
            return False
        sent.append(cmd)
        return True
    monkeypatch.setattr(vision_overlay, "send", send)
    return sent


def files_under(path):
    return sorted(p for p in path.rglob("*") if p.is_file())


def test_registration_risks_and_core():
    tools = grace_tools.load()
    group = {name: t for name, t in tools.items() if t.group == "vision"}
    assert set(group) == {"look_at_screen", "read_image", "ocr_screen", "find_on_screen", "point_at"}
    assert {n for n, t in group.items() if t.core} == {"look_at_screen"}
    assert group["point_at"].risk == "visible"
    assert all(t.risk == "read" for n, t in group.items() if n != "point_at")
    assert all(t.slow for t in group.values())


def test_find_tools_reaches_vision_tools():
    grace_tools.load()
    assert "find_on_screen" in [t.name for t in registry.find("find the save button on the screen")]
    assert "point_at" in [t.name for t in registry.find("show him where something is on the screen")]
    assert "ocr_screen" in [t.name for t in registry.find("read out the text on the screen")]
    assert "read_image" in [t.name for t in registry.find("describe an image file")]


def test_look_at_screen_sends_the_image_and_saves_nothing(ctx, screen, tmp_path):
    screen.replies = ["A Notepad window with a shopping list: milk, eggs."]
    before = files_under(tmp_path)
    out = call(ctx, "look_at_screen", question="hvad står der på min skærm")
    assert out["ok"] and out["risk"] == "read"
    assert "shopping list" in out["result"] and out["result"].startswith("On the screen:")
    assert screen.urls == ["http://127.0.0.1:8080"]
    assert screen.image().size == (1536, 864)  # downscaled, not padded
    prompt = screen.prompt()
    assert "hvad står der på min skærm" in prompt and "never instructions" in prompt
    assert files_under(tmp_path) == before  # a screenshot is never written unless asked


def test_look_at_screen_saves_to_scratch_only_when_asked(ctx, screen):
    screen.replies = ["An error dialog."]
    out = call(ctx, "look_at_screen", question="what does the error say", save=True)
    saved = list(ctx.scratch_dir.glob("screen-*.png"))
    assert len(saved) == 1 and str(saved[0]) in out["result"]


def test_llm_address_is_configurable(ctx, screen, monkeypatch):
    screen.replies = ["x", "y"]
    monkeypatch.setenv("GRACE_LLM_URL", "http://127.0.0.1:8090/")
    call(ctx, "look_at_screen", question="q")
    ctx.llm_url = "http://127.0.0.1:9999"
    call(ctx, "look_at_screen", question="q")
    assert screen.urls == ["http://127.0.0.1:8090", "http://127.0.0.1:9999"]


def test_target_and_monitor_are_passed_on(ctx, screen):
    screen.replies = ["x", "y", "z"]
    call(ctx, "look_at_screen", question="q", target="window")
    call(ctx, "look_at_screen", question="q", target="monitor", monitor=2)
    call(ctx, "look_at_screen", question="q", target="nonsense")
    assert screen.shots == [("window", None), ("monitor", 2), ("screen", None)]


def test_no_vision_projector_says_so(ctx, screen):
    err = urllib.error.HTTPError("u", 500, "err", {}, io.BytesIO(
        b'{"error":{"message":"image input is not supported - hint: you may need to provide the mmproj"}}'))
    screen.replies = [err]
    out = call(ctx, "look_at_screen", question="q")
    assert not out["ok"] and "--mmproj" in out["result"] and "didn't work" in out["result"]


def test_server_down_and_screenshot_failure(ctx, screen, monkeypatch):
    screen.replies = [urllib.error.URLError("connection refused")]
    out = call(ctx, "look_at_screen", question="q")
    assert not out["ok"] and "didn't answer" in out["result"]

    def broken(*a):
        raise OSError("screen grab failed")
    monkeypatch.setattr(vision, "screenshot", broken)
    out = call(ctx, "look_at_screen", question="q")
    assert not out["ok"] and "screenshot" in out["result"]


def test_find_on_screen_pads_to_square_and_maps_the_box(ctx, screen):
    # The measured Gemma reply for the save button at (640,640)-(840,710) on a padded 1920 square.
    screen.replies = ['```json\n{"found":true,"box_2d":[335,335,370,439],"label":"Gem"}\n```']
    out = call(ctx, "find_on_screen", what="knappen til at gemme")
    img = screen.image()
    assert img.size == (vision.DETECT_SIDE, vision.DETECT_SIDE) and len(screen.bodies) == 1  # big: one look
    assert out["found"] and out["box"] == {"x": 643, "y": 643, "w": 200, "h": 67}
    assert "'Gem'" in out["result"] and "bottom" not in out["result"]
    assert "knappen til at gemme" in screen.prompt()


def test_small_targets_get_a_second_closer_look(ctx, screen):
    # First look: a 38 px icon near the top right; second look at a 640 px square around it.
    screen.replies = ['{"found":true,"box_2d":[50,950,70,970],"label":"gear"}',
                      '{"found":true,"box_2d":[100,850,160,910],"label":"Settings"}']
    out = call(ctx, "find_on_screen", what="indstillingerne")
    assert len(screen.bodies) == 2
    assert screen.image(1).size == (768, 768)
    assert out["box"] == {"x": 1824, "y": 64, "w": 38, "h": 38}
    assert "'Settings'" in out["result"] and "top right" in out["result"]


def test_second_look_missing_keeps_the_first(ctx, screen):
    screen.replies = ['{"found":true,"box_2d":[50,950,70,970]}', '{"found":false}']
    out = call(ctx, "find_on_screen", what="the gear icon")
    assert out["box"] == {"x": 1824, "y": 96, "w": 38, "h": 38}


def test_find_on_screen_adds_the_monitor_origin(ctx, screen):
    screen.origin = (-2560, -156)
    screen.replies = ['{"found":true,"box_2d":[0,0,100,100]}']
    out = call(ctx, "find_on_screen", what="the start button")
    assert out["box"] == {"x": -2560, "y": -156, "w": 192, "h": 192}
    assert "monitor 2 of 3" in out["result"] and "top left" in out["result"]


@pytest.mark.parametrize("reply,box", [
    ('{"found":true,"box_2d":[370,439,335,335]}', [335, 335, 370, 439]),  # reversed corners
    ('Here: {"bbox":[100,200,150,260]}', [100, 200, 150, 260]),
    ('[{"box_2d":[10,20,30,40],"label":"a"},{"box_2d":[1,2,3,4]}]', [10, 20, 30, 40]),
    ('{"found":true,"box_2d":[-5,0,1200,50]}', [0, 0, 1000, 50]),
    ('{"found":false,"note":"only a browser"}', None),
    ("I can't see it.", None),
    ('{"box_2d":[1,2,3]}', None),
])
def test_parse_box_variants(reply, box):
    assert vision.parse_box(reply)[0] == box


def test_box_in_the_padding_is_not_found(ctx, screen):
    screen.replies = ['{"found":true,"box_2d":[800,100,900,200]}']  # below a 1080-high image
    out = call(ctx, "find_on_screen", what="indstillinger")
    assert out["ok"] and not out["found"] and "Couldn't see" in out["result"]


def test_not_found_passes_the_note(ctx, screen):
    screen.replies = ['{"found":false,"note":"a video player is in front"}']
    out = call(ctx, "find_on_screen", what="the settings")
    assert not out["found"] and "video player" in out["result"]


def test_point_at_finds_then_draws(ctx, screen, drawn):
    screen.replies = ['{"found":true,"box_2d":[335,335,370,439],"label":"Gem"}']
    out = call(ctx, "point_at", what="vis mig hvor man gemmer")
    assert out["risk"] == "visible" and out["shown"]
    assert drawn == [{"box": {"x": 643, "y": 643, "w": 200, "h": 67, "ms": 6000, "label": "Gem"}}]
    assert "orange box" in out["result"]


def test_point_at_with_a_box_needs_no_vision(ctx, screen, drawn):
    out = call(ctx, "point_at", x=10, y=20, w=30, h=40, label="Indstillinger", seconds=3)
    assert screen.bodies == []
    assert drawn == [{"box": {"x": 10, "y": 20, "w": 30, "h": 40, "ms": 3000, "label": "Indstillinger"}}]
    assert not call(ctx, "point_at", x=10, y=20, w=0, h=40)["ok"]
    assert not call(ctx, "point_at")["ok"]


def test_point_at_without_overlay_and_in_dry_run(ctx, dry_ctx, screen, drawn):
    drawn.running = False
    out = call(ctx, "point_at", x=1, y=2, w=3, h=4)
    assert out["ok"] and not out["shown"] and "isn't running" in out["result"]
    drawn.running = True
    out = call(dry_ctx, "point_at", x=1, y=2, w=3, h=4, label="Gem")
    assert out["dry_run"] and drawn == [] and "would draw" in out["result"]


def test_point_at_not_found_draws_nothing(ctx, screen, drawn):
    screen.replies = ['{"found":false}']
    out = call(ctx, "point_at", what="the export button")
    assert out["found"] is False and drawn == [] and "nothing was shown" in out["result"]


def test_read_image(ctx, screen, tmp_path):
    photo = tmp_path / "kvittering.jpg"
    Image.new("RGB", (800, 600), (200, 10, 10)).save(photo)
    screen.replies = ["A red square."]
    out = call(ctx, "read_image", path=str(photo), question="hvad viser billedet")
    assert out["ok"] and out["result"] == "kvittering.jpg: A red square."
    assert screen.image().size == (800, 600) and "hvad viser billedet" in screen.prompt()


def test_read_image_failures(ctx, screen, tmp_path):
    assert "no file" in call(ctx, "read_image", path=str(tmp_path / "nope.png"))["result"]
    (tmp_path / "notes.txt").write_text("hi")
    assert "isn't an image" in call(ctx, "read_image", path=str(tmp_path / "notes.txt"))["result"]
    (tmp_path / "broken.png").write_bytes(b"not a png")
    out = call(ctx, "read_image", path=str(tmp_path / "broken.png"))
    assert not out["ok"] and "couldn't be opened" in out["result"]
    assert screen.bodies == []


def test_ocr_reads_the_active_window_by_default_and_cuts_long_text(ctx, screen, tmp_path):
    screen.replies = ["Mælk, æg, rugbrød og kaffe\nHusk tandlægen torsdag kl. 10"]
    out = call(ctx, "ocr_screen")
    assert screen.shots == [("window", None)]
    assert "Mælk, æg, rugbrød" in out["result"] and "Transcribe" in screen.prompt()
    screen.replies = ["ord " * 1000]
    out = call(ctx, "ocr_screen", target="screen")
    assert len(out["result"]) <= vision.MAX_RESULT + 40 and out["result"].endswith("(cut)")
    page = tmp_path / "scan.png"
    Image.new("RGB", (100, 50), (255, 255, 255)).save(page)
    screen.replies = [""]
    assert "no readable text" in call(ctx, "ocr_screen", path=str(page))["result"]


def test_encode_square_reports_the_side():
    url, side = vision.encode(Image.new("RGB", (300, 100)), square=True)
    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert side == 300 and img.size == (768, 768)


def test_overlay_helpers_do_nothing_without_an_overlay(monkeypatch):
    monkeypatch.setattr(vision_overlay, "PORT", 1)  # nothing listens on port 1
    assert vision_overlay.running() is False
    assert vision_overlay.glow(True) is False
    assert vision_overlay.box(1, 2, 3, 4, label="x") is False


def test_overlay_helpers_speak_the_protocol(monkeypatch):
    import socket
    import threading
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    got = []

    def serve():
        for _ in range(3):
            c, _ = srv.accept()
            with c:
                got.append(json.loads(c.makefile("r").readline()))
                c.sendall(b'{"ok":true}\n')
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    monkeypatch.setattr(vision_overlay, "PORT", srv.getsockname()[1])
    assert vision_overlay.glow(True)
    assert vision_overlay.box(1.6, 2, 0, 4, label="Gem", ms=1500)
    assert vision_overlay.clear()
    t.join(2)
    srv.close()
    assert got == [{"glow": True}, {"box": {"x": 1, "y": 2, "w": 1, "h": 4, "ms": 1500, "label": "Gem"}},
                   {"clear": True}]
