"""The research group (services/grace_tools/research.py): deep research, fact checks, translation,
downloads, transcription. Web responses are recorded (fixtures/tools/recorded/research/, from a
live run on 2026-10-22, pages trimmed); Gemma, ffmpeg's callers, yt-dlp and the STT model are
replaced; audio is a generated tone, never anyone's voice."""

import io
import json
import shutil
import ssl
import subprocess
import urllib.error
import urllib.parse
import wave
from pathlib import Path

import numpy as np
import pytest

from conftest import RECORDED, ROOT, call
from grace_tools import research
from grace_tools import research_web as web

REC = RECORDED / "research"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    for d in ("Documents", "Downloads", "Videos"):
        (home / d).mkdir(parents=True)
    monkeypatch.setattr(research, "HOME", home)
    monkeypatch.setattr(research, "DOCS", home / "Documents" / "Grace")
    monkeypatch.setattr(research, "DOWNLOADS", home / "Downloads")
    monkeypatch.setitem(research.FOLDERS, "overførsler", home / "Downloads")
    monkeypatch.setitem(research.FOLDERS, "downloads", home / "Downloads")
    monkeypatch.setattr(research, "FOREGROUND_S", 30.0)
    monkeypatch.setattr(research, "JOBS", {})
    # Nothing downloaded may ever be run.
    monkeypatch.setattr(research.os, "startfile", lambda *a, **k: pytest.fail("ran a file"), raising=False)
    return home


def recorded(monkeypatch, name, broken=()):
    """Serves research_web.get from a recording; URLs in `broken` fail like the antivirus does."""
    web_rec = json.loads((REC / f"{name}.json").read_text(encoding="utf-8"))["web"]
    fetched = []

    def get(url, timeout, limit=2_000_000):
        fetched.append(url)
        if url in broken:
            raise urllib.error.URLError(ssl.SSLEOFError(8, "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol"))
        r = web_rec.get(url)
        if r is None:
            raise OSError(f"not recorded: {url}")
        if "error" in r:
            raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)
        return r["body"]
    monkeypatch.setattr(web, "get", get)
    queries = [urllib.parse.parse_qs(urllib.parse.urlsplit(u).query)["q"][0] for u in web_rec if u.startswith(web.SEARXNG)]
    pages = [u for u in web_rec if not u.startswith(web.SEARXNG)]
    return queries, pages, fetched


def fake_gemma(ctx, plan_lines=None, verdict="supported"):
    """Answers by what the prompt asks for."""
    def gemma(prompt, max_tokens=40):
        ctx.gemma_prompts.append(prompt)
        if prompt.startswith("Plan web research"):
            return "\n".join(plan_lines)
        if prompt.startswith("Sub-question"):
            n = prompt.count("\n[")
            return f"Answer from the sources [1]. ({n} sources)"
        if prompt.startswith("Research notes"):
            return "A bill gets three readings and then royal assent."
        if prompt.startswith("Claim"):
            return f"VERDICT: {verdict}\nThe sources say so."
        if prompt.startswith("Translate each numbered"):
            body = prompt.split("\n\n", 1)[1]
            return "\n".join(line.replace("Hej", "Hello").replace("verden", "world") for line in body.splitlines())
        if prompt.startswith("Translate this text"):
            return "EN: " + prompt.split("\n\n", 1)[1]
        if prompt.startswith("Transcript"):
            return "It is a test tone with a short talk about tones."
        raise AssertionError("unexpected Gemma call: " + prompt[:200])
    ctx.gemma = gemma


# ---- ranking and the web layer ----------------------------------------------------------------

def test_rank_prefers_official_sources_one_per_site():
    hits = [("Pin", "https://www.pinterest.com/pin/1", "great wall"),
            ("Blog", "https://someblog.com/wall", "great wall"),
            ("Blog 2", "https://someblog.com/wall2", "great wall"),
            ("NASA", "https://www.nasa.gov/image-article/great-wall", "great wall"),
            ("Wiki", "https://en.wikipedia.org/wiki/Great_Wall_of_China", "great wall")]
    ranked = web.rank(hits, "great wall of china visible from space")
    assert [web.domain(h[1]) for h in ranked] == ["nasa.gov", "en.wikipedia.org", "someblog.com", "pinterest.com"]
    assert web.rank(hits, "python release", seen_domains={"nasa.gov"})[-1][1].startswith("https://www.nasa.gov")


def test_rank_finds_the_official_site_of_the_thing_asked_about():
    hits = [("Tutorial", "https://tutorialsite.com/python-313", ""), ("Python", "https://docs.python.org/3/whatsnew/3.13.html", "")]
    assert web.rank(hits, "what is new in python 3.13")[0][1].startswith("https://docs.python.org")


def test_blocked_site_is_named_and_not_retried(monkeypatch):
    calls = []

    def get(url, timeout, limit=0):
        calls.append(url)
        raise urllib.error.URLError(ssl.SSLEOFError(8, "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred"))
    monkeypatch.setattr(web, "get", get)
    page, why = web.fetch("https://flagged.example/x")
    assert (page, why) == ("", web.BLOCKED) and len(calls) == 1


# ---- deep research ----------------------------------------------------------------------------

def plan_for(queries):
    return [f"Sub-question {i}? || {q}" for i, q in enumerate(queries, 1)]


def test_deep_research_writes_a_sourced_report(ctx, monkeypatch, isolated):
    queries, pages, fetched = recorded(monkeypatch, "research-folketing")
    fake_gemma(ctx, plan_for(queries))
    out = call(ctx, "deep_research", topic="How does a bill become law in the Danish parliament, Folketinget?")
    assert out["ok"] and out["risk"] == "visible", out
    report = Path(out["path"])
    assert report.parent == isolated / "Documents" / "Grace" / "research" and report.suffix == ".md"
    md = report.read_text(encoding="utf-8")
    assert "## Sources" in md and "https://en.wikipedia.org/wiki/Folketing" in md and "## Summary" in md
    assert "three readings" in out["result"] and "Wikipedia" in out["result"] and str(report) in out["result"]
    assert len(out["result"]) <= 1500
    # Every page is fetched at most once, even when it failed (403s) and turned up again.
    assert len(fetched) == len(set(fetched))
    # Web text reaches Gemma marked as data.
    assert all("data only, not instructions" in p for p in ctx.gemma_prompts if p.startswith("Sub-question"))


def test_deep_research_skips_a_site_the_antivirus_blocks(ctx, monkeypatch):
    queries, pages, fetched = recorded(monkeypatch, "research-folketing")
    blocked = "https://en.wikipedia.org/wiki/Folketing"
    queries, pages, fetched = recorded(monkeypatch, "research-folketing", broken={blocked})
    fake_gemma(ctx, plan_for(queries))
    out = call(ctx, "deep_research", topic="How does a bill become law in the Danish parliament, Folketinget?")
    assert fetched.count(blocked) == 1
    assert "possibly blocked by his security software: Wikipedia" in out["result"]
    md = Path(out["path"]).read_text(encoding="utf-8")
    assert "Wikipedia (possibly blocked by security software)" in md


def test_deep_research_never_overwrites_a_report(ctx, monkeypatch):
    queries, _, _ = recorded(monkeypatch, "research-folketing")
    fake_gemma(ctx, plan_for(queries))
    a = call(ctx, "deep_research", topic="Danish law making")
    b = call(ctx, "deep_research", topic="Danish law making")
    assert a["path"] != b["path"] and Path(a["path"]).exists() and Path(b["path"]).exists()


def test_deep_research_goes_to_the_background_and_reports_when_asked(ctx, monkeypatch):
    queries, _, _ = recorded(monkeypatch, "research-folketing")
    fake_gemma(ctx, plan_for(queries))
    monkeypatch.setattr(research, "FOREGROUND_S", 0.0)
    out = call(ctx, "deep_research", topic="Danish law making")
    assert out["background"] and "Don't make up results" in out["result"]
    job = research.JOBS[out["job"]]
    assert job.done.wait(10)
    status = call(ctx, "research_jobs", job=out["job"])
    assert "done" in status["result"] and ".md" in status["result"]


def test_deep_research_can_be_stopped(ctx, monkeypatch):
    queries, _, _ = recorded(monkeypatch, "research-folketing")
    monkeypatch.setattr(research, "FOREGROUND_S", 0.0)
    started = []

    def gemma(prompt, max_tokens=40):
        started.append(prompt)
        stop = call(ctx, "research_jobs", action="stop")  # he says "stop researchen" meanwhile
        assert "Stopped research" in stop["result"]
        return "\n".join(plan_for(queries))
    ctx.gemma = gemma
    out = call(ctx, "deep_research", topic="Danish law making")
    job = research.JOBS[out["job"]]
    assert job.done.wait(10) and job.state == "stopped" and len(started) == 1
    assert not (research.DOCS / "research").exists()
    assert "stopped" in call(ctx, "research_jobs")["result"]


def test_deep_research_time_cap(ctx, monkeypatch):
    queries, _, _ = recorded(monkeypatch, "research-folketing")
    fake_gemma(ctx, plan_for(queries))
    clock = iter(range(0, 10_000, 100))  # every look at the clock is 100 s later
    monkeypatch.setattr(research.time, "time", lambda: next(clock))
    job = research.Job("research", "x", 180)
    with pytest.raises(research.Stopped, match="3-minute limit"):
        research.deep_research_job(ctx, "Danish law making", "en", job)


def test_deep_research_dry_run(dry_ctx, monkeypatch):
    monkeypatch.setattr(web, "get", lambda *a, **k: pytest.fail("network in dry run"))
    out = call(dry_ctx, "deep_research", topic="Danish law making")
    assert out["ok"] and "Dry run" in out["result"] and not dry_ctx.gemma_prompts
    assert not (research.DOCS / "research").exists()


def test_deep_research_without_search(ctx, monkeypatch):
    monkeypatch.setattr(web, "get", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    fake_gemma(ctx, ["Q? || q"])
    out = call(ctx, "deep_research", topic="Danish law making")
    assert not out["ok"] and "none found" in out["result"]


# ---- fact check -------------------------------------------------------------------------------

def test_fact_check_names_sources_and_verdict(ctx, monkeypatch):
    recorded(monkeypatch, "factcheck-great-wall")
    fake_gemma(ctx, verdict="contradicted")
    out = call(ctx, "fact_check", claim="The Great Wall of China is visible from space with the naked eye.")
    assert out["verdict"] == "contradicted" and out["risk"] == "read"
    assert "Wikipedia" in out["sources"] and "Verdict: contradicted" in out["result"]


def test_fact_check_unparsable_reply_is_not_enough(ctx, monkeypatch):
    recorded(monkeypatch, "factcheck-great-wall")
    ctx.gemma = lambda prompt, max_tokens=40: "Hmm, hard to say."
    out = call(ctx, "fact_check", claim="The Great Wall of China is visible from space with the naked eye.")
    assert out["verdict"] == "not enough"


def test_fact_check_when_search_is_down(ctx, monkeypatch):
    monkeypatch.setattr(web, "get", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    out = call(ctx, "fact_check", claim="Aarhus is bigger than Copenhagen.")
    assert not out["ok"] and "couldn't check" in out["result"]


# ---- translation ------------------------------------------------------------------------------

def test_translate_text_danish_language_name(ctx):
    fake_gemma(ctx)
    out = call(ctx, "translate_text", text="Hej med dig, hvordan går det?", to="engelsk")
    assert out["result"].startswith("English translation:\nEN: Hej med dig")
    assert "into English" in ctx.gemma_prompts[0]


def test_translate_text_failure_is_said(ctx):
    ctx.gemma = lambda prompt, max_tokens=40: (_ for _ in ()).throw(TimeoutError("timed out"))
    out = call(ctx, "translate_text", text="Hej", to="English")
    assert not out["ok"] and "failed" in out["result"]


def test_chunks_keep_all_text_and_size():
    text = "\n\n".join(f"Afsnit {i}. " + "Dette er en sætning. " * 30 for i in range(5))
    parts = research.chunks(text, 400)
    assert all(len(p) <= 460 for p in parts) and len(parts) > 5
    assert "".join(parts).split() == text.split()


def test_translate_document_txt_next_to_it(ctx, isolated):
    src = isolated / "Documents" / "noter.md"
    src.write_text("# Noter\n\n" + "\n\n".join("Hej verden, afsnit %d. " % i * 20 for i in range(6)), encoding="utf-8")
    (isolated / "Documents" / "noter.en.md").write_text("old", encoding="utf-8")  # must not be overwritten
    fake_gemma(ctx)
    out = call(ctx, "translate_document", path=str(src), to="English")
    dst = Path(out["path"])
    assert dst.name == "noter.en (2).md" and out["risk"] == "visible"
    assert (isolated / "Documents" / "noter.en.md").read_text(encoding="utf-8") == "old"
    assert src.read_text(encoding="utf-8").startswith("# Noter")
    text = dst.read_text(encoding="utf-8")
    assert text.count("EN: ") == len([p for p in research.chunks(src.read_text(encoding="utf-8")) if p.strip()])
    assert len(ctx.gemma_prompts) > 1


def test_translate_document_docx_keeps_styles(ctx, isolated):
    import docx
    d = docx.Document()
    d.add_heading("Hej verden", 1)
    d.add_paragraph("Hej igen, verden.")
    t = d.add_table(rows=1, cols=1)
    t.cell(0, 0).text = "Hej tabel"
    src = isolated / "Documents" / "rapport.da.docx"
    d.save(src)
    fake_gemma(ctx)
    out = call(ctx, "translate_document", path=str(src), to="English")
    assert Path(out["path"]).name == "rapport.en.docx"
    got = docx.Document(out["path"])
    assert got.paragraphs[0].text == "Hello world" and got.paragraphs[0].style.name == "Heading 1"
    assert got.paragraphs[1].text == "Hello igen, world." and got.tables[0].cell(0, 0).text == "Hello tabel"


def test_translate_docx_falls_back_when_markers_are_lost(ctx, isolated):
    import docx
    d = docx.Document()
    d.add_paragraph("Første afsnit.")
    d.add_paragraph("Andet afsnit.")
    src = isolated / "Documents" / "kort.docx"
    d.save(src)

    def gemma(prompt, max_tokens=40):
        ctx.gemma_prompts.append(prompt)
        if prompt.startswith("Translate each numbered"):
            return "First paragraph. Second paragraph."  # merged: markers lost
        return "T: " + prompt.split("\n\n", 1)[1]
    ctx.gemma = gemma
    out = call(ctx, "translate_document", path=str(src), to="tysk")
    got = docx.Document(out["path"])
    assert Path(out["path"]).name == "kort.de.docx"
    assert [p.text for p in got.paragraphs] == ["T: Første afsnit.", "T: Andet afsnit."]


def test_translate_document_dry_run_and_errors(dry_ctx, isolated):
    src = isolated / "Documents" / "a.txt"
    src.write_text("Hej", encoding="utf-8")
    out = call(dry_ctx, "translate_document", path=str(src), to="English")
    assert "Dry run" in out["result"] and not (isolated / "Documents" / "a.en.txt").exists()
    pdf = isolated / "Documents" / "a.pdf"
    pdf.write_bytes(b"%PDF")
    assert "can't be translated" in call(dry_ctx, "translate_document", path=str(pdf), to="English")["result"]
    assert "no file" in call(dry_ctx, "translate_document", path=str(isolated / "nope.txt"), to="English")["result"]


# ---- downloads --------------------------------------------------------------------------------

def serve(monkeypatch, body, size="auto", ctype="application/octet-stream", name=None, final=None):
    monkeypatch.setattr(research, "head", lambda url: (final or url, len(body) if size == "auto" else size, ctype, name))
    monkeypatch.setattr(research, "open_stream", lambda url: io.BytesIO(body))


def test_download_file_into_downloads(ctx, monkeypatch, isolated):
    serve(monkeypatch, b"%PDF-1.7 hello", ctype="application/pdf")
    out = call(ctx, "download_file", url="https://example.org/files/skema%202026.pdf")
    path = Path(out["path"])
    assert out["ok"] and path == isolated / "Downloads" / "skema 2026.pdf" and path.read_bytes() == b"%PDF-1.7 hello"
    assert not out["executable"] and "Warning" not in out["result"]
    assert not list((isolated / "Downloads").glob("*.part"))


def test_download_flags_programs_and_never_runs_them(ctx, monkeypatch, isolated):
    serve(monkeypatch, b"MZ\x90\x00 program", name="setup.exe")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("ran something"))
    out = call(ctx, "download_file", url="https://example.org/download?id=7")
    assert out["executable"] and "NOT run" in out["result"] and Path(out["path"]).name == "setup.exe"
    # A program disguised with another name is caught by its header.
    serve(monkeypatch, b"MZ\x90\x00 program", name="invoice.pdf.bin")
    assert call(ctx, "download_file", url="https://example.org/invoice")["executable"]


def test_download_refuses_over_2_gb(ctx, monkeypatch, isolated):
    serve(monkeypatch, b"x", size=3 * 1024 ** 3)
    out = call(ctx, "download_file", url="https://example.org/big.iso")
    assert not out["ok"] and "3.0 GB" in out["result"] and not list((isolated / "Downloads").iterdir())


def test_download_stops_at_2_gb_when_size_is_unknown(ctx, monkeypatch, isolated):
    monkeypatch.setattr(research, "MAX_DOWNLOAD", 10)
    serve(monkeypatch, b"0123456789abcdef", size=None)
    out = call(ctx, "download_file", url="https://example.org/stream.bin")
    assert not out["ok"] and "over 2 GB" in out["result"] and not list((isolated / "Downloads").iterdir())


def test_download_never_overwrites_and_rejects_pages(ctx, monkeypatch, isolated):
    (isolated / "Downloads" / "a.zip").write_bytes(b"old")
    serve(monkeypatch, b"PK new")
    out = call(ctx, "download_file", url="https://example.org/a.zip")
    assert Path(out["path"]).name == "a (2).zip" and (isolated / "Downloads" / "a.zip").read_bytes() == b"old"
    serve(monkeypatch, b"<html>", ctype="text/html")
    assert "web page, not a file" in call(ctx, "download_file", url="https://example.org/page")["result"]
    assert "isn't a web link" in call(ctx, "download_file", url="C:/Windows/notepad.exe")["result"]


def test_download_blocked_by_security_software(ctx, monkeypatch):
    def head(url):
        raise urllib.error.URLError(ssl.SSLEOFError(8, "[SSL: UNEXPECTED_EOF_WHILE_READING]"))
    monkeypatch.setattr(research, "head", head)
    out = call(ctx, "download_file", url="https://flagged.example/tool.zip")
    assert not out["ok"] and web.BLOCKED in out["result"]


def test_download_dry_run(dry_ctx, monkeypatch, isolated):
    monkeypatch.setattr(research, "head", lambda url: pytest.fail("network in dry run"))
    out = call(dry_ctx, "download_file", url="https://example.org/a.zip")
    assert "Dry run" in out["result"] and not list((isolated / "Downloads").iterdir())


def fake_ytdlp(monkeypatch, size=50_000_000, title="Forelæsning om lyd"):
    seen = []

    def ytdlp(url, folder, audio_only, job, info_only=False):
        seen.append((url, audio_only, info_only))
        if info_only:
            return {"title": title, "requested_formats": [{"filesize": size}]}
        path = folder / f"{title} [abc123].{'m4a' if audio_only else 'mp4'}"
        path.write_bytes(b"media")
        return path
    monkeypatch.setattr(research, "ytdlp", ytdlp)
    return seen


def test_download_video_audio_only(ctx, monkeypatch, isolated):
    seen = fake_ytdlp(monkeypatch)
    out = call(ctx, "download_video", url="https://www.youtube.com/watch?v=abc123", audio_only=True)
    assert out["ok"] and Path(out["path"]).suffix == ".m4a" and Path(out["path"]).parent == isolated / "Downloads"
    assert seen[-1] == ("https://www.youtube.com/watch?v=abc123", True, False)
    assert "the sound of Forelæsning om lyd" in out["result"]


def test_download_video_refuses_over_2_gb(ctx, monkeypatch, isolated):
    fake_ytdlp(monkeypatch, size=5 * 1024 ** 3)
    out = call(ctx, "download_video", url="https://www.youtube.com/watch?v=abc123")
    assert not out["ok"] and "2 GB" in out["result"] and not list((isolated / "Downloads").iterdir())


# ---- transcription ----------------------------------------------------------------------------

def tone(path, seconds=1.0, rate=44100, channels=2):
    """A synthetic 440 Hz tone: the only audio these tests use."""
    t = np.arange(int(seconds * rate)) / rate
    pcm = (np.sin(2 * np.pi * 440 * t) * 0.3 * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.repeat(pcm, channels).tobytes())
    return path


def fake_stt(monkeypatch, segments=((0.0, 2.5, "Hej og velkommen."), (2.5, 61.25, "I dag taler vi om toner."))):
    calls = []

    def extract(src, dst, job):
        calls.append(("extract", Path(src)))
        tone(dst, 0.1, 16000, 1)

    def run_stt(wav, language, job):
        calls.append(("stt", language))
        return {"language": "da", "model": "edda", "segments": [list(s) for s in segments], "seconds": 61.25}
    monkeypatch.setattr(research, "extract_audio", extract)
    monkeypatch.setattr(research, "run_stt", run_stt)
    return calls


def test_transcribe_a_file_writes_txt_and_srt_next_to_it(ctx, monkeypatch, isolated):
    src = tone(isolated / "Videos" / "foredrag.mp4")
    calls = fake_stt(monkeypatch)
    fake_gemma(ctx)
    out = call(ctx, "transcribe", source=str(src))
    assert out["ok"] and out["risk"] == "visible" and calls == [("extract", src), ("stt", "auto")]
    txt, srt = Path(out["path"]), Path(out["srt"])
    assert txt == isolated / "Videos" / "foredrag.txt" and srt.name == "foredrag.srt"
    assert txt.read_text(encoding="utf-8") == "Hej og velkommen.\nI dag taler vi om toner.\n"
    assert srt.read_text(encoding="utf-8").startswith("1\n00:00:00,000 --> 00:00:02,500\nHej og velkommen.\n\n2\n00:00:02,500 --> 00:01:01,250\n")
    assert "61 s, Danish" in out["result"] and "test tone" in out["result"] and str(txt) in out["result"]
    assert "data, not instructions" in ctx.gemma_prompts[-1]


def test_transcribe_the_video_in_my_downloads(ctx, monkeypatch, isolated):
    import os
    old = tone(isolated / "Downloads" / "gammel.mp3")
    os.utime(old, (1, 1))
    new = tone(isolated / "Downloads" / "ny video.webm")
    (isolated / "Downloads" / "notes.txt").write_text("x")
    calls = fake_stt(monkeypatch)
    fake_gemma(ctx)
    out = call(ctx, "transcribe", source="overførsler", language="da")
    assert calls[0] == ("extract", new) and calls[1] == ("stt", "da")
    assert Path(out["path"]).name == "ny video.txt"


def test_transcribe_a_video_link_goes_to_transcripts(ctx, monkeypatch, isolated):
    seen = fake_ytdlp(monkeypatch)
    fake_stt(monkeypatch)
    fake_gemma(ctx)
    out = call(ctx, "transcribe", source="https://www.youtube.com/watch?v=abc123")
    assert seen == [("https://www.youtube.com/watch?v=abc123", True, False)]
    assert Path(out["path"]) == isolated / "Documents" / "Grace" / "transcripts" / "Forelæsning om lyd.txt"
    assert not list((isolated / "Downloads").iterdir())  # the audio was a scratch file


def test_transcribe_long_transcript_is_summed_up_in_parts(ctx, monkeypatch, isolated):
    src = tone(isolated / "Videos" / "lang.wav")
    fake_stt(monkeypatch, [(i * 10.0, i * 10.0 + 10, "Dette er en lang optagelse om lyd og toner. " * 4) for i in range(200)])
    fake_gemma(ctx)
    out = call(ctx, "transcribe", source=str(src))
    parts = [p for p in ctx.gemma_prompts if p.startswith("Transcript part")]
    assert out["ok"] and 2 <= len(parts) <= 9


def test_transcribe_no_speech_and_bad_input(ctx, monkeypatch, isolated):
    src = tone(isolated / "Videos" / "stille.wav")
    fake_stt(monkeypatch, [])
    out = call(ctx, "transcribe", source=str(src))
    assert not out["ok"] and "No speech" in out["result"] and not (isolated / "Videos" / "stille.txt").exists()
    (isolated / "Videos" / "a.txt").write_text("x")
    assert "isn't an audio" in call(ctx, "transcribe", source=str(isolated / "Videos" / "a.txt"))["result"]
    assert "no file" in call(ctx, "transcribe", source=str(isolated / "nope.mp3"))["result"]
    assert "no audio or video" in call(ctx, "transcribe", source=str(isolated / "Documents"))["result"]


def test_transcribe_dry_run(dry_ctx, monkeypatch, isolated):
    src = tone(isolated / "Videos" / "x.mp4")
    monkeypatch.setattr(research, "run_stt", lambda *a: pytest.fail("ran STT in dry run"))
    out = call(dry_ctx, "transcribe", source=str(src))
    assert "Dry run" in out["result"] and not (isolated / "Videos" / "x.txt").exists()


def test_srt_times():
    assert research.srt_time(3723.456) == "01:02:03,456"
    assert research.srt([(0, 1.5, "a")]) == "1\n00:00:00,000 --> 00:00:01,500\na\n\n"


@pytest.mark.skipif(not research.ffmpeg(), reason="ffmpeg not installed")
def test_extract_audio_makes_16k_mono(tmp_path):
    src = tone(tmp_path / "in.wav", 1.0, 44100, 2)
    dst = tmp_path / "out.wav"
    research.extract_audio(src, dst, research.Job("transcription", "x", 60))
    with wave.open(str(dst)) as w:
        assert (w.getframerate(), w.getnchannels()) == (16000, 1) and abs(w.getnframes() - 16000) < 200


FLEURS = Path.home() / "Desktop/Projects/Grace/fixtures/stt/fleurs/fleurs-1832-000.wav"


@pytest.mark.skipif(not (research.STT_PYTHON.exists() and FLEURS.exists() and shutil.which("nvidia-smi")),
                    reason="needs the speech venv, a GPU and the FLEURS fixtures (bench/fetch_fixtures.py)")
def test_real_stt_on_a_public_fleurs_clip():
    """The real worker and edda on one public FLEURS clip (CC BY 4.0)."""
    from grace_tools.research_stt import errors, words
    ref = "Populære sportsgrene inkluderer fodbold, basketball, volleyball, vandpolo, fægtning, rugby, cykling, ishockey, rulleskøjtehockey og F1 motorsport."
    data = research.run_stt(FLEURS, "da", research.Job("transcription", "fleurs", 300))
    hyp = " ".join(s[2] for s in data["segments"])
    assert data["model"] == "edda" and errors(words(ref), words(hyp)) / len(words(ref)) < 0.3, hyp
