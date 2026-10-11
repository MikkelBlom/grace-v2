"""Grace speech server: keeps the STT and TTS models warm so the engine can restart in a second.

  bench/.venv/Scripts/python.exe services/speech_server.py

Two localhost TCP ports, newline-delimited JSON, audio as base64 little-endian float32 mono:

  STT 127.0.0.1:7810   {"id": 1, "rate": 16000, "audio": "<b64>"}
                       -> {"id": 1, "text": "...", "ms": 210.5}
  TTS 127.0.0.1:7811   {"id": 2, "text": "Hello there."}
                       -> {"id": 2, "rate": 24000, "pcm": "<b64>"}   (streamed, several)
                       -> {"id": 2, "done": true, "first_ms": 90.1, "total_ms": 400.2}
                       {"cancel": 2}   stops request 2 between chunks

Models: edda-v0.2 (Danish STT, CTranslate2 int8 on CUDA) and Kyutai Pocket TTS (English, CPU,
streaming), per docs/MODELS.md. Each connection is served on its own thread; one STT and one TTS
model are shared behind locks.
"""

import base64
import json
import socketserver
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STT_MODEL = ROOT / "bench" / ".models" / "ct2" / "danish-foundation-models--edda-v0.2-int8_float16"
STT_PORT, TTS_PORT = 7810, 7811


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class Stt:
    def __init__(self):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(str(STT_MODEL), device="cuda", compute_type="int8_float16")
        self.lock = threading.Lock()
        self.transcribe(np.zeros(16000, np.float32))  # warm up CUDA kernels

    def transcribe(self, audio):
        with self.lock:
            segs, _ = self.model.transcribe(
                audio, language="da", beam_size=1, vad_filter=False,
                without_timestamps=True, condition_on_previous_text=False,
            )
            segs = list(segs)
            text = " ".join(s.text.strip() for s in segs).strip()
            # Whisper's own doubt signals, so the engine can drop hallucinations on noise.
            avg_logprob = min((s.avg_logprob for s in segs), default=0.0)
            no_speech = max((s.no_speech_prob for s in segs), default=1.0)
            return text, avg_logprob, no_speech


class Tts:
    def __init__(self, voice):
        from pocket_tts import TTSModel

        self.model = TTSModel.load_model()
        self.state = self.model.get_state_for_audio_prompt(voice)
        self.rate = self.model.sample_rate
        self.lock = threading.Lock()
        for _ in self.stream("Ready.", threading.Event()):  # warm up
            pass

    def stream(self, text, stop):
        with self.lock:
            for chunk in self.model.generate_audio_stream(self.state, text, stop=stop):
                yield chunk.float().numpy().astype(np.float32)


def b64(pcm):
    return base64.b64encode(np.ascontiguousarray(pcm, dtype="<f4").tobytes()).decode()


class Handler(socketserver.StreamRequestHandler):
    def finish(self):
        # The engine closes its connections abruptly when it exits; that is normal.
        try:
            super().finish()
        except OSError:
            pass

    def send(self, obj):
        self.wfile.write((json.dumps(obj) + "\n").encode())
        self.wfile.flush()


class SttHandler(Handler):
    def handle(self):
        for line in self.rfile:
            req = json.loads(line)
            started = time.perf_counter()
            try:
                audio = np.frombuffer(base64.b64decode(req["audio"]), dtype="<f4")
                assert req.get("rate", 16000) == 16000, "STT expects 16 kHz"
                text, avg_logprob, no_speech = self.server.stt.transcribe(audio)
                self.send({"id": req["id"], "text": text, "avg_logprob": avg_logprob,
                           "no_speech_prob": no_speech, "ms": (time.perf_counter() - started) * 1000})
            except Exception as e:  # report, keep serving
                self.send({"id": req.get("id"), "error": f"{type(e).__name__}: {e}"})


class TtsHandler(Handler):
    def handle(self):
        # A reader thread watches for cancels while the main loop streams audio.
        stops, current = {}, {}
        lock = threading.Lock()
        queue, cond = [], threading.Condition()

        def reader():
            for line in self.rfile:
                msg = json.loads(line)
                if "cancel" in msg:
                    with lock:
                        if msg["cancel"] in stops:
                            stops[msg["cancel"]].set()
                else:
                    with lock:
                        stops[msg["id"]] = threading.Event()
                    with cond:
                        queue.append(msg)
                        cond.notify()
            with cond:
                queue.append(None)
                cond.notify()

        threading.Thread(target=reader, daemon=True).start()
        while True:
            with cond:
                while not queue:
                    cond.wait()
                req = queue.pop(0)
            if req is None:
                return
            rid, stop = req["id"], stops[req["id"]]
            started = time.perf_counter()
            first = None
            try:
                if not stop.is_set():
                    for pcm in self.server.tts.stream(req["text"], stop):
                        if stop.is_set():
                            break
                        if first is None:
                            first = (time.perf_counter() - started) * 1000
                        self.send({"id": rid, "rate": self.server.tts.rate, "pcm": b64(pcm)})
                self.send({"id": rid, "done": True, "cancelled": stop.is_set(),
                           "first_ms": first, "total_ms": (time.perf_counter() - started) * 1000})
            except Exception as e:
                self.send({"id": rid, "done": True, "error": f"{type(e).__name__}: {e}"})
            with lock:
                stops.pop(rid, None)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            log(f"client {client_address[1]} disconnected")
            return
        super().handle_error(request, client_address)


def main():
    started = time.perf_counter()
    log("loading STT (edda-v0.2) ...")
    stt = Stt()
    cfg_path = ROOT / "config" / "speech.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    voice = cfg.get("tts_voice", "jane")
    log(f"loading TTS (Pocket TTS, voice {voice}) ...")
    tts = Tts(voice)
    servers = []
    for port, handler, attr, model in [(STT_PORT, SttHandler, "stt", stt), (TTS_PORT, TtsHandler, "tts", tts)]:
        srv = Server(("127.0.0.1", port), handler)
        setattr(srv, attr, model)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
    log(f"speech server ready in {time.perf_counter() - started:.1f} s (STT :{STT_PORT}, TTS :{TTS_PORT})")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
