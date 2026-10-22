"""Grace embed server: turns text into vectors for memory search (ADR 0007, docs/MODELS.md).

  bench/.venv-embed/Scripts/python.exe services/embed_server.py

Localhost TCP 127.0.0.1:7812, newline-delimited JSON:
  {"id": 1, "kind": "query", "texts": ["hvad hedder min søster"]}
  {"id": 2, "kind": "document", "texts": ["Mikkel's sister is called Sofie."]}
  -> {"id": 1, "model": "...", "dim": 640, "vectors": [[...]], "ms": 48.1}

Runs on the CPU (4 threads) so it never competes with the LLM for the GPU. Vectors are
L2-normalised; queries get the retrieval instruction, documents don't (as harrier expects).
"""

import json
import socketserver
import sys
import threading
import time

import torch
from sentence_transformers import SentenceTransformer

MODEL = "microsoft/harrier-oss-v1-270m"
PORT = 7812
QUERY_INSTRUCT = "Instruct: Given a spoken question, retrieve the stored fact that answers it\nQuery: "


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            req = json.loads(line)
            started = time.perf_counter()
            try:
                prompt = QUERY_INSTRUCT if req.get("kind") == "query" else None
                with self.server.lock:
                    vecs = self.server.model.encode(req["texts"], prompt=prompt, normalize_embeddings=True)
                reply = {"id": req["id"], "model": MODEL, "dim": int(vecs.shape[1]),
                         "vectors": [[round(float(x), 6) for x in v] for v in vecs],
                         "ms": (time.perf_counter() - started) * 1000}
            except Exception as e:
                reply = {"id": req.get("id"), "error": f"{type(e).__name__}: {e}"}
            self.wfile.write((json.dumps(reply) + "\n").encode())
            self.wfile.flush()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def main():
    started = time.perf_counter()
    torch.set_num_threads(4)
    srv = Server(("127.0.0.1", PORT), Handler)
    srv.model = SentenceTransformer(MODEL, device="cpu")
    srv.model.encode(["warm up"], normalize_embeddings=True)
    srv.lock = threading.Lock()
    log(f"embed server ready in {time.perf_counter() - started:.1f} s ({MODEL} on :{PORT})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
