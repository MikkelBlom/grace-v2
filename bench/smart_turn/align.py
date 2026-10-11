"""Word-level CTC forced alignment of Danish audio to its reference text with Røst wav2vec2 315m v3
(CoRal-project, OpenRAIL). Used only to find word boundaries for cutting; never for synthesis.
"""
import json
import re

import numpy as np
import torch
import torchaudio.functional as taf

MODEL_DIR = "fixtures/corpora/smart-turn/models/roest-wav2vec2-315m-v3"
FRAME_S = 0.02  # wav2vec2 stride 320 samples at 16 kHz
MAP = {"ä": "æ", "ö": "ø", "é": "é", "è": "e", "á": "a", "à": "a", "ü": "ü"}


class Aligner:
    def __init__(self, device="cuda"):
        from transformers import Wav2Vec2ForCTC
        self.vocab = json.load(open(f"{MODEL_DIR}/vocab.json", encoding="utf-8"))
        self.blank = 45  # <pad>
        self.sep = self.vocab["|"]
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to(device).eval()
        self.device = device

    def words(self, text):
        """Reference words: (original word, normalised chars). Words that normalise to nothing are dropped."""
        out = []
        for w in text.split():
            n = "".join(MAP.get(c, c) for c in w.lower())
            n = "".join(c for c in n if c in self.vocab and c != "|")
            if n:
                out.append((re.sub(r"[^\w]", "", w.lower()), n))
        return out

    @torch.no_grad()
    def align(self, audio, text):
        """-> list of (word, start_s, end_s, score) or None if alignment is impossible/poor."""
        ws = self.words(text)
        if not ws:
            return None
        x = torch.from_numpy((audio - audio.mean()) / (audio.std() + 1e-7)).to(self.device, torch.float16)[None]
        logits = self.model(x).logits.float()
        lp = torch.log_softmax(logits, -1)
        tokens = []
        for i, (_, n) in enumerate(ws):
            if i:
                tokens.append(self.sep)
            tokens += [self.vocab[c] for c in n]
        T = lp.shape[1]
        if len(tokens) > T:
            return None
        tgt = torch.tensor([tokens], dtype=torch.int32, device=self.device)
        try:
            ali, scores = taf.forced_align(lp, tgt, blank=self.blank)
        except RuntimeError:
            return None
        ali = ali[0].cpu().numpy(); scores = scores[0].exp().cpu().numpy()
        # token index per frame: walk the path
        spans = []  # (token_pos, start_frame, end_frame)
        pos = -1; prev = self.blank
        for t, tok in enumerate(ali):
            if tok == self.blank:
                prev = tok
                continue
            if tok != prev or not spans:
                pos += 1
                spans.append([pos, t, t + 1, [scores[t]]])
            else:
                spans[-1][2] = t + 1
                spans[-1][3].append(scores[t])
            prev = tok
        if len(spans) != len(tokens):
            return None
        out = []
        k = 0
        for i, (orig, n) in enumerate(ws):
            if i:
                k += 1  # separator token
            first, last = spans[k], spans[k + len(n) - 1]
            sc = float(np.mean([s for sp in spans[k:k + len(n)] for s in sp[3]]))
            out.append((orig, first[1] * FRAME_S, last[2] * FRAME_S, sc))
            k += len(n)
        return out
