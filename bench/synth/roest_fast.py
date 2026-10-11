"""Fast inference for CoRal Røst v3 Chatterbox (350m Turbo / 500m Multilingual).

The stock `chatterbox-tts` decode loop is launch-bound: ~20-25 ms per speech token on the
RTX 5090 Laptop, while one second of audio is 25 tokens. This module keeps the same weights
and sampling rules but runs the T3 backbone with a static KV cache and replays each decode
step (backbone + speech head + sampler) as one CUDA graph. No Triton needed, so it works on
Windows where torch.compile does not.

Speech tokens are turned into audio by S3Gen in chunks so the first audio leaves early
(`stream()`), or all at once (`synthesize()`).

Differences from the stock loop, on purpose:
- The 500m alignment analyser (attention-based early stop / anti-repeat) is not run: it needs
  eager attention on every step. A hard token budget per character replaces it.
- Repetition penalty is applied before top-k/top-p (stock Turbo applies it last).
- The Perth watermark is optional (`watermark=True`); the licence does not require it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPOS = {
    "350m": "CoRal-project/roest-v3-chatterbox-350m",
    "500m": "CoRal-project/roest-v3-chatterbox-500m",
    # Resemble AI's own multilingual Chatterbox (MIT): the 500m's base, Danish among 23 languages.
    "mtl-base": "ResembleAI/chatterbox",
}
START_SPEECH = 6561
STOP_SPEECH = 6562
S3_VOCAB = 6561
TOKENS_PER_SECOND = 25


def model_dir(name: str) -> Path:
    from huggingface_hub import snapshot_download

    if name == "mtl-base":  # the repo holds several generations; fetch only what from_local reads
        pats = ["t3_mtl23ls_v2.safetensors", "s3gen.pt", "ve.pt", "conds.pt", "grapheme_mtl_merged_expanded_v1.json",
                "mtl_tokenizer.json", "tokenizer.json", "Cangjie5_TC.json"]
    else:
        pats = ["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
    return Path(snapshot_download(REPOS[name], allow_patterns=pats))


# --------------------------------------------------------------------------------------------
# Backbones with a static KV cache. forward(x, pos, mask) works for prefill (T>1) and step (T=1).
# --------------------------------------------------------------------------------------------


def attend(q, k, v, mask):
    """SDPA's fused kernels are slow for one query over a long masked cache (~95 us/layer);
    two small matmuls are ~5x faster there."""
    if q.size(2) > 1:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    s = torch.matmul(q, k.transpose(-1, -2)).float() * (q.size(-1) ** -0.5)
    s = s.masked_fill(~mask, float("-inf")).softmax(-1)
    return torch.matmul(s.to(v.dtype), v)


class StaticGpt2:
    def __init__(self, tfmr, batch, max_len, dtype):
        cfg = tfmr.config
        self.h, self.d = cfg.n_head, cfg.n_embd // cfg.n_head
        self.eps = cfg.layer_norm_epsilon
        c = lambda t: t.detach().to(dtype)
        self.wpe = c(tfmr.wpe.weight)
        self.layers = [
            dict(
                ln1=(c(b.ln_1.weight), c(b.ln_1.bias)),
                attn=(c(b.attn.c_attn.weight), c(b.attn.c_attn.bias)),
                aproj=(c(b.attn.c_proj.weight), c(b.attn.c_proj.bias)),
                ln2=(c(b.ln_2.weight), c(b.ln_2.bias)),
                fc=(c(b.mlp.c_fc.weight), c(b.mlp.c_fc.bias)),
                mproj=(c(b.mlp.c_proj.weight), c(b.mlp.c_proj.bias)),
            )
            for b in tfmr.h
        ]
        self.lnf = (c(tfmr.ln_f.weight), c(tfmr.ln_f.bias))
        dev = self.wpe.device
        self.k = torch.zeros(len(self.layers), batch, self.h, max_len, self.d, dtype=dtype, device=dev)
        self.v = torch.zeros_like(self.k)
        self.dtype = dtype

    def forward(self, x, pos, mask, kv_len=None):
        B, T, C = x.shape
        kv_len = kv_len or self.k.size(3)
        x = x.to(self.dtype) + self.wpe[pos].unsqueeze(0)
        for i, L in enumerate(self.layers):
            a = F.layer_norm(x, (C,), *L["ln1"], self.eps)
            qkv = torch.addmm(L["attn"][1], a.view(-1, C), L["attn"][0]).view(B, T, 3, self.h, self.d)
            q, k, v = qkv.unbind(2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            self.k[i].index_copy_(2, pos, k)
            self.v[i].index_copy_(2, pos, v)
            o = attend(q, self.k[i][:, :, :kv_len], self.v[i][:, :, :kv_len], mask)
            o = o.transpose(1, 2).reshape(B * T, C)
            x = x + torch.addmm(L["aproj"][1], o, L["aproj"][0]).view(B, T, C)
            m = F.layer_norm(x, (C,), *L["ln2"], self.eps).view(-1, C)
            m = F.gelu(torch.addmm(L["fc"][1], m, L["fc"][0]), approximate="tanh")
            x = x + torch.addmm(L["mproj"][1], m, L["mproj"][0]).view(B, T, C)
        return F.layer_norm(x, (C,), *self.lnf, self.eps)


class StaticLlama:
    def __init__(self, tfmr, batch, max_len, dtype):
        cfg = tfmr.config
        self.h, self.d = cfg.num_attention_heads, cfg.head_dim
        self.eps = cfg.rms_norm_eps
        c = lambda t: t.detach().to(dtype)
        self.layers = [
            dict(
                n1=c(b.input_layernorm.weight),
                qkv=torch.cat([c(b.self_attn.q_proj.weight), c(b.self_attn.k_proj.weight), c(b.self_attn.v_proj.weight)]),
                o=c(b.self_attn.o_proj.weight),
                n2=c(b.post_attention_layernorm.weight),
                gu=torch.cat([c(b.mlp.gate_proj.weight), c(b.mlp.up_proj.weight)]),
                down=c(b.mlp.down_proj.weight),
            )
            for b in tfmr.layers
        ]
        self.norm = c(tfmr.norm.weight)
        dev = self.norm.device
        with torch.no_grad():
            ids = torch.arange(max_len, device=dev).unsqueeze(0)
            cos, sin = tfmr.rotary_emb(torch.zeros(1, device=dev, dtype=torch.float32), ids)
        self.cos, self.sin = cos[0].to(dtype), sin[0].to(dtype)  # (max_len, D)
        self.k = torch.zeros(len(self.layers), batch, self.h, max_len, self.d, dtype=dtype, device=dev)
        self.v = torch.zeros_like(self.k)
        self.dtype = dtype

    def rms(self, x, w):
        xf = x.float()
        return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(self.dtype)

    @staticmethod
    def rope(x, cos, sin):
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    def forward(self, x, pos, mask, kv_len=None):
        B, T, C = x.shape
        kv_len = kv_len or self.k.size(3)
        x = x.to(self.dtype)
        cos, sin = self.cos[pos].unsqueeze(0).unsqueeze(0), self.sin[pos].unsqueeze(0).unsqueeze(0)
        for i, L in enumerate(self.layers):
            a = self.rms(x, L["n1"])
            q, k, v = F.linear(a, L["qkv"]).view(B, T, 3, self.h, self.d).unbind(2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            q, k = self.rope(q, cos, sin), self.rope(k, cos, sin)
            self.k[i].index_copy_(2, pos, k)
            self.v[i].index_copy_(2, pos, v)
            o = attend(q, self.k[i][:, :, :kv_len], self.v[i][:, :, :kv_len], mask)
            x = x + F.linear(o.transpose(1, 2).reshape(B, T, C), L["o"])
            m = self.rms(x, L["n2"])
            g, u = F.linear(m, L["gu"]).chunk(2, dim=-1)
            x = x + F.linear(F.silu(g) * u, L["down"])
        return self.rms(x, self.norm)


# --------------------------------------------------------------------------------------------
# Sampling (graph-safe: no host sync)
# --------------------------------------------------------------------------------------------


@dataclass
class Sampling:
    temperature: float = 0.8
    top_k: int = 1000
    top_p: float = 0.95
    min_p: float = 0.0
    repetition_penalty: float = 1.2
    cfg_weight: float = 0.5  # 500m only


def sample(logits, seen, s: Sampling):
    """logits (1, V) float32 -> token (1, 1) long. `seen` (V,) bool marks penalised tokens."""
    if s.repetition_penalty != 1.0:
        p = s.repetition_penalty
        logits = torch.where(seen, torch.where(logits > 0, logits / p, logits * p), logits)
    if s.temperature != 1.0:
        logits = logits / s.temperature
    if s.top_k and s.top_k < logits.size(-1):
        vals, idx = logits.topk(s.top_k, dim=-1)
    else:
        vals, idx = logits.sort(dim=-1, descending=True)
    probs = vals.softmax(-1)
    if s.min_p > 0:
        probs = probs.masked_fill(probs < s.min_p * probs[:, :1], 0.0)
    if s.top_p < 1.0:
        cum = probs.cumsum(-1)
        probs = probs.masked_fill(cum - probs > s.top_p, 0.0)
    choice = torch.multinomial(probs, 1)
    return idx.gather(-1, choice)


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------


class Roest:
    """One Røst model with graph-captured decoding. `name` is '350m' or '500m'."""

    def __init__(self, name="350m", dtype=torch.bfloat16, device="cuda", max_len=1600, s3gen="auto",
                 graphs=True, sampling: Sampling | None = None, cfg=True):
        self.name, self.device, self.dtype, self.max_len, self.graphs = name, device, dtype, max_len, graphs
        d = model_dir(name)
        if name == "350m":
            from chatterbox.tts_turbo import ChatterboxTurboTTS as Tts
            self.tts = Tts.from_local(d, device)
            self.batch = 1
            self.sampling = sampling or Sampling(0.8, 1000, 0.95, 0.0, 1.2)
        else:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS as Tts
            self.tts = Tts.from_local(d, device=device)
            self.sampling = sampling or Sampling(0.8, 0, 1.0, 0.05, 2.0, 0.5)
            # classifier-free guidance runs a cond and an uncond row; cfg=False drops the second
            self.batch = 2 if cfg and self.sampling.cfg_weight > 0 else 1
        self.t3 = self.tts.t3
        self.sr = self.tts.sr
        # S3Gen: the 350m ships a 2-step "meanflow" decoder; the 500m ships the 10-step one.
        # Both read the same S3 speech tokens, so either can follow either T3.
        self.s3_steps = None
        if s3gen == "meanflow" and name != "350m":
            from chatterbox.models.s3gen import S3Gen
            from safetensors.torch import load_file

            from huggingface_hub import hf_hub_download

            # Resemble's MIT copy; byte-identical to the one in the Røst 350m repo (same sha256).
            g = S3Gen(meanflow=True)
            g.load_state_dict(load_file(hf_hub_download("ResembleAI/chatterbox-turbo", "s3gen_meanflow.safetensors")),
                              strict=True)
            self.tts.s3gen = g.to(device).eval()
        elif s3gen.isdigit():
            self.s3_steps = int(s3gen)
        self.s3gen = self.tts.s3gen
        Core = StaticGpt2 if self.t3.is_gpt else StaticLlama
        self.core = Core(self.t3.tfmr, self.batch, max_len, dtype)
        c = lambda t: t.detach().to(dtype)
        self.speech_emb = c(self.t3.speech_emb.weight)
        self.speech_pos = c(self.t3.speech_pos_emb.emb.weight) if self.t3.speech_pos_emb is not None else None
        self.head_w = c(self.t3.speech_head.weight)
        self.head_b = c(self.t3.speech_head.bias) if self.t3.speech_head.bias is not None else None
        V = self.head_w.size(0)
        dev = device
        # static step state
        self.cur = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.spos = torch.zeros(1, dtype=torch.long, device=dev)  # speech position (500m learned PE)
        self.n = torch.zeros(1, dtype=torch.long, device=dev)
        self.out = torch.zeros(max_len, dtype=torch.long, device=dev)
        self.seen = torch.zeros(V, dtype=torch.bool, device=dev)
        self.ar = torch.arange(max_len, device=dev)
        self.graph = None
        self.voice = None
        self.set_voice(None)

    # ---------------------------------------------------------------- voices
    def set_voice(self, prompt_wav: str | Path | None):
        """None = the repo's conds.pt (Resemble's default speaker). Otherwise a >5 s clip to clone."""
        if prompt_wav is None:
            from chatterbox.tts_turbo import Conditionals

            self.tts.conds = Conditionals.load(model_dir(self.name) / "conds.pt").to(self.device)
        else:
            if self.name == "350m":
                self.tts.prepare_conditionals(str(prompt_wav))
            else:
                self.tts.prepare_conditionals(str(prompt_wav), exaggeration=0.5)
        self.voice = str(prompt_wav) if prompt_wav else "builtin"

    def voice_conditionals(self):
        return self.tts.conds

    # ---------------------------------------------------------------- T3
    def logits(self, h):
        lg = F.linear(h, self.head_w, self.head_b).float()  # (B, V)
        if self.batch == 2:
            w = self.sampling.cfg_weight
            lg = lg[0:1] + w * (lg[0:1] - lg[1:2])
        return lg

    def step_body(self, L=None):
        L = L or self.max_len
        e = self.speech_emb[self.cur.view(-1)].view(1, 1, -1)
        if self.speech_pos is not None:
            e = e + self.speech_pos[self.spos].view(1, 1, -1)
        e = e.expand(self.batch, 1, -1)
        mask = (self.ar[:L] <= self.pos).view(1, 1, 1, -1)
        h = self.core.forward(e, self.pos, mask, L)[:, -1]
        tok = sample(self.logits(h), self.seen, self.sampling)
        self.cur.copy_(tok)
        self.out.index_copy_(0, self.n, tok.view(1))
        self.seen.index_fill_(0, tok.view(1), True)
        self.pos.add_(1)
        self.spos.add_(1)
        self.n.add_(1)

    def buckets(self):
        """Attention spans with their own graph: a step only reads the KV cache up to its bucket."""
        return sorted({*range(256, self.max_len, 256), self.max_len})

    def capture(self):
        """One graph per attention bucket. Clobbers the KV cache, so the prefix cache is dropped."""
        self.graph = {}
        for L in self.buckets():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.inference_mode():
                self.pos.fill_(L - 8)
                for _ in range(3):
                    self.step_body(L)
            torch.cuda.current_stream().wait_stream(s)
            self.pos.fill_(L - 8)
            self.n.zero_()
            g = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(g):
                self.step_body(L)
            self.graph[L] = g
        # prefill graphs: text + start tokens padded up to a bucket. Padding writes KV entries
        # past the real prompt; decode overwrites each of them before any step can attend to it.
        self.pf = {}
        B, D = self.batch, self.speech_emb.size(1)
        for Tb in PREFILL_BUCKETS:
            st = dict(x=torch.zeros(B, Tb, D, dtype=self.dtype, device=self.device),
                      pos=self.ar[self.max_len - Tb - 8: self.max_len - 8].clone(),
                      idx=torch.zeros(1, dtype=torch.long, device=self.device))

            def body(st=st, Tb=Tb):
                mask = (self.ar.view(1, -1) <= st["pos"].view(-1, 1)).view(1, 1, Tb, -1)
                h = self.core.forward(st["x"], st["pos"], mask)
                st["h"] = h.index_select(1, st["idx"])[:, 0]

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.inference_mode():
                for _ in range(2):
                    body()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(g):
                body()
            st["graph"] = g
            self.pf[Tb] = st
        self.prefix_owner = None

    def text_tokens(self, text):
        from chatterbox.tts_turbo import punc_norm

        text = punc_norm(text)
        if self.name == "350m":
            return self.tts.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        tt = self.tts.tokenizer.text_to_tokens(text, language_id="da").to(self.device)
        if self.batch == 2:
            tt = torch.cat([tt, tt])
        tt = F.pad(tt, (1, 0), value=self.t3.hp.start_text_token)
        return F.pad(tt, (0, 1), value=self.t3.hp.stop_text_token)

    @torch.inference_mode()
    def prefill(self, text):
        """Same input layout as stock T3: [voice conditioning][text][start]. The conditioning part
        is identical for every sentence in a voice, so its KV entries are computed once and kept."""
        t3, cond = self.t3, self.tts.conds.t3
        tt = self.text_tokens(text)
        B = tt.size(0)
        text_emb = t3.text_emb(tt)
        if self.batch == 2:
            text_emb[1].zero_()  # CFG uncond row
        if t3.hp.input_pos_emb == "learned":
            text_emb = text_emb + t3.text_pos_emb(tt)
        start = self.speech_emb[START_SPEECH].view(1, 1, -1).to(text_emb.dtype)
        if self.speech_pos is not None:
            start = start + self.speech_pos[0].to(text_emb.dtype)
        rest = torch.cat([text_emb, start.expand(B, 1, -1)], dim=1)
        if self.name != "350m":  # stock 500m loop appends one more BOS with speech position 0
            rest = torch.cat([rest, start.expand(B, 1, -1)], dim=1)
        if getattr(self, "prefix_owner", None) is not cond:
            cond_emb = t3.prepare_conditioning(cond).expand(B, -1, -1)
            Lc = cond_emb.size(1)
            pos = self.ar[:Lc]
            self.core.forward(cond_emb, pos, (self.ar.view(1, -1) <= pos.view(-1, 1)).view(1, 1, Lc, -1))
            self.prefix_owner, self.prefix_len = cond, Lc
        Lc = self.prefix_len
        T = Lc + rest.size(1)
        assert T + 8 < self.max_len, "prompt too long"
        n_rest = rest.size(1)
        Tb = next((b for b in PREFILL_BUCKETS if b >= n_rest), None)
        if getattr(self, "pf", None) and Tb is not None and Lc + Tb + 8 < self.max_len:
            st = self.pf[Tb]
            st["x"].zero_()
            st["x"][:, :n_rest] = rest
            st["pos"].copy_(self.ar[Lc: Lc + Tb])
            st["idx"].fill_(n_rest - 1)
            st["graph"].replay()
            h = st["h"]
        else:
            pos = self.ar[Lc:T]
            mask = (self.ar.view(1, -1) <= pos.view(-1, 1)).view(1, 1, T - Lc, -1)
            h = self.core.forward(rest, pos, mask)[:, -1]
        self.host_pos = T
        self.seen.zero_()
        if self.name != "350m":
            self.seen[START_SPEECH] = True
        tok = sample(self.logits(h), self.seen, self.sampling)
        self.seen.index_fill_(0, tok.view(1), True)
        self.cur.copy_(tok)
        self.out.zero_()
        self.out[0] = tok.view(())
        self.n.fill_(1)
        self.pos.fill_(T)
        self.spos.fill_(1)
        return len(text)

    def decode(self, max_tokens, burst):
        """Yields the full token list so far (1D long tensor, stop token removed) after each burst."""
        n = 1
        max_tokens = min(max_tokens, self.max_len - self.host_pos - 1)
        buckets = self.buckets()
        while n < max_tokens:
            k = min(burst, max_tokens - n)
            for _ in range(k):
                pos = self.host_pos + n - 1  # position this step writes
                L = next(b for b in buckets if b > pos)
                if self.graph is not None:
                    self.graph[L].replay()
                else:
                    with torch.inference_mode():
                        self.step_body(L)
                n += 1
            toks = self.out[:n]
            stop = (toks == STOP_SPEECH).nonzero()
            if stop.numel():
                yield toks[: int(stop[0, 0])], True
                return
            yield toks, False
        yield self.out[:n], True

    # ---------------------------------------------------------------- S3Gen
    @torch.inference_mode()
    def tokens_to_wav(self, toks, finalize, noise=None):
        # The flow's own finalize=False path is broken upstream (mask length mismatch), so the
        # flow always runs as if final; stream() holds back the last tokens' audio instead.
        toks = toks[toks < S3_VOCAB]
        if finalize:
            sil = torch.full((3,), 4299, dtype=torch.long, device=self.device)  # S3GEN_SIL
            toks = torch.cat([toks, sil]) if self.name == "350m" else toks
        g = self.s3gen
        steps = self.s3_steps or (2 if g.meanflow else 10)
        n_noise = None
        if g.meanflow:
            n_mel = 2 * toks.numel()
            n_noise = noise[:, :, :n_mel] if noise is not None else torch.randn(1, 80, n_mel, device=self.device, dtype=g.dtype)
        mels = super(type(g), g).forward(toks, ref_wav=None, ref_sr=None, ref_dict=self.tts.conds.gen,
                                        n_cfm_timesteps=steps, finalize=True, noised_mels=n_noise)
        wav, _ = g.hift_inference(mels.to(g.dtype), None)
        return wav[0]

    # ---------------------------------------------------------------- public
    def max_tokens_for(self, text):
        # ~25 tokens/s; Danish speech is ~12-16 characters/s. 3 tokens/char + 40 is a generous cap.
        return min(self.max_len - 600, 40 + 3 * len(text))

    @torch.inference_mode()
    def stream_eager(self, text, first_chunk=20, chunk=40, holdback=3, xfade=240):
        """Reference streaming without S3Gen graphs: re-vocodes the whole prefix each chunk."""
        self.prefill(text)
        if self.graphs and self.graph is None:
            self.capture()
            self.prefill(text)  # capture clobbered state
        noise = torch.randn(1, 80, 2 * self.max_len, device=self.device, dtype=self.s3gen.dtype)
        emitted = 0  # samples handed out
        tail = None
        next_at = first_chunk
        trim = self.s3gen.trim_fade
        for toks, done in self.decode(self.max_tokens_for(text), burst=4):
            if not done and toks.numel() < next_at:
                continue
            wav = self.tokens_to_wav(toks, finalize=done, noise=noise)
            wav[: len(trim)] *= trim
            end = wav.numel() if done else max(0, wav.numel() - holdback * 960)
            if end <= emitted:
                continue
            seg = wav[emitted:end].float()
            if tail is not None and seg.numel() >= xfade:
                ramp = torch.linspace(0, 1, xfade, device=seg.device)
                seg[:xfade] = seg[:xfade] * ramp + tail * (1 - ramp)
            if done:
                out, tail = seg, None
            else:
                out, tail = seg[:-xfade], seg[-xfade:].clone()
                end -= xfade
            yield out.cpu().numpy()
            emitted = end
            next_at = toks.numel() + chunk
            if done:
                return

    @torch.inference_mode()
    def synthesize(self, text, watermark=False):
        """Whole utterance (offline use). Returns float32 numpy at self.sr."""
        self.prefill(text)
        if self.graphs and self.graph is None:
            self.capture()
            self.prefill(text)
        toks = None
        for toks, done in self.decode(self.max_tokens_for(text), burst=16):
            if done:
                break
        wav = self.tokens_to_wav(toks, finalize=True)
        wav = wav.float().cpu().numpy()
        if watermark:
            wav = self.tts.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return wav, int(toks.numel())

    # ---------------------------------------------------------------- graph streaming
    @torch.inference_mode()
    def stream(self, text, first=16, chunk=32, ctx=8, xfade=480, prompt_tokens=None):
        """Streams float32 numpy chunks at self.sr. The first chunk covers `first` tokens
        (40 ms each), later ones `chunk` tokens. S3Gen runs as fixed-shape CUDA graphs.
        `prompt_tokens` keeps only the last N tokens (N/25 s) of the voice prompt for S3Gen: cheaper
        windows, slightly less voice detail."""
        key = (self.voice, first, chunk, ctx, prompt_tokens)
        if getattr(self, "voc", None) is None or self.voc.key != key:
            self.voc = ChunkVocoder(self, first, chunk, ctx, prompt_tokens)
        self.prefill(text)
        if self.graphs and self.graph is None:
            self.capture()
            self.prefill(text)
        yield from self.voc.run(self.decode(self.max_tokens_for(text), burst=4), xfade)


PREFILL_BUCKETS = (16, 32, 48, 64, 96, 128, 192, 256)
LOOKAHEAD = 3  # flow encoder's pre-lookahead, in tokens
SAMPLES_PER_TOKEN = 960  # 24 kHz / 25 Hz


class ChunkVocoder:
    """S3Gen (token -> mel -> wav) over fixed windows, each captured once as a CUDA graph.

    Window = [voice prompt][ctx already-voiced tokens][n new tokens]. The ctx tokens enter as
    prompt too, with the mels we generated for them, so the flow continues smoothly. The last
    LOOKAHEAD new tokens lack right context, so their audio is redone by the next window.
    """

    def __init__(self, m: Roest, first, chunk, ctx, prompt_tokens=None):
        assert ctx <= first - LOOKAHEAD
        self.m, self.first, self.chunk, self.ctx = m, first, chunk, ctx
        self.key = (m.voice, first, chunk, ctx, prompt_tokens)
        g = m.s3gen
        self.flow = g.flow
        gen = m.tts.conds.gen
        dev = m.device
        self.p_tok = gen["prompt_token"].to(dev).long()
        self.p_feat = gen["prompt_feat"].to(dev).float()
        if prompt_tokens and prompt_tokens < self.p_tok.size(1):
            self.p_tok = self.p_tok[:, -prompt_tokens:]
            self.p_feat = self.p_feat[:, -2 * prompt_tokens:]
        emb = F.normalize(torch.atleast_2d(gen["embedding"].to(dev).float()), dim=1)
        self.spk = self.flow.spk_embed_affine_layer(emb)
        self.steps = m.s3_steps or (2 if g.meanflow else 10)
        self.meanflow = g.meanflow
        self.mels = torch.zeros(1, 80, 2 * m.max_len, device=dev)  # generated mels per token
        _patch_s3gen_syncs()
        g.mel2wav.stft_window = g.mel2wav.stft_window.to(dev)
        self.graphs ={(c, n): self._capture(c, n) for c, n in ((0, first), (ctx, chunk))}

    def _body(self, st):
        flow = self.flow
        x = flow.input_embedding(st["tok"])
        h, _ = flow.encoder(x, st["tok_len"])
        h = flow.encoder_proj(h)
        mel1 = st["feat"].size(1)
        conds = torch.zeros(1, h.size(1), 80, device=h.device, dtype=h.dtype)
        conds[:, :mel1] = st["feat"]
        mask = torch.ones(1, 1, h.size(1), device=h.device, dtype=h.dtype)
        feat, _ = flow.decoder(mu=h.transpose(1, 2).contiguous(), mask=mask, spks=self.spk,
                               cond=conds.transpose(1, 2), n_timesteps=self.steps,
                               noised_mels=st["noise"] if self.meanflow else None, meanflow=self.meanflow)
        new = feat[:, :, mel1:]
        st["new_mels"].copy_(new)
        c = st["c"]
        voc_in = torch.cat([st["feat"][:, mel1 - 2 * c:].transpose(1, 2), new], dim=2) if c else new
        wav, _ = self.m.s3gen.hift_inference(voc_in, None)
        st["wav"].copy_(wav[:, : st["wav"].size(1)])

    def _capture(self, c, n):
        dev = self.m.device
        P = self.p_tok.size(1)
        st = dict(
            c=c,
            tok=torch.cat([self.p_tok, torch.full((1, c + n), 100, dtype=torch.long, device=dev)], dim=1),
            tok_len=torch.tensor([P + c + n], device=dev),
            feat=torch.cat([self.p_feat, torch.zeros(1, 2 * c, 80, device=dev)], dim=1),
            noise=torch.randn(1, 80, 2 * n, device=dev),
            new_mels=torch.zeros(1, 80, 2 * n, device=dev),
            wav=torch.zeros(1, 2 * (c + n) * 480, device=dev),
        )
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._body(st)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._body(st)
        st["graph"] = graph
        return st

    def _window(self, toks, start, c, n):
        """Run the (c, n) graph for new tokens toks[start:start+n] (padded with silence)."""
        st = self.graphs[(c, n)]
        P = self.p_tok.size(1)
        new = toks[start: start + n]
        k = new.numel()
        st["tok"][0, P + c: P + c + k] = new
        if k < n:
            st["tok"][0, P + c + k:] = 4299  # S3GEN_SIL
        if c:
            st["tok"][0, P: P + c] = toks[start - c: start]
            st["feat"][0, -2 * c:] = self.mels[0, :, 2 * (start - c): 2 * start].T
        st["noise"].normal_()
        st["graph"].replay()
        self.mels[0, :, 2 * start: 2 * (start + n)] = st["new_mels"][0]
        return st["wav"][0]

    def run(self, decoder, xfade):
        tail = None
        done_tok = 0  # tokens whose audio is out
        trim = self.m.s3gen.trim_fade
        for toks, done in decoder:
            if done:
                toks = toks[toks < S3_VOCAB]
            while True:
                c, n = (0, self.first) if done_tok == 0 else (self.ctx, self.chunk)
                have = toks.numel() - done_tok
                if not done and have < n:
                    break
                wav = self._window(toks, done_tok, c, n)
                last = done and have + LOOKAHEAD <= n
                emit_tok = min(n, have + LOOKAHEAD) if last else n - LOOKAHEAD
                a = c * SAMPLES_PER_TOKEN
                b = a + emit_tok * SAMPLES_PER_TOKEN
                if done_tok == 0:
                    wav = wav.clone()
                    wav[: len(trim)] *= trim
                if tail is not None:
                    seg = wav[a - xfade: b].clone()
                    ramp = torch.linspace(0, 1, xfade, device=seg.device)
                    seg[:xfade] = seg[:xfade] * ramp + tail * (1 - ramp)
                else:
                    seg = wav[a:b].clone()
                if last:
                    yield seg.float().cpu().numpy()
                    return
                tail = seg[-xfade:].clone()
                yield seg[:-xfade].float().cpu().numpy()
                done_tok += emit_tok


_PATCHED = False


def _patch_s3gen_syncs():
    """Remove host syncs from S3Gen so it can be captured. add_optional_chunk_mask ends with an
    `if (...).item()` repair of all-false rows; this does the same repair without the sync."""
    global _PATCHED
    if _PATCHED:
        return
    from chatterbox.models.s3gen import decoder as dec
    from chatterbox.models.s3gen.transformer import upsample_encoder as ue
    from chatterbox.models.s3gen.utils.mask import subsequent_chunk_mask
    import chatterbox.models.s3gen.flow_matching as fm

    orig = ue.add_optional_chunk_mask

    def no_sync(xs, masks, use_dynamic_chunk, use_dynamic_left_chunk, decoding_chunk_size, static_chunk_size,
                num_decoding_left_chunks, *a, **k):
        if use_dynamic_chunk:
            return orig(xs, masks, use_dynamic_chunk, use_dynamic_left_chunk, decoding_chunk_size,
                        static_chunk_size, num_decoding_left_chunks, *a, **k)
        if static_chunk_size > 0:
            cm = masks & subsequent_chunk_mask(xs.size(1), static_chunk_size, num_decoding_left_chunks,
                                               xs.device).unsqueeze(0)
        else:
            cm = masks
        return cm | (cm.sum(dim=-1, keepdim=True) == 0)

    ue.add_optional_chunk_mask = no_sync
    dec.add_optional_chunk_mask = no_sync

    # HiFT's sine source builds tensors on the CPU and copies them over; build them on the GPU.
    import chatterbox.models.s3gen.hifigan as hg

    @torch.no_grad()
    def sine_forward(self, f0):
        H = self.harmonic_num + 1
        mult = torch.arange(1, H + 1, device=f0.device, dtype=f0.dtype).view(1, H, 1)
        theta = 2 * math.pi * (torch.cumsum(f0 * mult / self.sampling_rate, dim=-1) % 1)
        phase = (torch.rand(f0.size(0), H, 1, device=f0.device) * 2 - 1) * math.pi
        phase[:, 0, :] = 0
        sine = self.sine_amp * torch.sin(theta + phase)
        uv = self._f02uv(f0)
        noise = (uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * torch.randn_like(sine)
        return sine * uv + noise, uv, noise

    hg.SineGen.forward = sine_forward

    # torch.istft checks its window envelope on the host; this is the same maths without it.
    def istft(self, magnitude, phase):
        n_fft, hop = self.istft_params["n_fft"], self.istft_params["hop_len"]
        magnitude = torch.clip(magnitude, max=1e2)
        spec = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
        win = self.stft_window.to(magnitude.device)
        frames = torch.fft.irfft(spec, n=n_fft, dim=1) * win.view(1, -1, 1)  # (B, n_fft, T)
        T = frames.size(-1)
        out_len = n_fft + hop * (T - 1)
        y = F.fold(frames, (1, out_len), (1, n_fft), stride=(1, hop)).view(frames.size(0), -1)
        env = F.fold((win * win).view(1, -1, 1).expand(1, n_fft, T), (1, out_len), (1, n_fft),
                     stride=(1, hop)).view(1, -1)
        start = n_fft // 2
        end = start + hop * (T - 1)
        return y[:, start:end] / env[:, start:end].clamp_min(1e-11)

    hg.HiFTGenerator._istft = istft
    fm.print = lambda *a, **k: None
    fm.tqdm = lambda it, **k: it
    _PATCHED = True
