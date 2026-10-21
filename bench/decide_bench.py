"""Typed decisions (Jev-style) on Grace's scenarios: JevK5 4B vs 9B, with Gemma as a baseline.

Every backend gets the exact JevK5 prompt (evidence, criterion, lettered options) and is read the
same way: the next-token log-probabilities of the option letters, renormalised. Nothing is
generated. Scenarios: fixtures/decide/scenarios.yaml.

  bench/.venv/Scripts/python bench/decide_bench.py --backend jev --url http://127.0.0.1:8091 \
      --temperature 1.22 --label jev4b
  bench/.venv/Scripts/python bench/decide_bench.py --backend gemma --url http://127.0.0.1:8080 --label gemma

Results: bench-results/decide/<label>.jsonl (one row per decision) and a summary on stdout.
Start the servers yourself; `--vram` samples nvidia-smi during the run for the peak.
"""

import argparse
import json
import statistics as st
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml
from jevk5 import JevK5GGUF
from jevk5.prompt import messages

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent

LOOKUP_Q = (
    "Mikkel spoke this request to his voice assistant. Does answering it need a lookup (web, "
    "a store's hours, prices, news, anything current or obscure) rather than general knowledge, "
    "maths, advice or chat?"
)
LOOP = {
    "answer": "Enough reliable, matching evidence: answer now.",
    "search_again": "Evidence is missing, outdated, about the wrong thing, or sources disagree: search again with a better query.",
    "give_up": "The budget is used up or the information is clearly not published: stop and say it could not be found.",
}
LOOP_Q = "A search assistant has read the sources below. What should it do next?"
RANK = [
    "irrelevant to the question",
    "related topic, but does not answer the question",
    "partly answers the question",
    "directly answers the exact question",
]
ABOUT = {
    "length": "how long her replies are",
    "tone": "her mood or tone of voice",
    "humour": "how funny or teasing she is",
    "questions": "how many questions she asks",
    "phrasing": "specific words or phrases she uses",
    "formality": "how formal or polite she is",
    "listening": "she missed or misunderstood what he said",
    "resuming": "how she continues after he interrupted her",
    "one_task": "a request for this one answer only (a story of a given length, a format)",
    "one_topic": "a correction of what she assumed about one subject",
    "facts": "a factual correction",
    "other": "something else",
}
ABOUT_Q = "Mikkel gave his voice assistant Grace this feedback. What is it about?"
RISK = {
    "read_only": "Only reads or looks something up; changes nothing.",
    "visible": "Does something visible and harmless that is easy to undo (opens a program, sets a timer).",
    "changes": "Sends, deletes, buys, overwrites or commits to something; needs the user's explicit yes first.",
}
RISK_Q = "How risky is this action for a voice assistant to take on its own?"


def decisions(path: Path):
    """(id, kind, variant, state, question, expect) for every decision in the scenarios."""
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    tools = doc["tool_options"]
    for it in doc["items"]:
        if "group" in it:
            for r in it["results"]:
                q = {"type": "score", "instructions": f"How well does this search result answer: {it['q']}", "criteria": RANK}
                yield r["id"], "rank", it["group"], r["state"], q, r["rel"]
            continue
        k, e = it["kind"], it["expect"]
        if k == "tool":
            q = {"type": "choice", "instructions": doc["tool_q"], "criteria": tools}
            yield it["id"], k, "da", f'Mikkel said: "{it["state"]}"', q, e
            yield it["id"] + "en", k, "en", f'Mikkel said: "{it["en"]}"', q, e
        elif k == "lookup":
            yield it["id"], k, "", f'Mikkel said: "{it["state"]}"', {"type": "noul", "instructions": LOOKUP_Q}, e
        elif k == "answered":
            q = {"type": "noul", "instructions": f"Does the evidence directly state the answer to this exact question: {it['q']}"}
            yield it["id"], k, "", it["state"], q, e
        elif k == "file":
            q = {"type": "noul", "instructions": f"Is this the file Mikkel means? {it['q']}"}
            yield it["id"], k, "", it["state"], q, e
        elif k == "loop":
            yield it["id"], k, "", it["state"], {"type": "choice", "instructions": LOOP_Q, "criteria": LOOP}, e
        elif k == "about":
            yield it["id"], k, "", it["state"], {"type": "choice", "instructions": ABOUT_Q, "criteria": ABOUT}, e
        elif k == "risk":
            yield it["id"], k, "", it["state"], {"type": "choice", "instructions": RISK_Q, "criteria": RISK}, e


class GemmaReader(JevK5GGUF):
    """The JevK5 prompt and readout on Gemma through llama-server's chat endpoint (Gemma's own
    template), so the only difference from JevK5 is the model."""

    def _logprobs(self, prompt):  # prompt is ignored; rebuilt from the current question
        raise NotImplementedError

    def probabilities(self, state, question):
        from jevk5.prompt import LETTERS, decision_options

        options = decision_options(question)
        texts = [t for _, t in options]
        msgs = messages(state, question["instructions"], texts)
        t0 = time.perf_counter()
        out = self._post(
            "/v1/chat/completions",
            {"messages": msgs, "max_tokens": 1, "temperature": 0, "logprobs": True,
             "top_logprobs": 20, "cache_prompt": False},
        )
        self.last_seconds = time.perf_counter() - t0
        top = out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        seen = {}
        for entry in top:
            tok = entry["token"].strip()
            if tok in LETTERS and tok not in seen:
                seen[tok] = entry["logprob"]
        floor = min(seen.values(), default=0.0) - 2.0
        import math

        lps = [seen.get(LETTERS[i], floor) for i in range(len(texts))]
        m = max(lps)
        w = [math.exp(z - m) for z in lps]
        s = sum(w)
        tokens = out.get("usage", {}).get("prompt_tokens", 0)
        return {k: v / s for (k, _), v in zip(options, w)}, tokens


def vram_sampler(stop, peak):
    while not stop.is_set():
        try:
            used = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
            ).split()[0])
            peak[0] = max(peak[0], used)
        except Exception:
            pass
        stop.wait(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["jev", "gemma"], required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--knockout-temperature", type=float)
    ap.add_argument("--label", required=True)
    ap.add_argument("--scenarios", default=str(ROOT / "fixtures/decide/scenarios.yaml"))
    ap.add_argument("--vram", action="store_true")
    args = ap.parse_args()

    cls = JevK5GGUF if args.backend == "jev" else GemmaReader
    model = cls(url=args.url, temperature=args.temperature, knockout_temperature=args.knockout_temperature)
    items = list(decisions(Path(args.scenarios)))
    # One warm-up decision so the first timing doesn't include CUDA graph setup.
    model.decide(items[0][3], items[0][4])

    stop, peak = threading.Event(), [0]
    if args.vram:
        threading.Thread(target=vram_sampler, args=(stop, peak), daemon=True).start()
    rows = []
    for did, kind, variant, state, q, expect in items:
        t0 = time.perf_counter()
        a = model.decide(state, q)
        ms = (time.perf_counter() - t0) * 1000
        row = dict(id=did, kind=kind, variant=variant, expect=expect, ms=ms, tokens=a["input_tokens"],
                   confidence=a["confidence"])
        if q["type"] == "noul":
            row.update(p=a["noul"], got=a["noul"] > 0.5, ok=(a["noul"] > 0.5) == expect,
                       brier=(a["noul"] - (1.0 if expect else 0.0)) ** 2)
        elif q["type"] == "choice":
            row.update(got=a["choice"], ok=a["choice"] == expect, p_expect=a["probabilities"][expect])
        else:
            row.update(score=a["score"], got=max(a["probabilities"], key=a["probabilities"].get))
        rows.append(row)
    stop.set()

    out = ROOT / "bench-results" / "decide" / f"{args.label}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    summary = summarise(rows)
    summary.update(label=args.label, vram_peak_mib=peak[0] or None)
    (out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


def summarise(rows):
    s = {}
    kinds = ["tool:da", "tool:en", "lookup", "answered", "loop", "file", "about", "risk"]
    for k in kinds:
        kind, _, var = k.partition(":")
        rs = [r for r in rows if r["kind"] == kind and (not var or r["variant"] == var)]
        if not rs:
            continue
        e = dict(acc=f"{sum(r['ok'] for r in rs)}/{len(rs)}")
        if "brier" in rs[0]:
            e["brier"] = round(st.mean(r["brier"] for r in rs), 3)
        s[k] = e
    # Ranking: per group, does the result with the highest expected score have the highest rel?
    groups = {}
    for r in rows:
        if r["kind"] == "rank":
            groups.setdefault(r["variant"], []).append(r)
    top1 = sum(max(g, key=lambda r: r["score"])["expect"] == max(r["expect"] for r in g) for g in groups.values())
    mae = st.mean(abs(r["score"] - r["expect"]) for g in groups.values() for r in g)
    s["rank"] = dict(top1=f"{top1}/{len(groups)}", mae=round(mae, 2))
    ok = [r["ok"] for r in rows if "ok" in r]
    s["all_typed"] = f"{sum(ok)}/{len(ok)}"
    ms = sorted(r["ms"] for r in rows)
    s["ms_p50"] = round(st.median(ms), 1)
    s["ms_p95"] = round(ms[int(0.95 * (len(ms) - 1))], 1)
    s["tokens_p50"] = st.median(r["tokens"] for r in rows)
    s["decisions"] = len(rows)
    return s


if __name__ == "__main__":
    main()
