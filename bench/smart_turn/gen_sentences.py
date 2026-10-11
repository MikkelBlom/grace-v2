"""Generate Danish spoken-style sentences for synthetic Smart Turn data with the LOCAL Gemma on
llama-server (http://127.0.0.1:8080, OpenAI-compatible). No cloud model is used.

Each item: a complete utterance plus an incomplete variant cut mid-clause, and a filler variant
ending in "øh"/"altså"/"og så"/... (the Smart Turn v3 synthetic recipe: complete / incomplete /
end-filler, all voiced by TTS later).

  python bench/smart_turn/gen_sentences.py --n 3000 --out fixtures/corpora/smart-turn/da-synth/sentences.jsonl
"""
import argparse
import json
import random
import re
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
TOPICS = [
    "vejret", "madlavning og opskrifter", "indkøb", "kalender og aftaler", "musik", "nyheder", "sport",
    "familie", "arbejde", "skole og lektier", "rejser og transport", "togtider", "sundhed", "trening",
    "film og serier", "bøger", "smarte hjem og lys", "påmindelser", "penge og regninger", "hobbyer",
    "kæledyr", "have", "computer og programmering", "spil", "historie", "naturvidenskab", "politik",
    "weekendplaner", "fødselsdage", "biler", "cykling", "venner", "minder fra barndommen", "planer for i morgen",
]
STYLES = [
    "et kort spørgsmål til en stemmeassistent", "en kommando til en stemmeassistent",
    "en lidt længere forklaring i hverdagssprog", "en sætning fra en afslappet samtale med en ven",
    "et spørgsmål med en bisætning", "en sætning med tal eller tidspunkter", "et meget kort svar (1-3 ord)",
]
FILLERS = ["øh", "øhm", "altså", "og så", "men øh", "og øh", "fordi", "så øh", "eller", "ikke også og"]

PROMPT = """Skriv {k} forskellige, naturlige danske ytringer om emnet "{topic}". Stil: {style}.
Skriv som folk faktisk taler (talesprog, gerne "jo", "lige", "egentlig"), ingen navne på rigtige personer.
For hver ytring giv også en UFÆRDIG version, der stopper midt i en sætning, som hvis personen tøver
og ikke er færdig (fx "Kan du sætte en påmindelse til i morgen om at" eller "Jeg tror at vi skal").
Svar KUN med JSON-linjer, én per ytring: {{"complete": "...", "incomplete": "..."}}"""


def ask(prompt, temperature=1.0):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": 1500,
                       "temperature": temperature, "top_p": 0.95}).encode("utf-8")
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


def parse(text):
    out = []
    for line in text.splitlines():
        line = line.strip().strip(",")
        m = re.search(r"\{.*\}", line)
        if not m:
            continue
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        c, i = d.get("complete", "").strip(), d.get("incomplete", "").strip()
        if c and i and len(i) < len(c) + 40 and not i.endswith((".", "?", "!")):
            out.append((c, i))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    seen = set()
    n = 0
    with open(a.out, "a", encoding="utf-8") as f:
        while n < a.n:
            topic, style = rng.choice(TOPICS), rng.choice(STYLES)
            try:
                pairs = parse(ask(PROMPT.format(k=10, topic=topic, style=style)))
            except Exception as e:  # server busy/timeouts: report and retry
                print("llm error:", e, flush=True)
                continue
            for c, i in pairs:
                if c.lower() in seen:
                    continue
                seen.add(c.lower())
                # filler variant: a cut of the incomplete text ending in a filler
                words = i.split()
                cut = " ".join(words[: max(1, rng.randint(len(words) // 2, len(words)))])
                filler = f"{cut.rstrip(',')} {rng.choice(FILLERS)}"
                f.write(json.dumps({"complete": c, "incomplete": i, "filler": filler, "topic": topic,
                                    "style": style}, ensure_ascii=False) + "\n")
                n += 1
            f.flush()
            print(n, flush=True)


if __name__ == "__main__":
    main()
