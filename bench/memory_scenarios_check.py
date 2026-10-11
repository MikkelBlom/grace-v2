"""Validate the Danish memory scenarios in fixtures/memory/scenarios/ against their schema.

Schema: fixtures/memory/scenarios/README.md. Prints per-category counts and exits non-zero on
any error. Needs PyYAML (bench/.venv has it).

    bench/.venv/Scripts/python.exe bench/memory_scenarios_check.py [scenario_dir]
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "fixtures" / "memory" / "scenarios"

CATEGORIES = {
    "extraction",
    "multi_session",
    "temporal",
    "knowledge_update",
    "abstention",
    "commitment",
    "self_echo",
    "interrupted",
    "stt_garbled",
    "momentary_command",
    "duplicate",
    "dialect",
    "same_session_correction",
}
MIN_PER_CATEGORY = 5
KINDS = {"user", "preference", "grace", "environment", "followup"}
TOP_KEYS = {"id", "title", "categories", "persona", "live_failure", "sessions", "notes"}
SESSION_KEYS = {"start", "turns", "expect"}
TURN_KEYS = {"user", "grace", "gap_s", "interrupted", "stt_garbled", "stt_logprob"}
EXPECT_KEYS = {
    "facts_present",
    "facts_absent",
    "facts_invalidated",
    "commitments_open",
    "commitments_done",
    "probe_at",
    "probes",
}
ASSERT_KEYS = {
    "facts_present": {"assert", "match", "exclude", "max_count", "kind_hint"},
    "facts_absent": {"assert", "match", "exclude", "scope"},
    "facts_invalidated": {"assert", "match", "exclude"},
    "commitments_open": {"assert", "match", "exclude", "due"},
    "commitments_done": {"assert", "match", "exclude"},
}
PROBE_KEYS = {"question", "expect", "must_not_contain", "abstain", "proactive", "rubric"}


def last_sunday(year: int, month: int) -> int:
    nxt = datetime(year + (month == 12), month % 12 + 1, 1)
    d = nxt - timedelta(days=1)
    return d.day - (d.weekday() + 1) % 7


def copenhagen_offset(local: datetime) -> timedelta:
    """UTC offset of Europe/Copenhagen for a naive local time (EU rule, no tzdata needed)."""
    y = local.year
    start = datetime(y, 3, last_sunday(y, 3), 2)
    end = datetime(y, 10, last_sunday(y, 10), 3)
    return timedelta(hours=2 if start <= local < end else 1)


class Checker:
    def __init__(self, path: Path):
        self.path = path
        self.errors: list[str] = []

    def err(self, where: str, msg: str) -> None:
        self.errors.append(f"{self.path.name}: {where}: {msg}")

    def keys(self, where: str, obj: object, allowed: set[str], required: set[str]) -> bool:
        if not isinstance(obj, dict):
            self.err(where, f"expected a mapping, got {type(obj).__name__}")
            return False
        for k in obj.keys() - allowed:
            self.err(where, f"unknown key '{k}'")
        for k in required - obj.keys():
            self.err(where, f"missing key '{k}'")
        return True

    def text(self, where: str, v: object) -> bool:
        if not isinstance(v, str) or not v.strip():
            self.err(where, "expected a non-empty string")
            return False
        return True

    def time(self, where: str, v: object) -> datetime | None:
        if not isinstance(v, str):
            self.err(where, "datetime must be a quoted ISO string")
            return None
        try:
            t = datetime.fromisoformat(v)
        except ValueError:
            self.err(where, f"not ISO 8601: {v!r}")
            return None
        if t.tzinfo is None:
            self.err(where, f"missing UTC offset: {v!r}")
            return None
        want = copenhagen_offset(t.replace(tzinfo=None))
        if t.utcoffset() != want:
            self.err(where, f"{v} has offset {t.utcoffset()}, Europe/Copenhagen is {want}")
        return t

    def alternative(self, where: str, s: object) -> None:
        if not self.text(where, s):
            return
        if s.startswith("re:"):
            # The runner is Rust (`regex` crate): no lookaround, no backreferences.
            if re.search(r"\(\?<?[=!]|\\[1-9]", s[3:]):
                self.err(where, f"regex {s!r} uses lookaround/backreference (unsupported in Rust regex)")
            try:
                re.compile(s[3:], re.IGNORECASE)
            except re.error as e:
                self.err(where, f"bad regex {s!r}: {e}")

    def matcher(self, where: str, groups: object) -> None:
        if not isinstance(groups, list) or not groups:
            self.err(where, "match/expect must be a non-empty list of groups")
            return
        for i, g in enumerate(groups):
            alts = g if isinstance(g, list) else [g]
            if not alts:
                self.err(f"{where}[{i}]", "empty group")
            for a in alts:
                self.alternative(f"{where}[{i}]", a)

    def assertion(self, where: str, kind: str, a: object) -> None:
        if not self.keys(where, a, ASSERT_KEYS[kind], {"assert", "match"}):
            return
        self.text(f"{where}.assert", a["assert"])
        self.matcher(f"{where}.match", a["match"])
        if "exclude" in a:
            if not isinstance(a["exclude"], list) or not a["exclude"]:
                self.err(where, "exclude must be a non-empty list")
            else:
                for s in a["exclude"]:
                    self.alternative(f"{where}.exclude", s)
        if "max_count" in a and (not isinstance(a["max_count"], int) or a["max_count"] < 1):
            self.err(where, "max_count must be a positive integer")
        if "kind_hint" in a and a["kind_hint"] not in KINDS:
            self.err(where, f"kind_hint must be one of {sorted(KINDS)}")
        if "scope" in a and a["scope"] not in {"current", "ever"}:
            self.err(where, "scope must be 'current' or 'ever'")
        if "due" in a:
            try:
                datetime.strptime(str(a["due"]), "%Y-%m-%d")
            except ValueError:
                self.err(where, "due must be YYYY-MM-DD")

    def probe(self, where: str, p: object) -> None:
        if not self.keys(where, p, PROBE_KEYS, {"question"}):
            return
        self.text(f"{where}.question", p["question"])
        for flag in ("abstain", "proactive"):
            if flag in p and not isinstance(p[flag], bool):
                self.err(where, f"{flag} must be a boolean")
        if "expect" in p:
            self.matcher(f"{where}.expect", p["expect"])
        if "must_not_contain" in p:
            if not isinstance(p["must_not_contain"], list) or not p["must_not_contain"]:
                self.err(where, "must_not_contain must be a non-empty list")
            else:
                for s in p["must_not_contain"]:
                    self.alternative(f"{where}.must_not_contain", s)
        if p.get("abstain") and "expect" in p:
            self.err(where, "an abstain probe has no expect (abstention markers are the check)")
        if not any(k in p for k in ("expect", "must_not_contain", "abstain")) and "rubric" not in p:
            self.err(where, "probe has nothing to check")
        if not any(k in p for k in ("expect", "must_not_contain", "abstain")):
            self.err(where, "probe needs a mechanical check (expect, must_not_contain or abstain)")

    def run(self, doc: object) -> list[str]:
        if not self.keys("top", doc, TOP_KEYS, {"id", "title", "categories", "persona", "sessions"}):
            return []
        if doc["id"] != self.path.stem:
            self.err("id", f"{doc['id']!r} != file stem {self.path.stem!r}")
        self.text("title", doc["title"])
        self.text("persona", doc["persona"])
        cats = doc["categories"]
        if not isinstance(cats, list) or not 1 <= len(cats) <= 4:
            self.err("categories", "must be a list of 1-4 categories")
            cats = []
        for c in cats:
            if c not in CATEGORIES:
                self.err("categories", f"unknown category {c!r}")
        if len(set(cats)) != len(cats):
            self.err("categories", "duplicate category")
        sessions = doc["sessions"]
        if not isinstance(sessions, list) or not 2 <= len(sessions) <= 5:
            self.err("sessions", "must be a list of 2-5 sessions")
            return [c for c in cats if c in CATEGORIES]
        prev: datetime | None = None
        flags: Counter[str] = Counter()
        for si, s in enumerate(sessions):
            w = f"sessions[{si}]"
            if not self.keys(w, s, SESSION_KEYS, {"start", "turns"}):
                continue
            t = self.time(f"{w}.start", s["start"])
            if t and prev and t <= prev:
                self.err(f"{w}.start", "sessions must be in strictly increasing time order")
            turns = s["turns"]
            if not isinstance(turns, list) or not turns:
                self.err(f"{w}.turns", "must be a non-empty list")
                turns = []
            elapsed = 0
            for ti, turn in enumerate(turns):
                tw = f"{w}.turns[{ti}]"
                if not self.keys(tw, turn, TURN_KEYS, {"user", "grace"}):
                    continue
                self.text(f"{tw}.user", turn["user"])
                if not isinstance(turn["grace"], str):
                    self.err(f"{tw}.grace", "must be a string (may be empty if cut off at once)")
                for flag in ("interrupted", "stt_garbled"):
                    if flag in turn:
                        if turn[flag] is not True:
                            self.err(tw, f"{flag} must be true when present")
                        flags[flag] += 1
                if "gap_s" in turn and (
                    not isinstance(turn["gap_s"], (int, float)) or turn["gap_s"] <= 0
                ):
                    self.err(tw, "gap_s must be a positive number")
                if "stt_logprob" in turn and not isinstance(turn["stt_logprob"], (int, float)):
                    self.err(tw, "stt_logprob must be a number")
                elapsed += turn.get("gap_s", 20) if isinstance(turn.get("gap_s", 20), (int, float)) else 20
            if t:
                prev = t + timedelta(seconds=elapsed)
            exp = s.get("expect")
            if exp is None:
                continue
            if not self.keys(f"{w}.expect", exp, EXPECT_KEYS, set()):
                continue
            for kind in ASSERT_KEYS:
                items = exp.get(kind, [])
                if not isinstance(items, list):
                    self.err(f"{w}.expect.{kind}", "must be a list")
                    continue
                for ai, a in enumerate(items):
                    self.assertion(f"{w}.expect.{kind}[{ai}]", kind, a)
            probes = exp.get("probes", [])
            if not isinstance(probes, list):
                self.err(f"{w}.expect.probes", "must be a list")
                probes = []
            for pi, p in enumerate(probes):
                self.probe(f"{w}.expect.probes[{pi}]", p)
            last = si == len(sessions) - 1
            if "probe_at" in exp:
                if not last:
                    self.err(f"{w}.expect.probe_at", "only the last session sets probe_at")
                pt = self.time(f"{w}.expect.probe_at", exp["probe_at"])
                if pt and prev and pt <= prev:
                    self.err(f"{w}.expect.probe_at", "must be after the session's last turn")
            elif last and probes:
                self.err(f"{w}.expect", "the last session has probes but no probe_at")
        for flag in ("interrupted", "stt_garbled"):
            if flag in cats and not flags[flag]:
                self.err("categories", f"category {flag} but no turn has {flag}: true")
        return [c for c in cats if c in CATEGORIES]


def main() -> int:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    files = sorted(folder.glob("*.yaml"))
    if not files:
        print(f"no scenarios in {folder}")
        return 1
    counts: Counter[str] = Counter()
    errors: list[str] = []
    ids: set[str] = set()
    stats = Counter()
    for f in files:
        try:
            doc = yaml.safe_load(f.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            errors.append(f"{f.name}: YAML error: {e}")
            continue
        c = Checker(f)
        cats = c.run(doc)
        errors += c.errors
        counts.update(cats)
        if isinstance(doc, dict):
            if doc.get("id") in ids:
                errors.append(f"{f.name}: duplicate id")
            ids.add(doc.get("id"))
            for s in doc.get("sessions") or []:
                if not isinstance(s, dict):
                    continue
                stats["sessions"] += 1
                stats["turns"] += len(s.get("turns") or [])
                e = s.get("expect") or {}
                stats["state assertions"] += sum(len(e.get(k) or []) for k in ASSERT_KEYS)
                stats["probes"] += len(e.get("probes") or [])
            if doc.get("live_failure"):
                stats["live failures"] += 1
    print(f"{len(files)} scenarios in {folder}")
    print("  " + ", ".join(f"{k}: {v}" for k, v in stats.items()))
    print("categories:")
    for cat in sorted(CATEGORIES):
        n = counts[cat]
        mark = "" if n >= MIN_PER_CATEGORY else f"   <-- fewer than {MIN_PER_CATEGORY}"
        print(f"  {cat:<24}{n:>3}{mark}")
        if n < MIN_PER_CATEGORY:
            errors.append(f"category {cat} has {n} scenarios (< {MIN_PER_CATEGORY})")
    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print("  " + e)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
