"""Tool group `time`: the clock and calendar, reminders and timers, todos and notes, a focus
timer, unit and currency conversion, and watchers that report back when something happens
(docs/tools/time.md).

State lives in SQLite (ctx.data_dir/time.sqlite3), so reminders and watchers survive a restart.
Every function takes its clock from ctx.now when set (tests use a fake clock), else datetime.now.

Delivery. Reminders, timers, the focus timer and watchers put due events in an `events` table;
the engine fetches them with pending_events(ctx) and confirms each with ack_event(ctx, id)
(the tools-server request {"kind": "events"}, docs/tools/time.md). A background scheduler
thread fires due items and runs watcher checks; it starts when this module loads unless
GRACE_TOOLS_SCHEDULER=0 (and never under pytest, where tests call tick() with a fake clock).
"""

import json
import os
import re
import sqlite3
import sys
import threading
import urllib.parse
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, available_timezones

from . import time_convert as tc
from . import time_parse as tp
from . import time_watch as tw
from .registry import b, n, s, tool

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / ".runtime/grace-data"  # the tools server's ctx.data_dir
LOCAL_TZ = ZoneInfo("Europe/Copenhagen")  # Mikkel's clock; naive datetimes are local time
LATE_S = 120  # a reminder this late (the server was off) says when it was due
EVENT_TTL_H = 12  # unacked events older than this are not offered any more
MAX_ERRORS = 10  # a watcher whose check fails this often in a row gives up
MAX_AHEAD_DAYS = 400

SCHEMA = """
create table if not exists items(id integer primary key, kind text, text text, due text, created text,
  state text default 'pending', fired_at text, grp text, say text, snoozes integer default 0);
create table if not exists notes(id integer primary key, kind text, list text, title text, text text,
  created text, done_at text);
create table if not exists watchers(id integer primary key, kind text, target text, cond text, arg text,
  label text, interval real, created text, expires text, next_check text, state text,
  status text default 'active', errors integer default 0, last_error text, ended text, message text);
create table if not exists events(id integer primary key, created text, priority text, source text,
  ref integer, say text, acked text);
create index if not exists items_due on items(state, due);
create index if not exists events_open on events(acked, created);
"""


def log(*a):
    print("[time]", *a, file=sys.stderr, flush=True)


def now_of(ctx):
    f = getattr(ctx, "now", None) or datetime.now
    return f().replace(microsecond=0)


def iso(dt):
    return dt.replace(microsecond=0).isoformat()


def dt_of(text):
    return datetime.fromisoformat(text)


# ---- the store --------------------------------------------------------------------------------

class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self.db() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def db(self):
        with self.lock:
            c = sqlite3.connect(self.path, timeout=10)
            c.row_factory = sqlite3.Row
            try:
                yield c
                c.commit()
            finally:
                c.close()


STORES = {}
STORES_LOCK = threading.Lock()
BOUND = {"dir": None}  # the data dir the server's tools use (for the scheduler and summaries)


def store_at(data_dir):
    path = str(Path(data_dir) / "time.sqlite3")
    with STORES_LOCK:
        if path not in STORES:
            STORES[path] = Store(path)
        return STORES[path]


def store(ctx):
    BOUND["dir"] = ctx.data_dir
    return store_at(ctx.data_dir)


def add_event(c, now, priority, source, ref, say):
    cur = c.execute("insert into events(created, priority, source, ref, say) values (?,?,?,?,?)",
                    (iso(now), priority, source, ref, say))
    return cur.lastrowid


# ---- the scheduler ----------------------------------------------------------------------------

def tick(st, now, checks=None, watchers=True):
    """Fires everything due at `now`: due reminders, timers and focus steps become "now" events;
    watchers whose check is due are checked. Returns the new events' ids."""
    fired = []
    with st.db() as c:
        for r in c.execute("select * from items where state='pending' and due<=? order by due", (iso(now),)).fetchall():
            if c.execute("update items set state='fired', fired_at=? where id=? and state='pending'",
                         (iso(now), r["id"])).rowcount != 1:
                continue
            say = r["say"]
            due = dt_of(r["due"])
            if (now - due).total_seconds() > LATE_S:
                say += f" (It was due {tp.say_when(due, now)}; I couldn't say it then.)"
            fired.append(add_event(c, now, "now", r["kind"], r["id"], say))
    if watchers:
        fired += tick_watchers(st, now, checks if checks is not None else tw.CHECKS)
    return fired


def tick_watchers(st, now, checks):
    with st.db() as c:
        rows = [dict(r) for r in c.execute(
            "select * from watchers where status='active' and next_check<=? order by next_check", (iso(now),)).fetchall()]
    out = []
    for w in rows:
        if dt_of(w["expires"]) <= now:
            hours = (dt_of(w["expires"]) - dt_of(w["created"])).total_seconds() / 3600
            out += _end_watcher(st, w, now, "expired",
                                f"I've stopped watching {w['label']} after {hours:.0f} hours; {_wait_for(w)} didn't happen.")
            continue
        state = json.loads(w["state"] or "{}")
        try:
            hit, say, state = checks[w["kind"]](w, state, now)
        except Exception as e:  # a check that fails is counted; one bad answer doesn't end it
            errors = w["errors"] + 1
            log(f"watcher {w['id']} check failed ({errors}): {e!r}")
            if errors >= MAX_ERRORS:
                out += _end_watcher(st, w, now, "failed", f"I've stopped watching {w['label']}: checking kept failing ({e}).")
                continue
            with st.db() as c:
                c.execute("update watchers set errors=?, last_error=?, next_check=? where id=? and status='active'",
                          (errors, str(e)[:300], iso(now + timedelta(seconds=w["interval"] * min(errors, 5))), w["id"]))
            continue
        if hit:
            out += _end_watcher(st, w, now, "fired", say)
            continue
        with st.db() as c:
            c.execute("update watchers set state=?, errors=0, next_check=? where id=? and status='active'",
                      (json.dumps(state), iso(now + timedelta(seconds=w["interval"])), w["id"]))
    return out


def _end_watcher(st, w, now, status, say):
    with st.db() as c:
        if c.execute("update watchers set status=?, ended=?, message=? where id=? and status='active'",
                     (status, iso(now), say, w["id"])).rowcount != 1:
            return []  # stopped while it was being checked
        return [add_event(c, now, "next_pause", "watcher", w["id"], say)]


class Scheduler:
    """Fires due items every second and runs due watcher checks every two, in two threads so a
    slow web check never delays a timer."""

    def __init__(self):
        self.stop = threading.Event()
        self.threads = []

    def start(self):
        for watchers, period in ((False, 1.0), (True, 2.0)):
            t = threading.Thread(target=self.loop, args=(watchers, period), daemon=True,
                                 name=f"time-{'watchers' if watchers else 'timers'}")
            t.start()
            self.threads.append(t)

    def loop(self, watchers, period):
        while not self.stop.wait(period):
            try:
                st = store_at(BOUND["dir"] or DEFAULT_DATA_DIR)
                now = datetime.now().replace(microsecond=0)
                if watchers:
                    tick_watchers(st, now, tw.CHECKS)
                else:
                    tick(st, now, watchers=False)
            except Exception as e:
                log(f"scheduler: {e!r}")


SCHEDULER = Scheduler()
if os.environ.get("GRACE_TOOLS_SCHEDULER", "1") != "0" and "pytest" not in sys.modules:
    SCHEDULER.start()


# ---- the delivery channel ---------------------------------------------------------------------

def pending_events(ctx):
    """Events Grace should say, oldest first, "now" before "next_pause". They stay pending until
    ack_event; timers are fired here too, so a poll is never later than the scheduler."""
    st = store(ctx)
    now = now_of(ctx)
    tick(st, now, watchers=False)
    with st.db() as c:
        rows = c.execute("select * from events where acked is null and created>=? "
                         "order by case priority when 'now' then 0 else 1 end, id",
                         (iso(now - timedelta(hours=EVENT_TTL_H)),)).fetchall()
    return [{"id": r["id"], "priority": r["priority"], "say": r["say"], "source": r["source"],
             "ref": r["ref"], "created": r["created"]} for r in rows]


def ack_event(ctx, id):
    """Marks an event as said (or handed to speech). True if it was pending."""
    with store(ctx).db() as c:
        return c.execute("update events set acked=? where id=? and acked is null",
                         (iso(now_of(ctx)), int(id))).rowcount == 1


# ---- helpers ----------------------------------------------------------------------------------

def _first(a, *keys):
    for k in keys:
        v = a.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _parse_any(now, *texts, prefer="future"):
    for t in texts:
        if t:
            try:
                w = tp.parse(t, now, prefer=prefer)
            except (ValueError, KeyError):
                w = None
            if w is not None:
                return w, t
    return None, None


def _pick(rows, which, fields):
    """The rows `which` names: an id, or words found in the given fields. (rows, how)"""
    which = (which or "").strip().lower().lstrip("#")
    if not which:
        return rows, "all"
    if which.isdigit():
        return [r for r in rows if r["id"] == int(which)], "id"
    words = [w for w in tp.Scan(which).rest().split() if w not in ("the", "my", "min", "mit", "mine", "den", "det")]
    hits = [r for r in rows if all(w in " ".join(str(r[f] or "") for f in fields).lower() for w in words)]
    return hits, "words"


# ---- the clock and calendar -------------------------------------------------------------------

PLACES = {
    "danmark": "Europe/Copenhagen", "denmark": "Europe/Copenhagen", "københavn": "Europe/Copenhagen",
    "copenhagen": "Europe/Copenhagen", "aarhus": "Europe/Copenhagen", "odense": "Europe/Copenhagen",
    "london": "Europe/London", "england": "Europe/London", "uk": "Europe/London", "storbritannien": "Europe/London",
    "skotland": "Europe/London", "irland": "Europe/Dublin", "ireland": "Europe/Dublin", "dublin": "Europe/Dublin",
    "new york": "America/New_York", "ny": "America/New_York", "washington": "America/New_York",
    "boston": "America/New_York", "miami": "America/New_York", "chicago": "America/Chicago",
    "texas": "America/Chicago", "denver": "America/Denver", "los angeles": "America/Los_Angeles",
    "la": "America/Los_Angeles", "san francisco": "America/Los_Angeles", "californien": "America/Los_Angeles",
    "california": "America/Los_Angeles", "seattle": "America/Los_Angeles", "vancouver": "America/Vancouver",
    "toronto": "America/Toronto", "montreal": "America/Toronto", "mexico city": "America/Mexico_City",
    "mexico": "America/Mexico_City", "são paulo": "America/Sao_Paulo", "sao paulo": "America/Sao_Paulo",
    "brasilien": "America/Sao_Paulo", "brazil": "America/Sao_Paulo", "buenos aires": "America/Argentina/Buenos_Aires",
    "argentina": "America/Argentina/Buenos_Aires", "honolulu": "Pacific/Honolulu", "hawaii": "Pacific/Honolulu",
    "nuuk": "America/Nuuk", "grønland": "America/Nuuk", "greenland": "America/Nuuk", "tórshavn": "Atlantic/Faroe",
    "torshavn": "Atlantic/Faroe", "færøerne": "Atlantic/Faroe", "faroe islands": "Atlantic/Faroe",
    "reykjavik": "Atlantic/Reykjavik", "island": "Atlantic/Reykjavik", "iceland": "Atlantic/Reykjavik",
    "oslo": "Europe/Oslo", "norge": "Europe/Oslo", "norway": "Europe/Oslo", "stockholm": "Europe/Stockholm",
    "sverige": "Europe/Stockholm", "sweden": "Europe/Stockholm", "helsinki": "Europe/Helsinki",
    "finland": "Europe/Helsinki", "berlin": "Europe/Berlin", "tyskland": "Europe/Berlin", "germany": "Europe/Berlin",
    "paris": "Europe/Paris", "frankrig": "Europe/Paris", "france": "Europe/Paris", "amsterdam": "Europe/Amsterdam",
    "holland": "Europe/Amsterdam", "madrid": "Europe/Madrid", "spanien": "Europe/Madrid", "spain": "Europe/Madrid",
    "lissabon": "Europe/Lisbon", "lisbon": "Europe/Lisbon", "portugal": "Europe/Lisbon", "rom": "Europe/Rome",
    "rome": "Europe/Rome", "italien": "Europe/Rome", "italy": "Europe/Rome", "athen": "Europe/Athens",
    "athens": "Europe/Athens", "grækenland": "Europe/Athens", "greece": "Europe/Athens", "istanbul": "Europe/Istanbul",
    "tyrkiet": "Europe/Istanbul", "turkey": "Europe/Istanbul", "moskva": "Europe/Moscow", "moscow": "Europe/Moscow",
    "rusland": "Europe/Moscow", "kyiv": "Europe/Kyiv", "kiev": "Europe/Kyiv", "ukraine": "Europe/Kyiv",
    "cairo": "Africa/Cairo", "egypten": "Africa/Cairo", "egypt": "Africa/Cairo", "cape town": "Africa/Johannesburg",
    "sydafrika": "Africa/Johannesburg", "south africa": "Africa/Johannesburg", "nairobi": "Africa/Nairobi",
    "kenya": "Africa/Nairobi", "dubai": "Asia/Dubai", "emiraterne": "Asia/Dubai", "delhi": "Asia/Kolkata",
    "new delhi": "Asia/Kolkata", "mumbai": "Asia/Kolkata", "indien": "Asia/Kolkata", "india": "Asia/Kolkata",
    "bangkok": "Asia/Bangkok", "thailand": "Asia/Bangkok", "singapore": "Asia/Singapore", "hong kong": "Asia/Hong_Kong",
    "beijing": "Asia/Shanghai", "shanghai": "Asia/Shanghai", "kina": "Asia/Shanghai", "china": "Asia/Shanghai",
    "tokyo": "Asia/Tokyo", "japan": "Asia/Tokyo", "seoul": "Asia/Seoul", "korea": "Asia/Seoul",
    "sydkorea": "Asia/Seoul", "sydney": "Australia/Sydney", "melbourne": "Australia/Melbourne",
    "australien": "Australia/Sydney", "australia": "Australia/Sydney", "perth": "Australia/Perth",
    "auckland": "Pacific/Auckland", "new zealand": "Pacific/Auckland", "bali": "Asia/Makassar",
    "jakarta": "Asia/Jakarta", "manila": "Asia/Manila", "filippinerne": "Asia/Manila", "vietnam": "Asia/Ho_Chi_Minh",
}


def zone_for(place):
    p = re.sub(r"^(?:i|in)\s+", "", " ".join((place or "").lower().split()))
    if p in PLACES:
        return ZoneInfo(PLACES[p])
    want = p.replace(" ", "_")
    for z in sorted(available_timezones()):
        if z.lower() == want or z.lower().rsplit("/", 1)[-1] == want:
            return ZoneInfo(z)
    return None


@tool("time_now",
      "The current time, date, weekday and week number, here or in another city or country "
      "('hvad er klokken', 'hvilken dato er det', 'hvilken uge er det', 'hvad er klokken i Tokyo').",
      {"place": s("A city or country if he asks about the time somewhere else; leave out for here.")},
      core=True)
def time_now(ctx, a, alive):
    now = now_of(ctx)
    year, week, _ = now.isocalendar()
    place = _first(a, "place")
    here = f"{now:%H:%M} on {tp.say_date(now.date())}, week {week}"
    if not place or place.lower() in ("her", "here"):
        return {"result": f"It is {here}."}
    z = zone_for(place)
    if z is None:
        return {"ok": False, "result": f"I don't know the time zone of {place!r}. Here it is {here}."}
    there = now.replace(tzinfo=LOCAL_TZ).astimezone(z)
    diff = (there.utcoffset() - now.replace(tzinfo=LOCAL_TZ).utcoffset()).total_seconds() / 3600
    rel = "the same as Denmark" if diff == 0 else f"{abs(diff):g} hour{'s' * (abs(diff) != 1)} {'ahead of' if diff > 0 else 'behind'} Denmark"
    return {"result": f"In {place.strip().title()} it is {there:%H:%M} on {tp.say_date(there.date(), now.date())} "
                      f"({rel}). Here it is {now:%H:%M}."}


def _days_text(d, today):
    days = (d - today).days
    if days == 0:
        return "today"
    if days == 1:
        return "tomorrow, 1 day from today"
    if days == -1:
        return "yesterday, 1 day ago"
    span = f"{abs(days)} days"
    if abs(days) >= 14:
        w, r = divmod(abs(days), 7)
        span += f" ({w} weeks" + (f" and {r} day{'s' * (r != 1)}" if r else "") + ")"
    return f"{span} from today" if days > 0 else f"{span} ago"


@tool("date_info",
      "Calendar questions and date maths: days until a date or holiday ('hvor mange dage til jul'), "
      "which weekday a date is ('hvilken ugedag er 3. december'), the dates of a week number ('uge 45'), "
      "the date some days from now ('om 100 dage'), or the days between two dates.",
      {"date": s("The date in his own words and language, e.g. 'jul', '3. december', 'uge 45', 'om 100 dage', "
                 "'næste fredag'. Don't translate or convert it."),
       "until": s("A second date, only for 'how many days between X and Y'.")},
      ["date"])
def date_info(ctx, a, alive):
    now = now_of(ctx)
    today = now.date()
    phrase = _first(a, "date")
    w, used = _parse_any(now, phrase, _first(a, "heard"))
    if w is None:
        return {"ok": False, "result": f"I couldn't read a date in {phrase!r}. Ask him which date he means."}
    d = w.dt.date()
    y, week, _ = d.isocalendar()
    if a.get("until"):
        w2, _ = _parse_any(now, _first(a, "until"))
        if w2 is None:
            return {"ok": False, "result": f"I couldn't read the date {a['until']!r}. Ask him."}
        d2 = w2.dt.date()
        days = abs((d2 - d).days)
        return {"result": f"From {tp.say_date(d, today)} to {tp.say_date(d2, today)} is {days} days"
                          + (f" ({days // 7} weeks and {days % 7} days)" if days >= 14 else "") + "."}
    if re.search(r"\b(?:uge|week)\s+\S+", tp.norm(used)) and not re.search(tp.WEEKDAY, tp.norm(used)):
        end = d + timedelta(days=6)
        return {"result": f"Week {week} of {y} runs from {tp.say_date(d, today)} to {tp.say_date(end, today)}; "
                          f"it starts {_days_text(d, today)}."}
    name = f"{w.label} is " if w.label else ""
    when = f" at {w.dt:%H:%M}" if w.relative and w.has_time and abs((w.dt - now).total_seconds()) < 2 * 86400 else ""
    verb = "was" if d < today else "is"
    return {"result": f"{name or ('That ' + verb + ' ')}{tp.say_date(d, today if d.year == today.year else None)}{when}: "
                      f"{_days_text(d, today)}. It {verb} in week {week}."}


# ---- reminders and timers ---------------------------------------------------------------------

def _pending(c):
    return c.execute("select * from items where state='pending' order by due").fetchall()


def _describe_item(r, now):
    due = dt_of(r["due"])
    left = tp.say_span((due - now).total_seconds())
    what = {"timer": "timer", "focus": "focus timer", "reminder": "reminder"}.get(r["kind"], r["kind"])
    text = f" \"{r['text']}\"" if r["text"] else ""
    return f"[{r['id']}] {what}{text} {tp.say_when(due, now)} (in {left})"


@tool("set_reminder",
      "Set a reminder or a timer that Grace says out loud when it is due ('mind mig om at ringe til mor om en time', "
      "'sæt en timer på ti minutter', 'remind me on Friday at nine').",
      {"text": s("What to remind him of, short and in English (e.g. 'call mom'). Empty for a plain timer."),
       "when": s("When, in his own words and language exactly as he said it ('om tyve minutter', "
                 "'på fredag klokken ni', 'halv fire', 'i morgen tidlig'). Don't translate or convert it."),
       "kind": s("timer for a countdown (sæt en timer), else reminder.", ["reminder", "timer"])},
      ["when"], risk="visible", core=True)
def set_reminder(ctx, a, alive):
    now = now_of(ctx)
    text = _first(a, "text")
    heard = _first(a, "heard")  # his own words, when the engine adds them (as for calculate)
    w, used = _parse_any(now, _first(a, "when"), heard, text)
    if heard and used != heard and re.search(r"\b(?:halv|kvart)\b", tp.norm(heard)) \
            and not re.search(r"\b(?:halv|kvart)\b", tp.norm(used or "")):
        # Gemma translated a Danish clock phrase: "halv fire" became "half four" (16:30, British).
        w2, _ = _parse_any(now, heard)
        w = w2 or w
    if w is None:
        return {"ok": False, "result": f"I couldn't tell when from {a.get('when')!r}. Ask him when."}
    if w.dt <= now:
        return {"ok": False, "result": f"{tp.say_when(w.dt, now).capitalize()} has already passed. Ask him when he means."}
    if w.dt > now + timedelta(days=MAX_AHEAD_DAYS):
        return {"ok": False, "result": "That is more than a year ahead; reminders go up to a year. Say so."}
    if text and tp.parse(text, now) is not None:
        text = tp.without_time(text)  # "call mom in an hour" -> "call mom"
    kind = "timer" if a.get("kind") == "timer" or (not text and w.relative) else "reminder"
    span = tp.say_span((w.dt - now).total_seconds())
    if kind == "timer":
        say = f"The timer for {text} is done." if text else f"Your timer for {span} is done."
    else:
        say = f"Reminder: {text or 'you asked me to remind you now'}."
    note = "" if w.has_time else f" He gave no time, so it is set for {w.dt:%H:%M}; say so."
    if ctx.dry:
        return {"result": f"Dry run: would set a {kind} for {tp.say_when(w.dt, now)} (in {span})"
                          + (f": {text}" if text else "") + "." + note}
    with store(ctx).db() as c:
        dup = c.execute("select id from items where state='pending' and kind=? and coalesce(text,'')=? and due between ? and ?",
                        (kind, text, iso(w.dt - timedelta(minutes=1)), iso(w.dt + timedelta(minutes=1)))).fetchone()
        rid = dup["id"] if dup else c.execute(
            "insert into items(kind, text, due, created, say) values (?,?,?,?,?)",
            (kind, text, iso(w.dt), iso(now), say)).lastrowid
    return {"detail": {"id": rid, "due": iso(w.dt), "kind": kind},
            "result": f"{kind.capitalize()} set for {tp.say_when(w.dt, now)} (in {span})" + (f": {text}" if text else "")
                      + f". Id {rid}. I'll say it out loud then." + note}


@tool("list_reminders",
      "List his upcoming reminders, timers and focus timer, with the time left on each "
      "('hvilke påmindelser har jeg', 'hvor lang tid er der tilbage på timeren').", {})
def list_reminders(ctx, a, alive):
    now = now_of(ctx)
    with store(ctx).db() as c:
        rows = _pending(c)
    if not rows:
        return {"result": "He has no reminders or timers set."}
    lines = [_describe_item(r, now) for r in rows[:10]]
    more = f" And {len(rows) - 10} more." if len(rows) > 10 else ""
    return {"result": f"{len(rows)} upcoming: " + "; ".join(lines) + "." + more}


@tool("cancel_reminder",
      "Cancel one reminder or timer ('slet påmindelsen om mor', 'stop timeren').",
      {"which": s("Its id from list_reminders, or words from it ('mor', 'timer'). Empty if he has only one.")},
      risk="visible")
def cancel_reminder(ctx, a, alive):
    now = now_of(ctx)
    with store(ctx).db() as c:
        rows = [r for r in _pending(c) if r["kind"] != "focus"]
        hits, how = _pick(rows, _first(a, "which"), ("text", "kind"))
        if not hits:
            return {"ok": False, "result": "No reminder or timer matches that." + (
                " Upcoming: " + "; ".join(_describe_item(r, now) for r in rows[:5]) + "." if rows else " He has none.")}
        if len(hits) > 1:
            return {"ok": False, "result": "Several match; ask him which: " + "; ".join(_describe_item(r, now) for r in hits[:5]) + "."}
        r = hits[0]
        if ctx.dry:
            return {"result": f"Dry run: would cancel {_describe_item(r, now)}."}
        c.execute("update items set state='cancelled' where id=?", (r["id"],))
    return {"result": f"Cancelled {_describe_item(r, now)}."}


@tool("snooze_reminder",
      "Snooze a reminder or timer that just went off, or move one to a later time ('udsæt den ti minutter', 'snooze').",
      {"which": s("Its id or words from it; empty for the one that just went off."),
       "minutes": n("Minutes to snooze (default 10)."),
       "when": s("A new time in his own words instead of minutes ('klokken tre').")},
      risk="visible")
def snooze_reminder(ctx, a, alive):
    now = now_of(ctx)
    with store(ctx).db() as c:
        recent = c.execute("select * from items where state='fired' and kind!='focus' and fired_at>=? order by fired_at desc",
                           (iso(now - timedelta(hours=3)),)).fetchall()
        rows = recent + [r for r in _pending(c) if r["kind"] != "focus"]
        which = _first(a, "which")
        hits, _ = _pick(rows, which, ("text", "kind")) if which else (rows[:1], "latest")
        if not hits:
            return {"ok": False, "result": "There is no reminder or timer to snooze."}
        r = hits[0]
        if _first(a, "when"):
            w, _ = _parse_any(now, _first(a, "when"))
            if w is None or w.dt <= now:
                return {"ok": False, "result": f"I couldn't read a future time in {a['when']!r}. Ask him when."}
            due = w.dt
        else:
            try:
                mins = float(a.get("minutes") or 10)
            except (TypeError, ValueError):
                mins = 10
            due = now + timedelta(minutes=max(1, min(mins, 24 * 60)))
        if ctx.dry:
            return {"result": f"Dry run: would snooze it to {tp.say_when(due, now)}."}
        c.execute("update items set state='pending', due=?, snoozes=snoozes+1 where id=?", (iso(due), r["id"]))
        c.execute("update events set acked=? where source=? and ref=? and acked is null", (iso(now), r["kind"], r["id"]))
    label = f" \"{r['text']}\"" if r["text"] else ""
    return {"result": f"Snoozed the {r['kind']}{label} to {tp.say_when(due, now)} (in {tp.say_span((due - now).total_seconds())})."}


# ---- focus timer ------------------------------------------------------------------------------

@tool("focus_start",
      "Start a focus (pomodoro) timer: work for some minutes, then a break, for one or more rounds "
      "('start en fokustimer', 'pomodoro i en time').",
      {"minutes": n("Minutes of focus per round (default 25)."),
       "break_minutes": n("Minutes of break between rounds (default 5)."),
       "rounds": n("How many focus rounds (default 1, at most 8)."),
       "task": s("What he is working on, if he said, in English.")},
      risk="visible")
def focus_start(ctx, a, alive):
    now = now_of(ctx)

    def num(k, default, lo, hi):
        try:
            return max(lo, min(hi, float(a.get(k) or default)))
        except (TypeError, ValueError):
            return default
    focus, pause, rounds = num("minutes", 25, 1, 240), num("break_minutes", 5, 1, 60), int(num("rounds", 1, 1, 8))
    task = _first(a, "task")
    on = f" on {task}" if task else ""
    steps, t = [], now
    for i in range(1, rounds + 1):
        t += timedelta(minutes=focus)
        if i == rounds:
            total = focus * rounds
            steps.append((t, f"Your focus session is over: {total:g} minutes{on}. Time for a break."))
        else:
            steps.append((t, f"Focus round {i} of {rounds} is over. Take a {pause:g} minute break."))
            t += timedelta(minutes=pause)
            steps.append((t, f"Break's over. Round {i + 1} of {rounds}{on}."))
    plan = f"{rounds} round{'s' * (rounds != 1)} of {focus:g} minutes" + (f" with {pause:g} minute breaks" if rounds > 1 else "")
    if ctx.dry:
        return {"result": f"Dry run: would start a focus timer, {plan}, ending {tp.say_when(t, now)}."}
    with store(ctx).db() as c:
        if c.execute("select 1 from items where state='pending' and kind='focus'").fetchone():
            return {"ok": False, "result": "A focus timer is already running. Ask if he wants to stop it first (focus_stop)."}
        grp = f"focus-{iso(now)}"
        for due, say in steps:
            c.execute("insert into items(kind, text, due, created, say, grp) values ('focus',?,?,?,?,?)",
                      (task, iso(due), iso(now), say, grp))
    return {"result": f"Focus timer started{on}: {plan}. It ends {tp.say_when(t, now)}; I'll say when each part is over."}


@tool("focus_stop", "Stop the running focus (pomodoro) timer.", {}, risk="visible")
def focus_stop(ctx, a, alive):
    now = now_of(ctx)
    with store(ctx).db() as c:
        rows = c.execute("select * from items where state='pending' and kind='focus' order by due").fetchall()
        if not rows:
            return {"ok": False, "result": "No focus timer is running."}
        started = dt_of(rows[0]["created"])
        if ctx.dry:
            return {"result": "Dry run: would stop the focus timer."}
        c.execute("update items set state='cancelled' where grp=? and state='pending'", (rows[0]["grp"],))
    return {"result": f"Stopped the focus timer after {tp.say_span((now - started).total_seconds())}."}


# ---- todos and notes --------------------------------------------------------------------------

def _list_name(a):
    return (_first(a, "list") or "todo").lower()


def _todo_line(r):
    return f"[{r['id']}] {r['text']}"


@tool("todo_add",
      "Add an item to his todo list or another list such as shopping ('skriv mælk på indkøbslisten', 'tilføj en todo').",
      {"text": s("The item, short."), "list": s("Which list, e.g. 'indkøb' (shopping); default 'todo'.")},
      ["text"], risk="visible")
def todo_add(ctx, a, alive):
    now = now_of(ctx)
    text, lst = _first(a, "text"), _list_name(a)
    if not text:
        return {"ok": False, "result": "No item was given. Ask what to add."}
    if ctx.dry:
        return {"result": f"Dry run: would add \"{text}\" to the {lst} list."}
    with store(ctx).db() as c:
        dup = c.execute("select id from notes where kind='todo' and list=? and lower(text)=lower(?) and done_at is null",
                        (lst, text)).fetchone()
        if dup:
            return {"result": f"\"{text}\" is already on the {lst} list (id {dup['id']})."}
        rid = c.execute("insert into notes(kind, list, text, created) values ('todo',?,?,?)", (lst, text, iso(now))).lastrowid
        count = c.execute("select count(*) from notes where kind='todo' and list=? and done_at is null", (lst,)).fetchone()[0]
    return {"result": f"Added \"{text}\" to the {lst} list (id {rid}); it has {count} open item{'s' * (count != 1)}."}


@tool("todo_list", "Read his todo list or another list ('hvad står der på min todo liste', 'hvad skal jeg købe').",
      {"list": s("Which list; leave out for the todo list. 'all' for every list."),
       "include_done": b("Also the items that are done.")})
def todo_list(ctx, a, alive):
    lst = _list_name(a)
    with store(ctx).db() as c:
        if lst in ("all", "alle"):
            rows = c.execute("select * from notes where kind='todo' and done_at is null order by list, id").fetchall()
            if not rows:
                return {"result": "All his lists are empty."}
            by = {}
            for r in rows:
                by.setdefault(r["list"], []).append(_todo_line(r))
            return {"result": " ".join(f"{k} ({len(v)}): {'; '.join(v[:10])}." for k, v in by.items())}
        rows = c.execute("select * from notes where kind='todo' and list=? and done_at is null order by id", (lst,)).fetchall()
        done = c.execute("select * from notes where kind='todo' and list=? and done_at is not null order by done_at desc",
                         (lst,)).fetchall()
        lists = [r[0] for r in c.execute("select distinct list from notes where kind='todo' and done_at is null")]
    other = [x for x in lists if x != lst]
    tail = f" Other lists: {', '.join(other)}." if other else ""
    if not rows:
        return {"result": f"The {lst} list is empty." + tail}
    text = f"{len(rows)} open on the {lst} list: " + "; ".join(_todo_line(r) for r in rows[:15]) + "."
    if a.get("include_done") and done:
        text += " Done: " + "; ".join(_todo_line(r) for r in done[:10]) + "."
    elif done:
        text += f" ({len(done)} done.)"
    return {"result": text + tail}


@tool("todo_done", "Mark a todo item as done ('jeg har købt mælk', 'marker tandlæge som færdig').",
      {"which": s("Its id or words from it."), "list": s("Which list; default 'todo'.")},
      ["which"], risk="visible")
def todo_done(ctx, a, alive):
    now = now_of(ctx)
    lst = _list_name(a)
    with store(ctx).db() as c:
        rows = c.execute("select * from notes where kind='todo' and done_at is null and (list=? or ?='')",
                         (lst, "" if not a.get("list") else lst)).fetchall()
        hits, _ = _pick(rows, _first(a, "which"), ("text",))
        if len(hits) != 1:
            pool = hits or rows
            return {"ok": False, "result": ("Several match; ask which: " if hits else "No open item matches that. Open: ")
                    + "; ".join(_todo_line(r) for r in pool[:8]) + "." if pool else "The list is empty."}
        r = hits[0]
        if ctx.dry:
            return {"result": f"Dry run: would mark \"{r['text']}\" as done."}
        c.execute("update notes set done_at=? where id=?", (iso(now), r["id"]))
        left = c.execute("select count(*) from notes where kind='todo' and list=? and done_at is null", (r["list"],)).fetchone()[0]
    return {"result": f"Marked \"{r['text']}\" as done; {left} left on the {r['list']} list."}


def _bound_count(sql, args):
    """For a destructive summary, which only sees the args: the count in the server's store."""
    try:
        with store_at(BOUND["dir"] or DEFAULT_DATA_DIR).db() as c:
            return c.execute(sql, args).fetchone()[0]
    except Exception:
        return None


def _todo_delete_summary(a):
    lst = (a.get("list") or "").lower()
    if a.get("all") or (a.get("which") or "").strip().lower() in ("all", "alle", "alt"):
        if lst in ("", "all", "alle"):
            k = _bound_count("select count(*) from notes where kind='todo'", ())
            return f"delete all {k if k is not None else 'of his'} todo items on every list"
        k = _bound_count("select count(*) from notes where kind='todo' and list=?", (lst,))
        return f"delete all {k if k is not None else 'the'} items on the {lst} list"
    return f"delete the todo item \"{a.get('which')}\""


@tool("todo_delete",
      "Delete todo items for good: one item, or all of a list ('slet alle mine todos', 'ryd indkøbslisten').",
      {"which": s("Its id or words from it; 'all' for everything."), "list": s("Which list; leave out for all lists when deleting all."),
       "all": b("True to delete every item (on the list, or on every list).")},
      risk="destructive", summary=_todo_delete_summary)
def todo_delete(ctx, a, alive):
    lst = (a.get("list") or "").lower()
    every = a.get("all") or (a.get("which") or "").strip().lower() in ("all", "alle", "alt")
    with store(ctx).db() as c:
        if every:
            q, args = ("select * from notes where kind='todo'", ()) if lst in ("", "all", "alle") else \
                ("select * from notes where kind='todo' and list=?", (lst,))
            rows = c.execute(q, args).fetchall()
        else:
            pool = c.execute("select * from notes where kind='todo'" + (" and list=?" if lst else ""), (lst,) if lst else ()).fetchall()
            rows, _ = _pick(pool, _first(a, "which"), ("text",))
            if len(rows) != 1:
                return {"ok": False, "result": ("Several match, nothing was deleted; ask which: " if rows else
                                                "No item matches that; nothing was deleted. ")
                        + "; ".join(_todo_line(r) for r in (rows or pool)[:8]) + "."}
        if not rows:
            return {"result": "There was nothing to delete."}
        if ctx.dry:
            return {"result": f"Dry run: would delete {len(rows)} item{'s' * (len(rows) != 1)}."}
        c.executemany("delete from notes where id=?", [(r["id"],) for r in rows])
    return {"result": f"Deleted {len(rows)} item{'s' * (len(rows) != 1)}"
                      + (f": {rows[0]['text']}" if len(rows) == 1 else "") + "."}


@tool("note_add", "Save a short note for later ('skriv en note om at ...', 'husk at koden til cyklen er ...').",
      {"text": s("The note, in his words."), "title": s("A few words to find it by, optional.")},
      ["text"], risk="visible")
def note_add(ctx, a, alive):
    now = now_of(ctx)
    text = _first(a, "text")
    if not text:
        return {"ok": False, "result": "The note is empty. Ask what to write."}
    if ctx.dry:
        return {"result": f"Dry run: would save the note \"{text[:80]}\"."}
    with store(ctx).db() as c:
        rid = c.execute("insert into notes(kind, list, title, text, created) values ('note','notes',?,?,?)",
                        (_first(a, "title"), text, iso(now))).lastrowid
    return {"result": f"Saved the note (id {rid})."}


@tool("note_find", "Find or read his saved notes ('hvad skrev jeg om cyklen', 'læs mine noter').",
      {"query": s("Words to look for; leave out for the latest notes.")})
def note_find(ctx, a, alive):
    now = now_of(ctx)
    with store(ctx).db() as c:
        rows = c.execute("select * from notes where kind='note' order by id desc").fetchall()
    q = _first(a, "query")
    hits, _ = _pick(rows, q, ("title", "text")) if q else (rows, "all")
    if not hits:
        return {"result": f"No note matches {q!r}." if q else "He has no notes."}
    out = [f"[{r['id']}] {tp.say_when(dt_of(r['created']), now).split(' at ')[0]}: {r['text'][:300]}" for r in hits[:5]]
    return {"result": f"{len(hits)} note{'s' * (len(hits) != 1)}: " + " | ".join(out)}


@tool("note_delete", "Delete a saved note for good.", {"which": s("Its id or words from it.")}, ["which"],
      risk="destructive", summary=lambda a: f"delete the note \"{a.get('which')}\"")
def note_delete(ctx, a, alive):
    with store(ctx).db() as c:
        rows = c.execute("select * from notes where kind='note'").fetchall()
        hits, _ = _pick(rows, _first(a, "which"), ("title", "text"))
        if len(hits) != 1:
            return {"ok": False, "result": ("Several notes match; nothing was deleted. Ask which: " if hits else
                                            "No note matches that; nothing was deleted.")
                    + "; ".join(f"[{r['id']}] {r['text'][:60]}" for r in hits[:5])}
        if ctx.dry:
            return {"result": f"Dry run: would delete the note \"{hits[0]['text'][:80]}\"."}
        c.execute("delete from notes where id=?", (hits[0]["id"],))
    return {"result": f"Deleted the note \"{hits[0]['text'][:80]}\"."}


# ---- conversion -------------------------------------------------------------------------------

@tool("convert_units",
      "Convert units: length, weight, volume and cooking measures, temperature, speed, area, data, energy "
      "('hvor mange centimeter er fem fod', 'hvad er 180 fahrenheit i celsius', 'tre dl i spiseskeer').",
      {"value": s("The number, as he said it ('5', '3,5', 'fem')."), "from_unit": s("The unit he has, e.g. 'feet', 'dl'."),
       "to_unit": s("The unit he wants, e.g. 'cm', 'spsk'.")},
      ["value", "from_unit", "to_unit"])
def convert_units(ctx, a, alive):
    try:
        _, text = tc.convert_units(a.get("value"), _first(a, "from_unit"), _first(a, "to_unit"))
    except ValueError as e:
        return {"ok": False, "result": f"Couldn't convert: {e}. Say so."}
    return {"result": text}


@tool("convert_currency",
      "Convert money between currencies at today's ECB rate ('hvad er 100 euro i kroner', 'hvor meget er 50 dollars').",
      {"amount": s("The amount, as he said it."), "from_currency": s("The currency he has, e.g. 'euro', 'USD', 'svenske kroner'."),
       "to_currency": s("The currency he wants; default Danish kroner.")},
      ["amount", "from_currency"], slow=True)
def convert_currency(ctx, a, alive):
    try:
        _, text = tc.convert_currency(ctx.data_dir, now_of(ctx), a.get("amount"), _first(a, "from_currency"),
                                      _first(a, "to_currency") or "DKK")
    except (ValueError, RuntimeError) as e:
        return {"ok": False, "result": f"Couldn't convert: {e}. Say so."}
    return {"result": text}


# ---- watchers ---------------------------------------------------------------------------------

WAIT_FOR = {("file", "download_done"): "the download finishing", ("file", "new_file"): "a new file appearing",
            ("file", "changed"): "a change", ("file", "deleted"): "it being deleted",
            ("process", "starts"): "it starting", ("process", "stops"): "it stopping",
            ("web", "changes"): "the page changing", ("web", "contains"): "the text appearing",
            ("web", "loses"): "the text disappearing", ("github", "finished"): "the run finishing",
            ("url", "up"): "it coming up"}
# Seconds between checks: (default, minimum). Local checks are cheap; the web is someone else's server.
INTERVALS = {"file": (5, 2), "process": (5, 2), "url": (60, 60), "url_local": (5, 2), "web": (300, 60),
             "github": (30, 15)}


def _wait_for(w):
    return WAIT_FOR.get((w["kind"], w["cond"]), "it")


def home():
    return Path.home()


def folder_aliases():
    h = home()
    return {"skrivebord": h / "Desktop", "skrivebordet": h / "Desktop", "desktop": h / "Desktop",
            "dokumenter": h / "Documents", "documents": h / "Documents", "overførsler": h / "Downloads",
            "downloads": h / "Downloads", "download": h / "Downloads", "billeder": h / "Pictures",
            "pictures": h / "Pictures", "videoer": h / "Videos", "videos": h / "Videos", "musik": h / "Music",
            "music": h / "Music"}


def resolve_path(text):
    t = (text or "").strip().strip('"')
    key = t.lower().removeprefix("min ").removeprefix("my ").removeprefix("mappen ").removesuffix(" folder").removesuffix(" mappen").strip()
    aliases = folder_aliases()
    if key in aliases:
        return aliases[key]
    p = Path(os.path.expandvars(os.path.expanduser(t)))
    return p if p.is_absolute() else home() / p


def _interval(kind, a, key="seconds"):
    default, low = INTERVALS[kind]
    try:
        v = float(a.get(key)) * (60 if key == "minutes" else 1) if a.get(key) not in (None, "") else default
    except (TypeError, ValueError):
        v = default
    return max(low, v)


def _watch(ctx, kind, target, cond, arg, label, interval, a):
    now = now_of(ctx)
    try:
        hours = max(0.1, min(168.0, float(a.get("hours") or 24)))
    except (TypeError, ValueError):
        hours = 24.0
    w = {"kind": kind, "target": str(target), "cond": cond, "arg": arg or "", "label": label}
    every = tp.say_span(interval)
    if ctx.dry:
        return {"result": f"Dry run: would watch {label} for {_wait_for(w)}, checking every {every}, for up to {hours:g} hours."}
    st = store(ctx)
    with st.db() as c:
        dup = c.execute("select id from watchers where status='active' and kind=? and target=? and cond=? and arg=?",
                        (kind, w["target"], cond, w["arg"])).fetchone()
    if dup:
        return {"result": f"I'm already watching {label} for {_wait_for(w)} (id {dup['id']})."}
    try:
        state, already = tw.STARTS[kind](w, now)
    except Exception as e:
        return {"ok": False, "result": f"Couldn't start watching {label}: {e}. Say so."}
    if already:
        return {"already": True, "result": already + " So there's nothing to wait for; tell him."}
    with st.db() as c:
        rid = c.execute("insert into watchers(kind, target, cond, arg, label, interval, created, expires, next_check, state) "
                        "values (?,?,?,?,?,?,?,?,?,?)",
                        (kind, w["target"], cond, w["arg"], label, interval, iso(now), iso(now + timedelta(hours=hours)),
                         iso(now + timedelta(seconds=interval)), json.dumps(state))).lastrowid
    return {"detail": {"id": rid},
            "result": f"Watching {label} (id {rid}): I'll tell him when it happens ({_wait_for(w)}). "
                      f"Checked every {every}, for up to {hours:g} hours."}


HOURS = n("How many hours to keep watching (default 24, at most 168).")


@tool("watch_file",
      "Watch a folder or file and tell him when a download finishes, a new file appears, it changes or is deleted "
      "('sig til når min download er færdig', 'giv besked når der kommer en ny fil i mappen').",
      {"path": s("The folder or file: a full path, or 'downloads', 'skrivebord', 'dokumenter'. Default downloads."),
       "event": s("What to wait for.", ["download_done", "new_file", "changed", "deleted"]),
       "file_name": s("Part of the file name to look for, optional (e.g. 'pdf', 'rapport')."), "hours": HOURS},
      ["event"], risk="visible")
def watch_file(ctx, a, alive):
    event = a.get("event") if a.get("event") in ("download_done", "new_file", "changed", "deleted") else "changed"
    path = resolve_path(_first(a, "path") or "downloads")
    name = _first(a, "file_name")
    if not path.exists():
        if event in ("new_file", "download_done") and path.parent.is_dir():
            path, name = path.parent, name or path.name  # wait for that file to appear in its folder
        else:
            return {"ok": False, "result": f"{path} doesn't exist. Ask him which folder or file he means."}
    label = f"the {path.name} folder" if path.is_dir() else path.name
    return _watch(ctx, "file", path, event, name, label, _interval("file", a), a)


@tool("watch_process",
      "Watch a program and tell him when it starts or stops ('sig til når renderingen er færdig og blender lukker', "
      "'giv besked når spillet er lukket').",
      {"program": s("The program's name, e.g. 'blender', 'chrome', 'obs64'."),
       "event": s("starts or stops.", ["starts", "stops"]), "hours": HOURS},
      ["program", "event"], risk="visible")
def watch_process(ctx, a, alive):
    name = re.sub(r"(?i)\.exe$", "", _first(a, "program", "name"))
    if not name:
        return {"ok": False, "result": "No program was named. Ask which one."}
    event = "starts" if a.get("event") == "starts" else "stops"
    return _watch(ctx, "process", name.lower(), event, "", name[0].upper() + name[1:], _interval("process", a), a)


@tool("watch_web",
      "Watch a web page and tell him when it changes, or when some text appears on it or disappears "
      "('sig til når billetterne kommer til salg', 'hold øje med om siden ændrer sig'). Checks at most once a minute.",
      {"url": s("The page's address."), "event": s("What to wait for.", ["changes", "contains", "loses"]),
       "text": s("The text to look for, for contains or loses (e.g. 'udsolgt', 'på lager')."),
       "minutes": n("Minutes between checks (default 5, at least 1)."), "hours": HOURS},
      ["url", "event"], risk="visible")
def watch_web(ctx, a, alive):
    url = _first(a, "url")
    if not re.match(r"(?i)^https?://", url):
        url = "https://" + url if url else ""
    if not url:
        return {"ok": False, "result": "No page was given. Ask for the address."}
    event = a.get("event") if a.get("event") in ("changes", "contains", "loses") else "changes"
    text = _first(a, "text")
    if event != "changes" and not text:
        return {"ok": False, "result": "Which text should I look for on the page? Ask him."}
    host = urllib.parse.urlparse(url).hostname or url
    kind = "url_local" if tw.is_local(url) else "web"
    interval = max(_interval(kind, a, "minutes"), INTERVALS[kind][1])
    return _watch(ctx, "web", url, event, text, f"the page {host}", interval, a)


@tool("watch_github",
      "Watch a GitHub Actions run and tell him when it finishes and whether it passed ('sig til når CI er færdig').",
      {"repo": s("The repository as owner/name, or a local folder with the checkout."),
       "run": s("A run id or a branch name, optional; default the latest run that is still going."), "hours": HOURS},
      ["repo"], risk="visible")
def watch_github(ctx, a, alive):
    repo = _first(a, "repo")
    if not repo:
        return {"ok": False, "result": "Which repository? Ask him (owner/name)."}
    short = repo
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        repo = str(resolve_path(repo))
        short = Path(repo).name
    return _watch(ctx, "github", repo, "finished", _first(a, "run"), f"the GitHub Actions run in {short}",
                  _interval("github", a), a)


@tool("watch_url",
      "Watch an address and tell him when it comes up and answers (a website that is down, or a local dev server "
      "such as localhost:3000).",
      {"url": s("The address, e.g. 'localhost:3000' or 'https://example.com'."), "hours": HOURS},
      ["url"], risk="visible")
def watch_url(ctx, a, alive):
    url = _first(a, "url")
    if not url:
        return {"ok": False, "result": "No address was given. Ask for it."}
    if not re.match(r"(?i)^https?://", url):
        url = ("http://" if re.match(r"(?i)^(localhost|127\.|\[::1\])", url) else "https://") + url
    kind = "url_local" if tw.is_local(url) else "url"
    return _watch(ctx, "url", url, "up", "", f"the address {url}", _interval(kind, a), a)


@tool("list_watchers", "List what Grace is watching for him right now, and what fired lately.", {})
def list_watchers(ctx, a, alive):
    now = now_of(ctx)
    with store(ctx).db() as c:
        active = c.execute("select * from watchers where status='active' order by id").fetchall()
        recent = c.execute("select * from watchers where status!='active' and ended>=? order by ended desc",
                           (iso(now - timedelta(hours=12)),)).fetchall()
    if not active and not recent:
        return {"result": "Nothing is being watched."}
    lines = [f"[{w['id']}] {w['label']}: waiting for {_wait_for(w)}, checked every {tp.say_span(w['interval'])}, "
             f"until {tp.say_when(dt_of(w['expires']), now)}" for w in active[:10]]
    text = (f"Watching {len(active)}: " + "; ".join(lines) + ".") if active else "Nothing is being watched now."
    if recent:
        text += " Lately: " + "; ".join(f"[{w['id']}] {w['status']}: {w['message']}" for w in recent[:3])
    return {"result": text}


@tool("stop_watcher", "Stop watching something ('stop med at holde øje med downloads').",
      {"which": s("Its id from list_watchers or words from it; empty if only one is running.")}, risk="visible")
def stop_watcher(ctx, a, alive):
    now = now_of(ctx)
    with store(ctx).db() as c:
        rows = c.execute("select * from watchers where status='active' order by id").fetchall()
        hits, _ = _pick(rows, _first(a, "which"), ("label", "target", "arg", "kind"))
        if len(hits) != 1:
            if not rows:
                return {"ok": False, "result": "Nothing is being watched."}
            return {"ok": False, "result": ("Several match; ask which: " if hits else "None matches that. Watching: ")
                    + "; ".join(f"[{w['id']}] {w['label']}" for w in (hits or rows)[:6]) + "."}
        w = hits[0]
        if ctx.dry:
            return {"result": f"Dry run: would stop watching {w['label']}."}
        c.execute("update watchers set status='stopped', ended=?, message=? where id=?",
                  (iso(now), "stopped by him", w["id"]))
    return {"result": f"Stopped watching {w['label']}."}
