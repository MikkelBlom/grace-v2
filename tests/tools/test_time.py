"""The time group (services/grace_tools/time.py): clock and calendar, reminders and timers, todos
and notes, the focus timer, conversion and watchers. Every test runs on a fake clock and drives
the scheduler synchronously with gt.tick(); nothing touches the internet or real folders."""

import os
from datetime import datetime, timedelta

import pytest

os.environ.setdefault("GRACE_TOOLS_SCHEDULER", "0")

from conftest import RECORDED, call  # noqa: E402
from grace_tools import time as gt  # noqa: E402
from grace_tools import time_convert as tc  # noqa: E402
from grace_tools import time_parse as tp  # noqa: E402
from grace_tools import time_watch as tw  # noqa: E402

THU = datetime(2026, 10, 22, 10, 0)  # a Thursday, week 43


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, **kw):
        self.t += timedelta(**kw)
        return self.t


@pytest.fixture
def clock(ctx):
    c = Clock(THU)
    ctx.now = c
    return c


def tick(ctx, checks=None):
    return gt.tick(gt.store(ctx), gt.now_of(ctx), checks)


def says(ctx):
    return [e["say"] for e in gt.pending_events(ctx)]


def ack_all(ctx):
    for e in gt.pending_events(ctx):
        gt.ack_event(ctx, e["id"])


def restart():
    """A new server process: no cached store objects, the same database file."""
    gt.STORES.clear()


def test_scheduler_thread_is_off_in_tests():
    assert not gt.SCHEDULER.threads


def test_group_shape():
    import grace_tools
    tools = [t for t in grace_tools.load().values() if t.group == "time"]
    assert sum(t.core for t in tools) <= 2
    assert {t.name for t in tools if t.core} == {"time_now", "set_reminder"}
    risky = {t.name for t in tools if t.risk == "destructive"}
    assert risky == {"todo_delete", "note_delete"}


# ---- the parser -------------------------------------------------------------------------------

D = lambda *a: datetime(*a)  # noqa: E731


@pytest.mark.parametrize("phrase, want", [
    ("om tyve minutter", D(2026, 10, 22, 10, 20)),
    ("om en time", D(2026, 10, 22, 11, 0)),
    ("om halvanden time", D(2026, 10, 22, 11, 30)),
    ("om en halv time", D(2026, 10, 22, 10, 30)),
    ("om et kvarter", D(2026, 10, 22, 10, 15)),
    ("om tre kvarter", D(2026, 10, 22, 10, 45)),
    ("om 2 timer og 30 minutter", D(2026, 10, 22, 12, 30)),
    ("om to og en halv time", D(2026, 10, 22, 12, 30)),
    ("om fem og fyrre minutter", D(2026, 10, 22, 10, 45)),
    ("om femogfyrre minutter", D(2026, 10, 22, 10, 45)),
    ("halv fire", D(2026, 10, 22, 15, 30)),  # the Danish rule: half *to* four
    ("klokken halv fire", D(2026, 10, 22, 15, 30)),
    ("kvart over tre", D(2026, 10, 22, 15, 15)),
    ("kvart i fire", D(2026, 10, 22, 15, 45)),
    ("fem over halv fire", D(2026, 10, 22, 15, 35)),
    ("ti i halv fire", D(2026, 10, 22, 15, 20)),
    ("tyve over to", D(2026, 10, 22, 14, 20)),
    ("fem minutter i otte", D(2026, 10, 22, 19, 55)),
    ("halv et", D(2026, 10, 22, 12, 30)),
    ("klokken ni", D(2026, 10, 22, 21, 0)),  # 09:00 has passed; he means tonight
    ("klokken 11", D(2026, 10, 22, 11, 0)),
    ("klokken 14:30", D(2026, 10, 22, 14, 30)),
    ("kl. 16.45", D(2026, 10, 22, 16, 45)),
    ("klokken ni i aften", D(2026, 10, 22, 21, 0)),
    ("klokken tre i eftermiddag", D(2026, 10, 22, 15, 0)),
    ("i morgen tidlig", D(2026, 10, 23, 7, 0)),
    ("klokken syv i morgen tidlig", D(2026, 10, 23, 7, 0)),
    ("i morgen klokken tre", D(2026, 10, 23, 15, 0)),
    ("i morgen klokken 9", D(2026, 10, 23, 9, 0)),
    ("i morgen", D(2026, 10, 23, 9, 0)),
    ("i aften", D(2026, 10, 22, 19, 0)),
    ("på fredag klokken ni", D(2026, 10, 23, 9, 0)),
    ("på mandag", D(2026, 10, 26, 9, 0)),
    ("på torsdag", D(2026, 10, 29, 9, 0)),  # said on a Thursday: next week's
    ("søndag aften", D(2026, 10, 25, 19, 0)),
    ("fredag i næste uge", D(2026, 10, 30, 9, 0)),
    ("i overmorgen klokken halv otte", D(2026, 10, 24, 7, 30)),
    ("i nat klokken to", D(2026, 10, 23, 2, 0)),
    ("midnat", D(2026, 10, 23, 0, 0)),
    ("den 3. december klokken 12", D(2026, 12, 3, 12, 0)),
    ("den tredje december", D(2026, 12, 3, 9, 0)),
    ("24/12", D(2026, 12, 24, 9, 0)),
    ("1. januar", D(2027, 1, 1, 9, 0)),
    ("juleaften", D(2026, 12, 24, 9, 0)),
    ("uge 45", D(2026, 11, 2, 9, 0)),
    ("fredag i uge 45", D(2026, 11, 6, 9, 0)),
    ("i weekenden", D(2026, 10, 24, 9, 0)),
    ("om en uge", D(2026, 10, 29, 10, 0)),
    ("in 20 minutes", D(2026, 10, 22, 10, 20)),
    ("in an hour and a half", D(2026, 10, 22, 11, 30)),
    ("half an hour", D(2026, 10, 22, 10, 30)),
    ("tomorrow at 9am", D(2026, 10, 23, 9, 0)),
    ("tomorrow morning", D(2026, 10, 23, 8, 0)),
    ("on friday at 7 pm", D(2026, 10, 23, 19, 0)),
    ("half past three", D(2026, 10, 22, 15, 30)),  # English half is half *past*
    ("quarter to four", D(2026, 10, 22, 15, 45)),
    ("december 3rd", D(2026, 12, 3, 9, 0)),
    ("set a timer for ten minutes", D(2026, 10, 22, 10, 10)),
    ("sæt en timer på ti minutter", D(2026, 10, 22, 10, 10)),
    ("2026-11-05T08:15", D(2026, 11, 5, 8, 15)),
])
def test_phrases(phrase, want):
    w = tp.parse(phrase, THU)
    assert w is not None, phrase
    assert w.dt == want, (phrase, w)


@pytest.mark.parametrize("now, phrase, want", [
    (D(2026, 10, 22, 16, 0), "halv fire", D(2026, 10, 23, 15, 30)),  # not 03:30 in the night
    (D(2026, 10, 22, 21, 0), "klokken syv", D(2026, 10, 23, 7, 0)),
    (D(2026, 10, 22, 15, 0), "klokken 14", D(2026, 10, 23, 14, 0)),
    (D(2026, 10, 22, 20, 0), "tidlig", D(2026, 10, 23, 7, 0)),
    (D(2026, 12, 28, 12, 0), "jul", D(2027, 12, 24, 9, 0)),
    (D(2026, 12, 28, 12, 0), "uge 2", D(2027, 1, 11, 9, 0)),
])
def test_phrases_depend_on_now(now, phrase, want):
    assert tp.parse(phrase, now).dt == want


def test_past_phrases():
    assert tp.parse("i fredags", THU).dt.date() == D(2026, 10, 16).date()
    assert tp.parse("sidste mandag", THU).dt.date() == D(2026, 10, 19).date()
    assert tp.parse("for 3 dage siden", THU).dt.date() == D(2026, 10, 19).date()


def test_no_time_in_it():
    for text in ("hej grace", "ring til mor", "hvad laver du", ""):
        assert tp.parse(text, THU) is None, text


def test_holidays():
    assert tp.easter(2027) == D(2027, 3, 28).date()
    assert tp.parse("påske", THU).dt.date() == D(2027, 3, 28).date()
    assert tp.parse("langfredag", THU).dt.date() == D(2027, 3, 26).date()
    assert tp.parse("sankthans", THU).dt.date() == D(2027, 6, 23).date()
    assert tp.parse("nytårsaften", THU).dt.date() == D(2026, 12, 31).date()


@pytest.mark.parametrize("text, want", [
    ("ring til mor om en time", "ring til mor"),
    ("mind mig om at ringe til mor klokken fem", "ringe til mor"),
    ("remind me to call mom tomorrow at 9am", "call mom"),
    ("tag kagen ud halv fire", "tag kagen ud"),
])
def test_without_time(text, want):
    assert tp.without_time(text) == want


def test_spans():
    assert tp.say_span(600) == "10 minutes"
    assert tp.say_span(3725) == "1 hour and 2 minutes"
    assert tp.say_span(45) == "45 seconds"
    assert tp.parse_duration("halvanden time") == 5400


# ---- the clock and calendar -------------------------------------------------------------------

def test_time_now(ctx, clock):
    out = call(ctx, "time_now")
    assert out["result"] == "It is 10:00 on Thursday 22 October 2026, week 43."
    assert out["risk"] == "read"


def test_time_elsewhere(ctx, clock):
    assert "04:00" in call(ctx, "time_now", place="New York")["result"]
    assert "6 hours behind Denmark" in call(ctx, "time_now", place="new york")["result"]
    assert "17:00" in call(ctx, "time_now", place="i Tokyo")["result"]
    out = call(ctx, "time_now", place="Atlantis")
    assert not out["ok"] and "10:00" in out["result"]


def test_days_until_christmas(ctx, clock):
    out = call(ctx, "date_info", date="til jul")
    assert "Christmas Eve" in out["result"] and "63 days (9 weeks) from today" in out["result"]
    assert "week 52" in out["result"]


def test_weekday_of_a_date(ctx, clock):
    assert "Thursday 3 December" in call(ctx, "date_info", date="hvilken ugedag er 3. december")["result"]


def test_week_numbers(ctx, clock):
    out = call(ctx, "date_info", date="uge 45")
    assert "Week 45 of 2026 runs from Monday 2 November to Sunday 8 November" in out["result"]
    assert "week 43" in call(ctx, "date_info", date="i dag")["result"]


def test_date_maths(ctx, clock):
    assert "Saturday 30 January 2027" in call(ctx, "date_info", date="om 100 dage")["result"]
    assert "63 days" in call(ctx, "date_info", date="i dag", until="jul")["result"]
    out = call(ctx, "date_info", date="i fredags")
    assert "6 days ago" in out["result"] and "was" in out["result"]
    assert not call(ctx, "date_info", date="blabla")["ok"]


# ---- reminders and timers ---------------------------------------------------------------------

def test_reminder_fires_on_time(ctx, clock):
    out = call(ctx, "set_reminder", text="call mom", when="om tyve minutter")
    assert out["risk"] == "visible" and "today at 10:20" in out["result"] and "call mom" in out["result"]
    clock.advance(minutes=19, seconds=59)
    tick(ctx)
    assert says(ctx) == []
    clock.advance(seconds=1)
    tick(ctx)
    events = gt.pending_events(ctx)
    assert [(e["priority"], e["say"], e["source"]) for e in events] == [("now", "Reminder: call mom.", "reminder")]
    assert gt.ack_event(ctx, events[0]["id"]) and not gt.ack_event(ctx, events[0]["id"])
    assert says(ctx) == []
    tick(ctx)
    assert says(ctx) == []  # fires once


def test_pending_events_fires_timers_itself(ctx, clock):
    call(ctx, "set_reminder", when="om fem minutter", kind="timer")
    clock.advance(minutes=5)
    assert says(ctx) == ["Your timer for 5 minutes is done."]


def test_halv_fire_is_half_past_three(ctx, clock):
    out = call(ctx, "set_reminder", text="take the cake out", when="halv fire")
    assert "today at 15:30" in out["result"]


def test_heard_wins_over_a_translated_time(ctx, clock):
    # Gemma turned "halv fire" into "half four" (British: 16:30); his own words say 15:30.
    out = call(ctx, "set_reminder", text="pick up Ida", when="half four", heard="mind mig om at hente ida halv fire")
    assert "15:30" in out["result"]


def test_heard_is_not_used_when_when_is_fine(ctx, clock):
    out = call(ctx, "set_reminder", when="om ti minutter", kind="timer",
               heard="mind mig om mor klokken fem og sæt en timer på ti minutter")
    assert "10:10" in out["result"]


def test_timer_from_danish(ctx, clock):
    out = call(ctx, "set_reminder", when="sæt en timer på ti minutter")
    assert out["detail"]["kind"] == "timer" and "10:10" in out["result"]
    clock.advance(minutes=10)
    assert says(ctx) == ["Your timer for 10 minutes is done."]


def test_timer_with_a_label(ctx, clock):
    call(ctx, "set_reminder", text="the eggs", when="om syv minutter", kind="timer")
    clock.advance(minutes=7)
    assert says(ctx) == ["The timer for the eggs is done."]


def test_text_with_the_time_in_it(ctx, clock):
    out = call(ctx, "set_reminder", text="call mom in an hour", when="in an hour")
    assert out["result"].endswith("I'll say it out loud then.") and ": call mom." in out["result"]


def test_date_without_time_defaults_to_nine(ctx, clock):
    out = call(ctx, "set_reminder", text="pay rent", when="på fredag")
    assert "tomorrow at 09:00" in out["result"] and "no time" in out["result"]


def test_bad_times(ctx, clock):
    assert not call(ctx, "set_reminder", text="x", when="i går klokken 9")["ok"]
    out = call(ctx, "set_reminder", text="x", when="snart engang")
    assert not out["ok"] and "Ask him when" in out["result"]
    assert not call(ctx, "set_reminder", text="x", when="om 3 år")["ok"]


def test_duplicate_reminder_is_not_doubled(ctx, clock):
    a = call(ctx, "set_reminder", text="call mom", when="om en time")
    b = call(ctx, "set_reminder", text="call mom", when="om en time")
    assert a["detail"]["id"] == b["detail"]["id"]


def test_list_and_cancel(ctx, clock):
    call(ctx, "set_reminder", text="call mom", when="om en time")
    call(ctx, "set_reminder", when="om ti minutter", kind="timer")
    out = call(ctx, "list_reminders")
    assert out["result"].startswith("2 upcoming") and "(in 10 minutes)" in out["result"] and "call mom" in out["result"]
    assert "Several match" in call(ctx, "cancel_reminder")["result"]
    out = call(ctx, "cancel_reminder", which="mom")
    assert out["risk"] == "visible" and "Cancelled" in out["result"]
    assert not call(ctx, "cancel_reminder", which="tandlæge")["ok"]
    clock.advance(minutes=10)
    assert says(ctx) == ["Your timer for 10 minutes is done."]


def test_snooze(ctx, clock):
    call(ctx, "set_reminder", text="stretch", when="om fem minutter")
    clock.advance(minutes=5)
    assert says(ctx) == ["Reminder: stretch."]
    out = call(ctx, "snooze_reminder")
    assert "10:15" in out["result"]
    assert says(ctx) == []  # the unsaid event is dropped
    clock.advance(minutes=10)
    assert says(ctx) == ["Reminder: stretch."]
    ack_all(ctx)
    out = call(ctx, "snooze_reminder", which="stretch", when="klokken halv to")
    assert "13:30" in out["result"]


def test_dry_run_changes_nothing(dry_ctx):
    dry_ctx.now = Clock(THU)
    out = call(dry_ctx, "set_reminder", text="call mom", when="om en time")
    assert out["result"].startswith("Dry run")
    assert call(dry_ctx, "list_reminders")["result"] == "He has no reminders or timers set."
    assert call(dry_ctx, "todo_add", text="milk")["result"].startswith("Dry run")
    assert "empty" in call(dry_ctx, "todo_list")["result"]


def test_reminders_survive_a_restart(ctx, clock):
    call(ctx, "set_reminder", text="call mom", when="om en time")
    call(ctx, "todo_add", text="buy milk")
    restart()
    assert "call mom" in call(ctx, "list_reminders")["result"]
    assert "buy milk" in call(ctx, "todo_list")["result"]
    # The server was off when it was due: it still fires, saying it is late.
    restart()
    clock.advance(hours=3)
    tick(ctx)
    [say] = says(ctx)
    assert say.startswith("Reminder: call mom.") and "It was due today at 11:00" in say


def test_unacked_events_survive_a_restart_and_expire(ctx, clock):
    call(ctx, "set_reminder", text="a", when="om et minut")
    clock.advance(minutes=1)
    tick(ctx)
    restart()
    assert says(ctx) == ["Reminder: a."]
    clock.advance(hours=13)
    assert says(ctx) == []  # too old to say now


# ---- focus timer ------------------------------------------------------------------------------

def test_focus_rounds(ctx, clock):
    out = call(ctx, "focus_start", minutes=25, break_minutes=5, rounds=2, task="the report")
    assert "2 rounds of 25 minutes with 5 minute breaks" in out["result"] and "10:55" in out["result"]
    assert not call(ctx, "focus_start")["ok"]  # one at a time
    clock.advance(minutes=25)
    assert says(ctx) == ["Focus round 1 of 2 is over. Take a 5 minute break."]
    ack_all(ctx)
    clock.advance(minutes=5)
    assert says(ctx) == ["Break's over. Round 2 of 2 on the report."]
    ack_all(ctx)
    assert "focus timer" in call(ctx, "list_reminders")["result"]
    clock.advance(minutes=10)
    assert "after 40 minutes" in call(ctx, "focus_stop")["result"]
    clock.advance(hours=1)
    assert says(ctx) == []
    assert not call(ctx, "focus_stop")["ok"]


def test_focus_single_round(ctx, clock):
    call(ctx, "focus_start")
    clock.advance(minutes=25)
    assert says(ctx) == ["Your focus session is over: 25 minutes. Time for a break."]


# ---- todos and notes --------------------------------------------------------------------------

def test_todos(ctx, clock):
    assert "1 open item" in call(ctx, "todo_add", text="buy milk")["result"]
    assert "already" in call(ctx, "todo_add", text="Buy milk")["result"]
    call(ctx, "todo_add", text="call the dentist")
    call(ctx, "todo_add", text="æbler", list="Indkøb")
    out = call(ctx, "todo_list")
    assert "2 open on the todo list" in out["result"] and "Other lists: indkøb" in out["result"]
    out = call(ctx, "todo_done", which="dentist")
    assert "Marked \"call the dentist\" as done; 1 left" in out["result"]
    assert "(1 done.)" in call(ctx, "todo_list")["result"]
    assert "æbler" in call(ctx, "todo_list", list="indkøb")["result"]
    assert "indkøb (1)" in call(ctx, "todo_list", list="all")["result"]
    assert not call(ctx, "todo_done", which="unicorn")["ok"]


def test_delete_all_todos_asks_first(ctx, clock):
    for t in ("a thing", "b thing", "c thing"):
        call(ctx, "todo_add", text=t)
    out = call(ctx, "todo_delete", which="all")
    assert out["needs_confirm"] and "delete all 3 todo items" in out["confirm"]["summary"]
    assert "3 open" in call(ctx, "todo_list")["result"]  # nothing deleted yet
    out = call(ctx, "todo_delete", confirmed=True, which="all")
    assert out["result"] == "Deleted 3 items." and out["risk"] == "destructive"
    assert "empty" in call(ctx, "todo_list")["result"]


def test_delete_one_todo(ctx, clock, dry_ctx):
    call(ctx, "todo_add", text="buy milk")
    call(ctx, "todo_add", text="buy bread")
    assert "Several match" in call(ctx, "todo_delete", confirmed=True, which="buy")["result"]
    out = call(ctx, "todo_delete", which="bread")
    assert out["needs_confirm"] and "bread" in out["confirm"]["summary"]
    assert call(ctx, "todo_delete", confirmed=True, which="bread")["result"] == "Deleted 1 item: buy bread."


def test_delete_in_dry_run(dry_ctx):
    dry_ctx.now = Clock(THU)
    dry_ctx.dry = False
    call(dry_ctx, "todo_add", text="keep me")
    dry_ctx.dry = True
    assert call(dry_ctx, "todo_delete", confirmed=True, all=True)["result"].startswith("Dry run")
    assert "keep me" in call(dry_ctx, "todo_list")["result"]


def test_notes(ctx, clock):
    call(ctx, "note_add", text="The bike lock code is 4711", title="bike")
    call(ctx, "note_add", text="Ida likes tulips")
    assert "4711" in call(ctx, "note_find", query="bike")["result"]
    assert "No note matches" in call(ctx, "note_find", query="passport")["result"]
    assert "2 notes" in call(ctx, "note_find")["result"]
    assert call(ctx, "note_delete", which="tulips")["needs_confirm"]
    assert "Deleted the note" in call(ctx, "note_delete", confirmed=True, which="tulips")["result"]
    assert "1 note:" in call(ctx, "note_find")["result"]


# ---- conversion -------------------------------------------------------------------------------

@pytest.mark.parametrize("value, frm, to, want", [
    ("5", "fod", "cm", "5 feet = 152.4 centimeters."),
    ("180", "fahrenheit", "celsius", "180 degrees Fahrenheit = 82.22 degrees Celsius."),
    ("3", "dl", "spsk", "3 deciliters = 20 tablespoons."),
    ("fem", "km", "mil", "5 kilometers = 3.107 miles (English/US miles)."),
    ("10", "km", "miles", "10 kilometers = 6.214 miles."),
    ("3,5", "kg", "pund", "3.5 kilograms = 7.716 pounds."),
    ("100", "km/t", "m/s", "100 km/h = 27.78 meters per second."),
    ("1.500", "gram", "kilo", "1,500 grams = 1.5 kilograms."),
    ("to hundrede", "liter", "gallons", "200 liters = 52.83 US gallons."),
])
def test_units(ctx, value, frm, to, want):
    assert call(ctx, "convert_units", value=value, from_unit=frm, to_unit=to)["result"] == want


def test_units_that_dont_fit(ctx):
    assert "different things" in call(ctx, "convert_units", value="5", from_unit="kg", to_unit="meter")["result"]
    assert not call(ctx, "convert_units", value="5", from_unit="furlongs", to_unit="m")["ok"]
    assert not call(ctx, "convert_units", value="lots", from_unit="m", to_unit="cm")["ok"]


@pytest.fixture
def ecb(monkeypatch):
    xml = (RECORDED / "time/eurofxref-daily.xml").read_text(encoding="utf-8")
    calls = []

    def fetch():
        calls.append(1)
        if calls and getattr(fetch, "fail", False):
            raise OSError("offline")
        return xml
    monkeypatch.setattr(tc, "fetch_ecb", fetch)
    fetch.calls = calls
    return fetch


def test_currency(ctx, clock, ecb):
    out = call(ctx, "convert_currency", amount="100", from_currency="euro", to_currency="kroner")
    assert out["result"].startswith("100 euros = 747.46 Danish kroner (ECB reference rate of 2026-10-21")
    out = call(ctx, "convert_currency", amount="50", from_currency="dollars")
    assert "332.68 Danish kroner" in out["result"]
    assert "Swedish kronor" in call(ctx, "convert_currency", amount="1000", from_currency="svenske kroner",
                                    to_currency="norske kroner")["result"]
    assert len(ecb.calls) == 1  # cached for a day
    clock.advance(hours=25)
    call(ctx, "convert_currency", amount="1", from_currency="eur")
    assert len(ecb.calls) == 2


def test_currency_offline(ctx, clock, ecb):
    call(ctx, "convert_currency", amount="1", from_currency="eur")
    ecb.fail = True
    clock.advance(days=2)
    out = call(ctx, "convert_currency", amount="10", from_currency="eur")
    assert out["ok"] and "couldn't be reached" in out["result"]


def test_currency_no_rates_at_all(ctx, clock, ecb):
    ecb.fail = True
    out = call(ctx, "convert_currency", amount="10", from_currency="eur")
    assert not out["ok"] and "couldn't be fetched" in out["result"]
    assert not call(ctx, "convert_currency", amount="10", from_currency="doubloons")["ok"]


# ---- watchers ---------------------------------------------------------------------------------

@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    (h / "Downloads").mkdir(parents=True)
    monkeypatch.setattr(gt, "home", lambda: h)
    return h


def test_download_finishes(ctx, clock, home):
    dl = home / "Downloads"
    (dl / "old.zip").write_bytes(b"x")
    out = call(ctx, "watch_file", event="download_done")
    assert out["risk"] == "visible" and "the Downloads folder" in out["result"] and "5 seconds" in out["result"]
    (dl / "film.mp4.crdownload").write_bytes(b"x" * 1000)
    clock.advance(seconds=5)
    tick(ctx)
    assert says(ctx) == []
    (dl / "film.mp4.crdownload").rename(dl / "film.mp4")
    clock.advance(seconds=5)
    tick(ctx)
    assert says(ctx) == []  # its size has to hold still for one more check
    clock.advance(seconds=5)
    tick(ctx)
    [e] = gt.pending_events(ctx)
    assert e["priority"] == "next_pause" and e["source"] == "watcher"
    assert e["say"] == "Your download is finished: film.mp4 (1 kB), in the Downloads folder."
    assert "Nothing is being watched now" in call(ctx, "list_watchers")["result"]


def test_new_file_and_deleted(ctx, clock, home):
    out = call(ctx, "watch_file", path=str(home / "Downloads" / "rapport.pdf"), event="new_file")
    assert out["ok"], out
    (home / "Downloads" / "other.txt").write_text("x")
    clock.advance(seconds=5)
    tick(ctx)
    assert says(ctx) == []
    (home / "Downloads" / "rapport.pdf").write_text("x")
    clock.advance(seconds=5)
    tick(ctx)
    assert says(ctx) == ["A new file appeared in the Downloads folder: rapport.pdf."]
    ack_all(ctx)
    call(ctx, "watch_file", path=str(home / "Downloads" / "rapport.pdf"), event="deleted")
    (home / "Downloads" / "rapport.pdf").unlink()
    clock.advance(seconds=5)
    tick(ctx)
    assert says(ctx) == ["rapport.pdf was deleted or moved."]


def test_watch_missing_folder(ctx, clock, home):
    out = call(ctx, "watch_file", path=str(home / "nope" / "deeper"), event="changed")
    assert not out["ok"] and "doesn't exist" in out["result"]


def test_process_stops(ctx, clock, monkeypatch):
    running = {"explorer", "blender"}
    monkeypatch.setattr(tw, "process_names", lambda: running)
    assert "already running" in call(ctx, "watch_process", program="explorer", event="starts")["result"]
    out = call(ctx, "watch_process", program="Blender.exe", event="stops")
    assert out["ok"] and "Watching Blender" in out["result"]
    clock.advance(seconds=5)
    tick(ctx)
    assert says(ctx) == []
    running.discard("blender")
    clock.advance(seconds=5)
    tick(ctx)
    assert says(ctx) == ["Blender has stopped (it's no longer running)."]


def test_web_page_is_polite_and_finds_text(ctx, clock, monkeypatch):
    page = {"text": "Koncert 12. december\nUdsolgt"}
    fetches = []

    def fetch(url):
        fetches.append(clock())
        return page["text"]
    monkeypatch.setattr(tw, "fetch_text", fetch)
    out = call(ctx, "watch_web", url="billetto.dk/koncert", event="loses", text="udsolgt", minutes=0.1)
    assert "every 60 seconds" in out["result"]  # never more than once a minute
    for _ in range(11):
        clock.advance(seconds=5)
        tick(ctx)
    assert len(fetches) == 1  # only the baseline so far
    clock.advance(seconds=5)
    tick(ctx)
    assert len(fetches) == 2 and says(ctx) == []
    page["text"] = "Koncert 12. december\nKøb billet"
    clock.advance(seconds=60)
    tick(ctx)
    assert says(ctx) == ['The page billetto.dk no longer says "udsolgt".']


def test_web_page_changes(ctx, clock, monkeypatch):
    page = {"text": "Version 1.0\nOld notes"}
    monkeypatch.setattr(tw, "fetch_text", lambda url: page["text"])
    assert "already says" in call(ctx, "watch_web", url="https://x.dk", event="contains", text="old notes")["result"]
    call(ctx, "watch_web", url="https://x.dk", event="changes")
    page["text"] = "Version 1.1\nOld notes"
    clock.advance(minutes=5)
    tick(ctx)
    assert says(ctx) == ["The page x.dk has changed. New: Version 1.1"]


def test_github_run(ctx, clock, monkeypatch):
    runs = {"status": "in_progress", "conclusion": ""}

    def gh(args, cwd=None):
        assert args[-2:] == ["-R", "MikkelBlom/grace-v2"]
        row = {"databaseId": 7, "workflowName": "CI", "displayTitle": "Add time tools", "headBranch": "main", **runs}
        return [row] if args[1] == "list" else row
    monkeypatch.setattr(tw, "gh", gh)
    out = call(ctx, "watch_github", repo="MikkelBlom/grace-v2")
    assert out["ok"] and "30 seconds" in out["result"]
    clock.advance(seconds=30)
    tick(ctx)
    assert says(ctx) == []
    runs.update(status="completed", conclusion="failure")
    clock.advance(seconds=30)
    tick(ctx)
    assert says(ctx) == ['The GitHub Actions run CI "Add time tools" on main finished: failure. It failed.']
    ack_all(ctx)
    assert "already finished: failure" in call(ctx, "watch_github", repo="MikkelBlom/grace-v2")["result"]


def test_url_comes_up(ctx, clock, monkeypatch):
    answers = [None, None, 200]
    monkeypatch.setattr(tw, "http_status", lambda url: answers.pop(0))
    out = call(ctx, "watch_url", url="localhost:3000")
    assert "http://localhost:3000" in out["result"] and "5 seconds" in out["result"]
    clock.advance(seconds=5)
    tick(ctx)
    assert says(ctx) == []
    clock.advance(seconds=5)
    tick(ctx)
    assert says(ctx) == ["The address http://localhost:3000 is up now (it answers with 200)."]
    monkeypatch.setattr(tw, "http_status", lambda url: 200)
    assert "already up" in call(ctx, "watch_url", url="https://example.com")["result"]


def fake_watcher(ctx, monkeypatch, check, hours=24):
    monkeypatch.setitem(tw.STARTS, "process", lambda w, now: ({"n": 0}, None))
    out = call(ctx, "watch_process", program="thing", event="stops", hours=hours)
    assert out["ok"], out
    return {"process": check}


def test_fake_check_counts_and_fires(ctx, clock, monkeypatch):
    def check(w, state, now):
        state["n"] += 1
        return state["n"] == 3, "Thing happened.", state
    checks = fake_watcher(ctx, monkeypatch, check)
    for _ in range(2):
        clock.advance(seconds=5)
        tick(ctx, checks)
    assert says(ctx) == []
    clock.advance(seconds=5)
    tick(ctx, checks)
    assert says(ctx) == ["Thing happened."]


def test_failing_check_gives_up(ctx, clock, monkeypatch):
    def check(w, state, now):
        raise OSError("no network")
    checks = fake_watcher(ctx, monkeypatch, check)
    for _ in range(gt.MAX_ERRORS):
        clock.advance(minutes=1)  # failures back off, up to 5 intervals
        tick(ctx, checks)
    [say] = says(ctx)
    assert "checking kept failing (no network)" in say


def test_watcher_expires(ctx, clock, monkeypatch):
    checks = fake_watcher(ctx, monkeypatch, lambda w, s_, now: (False, None, s_), hours=1)
    clock.advance(minutes=61)
    tick(ctx, checks)
    assert says(ctx) == ["I've stopped watching Thing after 1 hours; it stopping didn't happen."]


def test_stop_watcher_and_restart(ctx, clock, monkeypatch, home):
    hits = []
    checks = fake_watcher(ctx, monkeypatch, lambda w, s_, now: (hits.append(1) or True, "Fired.", s_))
    call(ctx, "watch_file", event="new_file")
    restart()
    out = call(ctx, "list_watchers")
    assert "Watching 2" in out["result"] and "Thing" in out["result"] and "Downloads" in out["result"]
    assert "Several match" in call(ctx, "stop_watcher")["result"]
    assert "Stopped watching Thing" in call(ctx, "stop_watcher", which="thing")["result"]
    clock.advance(seconds=10)
    tick(ctx, checks | {"file": lambda w, s_, now: (False, None, s_)})
    assert hits == [] and says(ctx) == []


def test_duplicate_watcher(ctx, clock, home):
    a = call(ctx, "watch_file", event="download_done")
    assert "already watching" in call(ctx, "watch_file", event="download_done", path="downloads")["result"]
    assert a["detail"]["id"]


def test_watch_dry_run(dry_ctx, monkeypatch):
    dry_ctx.now = Clock(THU)
    monkeypatch.setattr(tw, "http_status", lambda url: (_ for _ in ()).throw(AssertionError("no checks in dry run")))
    assert call(dry_ctx, "watch_url", url="https://example.com")["result"].startswith("Dry run")
    assert call(dry_ctx, "list_watchers")["result"] == "Nothing is being watched."


def test_now_events_come_first(ctx, clock, monkeypatch):
    checks = fake_watcher(ctx, monkeypatch, lambda w, s_, now: (True, "Watcher fired.", s_))
    call(ctx, "set_reminder", text="tea", when="om fem sekunder")
    clock.advance(seconds=5)
    tick(ctx, checks)
    assert [e["priority"] for e in gt.pending_events(ctx)] == ["now", "next_pause"]
