# Tool group `time`: time, organisation and watchers

Code: `services/grace_tools/time.py` (tools, SQLite store, scheduler, events API), with
`time_parse.py` (Danish and English time phrases), `time_watch.py` (watcher checks and their OS
layer) and `time_convert.py` (units, currency). Tests: `tests/tools/test_time.py`. Sim scenarios:
`fixtures/tools/scenarios/time.yaml`. Recorded ECB rates: `fixtures/tools/recorded/time/`.

All state lives in `ctx.data_dir/time.sqlite3` (`.runtime/grace-data/` for the server), so
reminders, todos, notes and watchers survive a restart. Currency rates are cached in
`ctx.data_dir/ecb-rates.json`. Every function reads its clock from `ctx.now` when set (tests use
a fake clock), else `datetime.now`; naive local times, Europe/Copenhagen.

## Tools

| Tool | Risk | Core | What it does |
|---|---|---|---|
| `time_now` | read | yes | Time, date, weekday, ISO week; the time in another city or country |
| `set_reminder` | visible | yes | A reminder or timer from his own time words; spoken when due |
| `date_info` | read | | Days until a date or holiday, weekday of a date, dates of a week number, date maths |
| `list_reminders` | read | | Upcoming reminders, timers and focus steps with time left |
| `cancel_reminder` | visible | | Cancel one (by id or words); several matches -> asks which |
| `snooze_reminder` | visible | | Snooze the one that just went off (default 10 min) or move one |
| `focus_start` / `focus_stop` | visible | | Pomodoro: focus minutes, breaks, rounds |
| `todo_add` / `todo_list` / `todo_done` | visible / read / visible | | Todo and other lists (e.g. `indkøb`) |
| `todo_delete` | destructive | | One item, or all of a list or every list; asks first |
| `note_add` / `note_find` | visible / read | | Short notes |
| `note_delete` | destructive | | Asks first |
| `convert_units` | read | | Length, mass, volume (incl. dl, spsk, tsk, kop), temperature, speed, area, time, data, energy, pressure |
| `convert_currency` | read (slow) | | ECB daily reference rates, cached a day |
| `watch_file` | visible | | download_done, new_file, changed, deleted, for a folder or file |
| `watch_process` | visible | | A program starts or stops |
| `watch_web` | visible | | A page changes, or contains / loses some text |
| `watch_github` | visible | | A GitHub Actions run finishes (via `gh`) |
| `watch_url` | visible | | An address comes up (e.g. `localhost:3000`) |
| `list_watchers` / `stop_watcher` | read / visible | | |

Cancelling reminders and stopping watchers work on one item at a time (no "all"), so they can
stay `visible`; deleting todos and notes is `destructive`, and "slet alle mine todos" returns a
summary with the count ("delete all 5 todo items on every list") and waits for his yes.

## Time phrases (time_parse.py)

Gemma is told to pass his time words untranslated (`when`: "halv fire", "på fredag klokken ni").
Rules covered by tests:
- Relative: "om tyve minutter", "om halvanden time", "om en halv time", "om et kvarter",
  "om tre kvarter", "om 2 timer og 30 minutter", "om fem og fyrre minutter", "in an hour and a half".
- Danish clock: **halv fire = 15:30**, kvart over tre = 15:15, kvart i fire = 15:45,
  fem over halv fire = 15:35, ti i halv fire = 15:20, tyve over to, fem minutter i otte, halv et = 12:30.
  English "half past three" = 15:30 and British "half four" = 16:30 are read as English.
- A bare hour is read as people mean it: "klokken ni" at 10:00 is 21:00; at 21:00 "klokken syv"
  is tomorrow 07:00; on a named day 7-12 is morning and 1-6 afternoon ("i morgen klokken tre" = 15:00).
- Days: i dag, i morgen, i overmorgen, i morgen tidlig (07:00), i aften (19:00), i eftermiddag,
  i formiddag, i nat; weekdays ("på fredag" said on a Friday means next week's), "i fredags"
  (last Friday), "fredag i næste uge", "i weekenden"; dates ("3. december", "den tredje december",
  "24/12", "december 3rd"); week numbers ("uge 45", "fredag i uge 45", ISO weeks); holidays
  (jul = 24 December, nytår, påske and the days around it, pinse, sankthans, ...).
- A date without a time becomes 09:00, and the result tells Gemma to say so.

**Translated time words.** If the engine passes `heard` (his transcribed words, as it already
does for `calculate`), `set_reminder` uses it when Gemma's `when` has lost a Danish
"halv"/"kvart" phrase ("half four" for "halv fire"). Until the engine adds `heard` for
`set_reminder`, this protection is inactive and the tool relies on the parameter description.

## The delivery channel (for the organiser)

Due reminders, timers and focus steps, and watchers that fire, are written to an `events` table.
Two plain functions in `grace_tools.time`:

```python
pending_events(ctx) -> list[dict]   # unacked events of the last 12 h; also fires due timers
ack_event(ctx, id) -> bool          # True if it was pending
```

Each event:

```json
{"id": 12, "priority": "now", "say": "Reminder: call mom.", "source": "reminder",
 "ref": 4, "created": "2026-10-22T11:00:00"}
```

- `priority`: `"now"` for reminders, timers and focus steps (say it as soon as he isn't
  speaking); `"next_pause"` for watchers (say it at the next pause in the conversation).
  `pending_events` lists `now` events first, then by id.
- `say`: an English sentence ready to speak (ADR 0011). A reminder that fires late (the server
  was off) says so: "Reminder: call mom. (It was due today at 11:00; I couldn't say it then.)"
- `source`: `reminder`, `timer`, `focus` or `watcher`; `ref` is the item or watcher id.
- Delivery is at-least-once: an event is offered on every poll until it is acked. Ack it once it
  has been spoken (or handed to speech and not cut off). Unacked events older than 12 hours are
  not offered again.

Proposed tools-server request (tools_server.py `Handler.serve`, not edited by this group):

```
{"kind": "events", "ack": [12, 13]}  ->  {"acked": [12, 13], "events": [<event>, ...]}
{"kind": "events"}                   ->  {"acked": [], "events": [...]}
```

```python
elif kind == "events":  # due reminders, timers and watchers (docs/tools/time.md)
    from grace_tools import time as gt
    acked = [i for i in req.get("ack") or [] if gt.ack_event(srv.tools, i)]
    out = {"acked": acked, "events": gt.pending_events(srv.tools)}
```

The engine polls this about once a second (a local SQLite query) and, after speaking an event,
puts its `say` into the conversation history as Grace's line, so "udsæt den" / "snooze it"
has something to refer to.

**Scheduler.** When the module loads, two daemon threads start: one fires due items every
second, the other runs due watcher checks every two seconds (so a slow web check never delays a
timer). They don't start when `GRACE_TOOLS_SCHEDULER=0`, nor under pytest. They use the data dir
of the last ctx seen, else `.runtime/grace-data`. `pending_events` also fires due timers, so a
poll is never later than the scheduler.

## Watchers

A watcher fires once, then ends; it also ends after `hours` (default 24, at most 168) with a
`next_pause` event saying it didn't happen, or after 10 failed checks in a row (with back-off)
saying why. Creating one first checks whether the thing has already happened ("the page already
says that", "localhost:3000 is already up", "the latest run already finished: success") and then
creates nothing. Duplicates of an active watcher are not created.

Check intervals: files and processes every 5 s (min 2 s, a directory listing / process list);
web pages every 5 min, **never more than once a minute**; URLs once a minute (5 s for localhost);
GitHub runs every 30 s (min 15 s; `gh` uses his API quota, 5000/h).

- **Download finished**: a new file in the folder (default Downloads) that isn't a partial
  download (`.crdownload`, `.part`, `.tmp`, ...), no new or growing partial files, and the same
  size on two checks in a row. Only the folder's own entries are listed, not subfolders.
- **Web page changed**: the hash of the page's visible text. Pages with clocks, ads or
  rotating content will "change" on the first check; `contains` / `loses` with a word is the
  reliable way ("udsolgt", "på lager").
- **GitHub**: `repo` as `owner/name` or a local checkout folder; watches the given run id, the
  latest run still going on a branch, or the latest run still going.

## Tested (tests/tools/test_time.py, all on a fake clock)

60 time phrases at a fixed Thursday plus 6 that depend on the hour, past phrases, holidays,
leftover-text cleanup; time_now here and abroad; date_info (days to jul, weekday, week ranges,
date maths, between); reminders firing at the exact second, once; timers; Danish rule;
`heard` overriding a translated phrase; dates without times; past and too-far times;
duplicates; list, cancel, snooze; dry-run; **persistence across a restart** (new Store on the same
file) including late firing and unacked events; focus rounds and stop; todos, lists, notes;
delete-all asking first with a count; units; currency from the recorded ECB response, the
day's cache, a stale cache when offline, no rates at all; watchers on real temp folders
(download, new file, deleted), fake process lists, fake pages (politeness interval, contains,
loses, changes), fake `gh`, fake HTTP; fake check functions for firing, failing and expiry;
stop and list across a restart; event ordering.

## Not tested, and why

- The real OS calls: `psutil` process listing, `urllib` page fetches, `gh` itself, and the
  ECB download (tests replace them; the ECB fixture is a real response recorded on 2026-10-22).
- The background threads (off under pytest); `tick()` is what they call and is fully tested.
- Live use with speech: the events request isn't wired into the engine yet.
- Repeating reminders ("hver dag klokken 8") and spoken years ("tyve seksogtyve") aren't
  supported. Time zones beyond the built-in city list fall back to IANA names.
