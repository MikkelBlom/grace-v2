"""Danish and English time phrases -> local datetimes, for the time group (grace_tools/time.py).

Pure functions: every one takes `now` explicitly, so tests are deterministic. Times are naive
local wall-clock times (Mikkel's PC, Europe/Copenhagen).

Danish rules that English speakers (and Gemma) get wrong:
  halv fire          = 15:30 (half *to* four), not 16:30
  kvart i fire       = 15:45,  kvart over tre = 15:15
  fem over halv fire = 15:35,  ti i halv fire  = 15:20
  i fredags          = last Friday,  i morgen tidlig = tomorrow early (07:00)
  jul                = 24 December (Danes celebrate on the evening of the 24th)
A bare hour is read the way people speak: "klokken ni" at 10:00 means 21:00, not tomorrow 09:00.
"""

import re
from datetime import date, datetime, timedelta

# Copied from services/tools_server.py (it loads OpenVINO, so it can't be imported here).
DA_UNITS = {"nul": 0, "en": 1, "et": 1, "én": 1, "ét": 1, "to": 2, "tre": 3, "fire": 4, "fem": 5, "seks": 6,
            "syv": 7, "otte": 8, "ni": 9, "ti": 10, "elleve": 11, "tolv": 12, "tretten": 13, "fjorten": 14,
            "femten": 15, "seksten": 16, "sytten": 17, "atten": 18, "nitten": 19}
DA_TENS = {"tyve": 20, "tredive": 30, "fyrre": 40, "fyrretyve": 40, "halvtreds": 50, "halvtredsindstyve": 50,
           "tres": 60, "tresindstyve": 60, "halvfjerds": 70, "halvfjerdsindstyve": 70, "firs": 80,
           "firsindstyve": 80, "halvfems": 90, "halvfemsindstyve": 90}
DA_BIG = {"hundrede": 100, "hundred": 100, "tusind": 1000, "tusinde": 1000}
EN_UNITS = {"zero": 0, "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
            "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
           "ninety": 90}
HALVES = {"halvanden": 1.5, "en halv": 0.5, "et halvt": 0.5, "half an": 0.5, "half a": 0.5, "a half": 0.5}

DA_ORD = ["første", "anden", "tredje", "fjerde", "femte", "sjette", "syvende", "ottende", "niende", "tiende",
          "ellevte", "tolvte", "trettende", "fjortende", "femtende", "sekstende", "syttende", "attende",
          "nittende", "tyvende"]
EN_ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
          "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
          "eighteenth", "nineteenth", "twentieth"]
ORDINALS = {w: i + 1 for i, w in enumerate(DA_ORD)} | {w: i + 1 for i, w in enumerate(EN_ORD)}
for _u, _v in [("en", 1), ("to", 2), ("tre", 3), ("fire", 4), ("fem", 5), ("seks", 6), ("syv", 7), ("otte", 8), ("ni", 9)]:
    ORDINALS[f"{_u}ogtyvende"] = 20 + _v
ORDINALS |= {"tredivte": 30, "tredivende": 30, "enogtredivte": 31, "enogtredivende": 31, "thirtieth": 30,
             "thirty-first": 31, "thirty first": 31}
for _i, _w in enumerate(EN_ORD[:9]):
    ORDINALS[f"twenty-{_w}"] = ORDINALS[f"twenty {_w}"] = 21 + _i

MONTHS = {"januar": 1, "january": 1, "jan": 1, "februar": 2, "february": 2, "feb": 2, "marts": 3, "march": 3,
          "mar": 3, "april": 4, "apr": 4, "maj": 5, "may": 5, "juni": 6, "june": 6, "jun": 6, "juli": 7, "july": 7,
          "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9, "oktober": 10, "october": 10, "okt": 10,
          "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12}
WEEKDAYS = {w: i for i, w in enumerate(["mandag", "tirsdag", "onsdag", "torsdag", "fredag", "lørdag", "søndag"])}
WEEKDAYS |= {w: i for i, w in enumerate(["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])}
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]

# Parts of the day: (default hour, minute, how a bare 1-12 hour is read)
PARTS = {"early": (7, 0, "am"), "morning": (8, 0, "am"), "forenoon": (10, 0, "am"), "noon": (12, 0, None),
         "afternoon": (15, 0, "pm"), "evening": (19, 0, "pm"), "night": (23, 0, "night")}
DEFAULT_TIME = (9, 0)  # a reminder with only a date


def _alt(words):
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


# "femogfyrre", and "fem og fyrre" as STT sometimes splits it.
_DA_COMPOUND = r"(?:en|et|to|tre|fire|fem|seks|syv|otte|ni)(?: og |og)(?:tyve|tredive|fyrre|halvtreds|tres|halvfjerds|firs|halvfems)"
_EN_COMPOUND = r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[- ](?:one|two|three|four|five|six|seven|eight|nine)"
NUM = rf"(?:\d+(?:[.,]\d+)?|{_DA_COMPOUND}|{_EN_COMPOUND}|{_alt(set(DA_UNITS) | set(DA_TENS) | set(EN_UNITS) - {'a', 'an'} | set(EN_TENS) | set(DA_BIG))})"
_HOUR_WORDS = [w for w, v in DA_UNITS.items() if 1 <= v <= 19] + [w for w, v in EN_UNITS.items() if 1 <= v <= 12 and w not in ("a", "an")]
HOUR = rf"(?:\d{{1,2}}|(?:en|to|tre|fire)ogtyve|tyve|{_alt(_HOUR_WORDS)})"
ORD = rf"(?:{_alt(ORDINALS)})"
MONTH = rf"(?:{_alt(MONTHS)})"
WEEKDAY = rf"(?:{_alt(WEEKDAYS)})"


def number(word):
    """'treogtyve' -> 23, 'twenty-five' -> 25, '1,5' -> 1.5, 'halvanden' -> 1.5, else None."""
    w = word.strip().lower()
    if re.fullmatch(r"\d+", w):
        return int(w)
    if re.fullmatch(r"\d+[.,]\d+", w):
        return float(w.replace(",", "."))
    if w in HALVES:
        return HALVES[w]
    w = w.replace("-", " ").replace(" og ", "og")
    for table in (DA_UNITS, DA_TENS, DA_BIG, EN_UNITS, EN_TENS):
        if w in table:
            return table[w]
    u, sep, t = w.partition("og")
    if sep and u in DA_UNITS and t in DA_TENS:
        return DA_UNITS[u] + DA_TENS[t]
    parts = w.split()
    if len(parts) == 2 and parts[0] in EN_TENS and parts[1] in EN_UNITS:
        return EN_TENS[parts[0]] + EN_UNITS[parts[1]]
    for big, mult in DA_BIG.items():  # "tohundrede"
        if w.endswith(big) and w[: -len(big)] in DA_UNITS:
            return DA_UNITS[w[: -len(big)]] * mult
    return None


def norm(text):
    t = " " + (text or "").lower().replace("’", "'") + " "
    t = re.sub(r"\bkl\b\.?", " klokken ", t)
    t = re.sub(r"\bo'clock\b", " ", t)
    t = re.sub(r"(?<=\d)(am|pm)\b", r" \1", t)
    t = re.sub(r"\ba\.m\.", "am", t)
    t = re.sub(r"\bp\.m\.", "pm", t)
    t = re.sub(r"[!?;\"“”()]", " ", t)
    t = re.sub(r",(?!\d)|(?<!\d),", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?![\d\s])", " ", t)  # keep "3." "9.30" "24.12.2026"
    t = re.sub(r"\.(?=\s)", ". ", t)
    return " " + " ".join(t.split()) + " "


class Scan:
    """The phrase being read: each recognised piece is blanked out so it is read only once."""

    def __init__(self, text):
        self.t = norm(text)
        self.used = []

    def take(self, pattern):
        m = re.search(pattern, self.t)
        if m:
            self.used.append(m.group(0).strip())
            self.t = self.t[:m.start()] + " " * (m.end() - m.start()) + self.t[m.end():]
        return m

    def rest(self):
        return " ".join(self.t.split())


# ---- durations --------------------------------------------------------------------------------

DUR_UNITS = [("sekunder|sekund|sek|seconds|second|secs|sec", 1), ("minutter|minut|min|minutes|minute|mins", 60),
             ("kvarter", 900), ("timer|time|hours|hour|hrs|hr|t", 3600), ("døgn|dage|dag|days|day", 86400),
             ("uger|uge|weeks|week", 604800), ("måneder|måned|months|month", "month"), ("år|years|year", "year")]
_DUR_UNIT = "|".join(u for u, _ in DUR_UNITS)
_DUR = re.compile(
    rf"(?<![\w.])(?P<n>halvanden|en halv|et halvt|half an|half a|{NUM}|an|a)"
    rf"(?P<half1>\s+og\s+en\s+halv|\s+and\s+a\s+half)?\s+(?P<u>{_DUR_UNIT})\b"
    rf"(?P<half2>\s+(?:og|and)\s+(?:en|et|a)\s+(?:halv|half)\b)?")


def take_duration(scan):
    """Every '<number> <unit>' in the phrase, summed: (seconds, months). None if there is none."""
    secs, months, found = 0.0, 0.0, False
    while True:
        m = _DUR.search(scan.t)
        if not m:
            break
        n = number(m.group("n"))
        unit = m.group("u")
        # "en timer" / "a timer" is the kitchen timer, not hours ("sæt en timer på ti minutter").
        if n is None or (unit == "timer" and m.group("n") in ("en", "et", "a", "an", "én", "ét")):
            scan.t = scan.t[:m.start("u")] + unit.upper() + scan.t[m.end("u"):]  # hidden from the regex
            continue
        if m.group("half1") or m.group("half2"):
            n += 0.5
        size = next(v for u, v in DUR_UNITS if re.fullmatch(u, unit))
        if size == "month":
            months += n
        elif size == "year":
            months += 12 * n
        else:
            secs += n * size
        found = True
        scan.used.append(m.group(0).strip())
        scan.t = scan.t[:m.start()] + " " * (m.end() - m.start()) + scan.t[m.end():]
    scan.t = scan.t.lower()
    return (secs, months) if found else None


def parse_duration(text):
    """'ti minutter' -> 600.0 seconds, 'halvanden time' -> 5400.0; months count as 30 days."""
    d = take_duration(Scan(text))
    return None if d is None else d[0] + d[1] * 30 * 86400


def add_months(dt, months):
    whole = int(months)
    y, m = divmod(dt.month - 1 + whole, 12)
    y, m = dt.year + y, m + 1
    day = min(dt.day, [31, 29 if y % 4 == 0 and (y % 100 or y % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    out = dt.replace(year=y, month=m, day=day)
    return out + timedelta(days=round((months - whole) * 30))


# ---- holidays ---------------------------------------------------------------------------------

def easter(y):
    """Easter Sunday (the anonymous Gregorian algorithm)."""
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 19 * l) // 433
    month = (h + l - 7 * m + 90) // 25
    return date(y, month, (h + l - 7 * m + 33 * month + 19) % 32)


def _nth_weekday(y, month, weekday, n):
    d = date(y, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


# Most specific first ("2. juledag" before "jul", "anden påskedag" before "påske").
HOLIDAYS = [
    (r"(?:2\.?|anden)\s+juledag|boxing day", lambda y: date(y, 12, 26), "Boxing Day (2. juledag)"),
    (r"(?:1\.?|første)\s+juledag|juledag|christmas day", lambda y: date(y, 12, 25), "Christmas Day"),
    (r"juleaften|christmas eve|jul(?:en)?|christmas|xmas", lambda y: date(y, 12, 24), "Christmas Eve (Danish jul)"),
    (r"nytårsaften|new year'?s eve", lambda y: date(y, 12, 31), "New Year's Eve"),
    (r"nytårsdag|nytår|new year'?s day|new year", lambda y: date(y, 1, 1), "New Year's Day"),
    (r"(?:2\.?|anden)\s+påskedag|easter monday", lambda y: easter(y) + timedelta(1), "Easter Monday"),
    (r"langfredag|good friday", lambda y: easter(y) - timedelta(2), "Good Friday"),
    (r"skærtorsdag|maundy thursday", lambda y: easter(y) - timedelta(3), "Maundy Thursday"),
    (r"påskedag|påske(?:n)?|easter(?: sunday)?", easter, "Easter Sunday"),
    (r"kristi himmelfart(?:sdag)?|ascension day", lambda y: easter(y) + timedelta(39), "Ascension Day"),
    (r"(?:2\.?|anden)\s+pinsedag|whit monday", lambda y: easter(y) + timedelta(50), "Whit Monday"),
    (r"pinsedag|pinse(?:n)?|pentecost|whitsun", lambda y: easter(y) + timedelta(49), "Whit Sunday"),
    (r"fastelavn", lambda y: easter(y) - timedelta(49), "Fastelavn"),
    (r"sankthans(?:aften)?|sankt hans(?: aften)?|midsummer(?: eve)?", lambda y: date(y, 6, 23), "Sankthansaften"),
    (r"grundlovsdag|constitution day", lambda y: date(y, 6, 5), "Grundlovsdag"),
    (r"fars dag|father'?s day", lambda y: date(y, 6, 5), "Father's Day (Danish, 5 June)"),
    (r"mors dag|mother'?s day", lambda y: _nth_weekday(y, 5, 6, 2), "Mother's Day"),
    (r"halloween", lambda y: date(y, 10, 31), "Halloween"),
    (r"mortensaften", lambda y: date(y, 11, 10), "Mortensaften"),
    (r"valentinsdag|valentine'?s(?: day)?", lambda y: date(y, 2, 14), "Valentine's Day"),
]


# ---- the parse --------------------------------------------------------------------------------

class When:
    def __init__(self, dt, has_date, has_time, relative, phrase, label=None):
        self.dt, self.has_date, self.has_time, self.relative = dt, has_date, has_time, relative
        self.phrase, self.label = phrase, label

    def __repr__(self):
        return f"When({self.dt.isoformat()}, date={self.has_date}, time={self.has_time}, rel={self.relative})"


def _clock(scan):
    """(hour, minute, mode) where mode is '24' (unambiguous), 'am', 'pm' or None (a bare 1-12)."""
    k = r"(?:klokken\s+|at\s+|by\s+)?"
    mm = rf"(?P<m>{NUM})(?:\s+(?:minutter|minut|min|minutes))?"
    h = rf"(?P<h>{HOUR})\b"

    def hv(m):
        return number(m.group("h"))

    pats = [
        (rf"{k}{mm}\s+over\s+halv\s+{h}", lambda m: (hv(m) - 1, 30 + number(m.group("m")))),
        (rf"{k}{mm}\s+i\s+halv\s+{h}", lambda m: (hv(m) - 1, 30 - number(m.group("m")))),
        (rf"{k}kvart\s+over\s+{h}", lambda m: (hv(m), 15)),
        (rf"{k}kvart\s+i\s+{h}", lambda m: (hv(m) - 1, 45)),
        (rf"{k}halv\s+{h}", lambda m: (hv(m) - 1, 30)),  # Danish: halv fire = 3:30
        (rf"{k}(?<!om ){mm}\s+over\s+{h}", lambda m: (hv(m), number(m.group("m")))),
        (rf"{k}(?<!om )(?<!in ){mm}\s+i\s+{h}", lambda m: (hv(m) - 1, 60 - number(m.group("m")))),
        (rf"{k}(?:a\s+)?quarter\s+past\s+{h}", lambda m: (hv(m), 15)),
        (rf"{k}(?:a\s+)?quarter\s+to\s+{h}", lambda m: (hv(m) - 1, 45)),
        (rf"{k}half\s+past\s+{h}", lambda m: (hv(m), 30)),
        (rf"{k}half\s+{h}", lambda m: (hv(m), 30)),  # British: half four = 4:30
        (rf"{k}{mm}\s+past\s+{h}", lambda m: (hv(m), number(m.group("m")))),
        (rf"{k}(?<!in ){mm}\s+to\s+{h}", lambda m: (hv(m) - 1, 60 - number(m.group("m")))),
    ]
    for p, f in pats:
        m = scan.take(p)
        if m:
            try:
                hh, mi = f(m)
            except TypeError:
                continue
            if hh is None or mi is None or not (0 <= mi < 60):
                continue
            return (hh % 24) or 12, int(mi), None  # "halv et" is 12:30
    m = scan.take(r"(?:klokken\s+|at\s+|by\s+)?(?<![\d.:/-])(?P<h>\d{1,2})(?::(?P<m>\d{2})|(?<=\d)\.(?P<m2>\d{2})(?!\.\d))?"
                  r"\s*(?P<ap>am|pm)\b")
    if m:
        hh, mi = int(m.group("h")), int(m.group("m") or m.group("m2") or 0)
        if m.group("ap") == "pm" and hh < 12:
            hh += 12
        if m.group("ap") == "am" and hh == 12:
            hh = 0
        return hh, mi, "24"
    m = (scan.take(r"(?:klokken|at|by)\s+(?P<h>\d{1,2})(?:[:.](?P<m>\d{2}))?(?![\d.:/-]*\d)")
         or scan.take(r"(?<![\d.:/-])(?P<h>\d{1,2}):(?P<m>\d{2})\b"))
    if m:
        raw = m.group("h")
        hh, mi = int(raw), int(m.group("m") or 0)
        if hh > 24 or mi > 59:
            return None
        return hh % 24, mi, "24" if hh == 0 or hh > 12 or (raw.startswith("0") and len(raw) == 2) else None
    m = scan.take(rf"klokken\s+(?P<h>{HOUR})\b(?:\s+(?P<m>{NUM})\b(?!\s+(?:{_DUR_UNIT})\b))?")
    if m:
        hh = number(m.group("h"))
        mi = number(m.group("m")) if m.group("m") else 0
        if hh is None or mi is None or hh > 24 or not (0 <= mi < 60):
            return None
        return hh % 24, int(mi), "24" if hh > 12 else None
    if scan.take(r"\b(?:noon|midday|middagstid|frokosttid|frokost|klokken tolv middag)\b"):
        return 12, 0, "24"
    if scan.take(r"\b(?:midnight|midnat)\b"):
        return 0, 0, "24"
    return None


def _part(scan):
    pats = [("early", r"\b(?:tidlig|early)\b"), ("forenoon", r"\b(?:i\s+)?formiddag(?:en)?\b|\bom formiddagen\b"),
            ("afternoon", r"\b(?:i\s+|om\s+)?eftermiddag(?:en)?\b|\b(?:this |in the )?afternoon\b"),
            ("evening", r"\b(?:i\s+|om\s+)?aften(?:en)?\b|\btonight\b|\b(?:this |in the )?evening\b"),
            ("morning", r"\bi\s+morges\b|\bom\s+morgenen\b|\bmorgen(?:en)?\b|\b(?:this |in the )?morning\b"),
            ("night", r"\b(?:i\s+|om\s+)?nat(?:ten)?\b|\b(?:at |in the )?night\b")]
    for name, p in pats:
        m = scan.take(p)
        if m:
            return name, m.group(0).strip()
    return None, None


def _day_offset(scan):
    for p, off in [(r"\bi\s+forgårs\b|\bthe day before yesterday\b", -2), (r"\b(?:i\s+)?overmorgen\b|\b(?:the )?day after tomorrow\b", 2),
                   (r"\bi\s*morgen\b|\btomorrow\b", 1), (r"\bi\s*går\b|\byesterday\b", -1),
                   (r"\bi\s*dag\b|\btoday\b|\bnu\b|\bnow\b", 0)]:
        if scan.take(p):
            return off
    if re.search(r"\btonight\b", scan.t):
        return 0
    return None


def _year_for(make, today, prefer):
    """The next (or, when asking about the past, the last) year in which make(year) is a date."""
    if prefer == "past":
        for y in (today.year, today.year - 1):
            d = make(y)
            if d and d <= today:
                return d
    for y in (today.year, today.year + 1, today.year + 2):
        try:
            d = make(y)
        except ValueError:
            continue
        if d >= today:
            return d
    return make(today.year)


def _date(scan, today, prefer):
    """(date, label) from an explicit date, holiday, week number or weekday; or (None, None)."""
    m = scan.take(r"\b(?P<y>\d{4})-(?P<mo>\d{1,2})-(?P<d>\d{1,2})\b")
    if m:
        return date(int(m["y"]), int(m["mo"]), int(m["d"])), None
    m = (scan.take(r"(?<![\d:.])(?P<d>\d{1,2})/(?P<mo>\d{1,2})(?:/(?P<y>\d{2,4}))?\b")
         or scan.take(r"(?<![\d:.])(?P<d>\d{1,2})[.-](?P<mo>\d{1,2})[.-](?P<y>\d{2,4})\b"))
    if m:
        d, mo = int(m["d"]), int(m["mo"])
        if m["y"]:
            y = int(m["y"]) + (2000 if len(m["y"]) == 2 else 0)
            return date(y, mo, d), None
        return _year_for(lambda y: date(y, mo, d), today, prefer), None
    m = (scan.take(rf"\b(?:den\s+|d\s*\.\s*|the\s+)?(?P<d>\d{{1,2}})(?:\.|st|nd|rd|th)?\s+(?:of\s+)?(?P<mo>{MONTH})\b(?:\s+(?P<y>\d{{4}}))?")
         or scan.take(rf"\b(?:den\s+|the\s+)?(?P<o>{ORD})\s+(?:of\s+)?(?P<mo>{MONTH})\b(?:\s+(?P<y>\d{{4}}))?")
         or scan.take(rf"\b(?P<mo>{MONTH})\s+(?:the\s+)?(?:(?P<d>\d{{1,2}})(?:st|nd|rd|th)?|(?P<o>{ORD}))\b(?:,?\s+(?P<y>\d{{4}}))?"))
    if m:
        d = int(m["d"]) if m.groupdict().get("d") else ORDINALS[m["o"].replace("-", " ") if m["o"] not in ORDINALS else m["o"]]
        mo = MONTHS[m["mo"]]
        if m["y"]:
            return date(int(m["y"]), mo, d), None
        return _year_for(lambda y: date(y, mo, d), today, prefer), None
    for p, make, label in HOLIDAYS:
        if scan.take(rf"\b(?:til\s+|to\s+|until\s+)?(?:{p})\b"):
            return _year_for(make, today, prefer), label
    m = scan.take(rf"\b(?:(?P<wd>{WEEKDAY})(?:en)?\s+(?:i\s+|in\s+|of\s+)?)?(?:uge|week)\s+(?P<w>\d{{1,2}}|{NUM})\b(?:\s+(?P<y>\d{{4}}))?")
    if m:
        w = number(m["w"])
        wd = WEEKDAYS[m["wd"]] if m["wd"] else 0
        if m["y"]:
            return date.fromisocalendar(int(m["y"]), int(w), wd + 1), None
        def make(y):
            return date.fromisocalendar(y, int(w), wd + 1)
        # A week counts as "this one" until its Sunday has passed.
        for y in (today.year, today.year + 1):
            try:
                if date.fromisocalendar(y, int(w), 7) >= today or prefer == "past":
                    return make(y), None
            except ValueError:
                continue
        return make(today.year + 1), None
    m = scan.take(rf"\bi\s+(?P<wd>{WEEKDAY})s\b")  # "i fredags" = last Friday
    if m:
        wd = WEEKDAYS[m["wd"]]
        return today - timedelta(days=(today.weekday() - wd - 1) % 7 + 1), None
    m = scan.take(rf"\b(?:(?P<mod>på|on|næste|next|kommende|this|denne|nu|sidste|last|forrige)\s+)?(?P<wd>{WEEKDAY})(?:en)?\b"
                  rf"(?P<nw>\s+(?:i\s+næste\s+uge|next\s+week|om\s+en\s+uge))?")
    if m:
        wd = WEEKDAYS[m["wd"]]
        if m["mod"] in ("sidste", "last", "forrige") or (prefer == "past" and not m["mod"]):
            return today - timedelta(days=(today.weekday() - wd - 1) % 7 + 1), None
        if m["nw"]:
            monday = today - timedelta(days=today.weekday()) + timedelta(weeks=1)
            return monday + timedelta(days=wd), None
        # The coming one; said on that weekday it means next week's ("på fredag" on a Friday).
        return today + timedelta(days=(wd - today.weekday() - 1) % 7 + 1), None
    if scan.take(r"\b(?:i\s+)?næste\s+uge\b|\bnext\s+week\b"):
        return today - timedelta(days=today.weekday()) + timedelta(weeks=1), None
    if scan.take(r"\b(?:i\s+)?(?:weekenden|weekend)\b|\b(?:this\s+|at\s+the\s+)?weekend\b"):
        return today + timedelta(days=(5 - today.weekday()) % 7 if today.weekday() < 5 else 0), None
    if scan.take(r"\b(?:i\s+)?næste\s+måned\b|\bnext\s+month\b"):
        return add_months(datetime(today.year, today.month, 1), 1).date(), None
    m = scan.take(rf"\b(?:den|the)\s+(?:(?P<d>\d{{1,2}})(?:\.|st|nd|rd|th)?|(?P<o>{ORD}))(?=\s|$)")
    if m:
        d = int(m["d"]) if m["d"] else ORDINALS[m["o"]]
        def make_month(k):
            y, mo = divmod(today.month - 1 + k, 12)
            return date(today.year + y, mo + 1, d)
        for k in ((0, -1) if prefer == "past" else (0, 1, 2)):
            try:
                c = make_month(k)
            except ValueError:
                continue
            if (c <= today) if prefer == "past" else (c >= today):
                return c, None
    return None, None


def _apply_hour(h, mi, mode, part, day, now, date_given):
    """The datetime for a clock time, reading a bare 1-12 hour the way people mean it."""
    if mode == "24" or h == 0 or h > 12:
        return datetime.combine(day, datetime.min.time()).replace(hour=h, minute=mi)
    if part:
        how = PARTS[part][2]
        if how == "pm" and h < 12:
            h += 12
        elif how == "am" and h == 12:
            h = 0
        elif how == "night":
            if 7 <= h < 12:
                h += 12
            elif h < 7 and not date_given:
                day = day + timedelta(days=1)
        return datetime.combine(day, datetime.min.time()).replace(hour=h, minute=mi)
    if date_given:
        # On a named day: 7-12 is the morning or noon, 1-6 the afternoon.
        if h < 7:
            h += 12
        return datetime.combine(day, datetime.min.time()).replace(hour=h, minute=mi)
    # No day: the next time that hour comes round in waking hours (07-22).
    cands = []
    for k in (0, 1):
        for hh in ((h, h + 12) if h < 12 else (12, 0)):
            dt = datetime.combine(now.date() + timedelta(days=k), datetime.min.time()).replace(hour=hh, minute=mi)
            if dt > now:
                cands.append(dt)
    awake = [c for c in cands if 7 <= c.hour <= 22]
    return min(awake or cands)


def parse(text, now, prefer="future", default_time=DEFAULT_TIME):
    """A time phrase -> When, or None if no time or date is in it.

    prefer="past" reads "fredag", "3. december" as the last one (for "hvilken dag var ...")."""
    text = (text or "").strip()
    if not text:
        return None
    iso = re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[t ]\d{2}:\d{2}(?::\d{2})?)?", text.lower())
    if iso:
        dt = datetime.fromisoformat(text.upper().replace(" ", "T")).replace(tzinfo=None)
        if len(text) == 10:
            dt = dt.replace(hour=default_time[0], minute=default_time[1])
        return When(dt, True, len(text) > 10, False, text)
    scan = Scan(text)
    if re.search(r"\b(?:var|was|sidste|last|siden|ago|i går|yesterday)\b", scan.t):
        prefer = "past"
    today = now.date()
    clock = _clock(scan)
    dur = take_duration(scan)
    ago = bool(scan.take(r"\b(?:siden|ago)\b"))
    off = _day_offset(scan)
    day, label = _date(scan, today, prefer)
    part, _ = _part(scan)
    phrase = " ".join(scan.used)
    if dur is not None:
        secs, months = dur
        sign = -1 if ago else 1
        base = add_months(now, sign * months) + timedelta(seconds=sign * secs)
        if clock is None and part is None and day is None:
            return When(base, False, True, True, phrase)
        day = day or base.date()
        if clock is None and part is None:
            return When(datetime.combine(day, now.time()).replace(microsecond=0), True, False, True, phrase)
    if off is not None and day is None:
        day = today + timedelta(days=off)
    if day is None and clock is None and part is None:
        return None
    date_given = day is not None and day != today
    if clock is not None:
        h, mi, mode = clock
        dt = _apply_hour(h, mi, mode, part, day or today, now, date_given)
        if day is None and dt <= now and mode == "24":
            dt += timedelta(days=1)  # "klokken 14" said at 15:00 means tomorrow
        return When(dt, day is not None, True, False, phrase, label)
    if part is not None:
        h, mi, _ = PARTS[part]
        d = day or today
        dt = datetime.combine(d, datetime.min.time()).replace(hour=h, minute=mi)
        if day is None and off is None and dt <= now and part in ("early", "morning"):
            dt += timedelta(days=1)  # "tidlig" said in the evening means tomorrow morning
        return When(dt, True, True, False, phrase, label)
    dt = datetime.combine(day, datetime.min.time()).replace(hour=default_time[0], minute=default_time[1])
    return When(dt, True, False, False, phrase, label)


def without_time(text):
    """The words left once the time phrase is removed: 'ring til mor om en time' -> 'ring til mor'."""
    scan = Scan(text)
    _clock(scan)
    take_duration(scan)
    scan.take(r"\b(?:siden|ago)\b")
    _day_offset(scan)
    try:
        _date(scan, date(2000, 1, 1), "future")
    except ValueError:
        pass
    _part(scan)
    rest = " " + scan.rest() + " "
    rest = re.sub(r"^ (?:mind mig om at|mind mig om|husk mig på at|husk mig på|husk at|remind me to|remind me|"
                  r"sæt en timer(?: på)?|set a timer(?: for)?|start en timer(?: på)?) ", " ", rest)
    edge = r"(?:om|in|på|on|at|for|og|and|klokken|til|ved|after|efter|kl)"
    while True:
        cut = re.sub(rf"^ {edge} | {edge} $", " ", rest)
        if cut == rest:
            break
        rest = cut
    return rest.strip()


# ---- saying it --------------------------------------------------------------------------------

def say_date(d, today=None):
    year = "" if today is not None and d.year == today.year else f" {d.year}"
    return f"{DAY_NAMES[d.weekday()]} {d.day} {MONTH_NAMES[d.month - 1]}{year}"


def say_when(dt, now):
    if dt.date() == now.date():
        day = "today"
    elif dt.date() == now.date() + timedelta(days=1):
        day = "tomorrow"
    else:
        day = say_date(dt.date(), now.date())
    return f"{day} at {dt:%H:%M}"


def say_span(seconds):
    """3725 -> '1 hour and 2 minutes'; under two minutes in seconds."""
    s = int(round(abs(seconds)))
    if s < 120:
        return f"{s} second{'s' * (s != 1)}"
    if s >= 3600:
        s = (s + 30) // 60 * 60  # whole minutes
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    parts = [f"{v} {u}{'s' * (v != 1)}" for v, u in ((d, "day"), (h, "hour"), (m, "minute"), (s, "second")) if v]
    if d:
        parts = parts[:2]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
