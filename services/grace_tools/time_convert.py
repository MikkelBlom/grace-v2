"""Unit and currency conversion for the time group (grace_tools/time.py).

Units are offline tables with the names he says (Danish and English). Currency uses the ECB's
daily euro reference rates (free, no key), cached for a day under ctx.data_dir; when the ECB
can't be reached, a stale cache is used and the answer says how old it is.
"""

import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

from . import time_parse as tp

ECB_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
CACHE_H = 24

# ---- units ------------------------------------------------------------------------------------

UNITS = {}  # alias -> (dimension, factor to the base unit, name to say)


def _add(dim, factor, name, *aliases):
    for a in (name, *aliases):
        UNITS[a] = (dim, factor, name)


for _f, _n, *_a in [
        (0.001, "millimeters", "mm", "millimeter", "millimetre", "millimetres"),
        (0.01, "centimeters", "cm", "centimeter", "centimetre", "centimetres"),
        (0.1, "decimeters", "dm", "decimeter"), (1, "meters", "m", "meter", "metre", "metres"),
        (1000, "kilometers", "km", "kilometer", "kilometre", "kilometres"),
        (0.0254, "inches", "inch", "in", "tomme", "tommer", '"'), (0.3048, "feet", "foot", "ft", "fod"),
        (0.9144, "yards", "yard", "yd"),
        (1609.344, "miles", "mile", "mi", "mil", "engelske mil", "engelsk mil"),
        (1852, "nautical miles", "nautical mile", "sømil", "nm")]:
    _add("length", _f, _n, *_a)
for _f, _n, *_a in [
        (1e-6, "milligrams", "mg", "milligram"), (0.001, "grams", "g", "gram", "gr"),
        (1, "kilograms", "kg", "kilo", "kilogram", "kilos"), (1000, "tonnes", "ton", "tons", "tonne", "t"),
        (0.45359237, "pounds", "pound", "lb", "lbs", "pund"), (0.028349523125, "ounces", "ounce", "oz", "unse", "unser"),
        (6.35029318, "stone", "stones", "st")]:
    _add("mass", _f, _n, *_a)
for _f, _n, *_a in [
        (0.001, "milliliters", "ml", "milliliter", "millilitre"), (0.01, "centiliters", "cl", "centiliter"),
        (0.1, "deciliters", "dl", "deciliter"), (1, "liters", "l", "liter", "litre", "litres"),
        (1000, "cubic meters", "m3", "m³", "kubikmeter", "cubic meter"),
        (0.005, "teaspoons", "tsk", "teske", "teskeer", "teaspoon", "tsp"),
        (0.015, "tablespoons", "spsk", "spiseske", "spiseskeer", "tablespoon", "tbsp"),
        (0.25, "Danish cups (kop)", "kop", "kopper"), (0.2365882365, "US cups", "cup", "cups"),
        (3.785411784, "US gallons", "gallon", "gallons", "gal"), (4.54609, "imperial gallons", "imperial gallon"),
        (0.473176473, "US pints", "pint", "pints"), (0.56826125, "UK pints", "uk pint", "britisk pint"),
        (0.0295735295625, "fluid ounces", "fl oz", "fluid ounce")]:
    _add("volume", _f, _n, *_a)
for _f, _n, *_a in [
        (1, "meters per second", "m/s", "meter i sekundet", "meter per sekund", "mps"),
        (1 / 3.6, "km/h", "km/t", "kmt", "kmh", "kph", "kilometer i timen", "km i timen", "kilometers per hour"),
        (0.44704, "miles per hour", "mph", "mil i timen"), (0.514444, "knots", "knob", "knot", "kn")]:
    _add("speed", _f, _n, *_a)
for _f, _n, *_a in [
        (1e-4, "square centimeters", "cm2", "cm²", "kvadratcentimeter"), (1, "square meters", "m2", "m²", "kvadratmeter", "kvm"),
        (1e6, "square kilometers", "km2", "km²", "kvadratkilometer"), (1e4, "hectares", "ha", "hektar", "hectare"),
        (4046.8564224, "acres", "acre"), (5516, "tønder land", "tønde land", "tdr land"),
        (0.09290304, "square feet", "sq ft", "ft2", "square foot")]:
    _add("area", _f, _n, *_a)
for _f, _n, *_a in [
        (1, "seconds", "s", "sek", "sekund", "sekunder", "second", "sec"), (60, "minutes", "min", "minut", "minutter", "minute"),
        (3600, "hours", "h", "time", "timer", "hour"), (86400, "days", "dag", "dage", "day", "døgn"),
        (604800, "weeks", "uge", "uger", "week"), (31556952, "years", "år", "year")]:
    _add("time", _f, _n, *_a)
for _f, _n, *_a in [
        (0.125, "bits", "bit"), (1, "bytes", "b", "byte"), (1e3, "kilobytes", "kb", "kilobyte"),
        (1e6, "megabytes", "mb", "megabyte"), (1e9, "gigabytes", "gb", "gigabyte"), (1e12, "terabytes", "tb", "terabyte"),
        (2 ** 20, "mebibytes", "mib"), (2 ** 30, "gibibytes", "gib")]:
    _add("data", _f, _n, *_a)
for _f, _n, *_a in [
        (1, "joules", "j", "joule"), (1000, "kilojoules", "kj", "kilojoule"), (4184, "kilocalories", "kcal", "kalorier",
        "kalorie", "calories", "calorie", "kilokalorier"), (3600, "watt hours", "wh"), (3.6e6, "kilowatt hours", "kwh",
        "kilowatttimer", "kilowatt-timer")]:
    _add("energy", _f, _n, *_a)
for _f, _n, *_a in [
        (1e5, "bar", "bars"), (100, "hectopascals", "hpa", "hektopascal", "millibar", "mbar"),
        (6894.757, "psi", "pounds per square inch"), (101325, "atmospheres", "atm")]:
    _add("pressure", _f, _n, *_a)
TEMPS = {"celsius": "C", "c": "C", "°c": "C", "grader": "C", "grader celsius": "C", "degrees celsius": "C",
         "fahrenheit": "F", "f": "F", "°f": "F", "grader fahrenheit": "F", "degrees fahrenheit": "F",
         "kelvin": "K", "k": "K"}
TEMP_NAMES = {"C": "degrees Celsius", "F": "degrees Fahrenheit", "K": "kelvin"}


def unit(name):
    u = " ".join((name or "").lower().replace("°", "° ").replace("° ", "°").split()).strip(" .")
    u = u.replace("per time", "i timen").replace("pr. time", "i timen")
    if u in TEMPS:
        return ("temp", TEMPS[u], TEMP_NAMES[TEMPS[u]])
    for cand in (u, u.rstrip("s"), re.sub(r"(er|e)$", "", u)):
        if cand in UNITS:
            return UNITS[cand]
    return None


def value_of(v):
    """A number from Gemma or his words: 5, '3,5', '1.500' (Danish thousands), 'fem', 'to hundrede'."""
    if isinstance(v, (int, float)):
        return float(v)
    t = str(v or "").strip().lower()
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+(,\d+)?", t):
        t = t.replace(".", "")
    t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        pass
    total, cur = 0, 0
    words = [w for w in t.replace("-", " ").split() if w not in ("og", "and")]
    if not words:
        return None
    for w in words:
        x = tp.number(w)
        if x is None:
            return None
        if x in (100, 1000):
            cur = (cur or 1) * x
            if x == 1000:
                total, cur = total + cur, 0
        else:
            cur += x
    return float(total + cur)


def money(v):
    return f"{v:,.0f}" if abs(v - round(v)) < 0.005 else f"{v:,.2f}"


def fmt(v):
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e15:
        return f"{int(round(v)):,}"
    if abs(v) >= 100:
        return f"{v:,.1f}"
    if abs(v) >= 0.01:
        return f"{v:.4g}" if abs(v) < 10 else f"{v:.2f}"
    return f"{v:.3g}"


def convert_units(value, frm, to):
    """(number, text) or raises ValueError with a reason Gemma can say."""
    x = value_of(value)
    if x is None:
        raise ValueError(f"{value!r} isn't a number")
    a, b = unit(frm), unit(to)
    if a is None or b is None:
        raise ValueError(f"I don't know the unit {frm if a is None else to!r}")
    if a[0] != b[0]:
        raise ValueError(f"{a[2]} and {b[2]} measure different things")
    if a[0] == "temp":
        c = {"C": x, "F": (x - 32) * 5 / 9, "K": x - 273.15}[a[1]]
        y = {"C": c, "F": c * 9 / 5 + 32, "K": c + 273.15}[b[1]]
    else:
        y = x * a[1] / b[1]
    # Danish "mil" is used for English miles today (the old Danish mil was 7.5 km): say which.
    note = " (English/US miles)" if "mil" in (frm.strip().lower(), to.strip().lower()) else ""
    return y, f"{fmt(x)} {a[2]} = {fmt(y)} {b[2]}{note}."


# ---- currency ---------------------------------------------------------------------------------

CURRENCIES = {
    "DKK": ("Danish kroner", ["kroner", "krone", "kr", "dkk", "danske kroner", "dansk krone", "danish kroner", "kroners"]),
    "EUR": ("euros", ["euro", "euros", "eur", "€"]),
    "USD": ("US dollars", ["dollar", "dollars", "usd", "$", "amerikanske dollar", "us dollar", "us dollars", "amerikanske dollars"]),
    "GBP": ("British pounds", ["pund", "gbp", "£", "britiske pund", "engelske pund", "pound", "pounds", "sterling", "pounds sterling"]),
    "SEK": ("Swedish kronor", ["svenske kroner", "svenske", "sek", "swedish kronor", "swedish krona", "svensk krone"]),
    "NOK": ("Norwegian kroner", ["norske kroner", "norske", "nok", "norwegian kroner", "norsk krone"]),
    "ISK": ("Icelandic krónur", ["islandske kroner", "isk", "icelandic kronur"]),
    "CHF": ("Swiss francs", ["franc", "francs", "schweizerfranc", "schweiziske franc", "chf", "swiss francs", "swiss franc"]),
    "JPY": ("Japanese yen", ["yen", "jpy", "japanske yen"]),
    "PLN": ("Polish złoty", ["zloty", "złoty", "pln", "polske zloty"]),
    "CZK": ("Czech koruna", ["czk", "koruna", "tjekkiske kroner"]),
    "HUF": ("Hungarian forint", ["huf", "forint"]),
    "RON": ("Romanian lei", ["ron", "lei", "leu"]),
    "TRY": ("Turkish lira", ["try", "lira", "tyrkiske lira"]),
    "CAD": ("Canadian dollars", ["cad", "canadiske dollar", "canadian dollars", "canadian dollar"]),
    "AUD": ("Australian dollars", ["aud", "australske dollar", "australian dollars", "australian dollar"]),
    "NZD": ("New Zealand dollars", ["nzd", "new zealandske dollar", "new zealand dollars"]),
    "CNY": ("Chinese yuan", ["cny", "yuan", "renminbi", "kinesiske yuan"]),
    "HKD": ("Hong Kong dollars", ["hkd", "hongkong dollar", "hong kong dollars"]),
    "SGD": ("Singapore dollars", ["sgd", "singapore dollar", "singapore dollars"]),
    "INR": ("Indian rupees", ["inr", "rupees", "rupee", "rupi", "indiske rupees"]),
    "IDR": ("Indonesian rupiah", ["idr", "rupiah"]), "ILS": ("Israeli shekels", ["ils", "shekel", "shekels"]),
    "KRW": ("South Korean won", ["krw", "won"]), "MXN": ("Mexican pesos", ["mxn", "mexicanske pesos", "pesos", "peso"]),
    "MYR": ("Malaysian ringgit", ["myr", "ringgit"]), "PHP": ("Philippine pesos", ["php", "filippinske pesos"]),
    "THB": ("Thai baht", ["thb", "baht"]), "ZAR": ("South African rand", ["zar", "rand"]),
    "BRL": ("Brazilian reais", ["brl", "real", "reais", "brasilianske real"]),
}
CURRENCY_ALIASES = {a: code for code, (_, aliases) in CURRENCIES.items() for a in aliases}


def currency(name):
    c = " ".join((name or "").lower().split()).strip(" .")
    if c.upper() in CURRENCIES:
        return c.upper()
    return CURRENCY_ALIASES.get(c) or CURRENCY_ALIASES.get(c.rstrip("s"))


def fetch_ecb():
    """The ECB's daily reference rates XML (the OS layer: tests replace this)."""
    req = urllib.request.Request(ECB_URL, headers={"User-Agent": "Grace (personal assistant)"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read(200_000).decode("utf-8")


def parse_ecb(xml):
    root = ET.fromstring(xml)
    rates, day = {"EUR": 1.0}, None
    for cube in root.iter():
        if cube.get("time"):
            day = cube.get("time")
        if cube.get("currency"):
            rates[cube.get("currency")] = float(cube.get("rate"))
    if len(rates) < 5:
        raise ValueError("the ECB answer had no rates")
    return day, rates


def rates(data_dir, now):
    """(rates per euro, the ECB's date, note) with a day's cache; a stale cache beats nothing."""
    path = Path(data_dir) / "ecb-rates.json"
    cached = None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    if cached and now - datetime.fromisoformat(cached["fetched"]) < timedelta(hours=CACHE_H):
        return cached["rates"], cached["date"], ""
    try:
        day, r = parse_ecb(fetch_ecb())
    except Exception as e:
        if cached:
            return cached["rates"], cached["date"], f" The ECB couldn't be reached ({e}), so these are the rates of {cached['date']}."
        raise RuntimeError(f"the ECB's exchange rates couldn't be fetched ({e})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fetched": now.isoformat(), "date": day, "rates": r}), encoding="utf-8")
    return r, day, ""


def convert_currency(data_dir, now, amount, frm, to):
    x = value_of(amount)
    if x is None:
        raise ValueError(f"{amount!r} isn't an amount")
    a, b = currency(frm), currency(to or "DKK")
    if a is None or b is None:
        raise ValueError(f"I don't know the currency {frm if a is None else to!r}")
    r, day, note = rates(data_dir, now)
    if a not in r or b not in r:
        raise ValueError(f"the ECB has no rate for {a if a not in r else b}")
    y = x / r[a] * r[b]
    return y, (f"{money(x)} {CURRENCIES[a][0]} = {money(y)} {CURRENCIES[b][0]} "
               f"(ECB reference rate of {day}; banks and cards add a margin).{note}")
