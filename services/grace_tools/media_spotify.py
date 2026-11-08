"""Spotify helpers for the built-in `media` tool (services/tools_server.py): which kind of thing he
asked to play (a song, a playlist, an album, an artist) and which search hit is it. Registers no
tools; it lives here so tests can import it without loading the tools server (TOOLS-ROADMAP P1:
"afspil noget Gilli", playlists and albums).

Opening a track link plays it. Opening a playlist, album or artist link only shows it in the
Spotify app (no Web API without a client id, TOOLS-ROADMAP P3), so the tool checks Spotify's
window title afterwards and says honestly whether it started.
"""

import re

KINDS = ("track", "playlist", "album", "artist")
# Words in his request that name the kind, Danish and English; removed from the search query.
KIND_WORDS = {
    "playlist": r"playlisten|playlist(?:en)?|spillelisten|spilleliste|afspilningslisten|afspilningsliste",
    "album": r"albummet|album(?:et)?|pladen|plade",
    "artist": r"kunstneren|artisten|artist(?:en)?|bandet",
}


def kind_of(query, kind=None):
    """The kind Gemma passed, else the one his words name ("playlisten chill"), else a track.
    "noget af/med X" (something by X) is an artist."""
    if kind in KINDS:
        return kind
    q = (query or "").lower()
    for k, pat in KIND_WORDS.items():
        if re.search(rf"(?<![a-zæøå])(?:{pat})(?![a-zæøå])", q):
            return k
    if re.match(r"\s*(noget|something)\b", q):
        return "artist"
    return "track"


def clean_query(query):
    """The words to search for: without kind words, 'noget af', 'by'/'af'/'med' and 'min/my'
    (articles stay: song titles have them)."""
    q = query or ""
    for pat in KIND_WORDS.values():
        q = re.sub(rf"(?i)(?<![a-zæøå])(?:{pat})(?![a-zæøå])", " ", q)
    q = re.sub(r"(?i)^\s*(noget|something)\b", " ", q)
    q = re.sub(r"(?i)\b(by|af|med|min|mine|my)\b", " ", q)
    return " ".join(q.split())


def pick(hits, kind):
    """(title, url, id) of the first search hit that is a Spotify page of that kind, or None.
    hits are (title, url, snippet) as tools_server.search returns them."""
    for title, url, *_ in hits:
        m = re.search(rf"open\.spotify\.com/(?:intl-[a-z]+/)?{kind}/([A-Za-z0-9]{{10,}})", url)
        if m:
            return clean_title(title), url, m.group(1)
    return None


def clean_title(title):
    """'Peace and Love - Album by Gilli | Spotify' -> 'Peace and Love by Gilli'."""
    t = re.split(r"\s+\|\s+Spotify", title or "")[0]
    t = re.sub(r"(?i)\s+-\s+(song|single|album|ep|playlist)\s+(by|af)\s+", " by ", t)
    t = re.sub(r"(?i)\s+-\s+(song|single|album|ep|playlist)\b.*$", "", t)
    return t.strip()


def started(before, after):
    """Spotify's window title went from `before` to a song (not paused, not the same song)."""
    return bool(after) and after != before
