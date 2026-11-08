"""services/grace_tools/media_spotify.py: what kind of thing he asked to play and which search hit
it is (the built-in media tool's playlists and albums)."""

import pytest

from grace_tools import media_spotify as M


@pytest.mark.parametrize("query,kind,want", [
    ("Peace and Love Gilli", None, "track"),
    ("playlisten chill vibes", None, "playlist"),
    ("spillelisten Danske hits", None, "playlist"),
    ("albummet Euforia af Gilli", None, "album"),
    ("noget af Gilli", None, "artist"),
    ("something by Kesi", None, "artist"),
    ("Gilli", "album", "album"),        # Gemma's choice wins
    ("Gilli", "bogus", "track"),        # an unknown kind is ignored
    ("Platformen", None, "track"),      # "plade" inside a word is not the album word
])
def test_kind_of(query, kind, want):
    assert M.kind_of(query, kind) == want


@pytest.mark.parametrize("query,want", [
    ("playlisten chill vibes", "chill vibes"),
    ("albummet Euforia af Gilli", "Euforia Gilli"),
    ("noget af Gilli", "Gilli"),
    ("min playlist Det Vi Har", "Det Vi Har"),
])
def test_clean_query(query, want):
    assert M.clean_query(query) == want


def test_pick_takes_the_first_hit_of_the_kind():
    hits = [
        ("Gilli - Wikipedia", "https://da.wikipedia.org/wiki/Gilli", ""),
        ("Euforia - song by Gilli | Spotify", "https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC", ""),
        ("Euforia - Album by Gilli | Spotify", "https://open.spotify.com/intl-da/album/1aBcDeFgHiJkLmNoPqRsTu?si=x", ""),
    ]
    assert M.pick(hits, "album") == ("Euforia by Gilli", hits[2][1], "1aBcDeFgHiJkLmNoPqRsTu")
    assert M.pick(hits, "track")[2] == "4uLU6hMCjMI75M1A2tKUQC"
    assert M.pick(hits, "playlist") is None


@pytest.mark.parametrize("title,want", [
    ("Danske Hits 2026 | Spotify Playlist", "Danske Hits 2026"),
    ("Chill Vibes - playlist by Spotify | Spotify", "Chill Vibes by Spotify"),
    ("Gilli | Spotify", "Gilli"),
])
def test_clean_title(title, want):
    assert M.clean_title(title) == want


def test_started_needs_a_new_song():
    assert M.started(None, "Gilli - Euforia")
    assert M.started("Kesi - Søvnløs", "Gilli - Euforia")
    assert not M.started("Gilli - Euforia", "Gilli - Euforia")
    assert not M.started("Gilli - Euforia", None)
