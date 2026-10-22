"""The system group (services/grace_tools/system.py) against a fake OS: no real window, setting,
process or network is touched. Every system_os function the tools use is replaced in `fake`."""

import pytest

import grace_tools
from grace_tools import registry
from grace_tools import system
from grace_tools import system_os as osl


def run(ctx, tool_name, confirmed=False, **args):
    """conftest.call, but these tools take a `name` argument, which collides with call's own."""
    grace_tools.load()
    out = registry.run(ctx, tool_name, args, lambda: True, confirmed)
    assert out is not None, f"no tool {tool_name}"
    assert isinstance(out.get("result"), str), out
    return out


def win(hwnd, title, exe, pid, minimized=False, maximized=False):
    return {"hwnd": hwnd, "title": title, "exe": exe, "pid": pid, "minimized": minimized, "maximized": maximized}


class FakeOS:
    def __init__(self):
        self.windows = [
            win(1, "Gmail - Indbakke – Google Chrome", "chrome.exe", 100),
            win(2, "eksamen.docx - Word", "WINWORD.EXE", 200),
            win(3, "Lommeregner", "ApplicationFrameHost.exe", 300),
            win(4, "Projects – Stifinder", "explorer.exe", 400),
            win(5, "Messenger – Google Chrome", "chrome.exe", 100),
            win(6, "eksamen - Google Søgning – Google Chrome", "chrome.exe", 100),
            win(7, "Windows PowerShell", "WindowsTerminal.exe", 700),
        ]
        self.fg = 1
        self.procs = [{"pid": 100, "name": "chrome.exe"}, {"pid": 101, "name": "chrome.exe"},
                      {"pid": 102, "name": "chrome.exe"}, {"pid": 200, "name": "WINWORD.EXE"},
                      {"pid": 400, "name": "explorer.exe"}, {"pid": 600, "name": "lsass.exe"},
                      {"pid": 610, "name": "llama-server.exe"}, {"pid": 620, "name": "python.exe"},
                      {"pid": 630, "name": "python.exe"}, {"pid": 640, "name": "MsMpEng.exe"},
                      {"pid": 650, "name": "svchost.exe"}, {"pid": 660, "name": "sursvc.exe"},
                      {"pid": 670, "name": "notepad.exe"}, {"pid": 680, "name": "seccenter.exe"},
                      {"pid": 990, "name": "python.exe"}]
        self.cmdlines = {620: r"C:\Users\mikke\Desktop\Projects\Grace\bench\.venv\python.exe services\speech_server.py",
                         630: r"python C:\Users\mikke\Documents\other\script.py"}
        self.paths = {680: r"C:\Program Files\Bitdefender\Bitdefender Security App\seccenter.exe",
                      100: r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                      400: r"C:\Windows\explorer.exe"}
        self.killed, self.launched, self.focused, self.closed, self.shown, self.rects = [], [], [], [], [], []
        self.monitors = [(0, 0, 2560, 1560), (2560, 0, 5120, 1400)]
        self.place = ((100, 100, 1100, 900), (0, 0, 2560, 1560))
        self.devices = [
            {"id": "spk", "name": "Speakers (Realtek(R) Audio)", "default": False, "connected": True},
            {"id": "logi", "name": "Højttalere (Logitech Z337)", "default": True, "connected": True},
            {"id": "mon", "name": "MSI MAG272QP (NVIDIA High Definition Audio)", "default": False, "connected": True},
            {"id": "jbl", "name": "Hovedtelefoner (JBL Tune 777NC)", "default": False, "connected": True},
            {"id": "jblhf", "name": "Headset (JBL Tune 777NC Hands-Free AG Audio)", "default": False, "connected": True},
            {"id": "buds", "name": "Hovedtelefoner (Mikkels Buds4 Pro)", "default": False, "connected": False},
            {"id": "jack", "name": "Headphones (Realtek(R) Audio)", "default": False, "connected": False},
        ]
        self.master, self.muted = (70, False)
        self.sessions = [{"pid": 900, "name": "Spotify.exe", "percent": 50, "muted": False},
                         {"pid": 100, "name": "chrome.exe", "percent": 100, "muted": False}]
        self.set_volume, self.set_app = [], []
        self.mic = ("DJI Mic (USB Audio Device)", False)
        self.mic_set = []
        self.radios = {"wifi": True, "bluetooth": True}
        self.radio_set = []
        self.bt = [{"name": "Logitech Z337", "connected": True}, {"name": "MX Master 3S", "connected": True},
                   {"name": "JBL Tune 777NC", "connected": False}]
        self.interfaces = ("    Navn                   : Wi-Fi\n    Tilstand               : tilsluttet\n"
                           "    SSID                   : HjemmeNet\n    Bånd                   : 5 GHz\n"
                           "    Signal                 : 88% \n")
        self.networks = ("SSID 1 : HjemmeNet\n    BSSID 1 : aa\n         Signal             : 88%  \n"
                         "    BSSID 2 : bb\n         Signal             : 40%  \n"
                         "SSID 2 : \n    BSSID 1 : cc\n         Signal             : 30%  \n"
                         "SSID 3 : Naboen\n    BSSID 1 : dd\n         Signal             : 61%  \n")
        self.dark, self.dark_set = True, []
        self.quiet, self.quiet_set, self.quiet_sticks = 0, [], True
        self.bright, self.bright_set = 80, []
        self.night = bytes.fromhex("434201000a0201002a06f2c4f5d5062a2b0e1343420100d00a02c6148af99cf899a2d4ee0100000000")
        self.online = True
        self.dns = {"www.google.com": ["142.250.74.4"], "dr.dk": ["95.100.155.139"], "facebook.com": ["157.240.1.35"]}
        self.http = {"https://dr.dk/": (200, 120.0, None), "https://facebook.com/": (503, 300.0, None)}
        self.addresses = [("Wi-Fi", "192.168.0.103"), ("VirtualBox", "192.168.56.1")]
        self.ports = [(135, "0.0.0.0", 1000, "svchost.exe"), (8080, "127.0.0.1", 610, "llama-server.exe"),
                      (7813, "127.0.0.1", 990, "python.exe")]
        self.clip, self.clip_seq, self.typed, self.copies = "gammelt klip", 1, [], 0
        self.selection = "Det markerede afsnit."
        self.locked = self.slept = False

    def install(self, mp):
        f = self
        mp.setattr(osl, "windows", lambda: [dict(w) for w in f.windows])
        mp.setattr(osl, "foreground", lambda: next((dict(w) for w in f.windows if w["hwnd"] == f.fg), None))
        mp.setattr(osl, "window_exists", lambda h: any(w["hwnd"] == h for w in f.windows))
        mp.setattr(osl, "focus", lambda h: (f.focused.append(h), setattr(f, "fg", h), True)[-1])
        mp.setattr(osl, "show", lambda h, how: f.shown.append((h, how)))
        mp.setattr(osl, "monitors", lambda: list(f.monitors))
        mp.setattr(osl, "window_place", lambda h: f.place)
        mp.setattr(osl, "set_rect", lambda h, r: f.rects.append((h, r)))
        mp.setattr(osl, "close", lambda h: f.closed.append(h))
        mp.setattr(osl, "minimize_all", lambda: f.shown.append(("all", "minimize")))
        mp.setattr(osl, "processes", lambda sample=0.5: [
            {**p, "cpu": {"chrome.exe": 4.0, "llama-server.exe": 30.0}.get(p["name"], 0.5),
             "ram_mb": {"chrome.exe": 900.0, "llama-server.exe": 4500.0, "WINWORD.EXE": 300.0}.get(p["name"], 50.0)}
            for p in f.procs])
        mp.setattr(osl, "process_names", lambda: [dict(p) for p in f.procs])
        mp.setattr(osl, "cmdline", lambda pid: f.cmdlines.get(pid))
        mp.setattr(osl, "exe_path", lambda pid: f.paths.get(pid))
        mp.setattr(osl, "own_pids", lambda: {990})
        mp.setattr(osl, "kill", lambda pid: (f.killed.append(pid), setattr(f, "procs", [p for p in f.procs if p["pid"] != pid]))[0] and None)
        mp.setattr(osl, "launch", lambda path: f.launched.append(path))
        mp.setattr(osl, "audio_devices", lambda kind="output": [dict(d) for d in f.devices])
        mp.setattr(osl, "set_default_audio", lambda i: [d.update(default=d["id"] == i) for d in f.devices])
        mp.setattr(osl, "master_volume", lambda: (f.master, f.muted))
        mp.setattr(osl, "set_master_volume", lambda percent=None, mute=None: f.set_volume.append((percent, mute)))
        mp.setattr(osl, "app_volumes", lambda: [dict(x) for x in f.sessions])
        mp.setattr(osl, "set_app_volume", lambda pid, percent=None, mute=None: f.set_app.append((pid, percent, mute)))
        mp.setattr(osl, "mic_state", lambda: f.mic)
        mp.setattr(osl, "set_mic_mute", lambda m: f.mic_set.append(m))
        mp.setattr(osl, "radio_state", lambda kind: f.radios.get(kind))
        mp.setattr(osl, "set_radio", lambda kind, on: f.radio_set.append((kind, on)))
        mp.setattr(osl, "bluetooth_devices", lambda: [dict(d) for d in f.bt])
        mp.setattr(osl, "netsh_interfaces", lambda: f.interfaces)
        mp.setattr(osl, "netsh_networks", lambda: f.networks)
        mp.setattr(osl, "dark_mode", lambda: f.dark)
        mp.setattr(osl, "set_dark_mode", lambda d: f.dark_set.append(d))
        mp.setattr(osl, "quiet_hours", lambda: f.quiet)

        def set_quiet(v):
            f.quiet_set.append(v)
            if f.quiet_sticks:
                f.quiet = v
        mp.setattr(osl, "set_quiet_hours", set_quiet)
        mp.setattr(osl, "brightness", lambda: f.bright)
        mp.setattr(osl, "set_brightness", lambda v: f.bright_set.append(v))
        mp.setattr(osl, "night_light_blob", lambda: f.night)
        mp.setattr(osl, "set_night_light_blob", lambda b: setattr(f, "night", b))
        mp.setattr(osl, "tcp_ok", lambda host, port, timeout=2.0: 12.0 if f.online else None)
        mp.setattr(osl, "resolve", lambda name: list(f.dns.get(name, [])) if f.online else [])
        mp.setattr(osl, "local_addresses", lambda: list(f.addresses))
        mp.setattr(osl, "http_check", lambda url, timeout=6: f.http.get(url, (None, 0.0, "timed out")))
        mp.setattr(osl, "download_mbps", lambda alive: 240.0)
        mp.setattr(osl, "listening_ports", lambda: list(f.ports))
        mp.setattr(osl, "clipboard_get", lambda: f.clip)

        def clip_set(t):
            f.clip = t
            f.clip_seq += 1
        mp.setattr(osl, "clipboard_set", clip_set)
        mp.setattr(osl, "clipboard_seq", lambda: f.clip_seq)

        def copy():
            f.copies += 1
            if f.selection is not None:
                clip_set(f.selection)
        mp.setattr(osl, "press_copy", copy)
        mp.setattr(osl, "type_text", lambda t: f.typed.append(t))
        mp.setattr(osl, "lock", lambda: setattr(f, "locked", True))
        mp.setattr(osl, "sleep", lambda: setattr(f, "slept", True))
        mp.setattr(osl, "cpu_ram", lambda: (23.0, 31.5, 63.6))
        mp.setattr(osl, "battery", lambda: (64, False, 135))
        mp.setattr(osl, "disks", lambda: [("C:", 161.0, 1878.0)])
        mp.setattr(osl, "gpus", lambda: [("NVIDIA GeForce RTX 5090 Laptop GPU", 31.0, 15800.0, 24463.0, 71.0)])
        mp.setattr(system.time, "sleep", lambda s: None)


@pytest.fixture
def fake(monkeypatch):
    f = FakeOS()
    f.install(monkeypatch)
    return f


def yes(ctx, out):
    """What the engine does after his yes: run the stored call, confirmed."""
    assert out.get("needs_confirm"), out
    name, args, summary, _ = ctx.confirmations.take(out["confirm"]["token"])
    return run(ctx, name, confirmed=True, **args)


# ---- windows ----------------------------------------------------------------------------------

def test_list_windows(ctx, fake):
    out = run(ctx, "list_windows")
    assert "7 windows" in out["result"] and "Google Chrome ('Gmail - Indbakke" in out["result"]
    assert "(in front)" in out["result"] and "Lommeregner" in out["result"]


def test_focus_danish_names(ctx, fake):
    out = run(ctx, "focus_window", name="skift til lommeregneren")
    assert out["ok"] and fake.focused == [3] and "Lommeregner" in out["result"]
    out = run(ctx, "focus_window", name="stifinderen")
    assert fake.focused[-1] == 4 and "File Explorer" in out["result"]
    out = run(ctx, "focus_window", name="word")
    assert fake.focused[-1] == 2


def test_focus_chrome_while_chrome_in_front_takes_the_next_one(ctx, fake):
    out = run(ctx, "focus_window", name="chrome")
    assert fake.focused == [5] and "Messenger" in out["result"]


def test_focus_ambiguous_title_asks(ctx, fake):
    out = run(ctx, "focus_window", name="eksamen")
    assert not out["ok"] and "Ask him which" in out["result"] and not fake.focused
    assert "Word" in out["result"] and "Google Chrome" in out["result"]


def test_focus_unknown_lists_windows(ctx, fake):
    out = run(ctx, "focus_window", name="photoshop")
    assert not out["ok"] and "open_app" in out["result"] and "Google Chrome" in out["result"]


def test_focus_dry_run(dry_ctx, fake):
    out = run(dry_ctx, "focus_window", name="word")
    assert out["detail"]["dry_run"] and not fake.focused and "Word" in out["result"]


def test_snap_and_other_monitor(ctx, fake):
    run(ctx, "window_action", action="snap_right", name="word")
    assert fake.rects[-1] == (2, (1280, 0, 2560, 1560))
    out = run(ctx, "window_action", action="other_monitor")  # the window in front
    assert fake.rects[-1] == (1, (2660, 100, 3660, 900)) and "other monitor" in out["result"]
    fake.monitors = fake.monitors[:1]
    out = run(ctx, "window_action", action="other_monitor")
    assert not out["ok"] and "one monitor" in out["result"]


def test_other_monitor_rect_fits_a_smaller_screen():
    assert system.other_monitor_rect((2000, 1000, 2500, 1500), (0, 0, 2560, 1560),
                                     [(0, 0, 2560, 1560), (2560, 0, 4480, 1080)]) == (4480 - 500, 1080 - 500, 4480, 1080)


def test_minimise_maximise(ctx, fake):
    run(ctx, "window_action", action="minimize", name="lommeregner")
    run(ctx, "window_action", action="maximize")
    assert fake.shown == [(3, "minimize"), (1, "maximize")]


def test_close_chrome_asks_first(ctx, fake):
    out = run(ctx, "close_window", name="luk chrome")
    assert out["needs_confirm"] and not fake.closed
    assert "Gmail" in out["confirm"]["summary"] and "unsaved" in out["confirm"]["summary"]
    fake.fg = 4  # he looks elsewhere before saying yes: the planned window still closes
    out = yes(ctx, out)
    assert fake.closed == [1] and "close" in out["result"]


def test_close_invented_approval_asks_again(ctx, fake):
    out = run(ctx, "close_window", name="chrome", approval="deadbeefdeadbeef")
    assert out["needs_confirm"] and not fake.closed


def test_approval_works_once(ctx, fake):
    out = run(ctx, "close_window", name="word")
    name, args, _, _ = ctx.confirmations.take(out["confirm"]["token"])
    run(ctx, name, confirmed=True, **args)
    again = run(ctx, name, confirmed=True, **args)
    assert fake.closed == [2] and again.get("needs_confirm")


def test_close_safe_app_needs_no_yes(ctx, fake):
    out = run(ctx, "close_window", name="lommeregner")
    assert out["ok"] and fake.closed == [3]


def test_close_all_chrome_windows(ctx, fake):
    out = run(ctx, "close_window", name="chrome", all=True)
    assert "all 3 Google Chrome windows" in out["confirm"]["summary"]
    yes(ctx, out)
    assert sorted(fake.closed) == [1, 5, 6]


def test_close_dry_run_after_yes(dry_ctx, fake):
    out = yes(dry_ctx, run(dry_ctx, "close_window", name="word"))
    assert out["detail"]["dry_run"] and not fake.closed


def test_show_desktop(ctx, dry_ctx, fake):
    run(dry_ctx, "show_desktop")
    assert not fake.shown
    run(ctx, "show_desktop")
    assert fake.shown == [("all", "minimize")]


# ---- sound ------------------------------------------------------------------------------------

def test_switch_to_headphones_danish(ctx, fake):
    out = run(ctx, "audio_output", action="switch", device="skift lyden til mine høretelefoner")
    assert out["ok"] and "JBL Tune 777NC" in out["result"]
    assert next(d for d in fake.devices if d["default"])["id"] == "jbl"  # stereo, not hands-free


def test_switch_to_speakers_prefers_external(ctx, fake):
    fake.devices[1]["default"], fake.devices[3]["default"] = False, True
    out = run(ctx, "audio_output", action="switch", device="højttalerne")
    assert "Logitech" in out["result"]
    out = run(ctx, "audio_output", action="switch", device="computerens højttalere")
    assert "Realtek" in out["result"]
    out = run(ctx, "audio_output", action="switch", device="skærmen")
    assert "MSI" in out["result"]


def test_switch_to_brand(ctx, fake):
    out = run(ctx, "audio_output", action="switch", device="logitech")
    assert "already" in out["result"]


def test_switch_disconnected_headphones(ctx, fake):
    fake.devices = [d for d in fake.devices if "JBL" not in d["name"]]
    out = run(ctx, "audio_output", action="switch", device="buds")
    assert not out["ok"] and "isn't connected" in out["result"]
    fake.bt.append({"name": "Sony WH-1000XM5", "connected": False})
    out = run(ctx, "audio_output", action="switch", device="sony")
    assert not out["ok"] and "paired but not connected" in out["result"]


def test_switch_dry_run(dry_ctx, fake):
    out = run(dry_ctx, "audio_output", action="switch", device="jbl")
    assert out["detail"]["dry_run"] and next(d for d in fake.devices if d["default"])["id"] == "logi"


def test_list_outputs(ctx, fake):
    out = run(ctx, "audio_output", action="list")
    assert "Højttalere (Logitech Z337) (in use)" in out["result"] and "Not connected: Hovedtelefoner (Mikkels" in out["result"]


def test_volume_master_and_app(ctx, fake):
    out = run(ctx, "volume", action="set", level=30)
    assert fake.set_volume[-1] == (30, None) and "30 %" in out["result"]
    out = run(ctx, "volume", action="down", app="spotify")
    assert fake.set_app[-1] == (900, 40, None) and "Spotify's volume is now 40 %" in out["result"]
    out = run(ctx, "volume", action="mute", app="chrome")
    assert fake.set_app[-1] == (100, None, True)
    out = run(ctx, "volume", action="up", app="teams")
    assert not out["ok"] and "Spotify" in out["result"]


def test_volume_up_unmutes(ctx, fake):
    fake.muted = True
    run(ctx, "volume", action="up")
    assert fake.set_volume[-1] == (80, False)


def test_mic_mute_asks_unmute_does_not(ctx, fake):
    out = run(ctx, "microphone", action="mute")
    assert out["needs_confirm"] and "DJI Mic" in out["confirm"]["summary"] and not fake.mic_set
    yes(ctx, out)
    assert fake.mic_set == [True]
    fake.mic = ("DJI Mic (USB Audio Device)", True)
    out = run(ctx, "microphone", action="unmute")
    assert out["ok"] and fake.mic_set == [True, False]


# ---- Wi-Fi, Bluetooth, display ----------------------------------------------------------------

def test_wifi_status_danish_netsh(ctx, fake):
    out = run(ctx, "wifi", action="status")
    assert "HjemmeNet" in out["result"] and "88 %" in out["result"] and "5 GHz" in out["result"]


def test_wifi_networks(ctx, fake):
    out = run(ctx, "wifi", action="networks")
    assert out["result"].index("HjemmeNet 88 % (connected)") < out["result"].index("Naboen 61 %")
    assert "(hidden network)" in out["result"]


def test_wifi_off_asks_on_does_not(ctx, fake):
    out = run(ctx, "wifi", action="off")
    assert out["needs_confirm"] and "web search" in out["confirm"]["summary"] and "HjemmeNet" in out["confirm"]["summary"]
    assert not fake.radio_set
    yes(ctx, out)
    assert fake.radio_set == [("wifi", False)]
    fake.radios["wifi"] = False
    out = run(ctx, "wifi", action="on")
    assert out["ok"] and fake.radio_set[-1] == ("wifi", True)


def test_bluetooth_off_names_what_disconnects(ctx, fake):
    out = run(ctx, "bluetooth", action="off")
    assert "Logitech Z337" in out["confirm"]["summary"] and "MX Master" in out["confirm"]["summary"]
    assert not fake.radio_set
    out = run(ctx, "bluetooth", action="devices")
    assert "Connected: Logitech Z337, MX Master 3S" in out["result"] and "JBL" in out["result"]


def test_dark_mode(ctx, dry_ctx, fake):
    run(dry_ctx, "dark_mode", mode="toggle")
    assert not fake.dark_set
    out = run(ctx, "dark_mode", mode="light")
    assert fake.dark_set == [False] and "light" in out["result"]
    out = run(ctx, "dark_mode", mode="dark")
    assert "already" in out["result"]


def test_do_not_disturb_checks_it_took(ctx, fake):
    out = run(ctx, "do_not_disturb", action="on")
    assert out["ok"] and fake.quiet_set == [1]
    fake.quiet, fake.quiet_sticks = 0, False
    out = run(ctx, "do_not_disturb", action="on")
    assert not out["ok"] and "didn't accept" in out["result"]


def test_brightness(ctx, fake):
    out = run(ctx, "brightness", action="down", level=30)
    assert fake.bright_set == [50] and "50 %" in out["result"]
    fake.bright = None
    out = run(ctx, "brightness", action="set", level=40)
    assert not out["ok"] and fake.bright_set == [50]


def test_night_light_blob_round_trip(ctx, fake):
    off = fake.night
    assert system.night_light_state(off) is False
    run(ctx, "night_light", action="on")
    assert system.night_light_state(fake.night) is True and len(fake.night) == len(off) + 2
    run(ctx, "night_light", action="off")
    assert system.night_light_state(fake.night) is False and len(fake.night) == len(off)
    fake.night = b"\x00" * 40  # unknown format: nothing is written
    out = run(ctx, "night_light", action="on")
    assert not out["ok"] and fake.night == b"\x00" * 40


# ---- network ----------------------------------------------------------------------------------

def test_network_status_up(ctx, fake):
    out = run(ctx, "network_status")
    assert "internet works" in out["result"] and "HjemmeNet" in out["result"] and "192.168.0.103" in out["result"]


def test_network_status_down(ctx, fake):
    fake.online = False
    out = run(ctx, "network_status")
    assert "internet is down" in out["result"] and "router" in out["result"]
    fake.addresses = []
    assert "isn't connected to any network" in run(ctx, "network_status")["result"]


def test_site_up_and_down(ctx, fake):
    assert "dr.dk is up" in run(ctx, "site_up", site="er dr.dk nede")["result"].replace("er ", "")
    out = run(ctx, "site_up", site="facebook")
    assert not out["ok"] and "at the site" in out["result"] and "facebook.com" in out["result"]
    out = run(ctx, "site_up", site="findesikke")
    assert "doesn't resolve" in out["result"]


def test_site_up_when_own_internet_is_down(ctx, fake):
    fake.online = False
    out = run(ctx, "site_up", site="dr.dk")
    assert "own internet" in out["result"]


def test_ping_parses_danish_output(ctx, fake, monkeypatch):
    danish = ("Pinger google.com [142.250.74.46] med 32 byte data:\n"
              "Svar fra 142.250.74.46: byte=32 tid=14ms TTL=117\nSvar fra 142.250.74.46: byte=32 tid=18ms TTL=117\n"
              "Anmodningen fik timeout.\nSvar fra 142.250.74.46: byte=32 tid<1ms TTL=117\n")
    monkeypatch.setattr(osl, "run", lambda cmd, timeout=15, encoding="utf-8": danish)
    assert osl.ping("google.com") == (3, [14, 18, 1])
    monkeypatch.setattr(osl, "ping", lambda host, count=4: (3, [14, 18, 1]))
    out = run(ctx, "ping_host", host="https://google.com/search")
    assert "google.com answered 3 of 4" in out["result"] and "11 ms" in out["result"]


def test_dns_and_ports(ctx, fake):
    assert "95.100.155.139" in run(ctx, "dns_lookup", name="dr.dk")["result"]
    out = run(ctx, "listening_ports")
    assert "8080 (llama-server)" in out["result"] and "135" not in out["result"].split("The others:")[1]
    assert "llama-server" in run(ctx, "listening_ports", port=8080)["result"]
    assert "Nothing" in run(ctx, "listening_ports", port=9999)["result"]


def test_speed_test(ctx, dry_ctx, fake):
    assert "240 Mbit/s" in run(ctx, "speed_test")["result"]
    assert run(dry_ctx, "speed_test")["detail"]["dry_run"]


# ---- processes --------------------------------------------------------------------------------

def test_top_by_ram_adds_up_an_app(ctx, fake):
    out = run(ctx, "top_processes", by="ram", count=2)
    assert out["result"].index("llama-server 4.4 GB") < out["result"].index("chrome 2.6 GB (3 processes)")
    assert "31.5 of 64 GB" in out["result"]


def test_is_running(ctx, fake):
    out = run(ctx, "is_running", name="kører chrome")
    assert "Yes" in out["result"] and "3 processes" in out["result"] and "Gmail" in out["result"]
    assert "No" in run(ctx, "is_running", name="spotify")["result"]


def test_end_chrome_summary_names_pids_and_windows(ctx, fake):
    out = run(ctx, "end_process", name="chrome")
    s = out["confirm"]["summary"]
    assert "all 3 chrome.exe processes" in s and "100, 101, 102" in s and "3 open windows" in s and "unsaved" in s
    assert not fake.killed
    yes(ctx, out)
    assert fake.killed == [100, 101, 102]


def test_end_by_pid(ctx, fake):
    out = run(ctx, "end_process", pid=670)
    assert out["confirm"]["summary"] == "end notepad.exe (PID 670)"
    yes(ctx, out)
    assert fake.killed == [670]


@pytest.mark.parametrize("name, why", [
    ("lsass", "Windows system process"), ("svchost", "Windows system process"), ("explorer", "Windows system process"),
    ("stifinderen", "Windows system process"), ("llama-server", "part of Grace"), ("msmpeng", "security software"),
    ("seccenter", "security software")])
def test_protected_never_ends(ctx, fake, name, why):
    out = run(ctx, "end_process", name=name)
    assert "not end anything" in out["confirm"]["summary"] and why in out["confirm"]["summary"]
    out = yes(ctx, out)  # even after a yes
    assert not out["ok"] and not fake.killed


def test_graces_own_pythons_are_protected(ctx, fake):
    out = run(ctx, "end_process", pid=620)
    assert "Grace's services" in out["confirm"]["summary"]
    out = run(ctx, "end_process", pid=990)
    assert "own tools server" in out["confirm"]["summary"]
    out = run(ctx, "end_process", pid=630)  # someone else's script is fine
    assert out["confirm"]["summary"] == "end python.exe (PID 630)"


def test_python_by_name_is_refused_while_one_is_grace(ctx, fake):
    out = run(ctx, "end_process", name="python")
    assert "not end anything" in out["confirm"]["summary"]


def test_ambiguous_name_asks_which(ctx, fake):
    out = run(ctx, "end_process", name="svc")
    s = out["confirm"]["summary"]
    assert "ask him which" in s and "sursvc.exe" in s and "svchost.exe" in s
    assert not yes(ctx, out)["ok"] and not fake.killed


def test_end_dry_run(dry_ctx, fake):
    out = yes(dry_ctx, run(dry_ctx, "end_process", name="word"))
    assert out["detail"]["dry_run"] and not fake.killed


def test_end_skips_processes_that_changed(ctx, fake):
    out = run(ctx, "end_process", name="chrome")
    fake.procs = [p for p in fake.procs if p["pid"] != 101]  # one exited before his yes
    yes(ctx, out)
    assert fake.killed == [100, 102]


def test_restart_explorer_is_allowed(ctx, fake):
    out = run(ctx, "restart_app", name="stifinder")
    assert out["confirm"]["summary"].startswith("restart explorer.exe (PID 400)")
    out = yes(ctx, out)
    assert fake.killed == [400] and fake.launched == [r"C:\Windows\explorer.exe"]


def test_restart_chrome(ctx, fake):
    out = yes(ctx, run(ctx, "restart_app", name="chrome"))
    assert fake.killed == [100, 101, 102] and fake.launched[0].endswith("chrome.exe") and "Restarted Google Chrome" in out["result"]


def test_restart_protected(ctx, fake):
    out = yes(ctx, run(ctx, "restart_app", name="lsass"))
    assert not out["ok"] and not fake.killed


# ---- clipboard, keyboard, status --------------------------------------------------------------

def test_clipboard(ctx, dry_ctx, fake):
    assert "gammelt klip" in run(ctx, "clipboard_read")["result"]
    run(dry_ctx, "clipboard_write", text="ny tekst")
    assert fake.clip == "gammelt klip"
    run(ctx, "clipboard_write", text="ny tekst")
    assert fake.clip == "ny tekst"


def test_type_text_shows_the_text_and_asks(ctx, fake):
    out = run(ctx, "type_text", text="Hej Anne, vi ses på fredag")
    assert out["confirm"]["summary"] == ("type \"Hej Anne, vi ses på fredag\" into Google Chrome "
                                         "('Gmail - Indbakke – Google Chrome')")
    assert not fake.typed
    yes(ctx, out)
    assert fake.typed == ["Hej Anne, vi ses på fredag"]


def test_type_text_never_presses_enter_in_a_terminal(ctx, fake):
    fake.fg = 7
    out = yes(ctx, run(ctx, "type_text", text="rm -rf build\n"))
    assert not out["ok"] and not fake.typed


def test_active_window(ctx, fake):
    assert "Google Chrome" in run(ctx, "active_window")["result"]


def test_selected_text_restores_the_clipboard(ctx, fake):
    out = run(ctx, "selected_text")
    assert "Det markerede afsnit." in out["result"] and fake.clip == "gammelt klip"


def test_selected_text_nothing_selected(ctx, fake):
    fake.selection = None
    out = run(ctx, "selected_text")
    assert not out["ok"] and "Nothing is selected" in out["result"]


def test_selected_text_not_in_a_terminal(ctx, dry_ctx, fake):
    fake.fg = 7
    out = run(ctx, "selected_text")
    assert not out["ok"] and fake.copies == 0
    fake.fg = 1
    assert run(dry_ctx, "selected_text")["detail"]["dry_run"] and fake.copies == 0


def test_system_status(ctx, fake):
    r = run(ctx, "system_status")["result"]
    for part in ("CPU 23 %", "RAM 31.5 of 64 GB", "RTX 5090 Laptop GPU 31 % load", "VRAM 15.4 of 24 GB",
                 "battery 64 %, on battery, about 2 h 15 min left", "disk C: 161 GB free"):
        assert part in r, part


def test_lock_and_sleep(ctx, fake):
    run(ctx, "lock_pc")
    assert fake.locked
    out = run(ctx, "sleep_pc")
    assert out["needs_confirm"] and not fake.slept
    yes(ctx, out)
    assert fake.slept


# ---- the group as a whole ---------------------------------------------------------------------

def test_risks_and_core():
    tools = {t.name: t for t in registry.TOOLS.values() if t.group in ("system", "system_os")}
    assert {n for n, t in tools.items() if t.core} == {"focus_window", "system_status"}
    assert {n for n, t in tools.items() if t.risk == "destructive"} == {"end_process", "restart_app", "type_text",
                                                                        "sleep_pc"}
    assert all(t.group == "system" for t in tools.values())


@pytest.mark.parametrize("need, name", [
    ("kill a process that hangs", "end_process"), ("switch sound output to headphones", "audio_output"),
    ("turn wifi off", "wifi"), ("which programs use the most ram", "top_processes"),
    ("check if the internet is down", "network_status"), ("close a window", "close_window"),
    ("snap window to the left", "window_action"), ("type text at the cursor", "type_text")])
def test_find_tools_finds_them(need, name):
    assert name in [t.name for t in registry.find(need)], need


def test_results_stay_short(ctx, fake):
    fake.windows = [win(i, f"Dokument nummer {i} med en lang titel – Word", "WINWORD.EXE", 200) for i in range(1, 200)]
    assert len(run(ctx, "list_windows")["result"]) <= 1500
    fake.ports = [(p, "0.0.0.0", 1, f"server{p}.exe") for p in range(1000, 1400)]
    assert len(run(ctx, "listening_ports")["result"]) <= 1500
