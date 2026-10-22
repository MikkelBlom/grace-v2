# Tool group: system

`services/grace_tools/system.py` (logic) and `services/grace_tools/system_os.py` (every Windows
call, replaced in tests). Tests: `tests/tools/test_system.py`. Sim scenarios:
`fixtures/tools/scenarios/system.yaml`. Packages: `services/grace_tools/system.requirements.txt`.

Core (offered every turn): `focus_window`, `system_status`. Everything else goes through
`find_tools` and `use_tool`.

## Tools

| Tool | Risk | What it does |
|---|---|---|
| list_windows | read | Open windows (app, title), which is in front |
| focus_window | visible, core | Switch to a window by app or title ("skift til chrome", "lommeregneren") |
| window_action | visible | Minimise, maximise, restore, snap left/right, move to the other monitor |
| close_window | visible, asks for apps that may hold unsaved work | WM_CLOSE, as the X button does; `all` closes every window of the app |
| show_desktop | visible | Minimise everything |
| audio_output | visible | List outputs; switch the default output ("høretelefonerne", "JBL", "højttalerne") |
| volume | visible | Master or one app's volume: get, set, up, down, mute, unmute |
| microphone | visible, mute asks | Status, mute, unmute the default microphone |
| wifi | visible, off asks | Status (network, signal, band), nearby networks, on/off |
| bluetooth | visible, off asks | Status, paired devices and which are connected, on/off |
| dark_mode | visible | Dark/light/toggle/status |
| do_not_disturb | visible | Focus assist / Do Not Disturb on/off/status |
| brightness | visible | The laptop display's brightness |
| night_light | visible | Night light on/off/status |
| network_status | read | Online?, DNS, Wi-Fi and signal, local IP, with a diagnosis when down |
| ping_host | read | 4 pings, replies and times |
| dns_lookup | read | A name's addresses |
| site_up | read | Is a site up, and is it the site or his connection |
| speed_test | read | Download speed from Cloudflare's free speed test (no account, no key) |
| listening_ports | read | Local listening TCP ports and their programs, or one port |
| top_processes | read | Top programs by CPU or RAM, instances of one app added up |
| is_running | read | Is a program running, its PIDs and window |
| end_process | destructive | Kill a program or PID; protected list; asks which on ambiguous names |
| restart_app | destructive | Kill all of a program and start it again (File Explorer allowed) |
| clipboard_read | read | The clipboard's text |
| clipboard_write | visible | Put text on the clipboard |
| type_text | destructive | Type text at the cursor of the window in front |
| active_window | read | The window in front |
| selected_text | read | Copies the selection (Ctrl+C), reads it, puts his clipboard back |
| system_status | read, core | CPU, RAM, GPU (nvidia-smi: load, VRAM, temperature), battery, disks |
| lock_pc | visible | Lock the PC |
| sleep_pc | destructive | Sleep |

### Visible tools that ask for one case

The registry has one risk per tool, but closing a browser window and closing the calculator are
not equally risky, and Wi-Fi on is harmless while Wi-Fi off cuts Grace's own web search. These
tools are registered `visible` and, for the risky case, return the same confirmation a
destructive tool returns (`ask_first` in system.py). The stored args carry a one-time approval
that only the server knows; the engine's confirm runs the tool again with it. An approval the
model invents is not in the table, so the tool just asks again. The risky cases:
- `close_window` for any app not in `SAFE_TO_CLOSE` (File Explorer, Calculator, Settings, Task
  Manager, Spotify, Photos, Steam, media players). Browsers, editors, Office and chat apps ask.
- `wifi off` (cuts Grace's web search), `bluetooth off` (his Logitech Z337 speakers and MX Master
  mouse are Bluetooth; the summary names what disconnects), `microphone mute` (she may go deaf).

### Process safeguards
- Protected, never ended even after a yes: Windows' own processes (csrss, wininit, winlogon,
  lsass, services, smss, svchost, dwm, ...; explorer except through `restart_app`), security
  software (by name, and by install folder: Bitdefender, Defender, Kaspersky, ESET, ...), and
  Grace (llama-server, grace-*, Docker and WSL, the tools server and its parents, and any Python
  whose command line mentions Grace or a `*_server.py`; an unreadable command line counts as hers).
- A name that matches several different programs ("svc") asks which. All instances of one
  program ("chrome", 41 processes) is one app and is ended together.
- The summary names the exact process and PIDs, and the windows that may hold unsaved work.
- The processes planned when Grace asked are the ones ended; any that exited or were replaced
  in the meantime are skipped.

## What was tested
78 deterministic tests (with parametrised cases) against a fake OS: Danish names (lommeregneren, stifinderen, høretelefoner,
højttalerne, computerens højttalere, skærmen, Danish `netsh` labels, Danish `ping` output),
ambiguity (windows, outputs, processes), the protected list, the confirm flow (asks first, runs
once after a yes, a forged approval asks again, planned window/PIDs kept), dry-run for every
changing tool, the night light blob round trip, result length.

Verified on the real PC, read-only (2026-10-22): every read tool and every status action, the
dry-run of every visible action (real lookups, nothing changed), and the confirmation summaries
of every destructive case (nothing confirmed). The speed test ran for real once (74 Mbit/s).
Process listing takes about 0.8 s (one `NtQuerySystemInformation` snapshot; psutil took 34 s on
the loaded PC).

## Not tested for real, and why
Nothing that changes his PC was run during development: focusing, moving, snapping or closing
windows; switching output, volume, mic mute; Wi-Fi/Bluetooth toggles; dark mode, Do Not Disturb,
brightness, night light; killing/restarting; typing, clipboard writes, Ctrl+C; lock and sleep.
Check these in the first live session. Known uncertainties:
- **Do Not Disturb** writes the Focus-assist WNF state (no public API). On Windows 11 24H2 the
  quick-settings DND may use another store; the tool reads the state back and says so if it
  didn't take.
- **Night light** edits the CloudStore blob by the known recipe (byte 18 0x13/0x15, `10 00` at 23,
  timestamp bump). The format is checked before writing and read back after; unknown formats are
  refused, never written.
- **focus_window**: Windows' foreground lock can refuse a background process; the tool borrows
  the foreground thread's input and falls back to an Alt tap, and reports when it still fails.
- **Brightness** reaches only the laptop's own display (WMI); external monitors (DDC/CI) aren't
  supported.
- **Switching the output** may also move Grace's own voice if the engine plays to the default
  device; check AEC still works after a switch.
- **Muting the microphone** mutes the default capture device, which may or may not be the one
  Grace listens on.
- Wi-Fi `networks` needs Windows' location permission on 24H2; the tool says so when it's missing.
- Upload speed isn't measured.
