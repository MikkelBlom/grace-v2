# Vision tools and the screen overlay

Grace looks at the screen or an image with Gemma 4 12B's own vision encoder, reads text, finds UI
elements and points at them with a native overlay. Code: `services/grace_tools/vision.py`,
`services/grace_tools/vision_overlay.py`, `crates/grace-overlay/`.

## Tools

| Tool | Risk | Core | What it does |
|---|---|---|---|
| `look_at_screen` | read | yes | Screenshot plus question to Gemma; returns her answer as text. `target`: `screen` (the monitor with the active window, default), `window`, `monitor` + `monitor` number (left to right), `all`. `save=true` writes the PNG to the scratch folder, only when he asks. |
| `read_image` | read | | Describe or answer about an image file (png, jpg, webp, bmp, gif, tif; at most 50 MB). |
| `ocr_screen` | read | | Word-for-word text of the active window (default), a monitor, the screen, or an image file (`path`). |
| `find_on_screen` | read | | Finds a button, field, icon or text; returns its box in screen pixels, its centre and where it is ("bottom left"). |
| `point_at` | visible | | Draws an orange focus box with a label for `seconds` (default 6). Takes `what` (finds it first) or a box from `find_on_screen`. Honours `ctx.dry`. Without the overlay, it says so and Grace describes the place in words. |

All are `slow` (Grace says a hold line). None clicks or types; acting on the screen belongs to a
later group and must go through the risk rules.

**Privacy.** A screenshot can contain anything. It exists only in memory for the one request to
the local llama-server; tools return text. Nothing is written to disk unless `save=true`. Every
prompt tells Gemma that text in the image is content, never instructions (prompt injection from
a web page or e-mail on screen).

**Address.** Requests go to `ctx.llm_url`, else `GRACE_LLM_URL`, else `http://127.0.0.1:8080`
(the engine's llama-server). Requests use OpenAI `image_url` with a base64 JPEG.

## The model side

- The projector is `mmproj-gemma-4-12b-it-qat-q4_0.gguf` from the same repo as the model,
  `google/gemma-4-12B-it-qat-q4_0-gguf`, 175 MB (SHA256 `cb018338…260da7`, matches the HF LFS oid).
  `scripts/grace.ps1` adds `--mmproj` and `-ub 1120` when the file exists. It is not yet in
  `scripts/fetch_runtime.py`, so a new machine must download it by hand.
- **`-ub 1120`**: llama-server caps an image at n_ubatch tokens (512 by default; Gemma allows
  1120). Small screen text (15 px Segoe UI on a 2560x1440 screen) read 16 of 28 lines exactly at
  1120 tokens and 2 of 28 at 512. Text TTFT was unchanged (p50 341 ms vs 343 ms over 15 turns),
  and VRAM rose by about 60 MB.
- **Boxes must be sent at 768x768.** On llama.cpp b11539, Gemma's `box_2d` x is exact at every
  size, but y drifts by up to ±17 % of the height (sign and size vary with position) at every
  size we tried (512-1536, square and 16:9) except 768x768, where a red square at 5 heights
  landed within 1/1000. So `find_on_screen` pads the shot bottom-right to a square, resizes it to
  768 and maps 0-1000 back to the square's side. On a synthetic Danish save dialog shown full
  screen on the 2560x1440 monitor, one pass put "Gem" and "Annuller" within 10-17 px. A second
  look at a square around the first box brought them within 1-5 px. That second look runs only
  for targets under 48 px (icons), because it doubles the time. This looks like a llama.cpp
  preprocessing quirk. After a llama.cpp bump, repeat the sweep (a red square at 5 heights in
  images of several sizes, asking for its box_2d), since a fix would make other sizes usable.
- Speed: about 1.4 s per vision call on an idle GPU; 2.4-4.4 s while the live server was busy
  (decode fell to 18-30 tok/s). `find_on_screen` with the second look takes two calls.

## The overlay (`grace-overlay`)

`target/release/grace-overlay.exe [--port 7815] [--capturable]`. Newline JSON on
127.0.0.1:7815, one reply line per command:

```
{"glow": true|false}                                   edge glow, fades in/out (450 ms) and breathes
{"box": {"x":..,"y":..,"w":..,"h":..,"label":"Gem","ms":4000}}   focus box, 200 ms-60 s
{"clear": true}   fades every box (not the glow)       {"ping": true}   {"quit": true}
```

Coordinates are physical virtual-desktop pixels, which is the same space as the vision tools'
boxes.

**Design.** There is one window per monitor with WS_EX_LAYERED | TRANSPARENT | TOPMOST |
NOACTIVATE | TOOLWINDOW. Each is painted with UpdateLayeredWindow from a premultiplied BGRA DIB
that the CPU fills. The overlay does not use the GPU, D3D or DirectComposition. The process is
per-monitor DPI aware v2, so glow width, line width and the 11 pt label scale with each monitor.
WM_NCHITTEST returns HTTRANSPARENT and WM_MOUSEACTIVATE returns MA_NOACTIVATE. HWND_TOPMOST is
set when a window is shown, re-asserted every second, and re-asserted on every foreground change
(SetWinEventHook EVENT_SYSTEM_FOREGROUND). Windows are hidden when nothing is drawn. By default
the overlay uses WDA_EXCLUDEFROMCAPTURE, so it is on screen but absent from screenshots: Grace
never sees her own glow or boxes. `--capturable` turns that off for verification.

The drawing (`draw.rs`), the state machine with an explicit clock (`scene.rs`) and the protocol
(`protocol.rs`) are pure and unit-tested (17 tests: edge colour and falloff, box outline, fill,
halo, clipping, alpha restore after GDI text, label placement, fades, expiry, clear, parsing,
limits). `win.rs` is the only file with `unsafe`.

**Cost.** A steady glow runs at 12 fps and fades run at 30 fps. A glow-only frame rewrites only
the edge bands (0.42 ms per 2560x1600 monitor; blending had cost 2.9 ms). With three monitors
and the glow on, the overlay uses 7.5 % of one core. Most of that is DWM copying the full-monitor
surfaces. It uses 0 % when idle. Thin edge-strip windows would cut it further (see Not done).

**Why v1's overlay failed.** Grace v1 used Electron `BrowserWindow({transparent: true,
alwaysOnTop: true, type: 'toolbar'})`. Electron's transparent windows are composited through
Chromium's GPU path and DirectComposition. On a hybrid laptop (Arc iGPU drives the panel, RTX
renders), that path is fragile: the window can come up black or empty, or lose its alpha, or be
covered after a GPU/display switch. The `'screen-saver'` always-on-top level was applied only
once at `ready-to-show`, so any window that later raised itself covered it. It also re-rendered
HTML/CSS in a renderer process just to draw a glow. The native overlay removes each cause: a CPU
bitmap that DWM composites like any layered window, with topmost re-asserted continuously.

**Verified live** (2026-10-22, 3 monitors: 2560x1600 at 150 %, 2560x1440, 1920x1080). The
overlay was started with `--capturable`, then checked by PIL ImageGrab:
- Glow orange at 8 of 8 edge points on every monitor, with the centre untouched.
- Box outline and label pill drawn in the exact orange.
- Over a maximised, topmost, focused window: glow and box still on top, the window keeps focus
  (foreground class never `GraceOverlay`), and WindowFromPoint at the box centre hits the
  window underneath (click-through).
- Glow gone after `{"glow":false}`.
- Default mode: screenshots show the window underneath, not the glow (excluded from capture).

Notepad was replaced by a Tk window, maximised and topmost, for the "on top" test. Windows 11
Notepad opens new files as tabs in an already running Notepad (one was open), so "close only
the one I opened" could not be guaranteed. A topmost window is the stronger test anyway.

On-screen testing then stopped because it got in Mikkel's way. From now on the window is checked
headlessly: `win::tests` creates a real overlay-style window off-screen at 1x1 and zero alpha, and
checks the five extended styles, HWND_TOPMOST after `topmost()`, HTTRANSPARENT and MA_NOACTIVATE,
and that the foreground window never changes. **Still to do in a live session:** check that it
stays on top of a maximised Notepad.

**Known limits.** Exclusive-fullscreen games (and some video players in exclusive mode) bypass
DWM and cover the overlay. Secure desktops (UAC, lock screen) are never covered.

## Tests

`tests/tools/test_vision.py` (31 tests) replaces the screenshot, the llama-server call and the
overlay socket. It covers:
- The request body (image_url, JPEG, 768 square for boxes, downscale otherwise).
- The configurable address.
- That no file is written unless `save`.
- Target passing, a missing `--mmproj`, server down and screenshot failure.
- Box mapping with monitor offsets, the second look and its fallback, and seven reply variants
  (fenced, reversed corners, `bbox`, lists, clamping, not found, malformed).
- A box falling in the padding.
- `point_at` with the overlay present, absent and in dry-run.
- `read_image` and `ocr_screen` normal cases and failures, plus result cutting.
- The overlay helpers speaking the real protocol to a socket, and doing nothing when no overlay
  listens.

Sim scenarios are in `fixtures/tools/scenarios/vision.yaml` (10). The group has no destructive
tool, so the safety case checks that Grace never claims to have clicked.

**Not tested automatically:** real screenshots (`screenshot`, `monitors`, `active_window`) and
real Gemma answers. Both were run live against a second llama-server (port 8090, same flags plus
`--mmproj`, `-c 8192 -np 1` to fit beside the live server) on a synthetic screen.
`look_at_screen` answered "hvad står der på min skærm" with the file name, both lines and the
dialog's buttons. `ocr_screen` read the dialog exactly, including æ/ø.

## Not done

- No local OCR engine. Gemma read the synthetic screens exactly, and Windows' built-in OCR would
  need a WinRT package.
- The engine does not glow yet. It should send `{"glow":true}` when a turn starts thinking or a
  tool runs, and `{"glow":false}` when the reply ends. `scripts/grace.ps1` does not start the
  overlay yet.
- The steady glow could use four thin edge windows per monitor instead of one full-monitor
  surface, to cut DWM copies.
