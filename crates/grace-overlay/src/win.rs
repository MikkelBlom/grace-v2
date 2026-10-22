//! The WinAPI layer: one layered, click-through, topmost, never-activated window per monitor,
//! painted with UpdateLayeredWindow from a CPU DIB. Everything here runs on the main thread; the
//! TCP threads only parse lines and post commands to it.
//!
//! Staying on top: HWND_TOPMOST is set when a window is shown, re-asserted every second and on
//! every foreground change (SetWinEventHook), because other topmost windows (and some apps that
//! briefly raise themselves) would otherwise cover the glow. Exclusive-fullscreen games bypass
//! the desktop compositor and will still cover it.
#![allow(unsafe_code)]

use std::cell::RefCell;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Instant;

use windows::Win32::Foundation::{COLORREF, HWND, LPARAM, LRESULT, POINT, RECT, SIZE, WPARAM};
use windows::Win32::Graphics::Gdi::{
    AC_SRC_ALPHA, AC_SRC_OVER, ANTIALIASED_QUALITY, BI_RGB, BITMAPINFO, BITMAPINFOHEADER,
    BLENDFUNCTION, CLIP_DEFAULT_PRECIS, CreateCompatibleDC, CreateDIBSection, CreateFontW,
    DEFAULT_CHARSET, DEFAULT_PITCH, DIB_RGB_COLORS, DeleteDC, DeleteObject, EnumDisplayMonitors,
    FF_SWISS, FW_SEMIBOLD, GdiFlush, GetDC, GetMonitorInfoW, GetTextExtentPoint32W, HBITMAP, HDC,
    HFONT, HGDIOBJ, HMONITOR, MONITORINFO, OUT_DEFAULT_PRECIS, ReleaseDC, SelectObject, SetBkMode,
    SetTextColor, TRANSPARENT, TextOutW,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::GetCurrentThreadId;
use windows::Win32::UI::Accessibility::{HWINEVENTHOOK, SetWinEventHook, UnhookWinEvent};
use windows::Win32::UI::HiDpi::{
    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, GetDpiForMonitor, MDT_EFFECTIVE_DPI,
    SetProcessDpiAwarenessContext,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, EVENT_SYSTEM_FOREGROUND,
    GetMessageW, HTTRANSPARENT, HWND_TOPMOST, KillTimer, MA_NOACTIVATE, MSG, PostQuitMessage,
    PostThreadMessageW, RegisterClassExW, SW_HIDE, SW_SHOWNOACTIVATE, SWP_NOACTIVATE, SWP_NOMOVE,
    SWP_NOOWNERZORDER, SWP_NOSIZE, SetTimer, SetWindowDisplayAffinity, SetWindowPos, ShowWindow,
    ULW_ALPHA, UpdateLayeredWindow, WDA_EXCLUDEFROMCAPTURE, WINDOW_EX_STYLE, WINEVENT_OUTOFCONTEXT,
    WM_APP, WM_DISPLAYCHANGE, WM_DPICHANGED, WM_MOUSEACTIVATE, WM_NCHITTEST, WM_TIMER, WNDCLASSEXW,
    WS_EX_LAYERED, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_EX_TOPMOST, WS_EX_TRANSPARENT, WS_POPUP,
};
use windows::core::{BOOL, PCWSTR, w};

use crate::draw::{self, Canvas};
use crate::protocol::{self, Command, Rect};
use crate::scene::Scene;

const WM_COMMANDS: u32 = WM_APP + 1;
const WM_REBUILD: u32 = WM_APP + 2;
const WM_TOPMOST: u32 = WM_APP + 3;
const FRAME_MS: u32 = 33;
/// While only the slow breath runs (2.4 s period), 12 fps looks the same.
const SLOW_FRAME_MS: u32 = 83;
const TOPMOST_MS: u32 = 1000;
const CLASS: PCWSTR = w!("GraceOverlay");
/// Labels are 11 pt Segoe UI Semibold.
const LABEL_PT: i32 = 11;

type Res<T> = Result<T, String>;

/// Click-through, never activated, above normal windows, not in Alt-Tab, per-pixel alpha.
const OVERLAY_EX: WINDOW_EX_STYLE = WINDOW_EX_STYLE(
    WS_EX_LAYERED.0
        | WS_EX_TRANSPARENT.0
        | WS_EX_TOPMOST.0
        | WS_EX_NOACTIVATE.0
        | WS_EX_TOOLWINDOW.0,
);

fn err(what: &str) -> impl Fn(windows::core::Error) -> String + '_ {
    move |e| format!("{what}: {e}")
}

/// A top-down 32-bit DIB selected into a memory DC.
struct Surface {
    dc: HDC,
    bmp: HBITMAP,
    old: HGDIOBJ,
    bits: *mut u32,
    w: i32,
    h: i32,
}

impl Surface {
    fn new(w: i32, h: i32) -> Res<Self> {
        unsafe {
            let screen = GetDC(None);
            let dc = CreateCompatibleDC(Some(screen));
            ReleaseDC(None, screen);
            let bmi = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: w,
                    biHeight: -h, // top-down: row 0 is the top of the screen
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB.0,
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut bits = std::ptr::null_mut();
            let bmp = CreateDIBSection(Some(dc), &bmi, DIB_RGB_COLORS, &mut bits, None, 0)
                .map_err(err("CreateDIBSection"))?;
            let old = SelectObject(dc, bmp.into());
            Ok(Self {
                dc,
                bmp,
                old,
                bits: bits.cast(),
                w,
                h,
            })
        }
    }

    fn pixels(&mut self) -> &mut [u32] {
        // The DIB lives as long as self and is only touched from this thread.
        unsafe { std::slice::from_raw_parts_mut(self.bits, (self.w * self.h) as usize) }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            SelectObject(self.dc, self.old);
            let _ = DeleteObject(self.bmp.into());
            let _ = DeleteDC(self.dc);
        }
    }
}

struct Monitor {
    rect: Rect,
    scale: f32,
    hwnd: HWND,
    surface: Surface,
    font: HFONT,
    shown: bool,
    boxes_drawn: bool,
}

impl Drop for Monitor {
    fn drop(&mut self) {
        unsafe {
            let _ = DestroyWindow(self.hwnd);
            let _ = DeleteObject(self.font.into());
        }
    }
}

struct App {
    mons: Vec<Monitor>,
    scene: Scene,
    rx: Receiver<Command>,
    start: Instant,
    frame_timer: usize,
    frame_ms: u32,
    capturable: bool,
}

thread_local! {
    static APP: RefCell<Option<App>> = const { RefCell::new(None) };
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

unsafe extern "system" fn monitor_cb(m: HMONITOR, _: HDC, _: *mut RECT, data: LPARAM) -> BOOL {
    unsafe { (*(data.0 as *mut Vec<HMONITOR>)).push(m) };
    BOOL(1)
}

/// The window procedure never touches APP (it can run inside CreateWindowExW or SetWindowPos,
/// while APP is borrowed); it only posts work to the main loop.
unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    match msg {
        WM_NCHITTEST => LRESULT(HTTRANSPARENT as isize),
        WM_MOUSEACTIVATE => LRESULT(MA_NOACTIVATE as isize),
        WM_DISPLAYCHANGE => {
            unsafe {
                let _ = PostThreadMessageW(GetCurrentThreadId(), WM_REBUILD, WPARAM(0), LPARAM(0));
            }
            LRESULT(0)
        }
        // Positions are set on every UpdateLayeredWindow; don't let Windows resize us.
        WM_DPICHANGED => LRESULT(0),
        _ => unsafe { DefWindowProcW(hwnd, msg, wp, lp) },
    }
}

unsafe extern "system" fn foreground_cb(
    _: HWINEVENTHOOK,
    _: u32,
    _: HWND,
    _: i32,
    _: i32,
    _: u32,
    _: u32,
) {
    unsafe {
        let _ = PostThreadMessageW(GetCurrentThreadId(), WM_TOPMOST, WPARAM(0), LPARAM(0));
    }
}

fn monitors(capturable: bool) -> Res<Vec<Monitor>> {
    let mut handles: Vec<HMONITOR> = Vec::new();
    unsafe {
        let _ = EnumDisplayMonitors(
            None,
            None,
            Some(monitor_cb),
            LPARAM(&mut handles as *mut _ as isize),
        );
    }
    let hinst = unsafe { GetModuleHandleW(None) }.map_err(err("GetModuleHandleW"))?;
    let ex = OVERLAY_EX;
    let mut out = Vec::new();
    for h in handles {
        let mut info = MONITORINFO {
            cbSize: size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        if !unsafe { GetMonitorInfoW(h, &mut info) }.as_bool() {
            continue;
        }
        let r = info.rcMonitor;
        let rect = Rect::new(r.left, r.top, r.right - r.left, r.bottom - r.top);
        let (mut dx, mut dy) = (96u32, 96u32);
        let _ = unsafe { GetDpiForMonitor(h, MDT_EFFECTIVE_DPI, &mut dx, &mut dy) };
        let scale = dx as f32 / 96.0;
        let hwnd = unsafe {
            CreateWindowExW(
                ex,
                CLASS,
                w!("Grace overlay"),
                WS_POPUP,
                rect.x,
                rect.y,
                rect.w,
                rect.h,
                None,
                None,
                Some(hinst.into()),
                None,
            )
        }
        .map_err(err("CreateWindowExW"))?;
        if !capturable {
            // Grace's own screenshots must not see her glow or boxes.
            let _ = unsafe { SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE) };
        }
        let px = (LABEL_PT as f32 * dx as f32 / 72.0).round() as i32;
        let font = unsafe {
            CreateFontW(
                -px,
                0,
                0,
                0,
                FW_SEMIBOLD.0 as i32,
                0,
                0,
                0,
                DEFAULT_CHARSET,
                OUT_DEFAULT_PRECIS,
                CLIP_DEFAULT_PRECIS,
                ANTIALIASED_QUALITY,
                (DEFAULT_PITCH.0 | FF_SWISS.0) as u32,
                w!("Segoe UI"),
            )
        };
        let surface = Surface::new(rect.w, rect.h)?;
        out.push(Monitor {
            rect,
            scale,
            hwnd,
            surface,
            font,
            shown: false,
            boxes_drawn: false,
        });
    }
    if out.is_empty() {
        return Err("no monitors found".into());
    }
    Ok(out)
}

fn topmost(hwnd: HWND) {
    unsafe {
        let _ = SetWindowPos(
            hwnd,
            Some(HWND_TOPMOST),
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOOWNERZORDER,
        );
    }
}

impl App {
    fn now(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    fn drain(&mut self) -> bool {
        let now = self.now();
        let mut quit = false;
        while let Ok(cmd) = self.rx.try_recv() {
            quit |= cmd == Command::Quit;
            self.scene.apply(&cmd, now);
        }
        self.render();
        quit
    }

    fn rebuild(&mut self) {
        self.mons.clear();
        match monitors(self.capturable) {
            Ok(m) => self.mons = m,
            Err(e) => eprintln!("grace-overlay: {e}"),
        }
        self.render();
    }

    fn reassert(&self) {
        for m in self.mons.iter().filter(|m| m.shown) {
            topmost(m.hwnd);
        }
    }

    fn render(&mut self) {
        let now = self.now();
        let animating = self.scene.tick(now);
        let glow_a = self.scene.glow_alpha(now);
        for m in &mut self.mons {
            let room = ((draw::BOX_HALO_PX + draw::BOX_LINE_PX) as f32 * m.scale) as i32 + 60;
            let boxes: Vec<_> = self
                .scene
                .boxes
                .iter()
                .filter(|b| b.rect.inflate(room).intersects(&m.rect))
                .map(|b| {
                    (
                        b.rect.offset(-m.rect.x, -m.rect.y),
                        b.label.clone(),
                        b.alpha(now),
                    )
                })
                .collect();
            if glow_a <= 0.004 && boxes.is_empty() {
                if m.shown {
                    unsafe {
                        let _ = ShowWindow(m.hwnd, SW_HIDE);
                    }
                    m.shown = false;
                }
                continue;
            }
            paint(m, glow_a, &boxes);
            if !m.shown {
                unsafe {
                    let _ = ShowWindow(m.hwnd, SW_SHOWNOACTIVATE);
                }
                m.shown = true;
                topmost(m.hwnd);
            }
        }
        let want = match (animating, self.scene.fading(now)) {
            (false, _) => 0,
            (true, true) => FRAME_MS,
            (true, false) => SLOW_FRAME_MS,
        };
        if want != self.frame_ms {
            unsafe {
                if self.frame_timer != 0 {
                    let _ = KillTimer(None, self.frame_timer);
                    self.frame_timer = 0;
                }
                if want != 0 {
                    self.frame_timer = SetTimer(None, 0, want, None);
                }
            }
            self.frame_ms = want;
        }
    }
}

fn paint(m: &mut Monitor, glow_a: f32, boxes: &[(Rect, Option<String>, f32)]) {
    let (w, h, scale) = (m.rect.w, m.rect.h, m.scale);
    {
        let mut c = Canvas::new(m.surface.pixels(), w, h);
        // A glow-only frame rewrites just the edge bands; the interior needs clearing only
        // after boxes were drawn there (the DIB starts zeroed).
        if m.boxes_drawn {
            c.clear();
        }
        m.boxes_drawn = !boxes.is_empty();
        c.glow((draw::GLOW_PX as f32 * scale) as i32, glow_a);
        for (r, _, a) in boxes {
            c.focus_box(*r, scale, *a);
        }
    }
    let dc = m.surface.dc;
    let pad = (draw::LABEL_PAD_PX as f32 * scale) as i32;
    let gap = ((draw::BOX_LINE_PX + 3) as f32 * scale) as i32;
    for (r, label, a) in boxes {
        let Some(text) = label else { continue };
        let text = wide(text);
        let mut size = SIZE::default();
        unsafe {
            let old = SelectObject(dc, m.font.into());
            let _ = GetTextExtentPoint32W(dc, &text, &mut size);
            let pill = draw::label_rect(*r, size.cx, size.cy, pad, gap, Rect::new(0, 0, w, h));
            Canvas::new(m.surface.pixels(), w, h).fill(pill, draw::ORANGE, 1.0);
            SetBkMode(dc, TRANSPARENT);
            SetTextColor(dc, COLORREF(0x00FF_FFFF));
            let _ = TextOutW(dc, pill.x + pad, pill.y + pad / 2, &text);
            let _ = GdiFlush();
            SelectObject(dc, old);
            Canvas::new(m.surface.pixels(), w, h).restore_pill(pill, *a);
        }
    }
    let blend = BLENDFUNCTION {
        BlendOp: AC_SRC_OVER as u8,
        BlendFlags: 0,
        SourceConstantAlpha: 255,
        AlphaFormat: AC_SRC_ALPHA as u8,
    };
    let pos = POINT {
        x: m.rect.x,
        y: m.rect.y,
    };
    let size = SIZE { cx: w, cy: h };
    let src = POINT { x: 0, y: 0 };
    unsafe {
        let screen = GetDC(None);
        if let Err(e) = UpdateLayeredWindow(
            m.hwnd,
            Some(screen),
            Some(&pos),
            Some(&size),
            Some(dc),
            Some(&src),
            COLORREF(0),
            Some(&blend),
            ULW_ALPHA,
        ) {
            eprintln!("grace-overlay: UpdateLayeredWindow: {e}");
        }
        ReleaseDC(None, screen);
    }
}

fn serve(stream: TcpStream, tx: Sender<Command>, tid: u32) {
    let Ok(read) = stream.try_clone() else { return };
    let mut out = stream;
    for line in BufReader::new(read).lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let parsed = protocol::parse(&line);
        if let Ok(cmd) = &parsed {
            if tx.send(cmd.clone()).is_err() {
                break;
            }
            unsafe {
                let _ = PostThreadMessageW(tid, WM_COMMANDS, WPARAM(0), LPARAM(0));
            }
        }
        if writeln!(out, "{}", protocol::reply(&parsed)).is_err() {
            break;
        }
    }
}

pub fn run(port: u16, capturable: bool) -> Res<()> {
    // Bind first: a second overlay fails here instead of stacking windows.
    let listener = TcpListener::bind(("127.0.0.1", port))
        .map_err(|e| format!("can't listen on 127.0.0.1:{port}: {e}"))?;
    unsafe {
        // Physical pixels everywhere, so box coordinates match a full-desktop screenshot.
        let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        let hinst = GetModuleHandleW(None).map_err(err("GetModuleHandleW"))?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(wndproc),
            hInstance: hinst.into(),
            lpszClassName: CLASS,
            ..Default::default()
        };
        if RegisterClassExW(&class) == 0 {
            return Err("RegisterClassExW failed".into());
        }
    }
    let (tx, rx) = mpsc::channel();
    let mons = monitors(capturable)?;
    eprintln!(
        "grace-overlay: {} monitor(s) {:?}, listening on 127.0.0.1:{port}{}",
        mons.len(),
        mons.iter().map(|m| (m.rect, m.scale)).collect::<Vec<_>>(),
        if capturable { " (capturable)" } else { "" }
    );
    APP.with(|a| {
        *a.borrow_mut() = Some(App {
            mons,
            scene: Scene::new(),
            rx,
            start: Instant::now(),
            frame_timer: 0,
            frame_ms: 0,
            capturable,
        })
    });
    let tid = unsafe { GetCurrentThreadId() };
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let tx = tx.clone();
            std::thread::spawn(move || serve(stream, tx, tid));
        }
    });
    let hook = unsafe {
        SetWinEventHook(
            EVENT_SYSTEM_FOREGROUND,
            EVENT_SYSTEM_FOREGROUND,
            None,
            Some(foreground_cb),
            0,
            0,
            WINEVENT_OUTOFCONTEXT,
        )
    };
    let topmost_timer = unsafe { SetTimer(None, 0, TOPMOST_MS, None) };
    let mut msg = MSG::default();
    loop {
        let r = unsafe { GetMessageW(&mut msg, None, 0, 0) };
        if r.0 <= 0 {
            break;
        }
        if !msg.hwnd.0.is_null() {
            unsafe { DispatchMessageW(&msg) };
            continue;
        }
        APP.with(|a| {
            let mut a = a.borrow_mut();
            let Some(app) = a.as_mut() else { return };
            match msg.message {
                WM_COMMANDS => {
                    if app.drain() {
                        unsafe { PostQuitMessage(0) };
                    }
                }
                WM_TIMER if msg.wParam.0 == app.frame_timer => app.render(),
                WM_TIMER if msg.wParam.0 == topmost_timer => app.reassert(),
                WM_TOPMOST => app.reassert(),
                WM_REBUILD => app.rebuild(),
                _ => {}
            }
        });
    }
    unsafe {
        let _ = UnhookWinEvent(hook);
    }
    APP.with(|a| a.borrow_mut().take());
    Ok(())
}

/// Headless window check: a real overlay-style window, created off-screen at 1x1 and painted
/// fully transparent, so nothing ever shows on the desktop. Checks the styles, topmost z-order,
/// click-through hit testing and that it never takes the foreground.
#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::UI::WindowsAndMessaging::{
        GWL_EXSTYLE, GetForegroundWindow, GetWindowLongW, SendMessageW,
    };

    #[test]
    fn overlay_window_is_layered_topmost_click_through_and_never_foreground() {
        unsafe {
            let hinst = GetModuleHandleW(None).unwrap();
            let class = WNDCLASSEXW {
                cbSize: size_of::<WNDCLASSEXW>() as u32,
                lpfnWndProc: Some(wndproc),
                hInstance: hinst.into(),
                lpszClassName: w!("GraceOverlayTest"),
                ..Default::default()
            };
            assert_ne!(RegisterClassExW(&class), 0);
            let fg_before = GetForegroundWindow();
            let hwnd = CreateWindowExW(
                OVERLAY_EX,
                w!("GraceOverlayTest"),
                w!("Grace overlay test"),
                WS_POPUP,
                -32000,
                -32000,
                1,
                1,
                None,
                None,
                Some(hinst.into()),
                None,
            )
            .unwrap();
            let _ = SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE);
            // Zero-alpha pixel: a layered window needs one UpdateLayeredWindow before it shows.
            let surface = Surface::new(1, 1).unwrap();
            let blend = BLENDFUNCTION {
                BlendOp: AC_SRC_OVER as u8,
                BlendFlags: 0,
                SourceConstantAlpha: 0,
                AlphaFormat: AC_SRC_ALPHA as u8,
            };
            let screen = GetDC(None);
            UpdateLayeredWindow(
                hwnd,
                Some(screen),
                Some(&POINT {
                    x: -32000,
                    y: -32000,
                }),
                Some(&SIZE { cx: 1, cy: 1 }),
                Some(surface.dc),
                Some(&POINT { x: 0, y: 0 }),
                COLORREF(0),
                Some(&blend),
                ULW_ALPHA,
            )
            .unwrap();
            ReleaseDC(None, screen);
            let _ = ShowWindow(hwnd, SW_SHOWNOACTIVATE);
            topmost(hwnd);

            let ex = GetWindowLongW(hwnd, GWL_EXSTYLE) as u32;
            for (flag, name) in [
                (WS_EX_LAYERED, "LAYERED"),
                (WS_EX_TRANSPARENT, "TRANSPARENT"),
                (WS_EX_TOPMOST, "TOPMOST"),
                (WS_EX_NOACTIVATE, "NOACTIVATE"),
                (WS_EX_TOOLWINDOW, "TOOLWINDOW"),
            ] {
                assert!(ex & flag.0 != 0, "missing WS_EX_{name}: {ex:#x}");
            }
            let hit = SendMessageW(hwnd, WM_NCHITTEST, None, None);
            assert_eq!(hit.0, HTTRANSPARENT as isize);
            let act = SendMessageW(hwnd, WM_MOUSEACTIVATE, None, None);
            assert_eq!(act.0, MA_NOACTIVATE as isize);
            assert_ne!(GetForegroundWindow(), hwnd);
            assert_eq!(GetForegroundWindow(), fg_before);
            let _ = DestroyWindow(hwnd);
        }
    }
}
