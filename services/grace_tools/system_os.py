"""The system group's OS layer: every call into Windows that services/grace_tools/system.py makes,
each a small function that tests replace (tests/tools/test_system.py). Nothing here decides
anything; matching, safeguards and wording live in system.py.

Windows APIs through ctypes (user32, kernel32, ntdll, powrprof), audio through pycaw (Core Audio),
radios through WinRT (Windows.Devices.Radios, no admin needed), the rest through psutil, the
registry, netsh and PowerShell. Tool calls run on server threads, so COM is initialised per call.
"""

import asyncio
import ctypes
import json
import os
import re
import socket
import struct
import subprocess
import threading
import time
import urllib.request
import warnings
import winreg
from ctypes import wintypes

import psutil

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
dwmapi = ctypes.WinDLL("dwmapi")
ntdll = ctypes.WinDLL("ntdll")

HWND, LPARAM, UINT, DWORD = wintypes.HWND, wintypes.LPARAM, wintypes.UINT, wintypes.DWORD
WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, HWND, LPARAM)
user32.EnumWindows.argtypes = [WNDENUMPROC, LPARAM]
for f in ("IsWindowVisible", "IsIconic", "IsZoomed", "IsWindow", "SetForegroundWindow", "BringWindowToTop",
          "GetWindowTextLengthW"):
    getattr(user32, f).argtypes = [HWND]
user32.GetWindow.argtypes, user32.GetWindow.restype = [HWND, UINT], HWND
user32.GetWindowLongPtrW.argtypes, user32.GetWindowLongPtrW.restype = [HWND, ctypes.c_int], ctypes.c_ssize_t
user32.GetWindowTextW.argtypes = [HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowThreadProcessId.argtypes = [HWND, ctypes.POINTER(DWORD)]
user32.GetWindowThreadProcessId.restype = DWORD
user32.GetForegroundWindow.restype = HWND
user32.ShowWindow.argtypes = [HWND, ctypes.c_int]
user32.GetWindowRect.argtypes = [HWND, ctypes.POINTER(wintypes.RECT)]
user32.SetWindowPos.argtypes = [HWND, HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, UINT]
user32.PostMessageW.argtypes = [HWND, UINT, wintypes.WPARAM, LPARAM]
user32.MonitorFromWindow.argtypes, user32.MonitorFromWindow.restype = [HWND, DWORD], wintypes.HMONITOR
user32.AttachThreadInput.argtypes = [DWORD, DWORD, wintypes.BOOL]
user32.SendMessageTimeoutW.argtypes = [HWND, UINT, wintypes.WPARAM, wintypes.LPCWSTR, UINT, UINT,
                                       ctypes.POINTER(ctypes.c_size_t)]
user32.GetClipboardData.restype = ctypes.c_void_p
user32.SetClipboardData.argtypes, user32.SetClipboardData.restype = [UINT, ctypes.c_void_p], ctypes.c_void_p
user32.OpenClipboard.argtypes = [HWND]
user32.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
kernel32.GlobalAlloc.argtypes, kernel32.GlobalAlloc.restype = [UINT, ctypes.c_size_t], ctypes.c_void_p
kernel32.GlobalLock.argtypes, kernel32.GlobalLock.restype = [ctypes.c_void_p], ctypes.c_void_p
kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
dwmapi.DwmGetWindowAttribute.argtypes = [HWND, DWORD, ctypes.c_void_p, DWORD]

GW_OWNER, GWL_EXSTYLE, WS_EX_TOOLWINDOW, DWMWA_CLOAKED = 4, -20, 0x80, 14
SW_MINIMIZE, SW_MAXIMIZE, SW_RESTORE = 6, 3, 9
WM_CLOSE, WM_SETTINGCHANGE, HWND_BROADCAST = 0x0010, 0x001A, 0xFFFF
CF_UNICODETEXT, GMEM_MOVEABLE = 13, 0x0002
NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW: no console flashes up for netsh / powershell
DPI_PER_MONITOR_V2 = ctypes.c_void_p(-4)


def run(cmd, timeout=15, encoding="utf-8"):
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding=encoding,
                         errors="replace", creationflags=NO_WINDOW)
    return out.stdout


def powershell(script, timeout=20):
    return run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                "[Console]::OutputEncoding = [Text.Encoding]::UTF8; " + script], timeout)


class dpi_aware:
    """Real pixel coordinates for this thread only: the tools server has no windows of its own,
    and a DPI-unaware thread would see scaled rectangles on his 150 % display."""

    def __enter__(self):
        self.old = user32.SetThreadDpiAwarenessContext(DPI_PER_MONITOR_V2)

    def __exit__(self, *a):
        if self.old:
            user32.SetThreadDpiAwarenessContext(self.old)


# ---- windows ----------------------------------------------------------------------------------

def _title(hwnd):
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def _pid(hwnd):
    pid = DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def _exe(pid):
    try:
        return psutil.Process(pid).name()
    except Exception:
        return ""


def _rect(hwnd):
    r = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return (r.left, r.top, r.right, r.bottom)


def windows():
    """The windows on the taskbar, front to back: {hwnd, title, pid, exe, minimized, maximized}.
    Tool windows, owned popups and cloaked ones (other virtual desktops, suspended Store apps)
    are left out, as Alt+Tab does."""
    found = []

    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd) or user32.GetWindow(hwnd, GW_OWNER):
            return True
        if user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW:
            return True
        cloaked = ctypes.c_int(0)
        dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(cloaked), 4)
        title = _title(hwnd)
        if cloaked.value or not title or title in ("Program Manager", "Windows Input Experience"):
            return True
        found.append(hwnd)
        return True

    user32.EnumWindows(WNDENUMPROC(cb), 0)
    out = []
    for hwnd in found:
        pid = _pid(hwnd)
        out.append({"hwnd": hwnd, "title": _title(hwnd), "pid": pid, "exe": _exe(pid),
                    "minimized": bool(user32.IsIconic(hwnd)), "maximized": bool(user32.IsZoomed(hwnd))})
    return out


def foreground():
    """The window in front: {hwnd, title, pid, exe} or None."""
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    pid = _pid(hwnd)
    return {"hwnd": hwnd, "title": _title(hwnd), "pid": pid, "exe": _exe(pid)}


def window_exists(hwnd):
    return bool(user32.IsWindow(hwnd))


def focus(hwnd):
    """Brings a window to the front. Windows only lets the process with the last input do that,
    so this thread borrows the foreground thread's input state for the call."""
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    fg = user32.GetForegroundWindow()
    fg_thread = user32.GetWindowThreadProcessId(fg, None) if fg else 0
    me = kernel32.GetCurrentThreadId()
    attached = bool(fg_thread and fg_thread != me and user32.AttachThreadInput(me, fg_thread, True))
    try:
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(me, fg_thread, False)
    if user32.GetForegroundWindow() != hwnd:
        # A lone Alt press counts as input from us; then the switch is allowed.
        user32.keybd_event(0x12, 0, 0, 0)
        user32.keybd_event(0x12, 0, 2, 0)
        user32.SetForegroundWindow(hwnd)
    return user32.GetForegroundWindow() == hwnd


def show(hwnd, how):
    user32.ShowWindow(hwnd, {"minimize": SW_MINIMIZE, "maximize": SW_MAXIMIZE, "restore": SW_RESTORE}[how])


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT), ("dwFlags", DWORD)]


def _work_area(hmon):
    mi = MONITORINFO()
    mi.cbSize = ctypes.sizeof(MONITORINFO)
    user32.GetMonitorInfoW(hmon, ctypes.byref(mi))
    w = mi.rcWork
    return (w.left, w.top, w.right, w.bottom), bool(mi.dwFlags & 1)


def monitors():
    """Each monitor's work area (left, top, right, bottom), primary first."""
    with dpi_aware():
        found = []
        proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), LPARAM)

        def cb(hmon, hdc, rect, _):
            found.append(_work_area(hmon))
            return True
        user32.EnumDisplayMonitors(None, None, proc(cb), 0)
    return [w for w, primary in sorted(found, key=lambda m: not m[1])]


def window_place(hwnd):
    """(window rect, work area of its monitor), in real pixels."""
    with dpi_aware():
        return _rect(hwnd), _work_area(user32.MonitorFromWindow(hwnd, 2))[0]


def set_rect(hwnd, rect):
    left, top, right, bottom = rect
    with dpi_aware():
        if user32.IsZoomed(hwnd) or user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, SW_RESTORE)
        user32.SetWindowPos(hwnd, None, left, top, right - left, bottom - top, 0x0004 | 0x0040)  # NOZORDER|SHOWWINDOW


def close(hwnd):
    """Asks the window to close, as its X button does: the app can still ask to save."""
    user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)


def minimize_all():
    import comtypes.client
    comtypes.CoInitialize()
    comtypes.client.CreateObject("Shell.Application").MinimizeAll()


# ---- processes --------------------------------------------------------------------------------

class UNICODE_STRING(ctypes.Structure):
    _fields_ = [("Length", ctypes.c_ushort), ("MaximumLength", ctypes.c_ushort), ("Buffer", ctypes.c_void_p)]


class SYSTEM_PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("NextEntryOffset", ctypes.c_ulong), ("NumberOfThreads", ctypes.c_ulong),
                ("WorkingSetPrivateSize", ctypes.c_longlong), ("HardFaultCount", ctypes.c_ulong),
                ("NumberOfThreadsHighWatermark", ctypes.c_ulong), ("CycleTime", ctypes.c_ulonglong),
                ("CreateTime", ctypes.c_longlong), ("UserTime", ctypes.c_longlong), ("KernelTime", ctypes.c_longlong),
                ("ImageName", UNICODE_STRING), ("BasePriority", ctypes.c_long), ("UniqueProcessId", ctypes.c_void_p),
                ("InheritedFromUniqueProcessId", ctypes.c_void_p), ("HandleCount", ctypes.c_ulong),
                ("SessionId", ctypes.c_ulong), ("UniqueProcessKey", ctypes.c_size_t),
                ("PeakVirtualSize", ctypes.c_size_t), ("VirtualSize", ctypes.c_size_t),
                ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t)]


def _snapshot():
    """{pid: (name, cpu time in 100 ns, private working set bytes)} for every process in one
    system call. psutil asks each process separately, which took 34 s on a busy PC because
    protected processes fall back to slow paths."""
    size = 1 << 20
    while True:
        buf = ctypes.create_string_buffer(size)
        need = ctypes.c_ulong(0)
        r = ntdll.NtQuerySystemInformation(5, buf, size, ctypes.byref(need)) & 0xFFFFFFFF  # SystemProcessInformation
        if r == 0xC0000004:  # STATUS_INFO_LENGTH_MISMATCH: more processes than room
            size = max(size * 2, need.value + 65536)
            continue
        if r != 0:
            raise OSError(f"NtQuerySystemInformation failed ({r:#x})")
        break
    out, offset, base = {}, 0, ctypes.addressof(buf)
    while True:
        e = SYSTEM_PROCESS_INFORMATION.from_address(base + offset)
        pid = e.UniqueProcessId or 0
        name = (ctypes.wstring_at(e.ImageName.Buffer, e.ImageName.Length // 2) if e.ImageName.Buffer
                else "System Idle Process")
        out[pid] = (name, e.UserTime + e.KernelTime, e.WorkingSetPrivateSize)
        if not e.NextEntryOffset:
            return out
        offset += e.NextEntryOffset


def processes(sample=0.5):
    """[{pid, name, cpu (% of the whole CPU), ram_mb (private working set, as Task Manager shows)}],
    CPU measured over `sample` seconds."""
    first, t0 = _snapshot(), time.perf_counter()
    time.sleep(sample)
    second, t1 = _snapshot(), time.perf_counter()
    wall = (t1 - t0) * 1e7 * (os.cpu_count() or 1)  # 100 ns units across all cores
    out = []
    for pid, (name, cpu, ram) in second.items():
        before = first.get(pid)
        used = cpu - before[1] if before and before[0] == name else 0
        out.append({"pid": pid, "name": name, "cpu": max(0.0, used / wall * 100), "ram_mb": ram / 2**20})
    return out


def process_names():
    """[{pid, name}] without the CPU sample: for "is X running" and name lookups."""
    return [{"pid": pid, "name": v[0]} for pid, v in _snapshot().items()]


def cmdline(pid):
    """The command line, or None when Windows won't say (another user's or a protected process)."""
    try:
        return " ".join(psutil.Process(pid).cmdline())
    except psutil.Error:
        return None


def exe_path(pid):
    try:
        return psutil.Process(pid).exe()
    except psutil.Error:
        return None


def own_pids():
    """This server and the processes that started it (the engine or a terminal)."""
    pids = {os.getpid()}
    try:
        pids |= {p.pid for p in psutil.Process().parents()}
    except psutil.Error:
        pass
    return pids


def kill(pid):
    """Ends a process (TerminateProcess). Returns None, or why it failed."""
    try:
        p = psutil.Process(pid)
        p.terminate()
        p.wait(3)
    except psutil.NoSuchProcess:
        return None
    except psutil.TimeoutExpired:
        try:
            psutil.Process(pid).kill()
        except psutil.Error as e:
            return str(e)
    except psutil.AccessDenied:
        return "access denied (it runs as administrator or another user)"
    except psutil.Error as e:
        return str(e)
    return None


def launch(path):
    subprocess.Popen([path], creationflags=0x00000008 | 0x00000200, close_fds=True)  # DETACHED | NEW_GROUP


# ---- audio (Core Audio through pycaw) ---------------------------------------------------------

def _caw():
    import comtypes
    comtypes.CoInitialize()
    warnings.filterwarnings("ignore", module="pycaw")
    from pycaw import pycaw
    return pycaw


def audio_devices(kind="output"):
    """[{id, name, default, connected}] for outputs or inputs. Unplugged devices are included so a
    Bluetooth headset that is off can be named ("they aren't connected")."""
    caw = _caw()
    from pycaw.constants import DEVICE_STATE, EDataFlow, ERole
    flow = EDataFlow.eRender.value if kind == "output" else EDataFlow.eCapture.value
    try:
        default = caw.AudioUtilities.GetDeviceEnumerator().GetDefaultAudioEndpoint(flow, ERole.eMultimedia.value).GetId()
    except Exception:
        default = None
    out = []
    for d in caw.AudioUtilities.GetAllDevices(flow, DEVICE_STATE.ACTIVE.value | DEVICE_STATE.UNPLUGGED.value):
        if d is None or not d.FriendlyName:
            continue
        out.append({"id": d.id, "name": d.FriendlyName, "default": d.id == default,
                    "connected": d.state.value == DEVICE_STATE.ACTIVE.value})
    return out


def set_default_audio(device_id):
    """Makes a device the default for every role (console, multimedia, communications)."""
    caw = _caw()
    from pycaw.constants import ERole
    caw.AudioUtilities.SetDefaultDevice(device_id, [ERole.eConsole, ERole.eMultimedia, ERole.eCommunications])


def master_volume():
    """(percent, muted) of the default output."""
    v = _caw().AudioUtilities.GetSpeakers().EndpointVolume
    return round(v.GetMasterVolumeLevelScalar() * 100), bool(v.GetMute())


def set_master_volume(percent=None, mute=None):
    v = _caw().AudioUtilities.GetSpeakers().EndpointVolume
    if percent is not None:
        v.SetMasterVolumeLevelScalar(max(0.0, min(100.0, percent)) / 100, None)
    if mute is not None:
        v.SetMute(int(mute), None)


def app_volumes():
    """[{pid, name, percent, muted}] for every app with an audio session on the default output."""
    out = []
    for s in _caw().AudioUtilities.GetAllSessions():
        if not s.Process:
            continue
        v = s.SimpleAudioVolume
        out.append({"pid": s.ProcessId, "name": s.Process.name(), "percent": round(v.GetMasterVolume() * 100),
                    "muted": bool(v.GetMute())})
    return out


def set_app_volume(pid, percent=None, mute=None):
    for s in _caw().AudioUtilities.GetAllSessions():
        if s.ProcessId == pid:
            if percent is not None:
                s.SimpleAudioVolume.SetMasterVolume(max(0.0, min(100.0, percent)) / 100, None)
            if mute is not None:
                s.SimpleAudioVolume.SetMute(int(mute), None)


def _mic():
    caw = _caw()
    dev = caw.AudioUtilities.CreateDevice(caw.AudioUtilities.GetMicrophone())
    return dev.FriendlyName, dev.EndpointVolume


def mic_state():
    """(name of the default microphone, muted)."""
    name, v = _mic()
    return name, bool(v.GetMute())


def set_mic_mute(mute):
    _mic()[1].SetMute(int(mute), None)


# ---- radios, Wi-Fi, Bluetooth ------------------------------------------------------------------

def _winrt(coro_fn):
    """Runs a WinRT coroutine on its own thread with its own loop (the server's threads have none,
    and WinRT wants a fresh apartment)."""
    box = {}

    def go():
        try:
            box["v"] = asyncio.run(coro_fn())
        except Exception as e:
            box["e"] = e
    t = threading.Thread(target=go)
    t.start()
    t.join(15)
    if "e" in box:
        raise box["e"]
    return box.get("v")


def radio_state(kind):
    """True / False for the "wifi" or "bluetooth" radio, None when there is none."""
    from winrt.windows.devices.radios import Radio, RadioKind, RadioState
    want = RadioKind.WI_FI if kind == "wifi" else RadioKind.BLUETOOTH

    async def go():
        for r in await Radio.get_radios_async():
            if r.kind == want:
                return r.state == RadioState.ON
        return None
    return _winrt(go)


def set_radio(kind, on):
    """Turns a radio on or off as the quick settings toggle does. Returns None, or why not."""
    from winrt.windows.devices.radios import Radio, RadioAccessStatus, RadioKind, RadioState
    want = RadioKind.WI_FI if kind == "wifi" else RadioKind.BLUETOOTH

    async def go():
        if await Radio.request_access_async() != RadioAccessStatus.ALLOWED:
            return "Windows denied access to the radios (Settings > Privacy > Radios)"
        for r in await Radio.get_radios_async():
            if r.kind == want:
                status = await r.set_state_async(RadioState.ON if on else RadioState.OFF)
                return None if status == RadioAccessStatus.ALLOWED else f"Windows refused ({status.name})"
        return f"there is no {kind} radio"
    return _winrt(go)


def netsh_interfaces():
    return run(["netsh", "wlan", "show", "interfaces"], encoding="oem")


def netsh_networks():
    return run(["netsh", "wlan", "show", "networks", "mode=bssid"], encoding="oem")


def bluetooth_devices():
    """[{name, connected}] for paired Bluetooth devices (one row per device, not per service)."""
    out = powershell(
        "Get-PnpDevice -Class Bluetooth -ErrorAction SilentlyContinue | "
        "Where-Object { $_.InstanceId -match '^BTH(ENUM|LE)\\\\DEV_' } | ForEach-Object { "
        "$c = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName '{83DA6326-97A6-4088-9453-A1923F573B29} 15' "
        "-ErrorAction SilentlyContinue).Data; [pscustomobject]@{name=$_.FriendlyName; connected=[bool]$c} } | "
        "ConvertTo-Json -Compress", timeout=30)
    rows = json.loads(out or "[]")
    rows = rows if isinstance(rows, list) else [rows]
    merged = {}
    for r in rows:  # a device appears once for Classic and once for LE
        merged[r["name"]] = merged.get(r["name"], False) or bool(r["connected"])
    return [{"name": k, "connected": v} for k, v in merged.items()]


# ---- look and feel ----------------------------------------------------------------------------

PERSONALIZE = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
NIGHT_LIGHT = (r"Software\Microsoft\Windows\CurrentVersion\CloudStore\Store\DefaultAccount\Current"
               r"\default$windows.data.bluelightreduction.bluelightreductionstate"
               r"\windows.data.bluelightreduction.bluelightreductionstate")


def dark_mode():
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, PERSONALIZE) as k:
        return winreg.QueryValueEx(k, "AppsUseLightTheme")[0] == 0


def set_dark_mode(dark):
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, PERSONALIZE, 0, winreg.KEY_SET_VALUE) as k:
        for name in ("AppsUseLightTheme", "SystemUsesLightTheme"):
            winreg.SetValueEx(k, name, 0, winreg.REG_DWORD, 0 if dark else 1)
    # Open apps and the taskbar repaint when told the colour set changed.
    res = ctypes.c_size_t()
    user32.SendMessageTimeoutW(HWND_BROADCAST, WM_SETTINGCHANGE, 0, "ImmersiveColorSet", 2, 300, ctypes.byref(res))


# Focus assist / Do not disturb lives in a WNF state (no public API): 0 off, 1 priority only,
# 2 alarms only. The same state Windows' quick settings write.
WNF_QUIET_HOURS = 0x0D83063EA3BF1C75


def quiet_hours():
    state, buf = ctypes.c_uint64(WNF_QUIET_HOURS), ctypes.c_uint32(0)
    size, stamp = ctypes.c_ulong(4), ctypes.c_ulong(0)
    r = ntdll.NtQueryWnfStateData(ctypes.byref(state), None, None, ctypes.byref(stamp), ctypes.byref(buf),
                                  ctypes.byref(size))
    if r != 0:
        raise OSError(f"NtQueryWnfStateData failed ({r & 0xFFFFFFFF:#x})")
    return buf.value


def set_quiet_hours(value):
    state, buf = ctypes.c_uint64(WNF_QUIET_HOURS), ctypes.c_uint32(value)
    r = ntdll.NtUpdateWnfStateData(ctypes.byref(state), ctypes.byref(buf), 4, None, None, 0, 0)
    if r != 0:
        raise OSError(f"NtUpdateWnfStateData failed ({r & 0xFFFFFFFF:#x})")


def brightness():
    """The built-in display's brightness in percent, or None (external monitors aren't reachable)."""
    out = powershell("(Get-CimInstance -Namespace root/wmi -ClassName WmiMonitorBrightness -ErrorAction "
                     "SilentlyContinue | Select-Object -First 1).CurrentBrightness").strip()
    return int(out) if out.isdigit() else None


def set_brightness(percent):
    powershell("Get-CimInstance -Namespace root/wmi -ClassName WmiMonitorBrightnessMethods | "
               f"Invoke-CimMethod -MethodName WmiSetBrightness -Arguments @{{Timeout=0; Brightness={int(percent)}}}")


def night_light_blob():
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, NIGHT_LIGHT) as k:
            return bytes(winreg.QueryValueEx(k, "Data")[0])
    except OSError:
        return None


def set_night_light_blob(data):
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, NIGHT_LIGHT, 0, winreg.KEY_SET_VALUE) as k:
        winreg.SetValueEx(k, "Data", 0, winreg.REG_BINARY, data)


# ---- network ----------------------------------------------------------------------------------

def tcp_ok(host, port, timeout=2.0):
    """Milliseconds to open a TCP connection, or None."""
    t = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout):
            return (time.perf_counter() - t) * 1000
    except OSError:
        return None


def resolve(name):
    """The addresses a name resolves to (IPv4 first), or [] when it doesn't."""
    try:
        infos = socket.getaddrinfo(name, None)
    except OSError:
        return []
    seen = []
    for fam, *_, addr in sorted(infos, key=lambda i: i[0] != socket.AF_INET):
        if addr[0] not in seen:
            seen.append(addr[0])
    return seen


def ping(host, count=4):
    """(replies, [times ms]) from Windows' ping. Matching "TTL=" and "=12ms"/"<1ms" works in the
    Danish output too ("Svar fra ...: bytes=32 tid=12ms TTL=57")."""
    out = run(["ping", "-n", str(count), "-w", "1500", host], timeout=count * 2 + 5, encoding="oem")
    replies = len(re.findall(r"TTL=", out, re.I))
    times = [int(x) for x in re.findall(r"[=<]\s*(\d+)\s*ms\b.*TTL=", out, re.I)]
    return replies, times


def primary_ip():
    """The address the PC uses to reach the internet (a UDP "connect" sends nothing), or None."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(("1.1.1.1", 53))
            return sk.getsockname()[0]
    except OSError:
        return None


def local_addresses():
    """[(interface, IPv4)] for interfaces that are up, without loopback and link-local; the one
    that reaches the internet first (VirtualBox and WSL adapters have addresses too)."""
    stats = psutil.net_if_stats()
    out = []
    for name, addrs in psutil.net_if_addrs().items():
        if not stats.get(name) or not stats[name].isup:
            continue
        for a in addrs:
            if a.family == socket.AF_INET and not a.address.startswith(("127.", "169.254.")):
                out.append((name, a.address))
    main = primary_ip()
    return sorted(out, key=lambda x: x[1] != main)


def http_check(url, timeout=6):
    """(HTTP status or None, ms, error text or None)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/140"})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read(2048)
            return r.status, (time.perf_counter() - t) * 1000, None
    except urllib.error.HTTPError as e:
        return e.code, (time.perf_counter() - t) * 1000, None
    except Exception as e:
        return None, (time.perf_counter() - t) * 1000, str(getattr(e, "reason", e))


SPEED_URL = "https://speed.cloudflare.com/__down?bytes={}"  # free, no account; Cloudflare's own speed test


def download_mbps(alive, nbytes=25_000_000, budget_s=12.0):
    """Download speed in Mbit/s from Cloudflare's speed test, stopping early after budget_s."""
    req = urllib.request.Request(SPEED_URL.format(nbytes), headers={"User-Agent": "Mozilla/5.0"})
    got, t = 0, time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as r:
        while alive() and time.perf_counter() - t < budget_s:
            chunk = r.read(256 * 1024)
            if not chunk:
                break
            got += len(chunk)
    secs = time.perf_counter() - t
    return got * 8 / 1e6 / secs if secs > 0 and got else None


def listening_ports():
    """[(port, address, pid, process name)] for TCP ports in LISTEN state."""
    out = []
    for c in psutil.net_connections("tcp"):
        if c.status != psutil.CONN_LISTEN:
            continue
        out.append((c.laddr.port, c.laddr.ip, c.pid, _exe(c.pid) if c.pid else ""))
    return out


# ---- keyboard, clipboard, session -------------------------------------------------------------

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", DWORD), ("time", DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", DWORD), ("dwFlags", DWORD),
                ("time", DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class INPUT(ctypes.Structure):
    class U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]
    _fields_ = [("type", DWORD), ("u", U)]


user32.SendInput.argtypes = [UINT, ctypes.POINTER(INPUT), ctypes.c_int]


def _keys(events):
    """events: (vk, scan, flags) tuples."""
    arr = (INPUT * len(events))()
    for i, (vk, scan, flags) in enumerate(events):
        arr[i].type = 1
        arr[i].u.ki = KEYBDINPUT(vk, scan, flags, 0, 0)
    return user32.SendInput(len(events), arr, ctypes.sizeof(INPUT))


def type_text(text):
    """Types text at the cursor of whatever has focus, as Unicode keystrokes (Danish letters
    included, whatever the keyboard layout)."""
    events = []
    for ch in text.replace("\r\n", "\n"):
        if ch == "\n":
            events += [(0x0D, 0, 0), (0x0D, 0, 2)]
            continue
        for unit in struct.unpack(f"<{len(ch.encode('utf-16-le')) // 2}H", ch.encode("utf-16-le")):
            events += [(0, unit, 4), (0, unit, 4 | 2)]  # KEYEVENTF_UNICODE, then + KEYUP
    for i in range(0, len(events), 200):
        _keys(events[i:i + 200])
        time.sleep(0.01)


def press_copy():
    _keys([(0x11, 0, 0), (0x43, 0, 0), (0x43, 0, 2), (0x11, 0, 2)])  # Ctrl+C


def clipboard_seq():
    return user32.GetClipboardSequenceNumber()


def _open_clipboard():
    for _ in range(20):  # another app may hold it for a moment
        if user32.OpenClipboard(None):
            return True
        time.sleep(0.025)
    return False


def clipboard_get():
    """The clipboard's text, or None when it holds no text."""
    if not _open_clipboard():
        raise OSError("the clipboard is busy")
    try:
        h = user32.GetClipboardData(CF_UNICODETEXT)
        if not h:
            return None
        p = kernel32.GlobalLock(h)
        try:
            return ctypes.wstring_at(p)
        finally:
            kernel32.GlobalUnlock(h)
    finally:
        user32.CloseClipboard()


def clipboard_set(text):
    data = (text + "\0").encode("utf-16-le")
    h = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
    p = kernel32.GlobalLock(h)
    ctypes.memmove(p, data, len(data))
    kernel32.GlobalUnlock(h)
    if not _open_clipboard():
        kernel32.GlobalFree(h)
        raise OSError("the clipboard is busy")
    try:
        user32.EmptyClipboard()
        if not user32.SetClipboardData(CF_UNICODETEXT, h):
            kernel32.GlobalFree(h)
            raise OSError("SetClipboardData failed")
    finally:
        user32.CloseClipboard()


def lock():
    if not user32.LockWorkStation():
        raise OSError("LockWorkStation failed")


def sleep():
    ctypes.WinDLL("powrprof").SetSuspendState(False, False, False)


# ---- status -----------------------------------------------------------------------------------

def cpu_ram():
    """(CPU %, RAM used GB, RAM total GB)."""
    vm = psutil.virtual_memory()
    return psutil.cpu_percent(interval=0.3), (vm.total - vm.available) / 2**30, vm.total / 2**30


def battery():
    """(percent, plugged in, minutes left or None), or None on a desktop."""
    b = psutil.sensors_battery()
    if b is None:
        return None
    left = b.secsleft if isinstance(b.secsleft, int) and b.secsleft > 0 else None
    return round(b.percent), bool(b.power_plugged), (left // 60 if left else None)


def disks():
    """[(drive, free GB, total GB)] for fixed drives."""
    out = []
    for part in psutil.disk_partitions(all=False):
        if "fixed" not in part.opts and "rw" not in part.opts:
            continue
        try:
            u = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue
        out.append((part.mountpoint.rstrip("\\"), u.free / 2**30, u.total / 2**30))
    return out


def gpus():
    """[(name, load %, VRAM used MB, VRAM total MB, temperature C)] from nvidia-smi, [] without it."""
    try:
        out = run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                   "--format=csv,noheader,nounits"], timeout=8)
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 5:
            try:
                rows.append((parts[0], *(float(x) for x in parts[1:])))
            except ValueError:
                pass
    return rows
