//! Thin, safe wrappers over WASAPI shared-mode streams.
//!
//! COM objects here are not `Send`: open a stream on the thread that will use it, after
//! calling [`com_init`] on that thread.

use anyhow::{Context, bail};
use windows::{
    Win32::{
        Devices::FunctionDiscovery::PKEY_Device_FriendlyName,
        Media::{Audio::*, Multimedia::KSDATAFORMAT_SUBTYPE_IEEE_FLOAT},
        System::Com::*,
    },
    core::{HSTRING, Interface, PCWSTR},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Capture,
    Render,
}

#[derive(Debug, Clone)]
pub struct Endpoint {
    pub id: String,
    pub name: String,
    pub flow: Flow,
}

/// Initializes COM (multithreaded apartment) for the calling thread. Safe to call repeatedly.
pub fn com_init() -> anyhow::Result<()> {
    // S_FALSE (already initialized) is fine; only a real failure is an error.
    unsafe { CoInitializeEx(None, COINIT_MULTITHREADED).ok() }.context("CoInitializeEx")
}

fn enumerator() -> anyhow::Result<IMMDeviceEnumerator> {
    Ok(unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)? })
}

fn data_flow(flow: Flow) -> EDataFlow {
    match flow {
        Flow::Capture => eCapture,
        Flow::Render => eRender,
    }
}

fn describe(device: &IMMDevice, flow: Flow) -> anyhow::Result<Endpoint> {
    unsafe {
        let raw = device.GetId()?;
        let id = raw.to_string()?;
        CoTaskMemFree(Some(raw.0 as _));
        let name = device
            .OpenPropertyStore(STGM_READ)
            .and_then(|store| store.GetValue(&PKEY_Device_FriendlyName))
            .map(|v| v.to_string())
            .unwrap_or_default();
        Ok(Endpoint { id, name, flow })
    }
}

pub fn list(flow: Flow) -> anyhow::Result<Vec<Endpoint>> {
    unsafe {
        let devices = enumerator()?.EnumAudioEndpoints(data_flow(flow), DEVICE_STATE_ACTIVE)?;
        (0..devices.GetCount()?)
            .map(|i| describe(&devices.Item(i)?, flow))
            .collect()
    }
}

/// The default endpoint for `flow`; `communications` picks the voice-call default.
pub fn default_endpoint(flow: Flow, communications: bool) -> anyhow::Result<Endpoint> {
    let role = if communications {
        eCommunications
    } else {
        eConsole
    };
    let device = unsafe { enumerator()?.GetDefaultAudioEndpoint(data_flow(flow), role)? };
    describe(&device, flow)
}

/// First active endpoint whose friendly name contains any of `needles` (case-insensitive).
pub fn find(flow: Flow, needles: &[&str]) -> anyhow::Result<Endpoint> {
    let all = list(flow)?;
    for needle in needles {
        let needle = needle.to_lowercase();
        if let Some(ep) = all.iter().find(|e| e.name.to_lowercase().contains(&needle)) {
            return Ok(ep.clone());
        }
    }
    let names: Vec<_> = all.iter().map(|e| e.name.as_str()).collect();
    bail!("no {flow:?} endpoint matching {needles:?}; have {names:?}")
}

/// Preferred devices from `config/audio.json` (see that file). Missing file or no match falls
/// back to the Windows default, with a warning, rather than guessing a different device.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct DevicePrefs {
    #[serde(default)]
    pub speakers: Vec<String>,
    #[serde(default)]
    pub microphone: Vec<String>,
}

impl DevicePrefs {
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => Ok(serde_json::from_str(&text)
                .with_context(|| format!("parsing {}", path.display()))?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e.into()),
        }
    }

    pub fn speakers(&self) -> anyhow::Result<Endpoint> {
        resolve(Flow::Render, &self.speakers, false)
    }

    pub fn microphone(&self) -> anyhow::Result<Endpoint> {
        resolve(Flow::Capture, &self.microphone, true)
    }
}

fn resolve(flow: Flow, names: &[String], communications: bool) -> anyhow::Result<Endpoint> {
    if names.is_empty() {
        return default_endpoint(flow, communications);
    }
    let needles: Vec<&str> = names.iter().map(String::as_str).collect();
    find(flow, &needles).or_else(|e| {
        eprintln!("warning: {e}; using the Windows default {flow:?} device");
        default_endpoint(flow, communications)
    })
}

fn device(ep: &Endpoint) -> anyhow::Result<IMMDevice> {
    Ok(unsafe { enumerator()?.GetDevice(&HSTRING::from(ep.id.as_str()))? })
}

#[derive(Debug, Clone, Copy)]
struct Format {
    rate: u32,
    channels: u16,
    float: bool,
    bits: u16,
}

unsafe fn read_format(ptr: *const WAVEFORMATEX) -> Format {
    unsafe {
        let wf = std::ptr::read_unaligned(ptr);
        let float = match wf.wFormatTag as u32 {
            WAVE_FORMAT_IEEE_FLOAT => true,
            WAVE_FORMAT_EXTENSIBLE => {
                let ext = std::ptr::read_unaligned(ptr as *const WAVEFORMATEXTENSIBLE);
                let sub = ext.SubFormat;
                sub == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
            }
            _ => false,
        };
        Format {
            rate: wf.nSamplesPerSec,
            channels: wf.nChannels,
            float,
            bits: wf.wBitsPerSample,
        }
    }
}

const WAVE_FORMAT_IEEE_FLOAT: u32 = 3;
const WAVE_FORMAT_EXTENSIBLE: u32 = 0xFFFE;
const BUFFER_100NS: i64 = 1_000_000; // 100 ms shared-mode buffer

/// Downmixes interleaved device samples to mono f32 and appends them to `out`.
fn append_mono(bytes: &[u8], fmt: Format, out: &mut Vec<f32>) -> anyhow::Result<()> {
    let ch = fmt.channels as usize;
    match (fmt.float, fmt.bits) {
        (true, 32) => {
            for frame in bytes.chunks_exact(4 * ch) {
                let sum: f32 = frame
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .sum();
                out.push(sum / ch as f32);
            }
        }
        (false, 16) => {
            for frame in bytes.chunks_exact(2 * ch) {
                let sum: f32 = frame
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| i16::from_le_bytes(*b) as f32 / 32768.0)
                    .sum();
                out.push(sum / ch as f32);
            }
        }
        other => bail!("unsupported device sample format {other:?}"),
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub enum CaptureMode {
    /// Bypass the driver's voice processing (`AUDCLNT_STREAMOPTIONS_RAW`).
    Raw,
    /// Communications category: the OS applies its voice processing (AEC, NS, beamforming).
    /// With a reference, AEC cancels everything that render endpoint plays.
    Communications { aec_reference: Option<Endpoint> },
    /// Everything a render endpoint is playing.
    Loopback,
}

pub struct Capture {
    // Kept alive: the capture client is only valid while its audio client exists.
    client: IAudioClient,
    capture: IAudioCaptureClient,
    fmt: Format,
    timeline: Timeline,
    samples_read: u64,
}

/// Maps sample indices of one stream to the system performance clock, so streams from
/// different devices (mic, loopback) can be aligned exactly.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Timeline {
    pub rate: u32,
    /// (index of the packet's first sample, its capture time in 100 ns QPC units).
    pub stamps: Vec<(u64, u64)>,
    /// Packets the driver flagged as discontinuous (glitches): non-zero means samples were lost.
    pub discontinuities: u32,
}

impl Timeline {
    /// Capture time of sample `index`, in seconds on the QPC clock. Interpolates between packet
    /// stamps; extrapolates at the sample rate outside them.
    pub fn time_of(&self, index: f64) -> Option<f64> {
        let first = self.stamps.first()?;
        let pos = self.stamps.partition_point(|(i, _)| (*i as f64) <= index);
        let (i0, t0) = if pos == 0 {
            *first
        } else {
            self.stamps[pos - 1]
        };
        Some(t0 as f64 * 1e-7 + (index - i0 as f64) / self.rate as f64)
    }
}

impl Capture {
    /// Opens a capture stream, retrying while a (Bluetooth) device is waking or reconnecting.
    pub fn open(ep: &Endpoint, mode: &CaptureMode) -> anyhow::Result<Self> {
        with_retry(ep, || Self::open_once(ep, mode))
    }

    fn open_once(ep: &Endpoint, mode: &CaptureMode) -> anyhow::Result<Self> {
        unsafe {
            let dev = device(ep)?;
            let client: IAudioClient2 = dev.Activate(CLSCTX_ALL, None)?;
            let mut flags = 0;
            match mode {
                CaptureMode::Raw => {
                    let props = AudioClientProperties {
                        cbSize: size_of::<AudioClientProperties>() as u32,
                        bIsOffload: false.into(),
                        eCategory: AudioCategory_Other,
                        Options: AUDCLNT_STREAMOPTIONS_RAW,
                    };
                    client.SetClientProperties(&props)?;
                }
                CaptureMode::Communications { .. } => {
                    let props = AudioClientProperties {
                        cbSize: size_of::<AudioClientProperties>() as u32,
                        bIsOffload: false.into(),
                        eCategory: AudioCategory_Communications,
                        Options: AUDCLNT_STREAMOPTIONS_NONE,
                    };
                    client.SetClientProperties(&props)?;
                }
                CaptureMode::Loopback => {
                    anyhow::ensure!(ep.flow == Flow::Render, "loopback needs a render endpoint");
                    flags |= AUDCLNT_STREAMFLAGS_LOOPBACK;
                }
            }
            let format = client.GetMixFormat()?;
            let fmt = read_format(format);
            let init = client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                flags,
                BUFFER_100NS,
                0,
                format,
                None,
            );
            CoTaskMemFree(Some(format as _));
            init.context("IAudioClient::Initialize")?;
            if let CaptureMode::Communications {
                aec_reference: Some(reference),
            } = mode
            {
                let aec: IAcousticEchoCancellationControl = client
                    .GetService()
                    .context("driver does not expose AEC reference control")?;
                aec.SetEchoCancellationRenderEndpoint(PCWSTR(
                    HSTRING::from(reference.id.as_str()).as_ptr(),
                ))
                .context("setting AEC reference endpoint")?;
            }
            let capture: IAudioCaptureClient = client.GetService()?;
            Ok(Self {
                client: client.cast()?,
                capture,
                fmt,
                timeline: Timeline {
                    rate: fmt.rate,
                    ..Timeline::default()
                },
                samples_read: 0,
            })
        }
    }

    pub fn rate(&self) -> u32 {
        self.fmt.rate
    }

    pub fn timeline(&self) -> &Timeline {
        &self.timeline
    }

    pub fn start(&self) -> anyhow::Result<()> {
        unsafe { self.client.Start()? };
        Ok(())
    }

    /// Drains every available packet as mono f32 into `out`. Returns frames appended.
    pub fn read_mono(&mut self, out: &mut Vec<f32>) -> anyhow::Result<usize> {
        let before = out.len();
        unsafe {
            while self.capture.GetNextPacketSize()? > 0 {
                let mut data = std::ptr::null_mut();
                let mut frames = 0u32;
                let mut flags = 0u32;
                let mut qpc = 0u64;
                self.capture
                    .GetBuffer(&mut data, &mut frames, &mut flags, None, Some(&mut qpc))?;
                self.timeline.stamps.push((self.samples_read, qpc));
                if flags & (AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY.0 as u32) != 0 {
                    self.timeline.discontinuities += 1;
                }
                self.samples_read += frames as u64;
                let block = self.fmt.channels as usize * (self.fmt.bits as usize / 8);
                if flags & (AUDCLNT_BUFFERFLAGS_SILENT.0 as u32) != 0 {
                    out.extend(std::iter::repeat_n(0.0, frames as usize));
                } else {
                    let bytes = std::slice::from_raw_parts(data, frames as usize * block);
                    append_mono(bytes, self.fmt, out)?;
                }
                self.capture.ReleaseBuffer(frames)?;
            }
        }
        Ok(out.len() - before)
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        unsafe {
            let _ = self.client.Stop();
        }
    }
}

pub struct Render {
    client: IAudioClient,
    render: IAudioRenderClient,
    fmt: Format,
    buffer_frames: u32,
}

impl Render {
    /// Opens a render stream, retrying while a (Bluetooth) device is waking or reconnecting.
    pub fn open(ep: &Endpoint) -> anyhow::Result<Self> {
        with_retry(ep, || Self::open_once(ep))
    }

    fn open_once(ep: &Endpoint) -> anyhow::Result<Self> {
        anyhow::ensure!(ep.flow == Flow::Render, "not a render endpoint");
        unsafe {
            let client: IAudioClient = device(ep)?.Activate(CLSCTX_ALL, None)?;
            let format = client.GetMixFormat()?;
            let fmt = read_format(format);
            let init =
                client.Initialize(AUDCLNT_SHAREMODE_SHARED, 0, BUFFER_100NS, 0, format, None);
            CoTaskMemFree(Some(format as _));
            init.context("IAudioClient::Initialize (render)")?;
            anyhow::ensure!(
                fmt.float && fmt.bits == 32,
                "render mix format {fmt:?} not float32"
            );
            let buffer_frames = client.GetBufferSize()?;
            let render = client.GetService()?;
            Ok(Self {
                client,
                render,
                fmt,
                buffer_frames,
            })
        }
    }

    pub fn rate(&self) -> u32 {
        self.fmt.rate
    }

    /// Plays mono samples (already at [`Self::rate`]) to completion, blocking.
    pub fn play_blocking(&mut self, mono: &[f32]) -> anyhow::Result<()> {
        let ch = self.fmt.channels as usize;
        let mut pos = 0;
        unsafe {
            self.client.Start()?;
            while pos < mono.len() {
                let free = self.buffer_frames - self.client.GetCurrentPadding()?;
                let n = (free as usize).min(mono.len() - pos);
                if n > 0 {
                    let data = self.render.GetBuffer(n as u32)? as *mut f32;
                    let dst = std::slice::from_raw_parts_mut(data, n * ch);
                    for (frame, &s) in dst.chunks_exact_mut(ch).zip(&mono[pos..pos + n]) {
                        frame.fill(s);
                    }
                    self.render.ReleaseBuffer(n as u32, 0)?;
                    pos += n;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            while self.client.GetCurrentPadding()? > 0 {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            self.client.Stop()?;
        }
        Ok(())
    }
}

const TRANSIENT: [windows::core::HRESULT; 3] = [
    AUDCLNT_E_ENDPOINT_CREATE_FAILED,
    AUDCLNT_E_DEVICE_INVALIDATED,
    AUDCLNT_E_DEVICE_IN_USE,
];

/// Retries `open` on errors that a Bluetooth device produces while it wakes from standby,
/// reconnects, or is momentarily busy, then explains the failure in plain words.
fn with_retry<T>(ep: &Endpoint, mut open: impl FnMut() -> anyhow::Result<T>) -> anyhow::Result<T> {
    let mut last = None;
    for attempt in 0..5 {
        match open() {
            Ok(v) => return Ok(v),
            Err(e) => {
                let code = e
                    .chain()
                    .find_map(|c| c.downcast_ref::<windows::core::Error>())
                    .map(|w| w.code());
                if !code.is_some_and(|c| TRANSIENT.contains(&c)) {
                    return Err(e);
                }
                last = Some((e, code));
                std::thread::sleep(std::time::Duration::from_millis(300 * (attempt + 1)));
            }
        }
    }
    let (e, code) = last.expect("loop ran");
    let hint = match code {
        Some(c) if c == AUDCLNT_E_DEVICE_IN_USE => "another app holds it in exclusive mode",
        _ => {
            "it may be off, asleep, or crowded out by another Bluetooth audio device (e.g. earbuds); turn it on, disconnect other Bluetooth audio, and try again"
        }
    };
    Err(e.context(format!("could not open \"{}\": {hint}", ep.name)))
}

/// Linear-interpolation resampler; good enough for test signals, not for the voice path.
pub fn resample_linear(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let len = (input.len() as f64 / ratio) as usize;
    (0..len)
        .map(|i| {
            let x = i as f64 * ratio;
            let j = x as usize;
            let frac = (x - j as f64) as f32;
            let a = input[j];
            let b = *input.get(j + 1).unwrap_or(&a);
            a + (b - a) * frac
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_float_stereo() {
        let fmt = Format {
            rate: 48_000,
            channels: 2,
            float: true,
            bits: 32,
        };
        let bytes: Vec<u8> = [0.5f32, -0.5, 1.0, 0.0]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let mut out = Vec::new();
        append_mono(&bytes, fmt, &mut out).unwrap();
        assert_eq!(out, vec![0.0, 0.5]);
    }

    #[test]
    fn resample_halves_length() {
        let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let out = resample_linear(&input, 48_000, 24_000);
        assert_eq!(out.len(), 50);
        assert_eq!(out[10], 20.0);
    }
}
