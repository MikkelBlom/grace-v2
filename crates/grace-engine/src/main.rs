//! `grace`: the voice engine. Needs llama-server and services/speech_server.py running (they
//! hold the big models, so this process starts in about a second).
//!
//!   cargo run --release -p grace

mod audio_io;
mod clock;
mod conductor;
mod delay;
mod frontend;
mod speech;
mod timed;

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Instant,
};

use grace_audio::wasapi::{self, CaptureMode, DevicePrefs};
use grace_core::vad_gate::VadGateConfig;
use grace_llm::client::LlmClient;
use grace_vad::smart_turn::SmartTurn;

use crate::{
    audio_io::Source,
    frontend::{FrontendConfig, Shared},
    timed::TimedRing,
};

const LLM_URL: &str = "http://127.0.0.1:8080";
const STT_ADDR: &str = "127.0.0.1:7810";
const TTS_ADDR: &str = "127.0.0.1:7811";
const LEAD_MS: f64 = 30.0;

/// Refuses to start a second engine: two would fight over the mic and both answer.
fn single_instance() -> anyhow::Result<std::fs::File> {
    std::fs::create_dir_all("data")?;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open("data/engine.lock")?;
    if file.try_lock().is_err() {
        let pid = std::fs::read_to_string("data/engine.pid").unwrap_or_default();
        let pid = pid.trim();
        anyhow::bail!("Grace is already running (pid {pid}). Stop it with: taskkill /PID {pid} /F");
    }
    std::fs::write("data/engine.pid", std::process::id().to_string())?;
    Ok(file)
}

fn calibrated_delay(speakers: &str) -> Option<f64> {
    let text = std::fs::read_to_string("data/calibration.json").ok()?;
    let all: BTreeMap<String, serde_json::Value> = serde_json::from_str(&text).ok()?;
    all.get(speakers)?["delay_ms"].as_f64()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let started = Instant::now();
    let _instance = single_instance()?;
    let trace = grace_trace::init("engine", Path::new("traces"))?;

    let llm = LlmClient::new(LLM_URL);
    anyhow::ensure!(
        llm.healthy().await,
        "llama-server is not running at {LLM_URL} (see scripts/grace.ps1)"
    );
    let stt = Arc::new(speech::Stt::connect(STT_ADDR).await?);
    let tts = Arc::new(speech::Tts::connect(TTS_ADDR).await?);
    let smart_turn = Arc::new(Mutex::new(SmartTurn::new(Path::new(
        ".runtime/models/smart-turn-v3.2-fp32.onnx",
    ))?));

    wasapi::com_init()?;
    let prefs = DevicePrefs::load(Path::new("config/audio.json"))?;
    let (mic, speakers) = (prefs.microphone()?, prefs.speakers()?);
    let delay = calibrated_delay(&speakers.name).unwrap_or(150.0);
    tracing::info!(mic = %mic.name, speakers = %speakers.name, initial_delay_ms = delay, "devices");

    let stop = Arc::new(AtomicBool::new(false));
    let talking = Arc::new(AtomicBool::new(false));
    let (chunk_tx, chunk_rx) = mpsc::channel();
    let (front_tx, front_rx) = tokio::sync::mpsc::unbounded_channel();
    let (render_tx, render_rx) = mpsc::channel();
    let (render_ev_tx, render_ev_rx) = tokio::sync::mpsc::unbounded_channel();
    let shared = Arc::new(Mutex::new(Shared {
        aec16: TimedRing::new(16_000, 30.0),
        comms16: TimedRing::new(16_000, 30.0),
        speakers16: TimedRing::new(16_000, 30.0),
    }));

    let threads = vec![
        audio_io::spawn_render(
            speakers.clone(),
            render_rx,
            render_ev_tx,
            talking.clone(),
            stop.clone(),
        ),
        audio_io::spawn_capture(
            Source::Mic,
            mic.clone(),
            CaptureMode::Raw,
            chunk_tx.clone(),
            stop.clone(),
        ),
        audio_io::spawn_capture(
            Source::Comms,
            mic.clone(),
            CaptureMode::Communications {
                aec_reference: Some(speakers.clone()),
            },
            chunk_tx.clone(),
            stop.clone(),
        ),
        audio_io::spawn_capture(
            Source::Loopback,
            speakers.clone(),
            CaptureMode::Loopback,
            chunk_tx,
            stop.clone(),
        ),
        frontend::spawn(
            FrontendConfig {
                vad_model: PathBuf::from(".runtime/models/silero_vad_v6.2.3.onnx"),
                initial_delay_ms: delay,
                lead_ms: LEAD_MS,
                gate: VadGateConfig::default(),
            },
            chunk_rx,
            shared.clone(),
            front_tx,
            talking,
            stop.clone(),
        ),
    ];

    {
        // Ctrl+C, closing the window, or q + Enter all stop Grace; if a thread hangs on the way
        // out, the process exits anyway after 2 s (the first live run kept talking after Ctrl+C).
        let stop = stop.clone();
        tokio::spawn(async move {
            let mut close = tokio::signal::windows::ctrl_close().ok();
            // A plain thread, not spawn_blocking: a pending console read must never keep the
            // runtime from shutting down.
            let (quit_tx, quit_rx) = tokio::sync::oneshot::channel::<()>();
            std::thread::spawn(move || {
                let mut line = String::new();
                while std::io::stdin().read_line(&mut line).is_ok_and(|n| n > 0) {
                    if line.trim().eq_ignore_ascii_case("q") {
                        let _ = quit_tx.send(());
                        return;
                    }
                    line.clear();
                }
            });
            let closed = async {
                match close.as_mut() {
                    Some(c) => {
                        c.recv().await;
                    }
                    None => std::future::pending::<()>().await,
                }
            };
            let typed_q = async {
                if quit_rx.await.is_err() {
                    std::future::pending::<()>().await // stdin closed: not a quit
                }
            };
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = closed => {}
                _ = typed_q => {}
            }
            println!("stopping ...");
            stop.store(true, Ordering::Relaxed);
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            eprintln!("forced exit");
            std::process::exit(0);
        });
    }
    println!(
        "Grace is listening ({} → {}), ready in {:.1} s. Ctrl+C (or q + Enter) to stop.",
        mic.name,
        speakers.name,
        started.elapsed().as_secs_f64()
    );
    tracing::info!(
        startup_ms = started.elapsed().as_millis() as u64,
        "engine ready"
    );

    if std::env::args().any(|a| a == "--selftest") {
        // Silent check: streams and models run, nothing is answered or played.
        let mut front_rx = front_rx;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut events = Vec::new();
        while let Ok(Some(ev)) = tokio::time::timeout_at(deadline, front_rx.recv()).await {
            events.push(format!("{ev:?}"));
        }
        let (aec, comms) = {
            let s = shared.lock().expect("shared poisoned");
            (s.aec16.end(), s.comms16.end())
        };
        let now = clock::now();
        println!(
            "selftest: echo-cancelled mic {:.2} s behind now, comms mic {:.2} s behind now",
            now - aec,
            now - comms
        );
        println!("selftest: {} frontend events: {:?}", events.len(), events);
        let ok = (now - aec) < 0.5 && (now - comms) < 0.5;
        println!(
            "selftest: {}",
            if ok {
                "PASS (audio flowing on both paths)"
            } else {
                "FAIL (a capture path is not delivering)"
            }
        );
        stop.store(true, Ordering::Relaxed);
        trace.flush();
        std::process::exit(if ok { 0 } else { 1 });
    }
    let services = conductor::Services {
        llm,
        llm_model: "gemma-4-12b-qat".into(),
        stt,
        tts,
        smart_turn,
        shared,
        render: render_tx,
        system_prompt: PathBuf::from("config/prompts/system.md"),
    };
    // Threads that die early (a device unplugged, a model failing) end the engine visibly.
    let watchdog = {
        let stop = stop.clone();
        tokio::task::spawn_blocking(move || {
            for t in threads {
                let name = t.thread().name().unwrap_or("?").to_owned();
                match t.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => eprintln!("{name} stopped: {e:#}"),
                    Err(_) => eprintln!("{name} panicked"),
                }
                stop.store(true, Ordering::Relaxed);
            }
        })
    };
    let result = conductor::run(services, front_rx, render_ev_rx, stop.clone()).await;
    stop.store(true, Ordering::Relaxed);
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), watchdog).await;
    trace.flush();
    if let Err(e) = result {
        eprintln!("conductor: {e:#}");
        std::process::exit(1);
    }
    std::process::exit(0);
}
