//! Writes a few synthetic voice turns to `traces/` so the debug viewer has realistic data
//! before any model is wired up. Stage names match the ones the engine will emit.
//!
//!   cargo run -p grace-trace --example demo_turns

use std::{path::Path, thread::sleep, time::Duration};

use tracing::{info, info_span};

fn stage(name: &'static str, ms: u64, model: &str) {
    let span = info_span!("stage", stage = name, model = model);
    let _enter = span.enter();
    sleep(Duration::from_millis(ms));
}

fn main() -> anyhow::Result<()> {
    let guard = grace_trace::init("demo", Path::new("traces"))?;
    for (i, slow_tool) in [false, true, false].into_iter().enumerate() {
        let turn = info_span!("turn", turn = format!("demo-{i}"));
        let _turn = turn.enter();
        stage("vad.speech", 1200, "silero-v6");
        stage("turn.detect", 180, "smart-turn-v3");
        stage("stt", 140, "roest-v3");
        info!(text = "hvad er klokken", "transcript");
        {
            let llm = info_span!("stage", stage = "llm", model = "gemma4:e4b");
            let _llm = llm.enter();
            stage("llm.ttft", 120, "gemma4:e4b");
            if slow_tool {
                stage("tool.web_search", 900, "searxng");
            }
            stage("llm.decode", 260, "gemma4:e4b");
        }
        stage("tts.ttfa", 90, "piper-da");
        stage("tts.play", 1500, "piper-da");
        sleep(Duration::from_millis(400));
    }
    guard.flush();
    println!("wrote {}", guard.path.display());
    Ok(())
}
