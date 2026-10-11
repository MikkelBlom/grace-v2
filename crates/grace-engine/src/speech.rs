//! Clients for `services/speech_server.py` (newline-delimited JSON over localhost TCP).

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use anyhow::{Context, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::TcpStream,
    sync::{Mutex, mpsc},
};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn encode(pcm: &[f32]) -> String {
    let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
    STANDARD.encode(bytes)
}

fn decode(b64: &str) -> anyhow::Result<Vec<f32>> {
    let bytes = STANDARD.decode(b64)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

/// One request at a time over a persistent connection (the server serialises STT anyway).
pub struct Stt {
    conn: Mutex<(
        tokio::net::tcp::OwnedWriteHalf,
        BufReader<tokio::net::tcp::OwnedReadHalf>,
    )>,
}

impl Stt {
    pub async fn connect(addr: &str) -> anyhow::Result<Self> {
        let stream = TcpStream::connect(addr).await.with_context(|| {
            format!("speech server STT not reachable at {addr}; start services/speech_server.py")
        })?;
        stream.set_nodelay(true)?;
        let (r, w) = stream.into_split();
        Ok(Self {
            conn: Mutex::new((w, BufReader::new(r))),
        })
    }

    /// Transcribes 16 kHz mono audio.
    pub async fn transcribe(&self, audio: &[f32]) -> anyhow::Result<Transcript> {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let mut conn = self.conn.lock().await;
        let msg = json!({ "id": id, "rate": 16000, "audio": encode(audio) }).to_string()
            + "
";
        conn.0.write_all(msg.as_bytes()).await?;
        let mut line = String::new();
        conn.1.read_line(&mut line).await?;
        let v: Value = serde_json::from_str(&line).context("bad STT reply")?;
        if let Some(e) = v.get("error").and_then(Value::as_str) {
            bail!("STT: {e}");
        }
        Ok(Transcript {
            text: v["text"].as_str().unwrap_or("").to_owned(),
            check_text: v["check_text"].as_str().unwrap_or("").to_owned(),
            avg_logprob: v["avg_logprob"].as_f64().unwrap_or(0.0),
            no_speech_prob: v["no_speech_prob"].as_f64().unwrap_or(0.0),
            ms: v["ms"].as_f64().unwrap_or(0.0),
        })
    }
}

pub struct Transcript {
    pub text: String,
    /// What ekko-v1-tiny (a transducer, run alongside) heard: the hallucination check.
    pub check_text: String,
    pub avg_logprob: f64,
    pub no_speech_prob: f64,
    pub ms: f64,
}

impl Transcript {
    /// Whisper invents fluent sentences from noise ("Tvernes ændring er blevet fornøjet at vente
    /// en anden opslag", "Tester en to tre ..."), so its text is dropped when ekko, which rarely
    /// invents, heard no words at all. grace-sim (ADR 0016): catches 92/100 non-speech clips and
    /// drops 0/40 clean, 4/40 distant and 1/40 talked-over real utterances; the old avg_logprob
    /// < -0.55 rule caught 50/100 and dropped 16/40 distant and 15/40 talked-over utterances.
    pub fn looks_hallucinated(&self) -> bool {
        !self.text.trim().is_empty() && !self.check_text.chars().any(char::is_alphanumeric)
    }
}

pub enum TtsEvent {
    Audio {
        rate: u32,
        pcm: Vec<f32>,
    },
    Done {
        cancelled: bool,
        first_ms: Option<f64>,
    },
}

/// TTS over one connection; requests are queued server-side, replies routed by id.
pub struct Tts {
    writer: Mutex<tokio::net::tcp::OwnedWriteHalf>,
    routes: Arc<Mutex<std::collections::HashMap<u64, mpsc::UnboundedSender<TtsEvent>>>>,
}

impl Tts {
    pub async fn connect(addr: &str) -> anyhow::Result<Self> {
        let stream = TcpStream::connect(addr).await.with_context(|| {
            format!("speech server TTS not reachable at {addr}; start services/speech_server.py")
        })?;
        stream.set_nodelay(true)?;
        let (r, w) = stream.into_split();
        let routes: Arc<Mutex<std::collections::HashMap<u64, mpsc::UnboundedSender<TtsEvent>>>> =
            Arc::default();
        let table = routes.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(r).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(v) = serde_json::from_str::<Value>(&line) else {
                    continue;
                };
                let Some(id) = v["id"].as_u64() else { continue };
                let mut table = table.lock().await;
                let Some(tx) = table.get(&id) else { continue };
                if v["done"].as_bool() == Some(true) {
                    let _ = tx.send(TtsEvent::Done {
                        cancelled: v["cancelled"].as_bool().unwrap_or(false),
                        first_ms: v["first_ms"].as_f64(),
                    });
                    table.remove(&id);
                } else if let (Some(rate), Some(pcm)) = (v["rate"].as_u64(), v["pcm"].as_str())
                    && let Ok(pcm) = decode(pcm)
                {
                    let _ = tx.send(TtsEvent::Audio {
                        rate: rate as u32,
                        pcm,
                    });
                }
            }
        });
        Ok(Self {
            writer: Mutex::new(w),
            routes,
        })
    }

    /// Starts synthesising `text`; audio arrives on the returned channel.
    pub async fn speak(
        &self,
        text: &str,
    ) -> anyhow::Result<(u64, mpsc::UnboundedReceiver<TtsEvent>)> {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::unbounded_channel();
        self.routes.lock().await.insert(id, tx);
        let msg = json!({ "id": id, "text": text }).to_string() + "\n";
        self.writer.lock().await.write_all(msg.as_bytes()).await?;
        Ok((id, rx))
    }

    pub async fn cancel(&self, id: u64) {
        let msg = json!({ "cancel": id }).to_string() + "\n";
        let _ = self.writer.lock().await.write_all(msg.as_bytes()).await;
    }
}

pub fn write_wav(path: &std::path::Path, pcm: &[f32], rate: u32) -> anyhow::Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &s in pcm {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    Ok(())
}

/// Linear upsampling from the TTS rate to the speaker rate; fine for v0.1 (24→48 kHz is 2×).
pub fn resample(pcm: &[f32], from: u32, to: u32) -> Vec<f32> {
    grace_audio::wasapi::resample_linear(pcm, from, to)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(text: &str, check: &str) -> Transcript {
        Transcript {
            text: text.into(),
            check_text: check.into(),
            avg_logprob: -0.3,
            no_speech_prob: 0.0,
            ms: 0.0,
        }
    }

    #[test]
    fn dropped_only_when_the_check_heard_nothing() {
        assert!(t("Tester en to tre fire fem.", "").looks_hallucinated());
        assert!(t("Tester en to tre fire fem.", " . ").looks_hallucinated());
        assert!(!t("Min søster bor i Aarhus.", "min søster bor i århus").looks_hallucinated());
        assert!(!t("", "").looks_hallucinated(), "nothing to drop");
    }
}
