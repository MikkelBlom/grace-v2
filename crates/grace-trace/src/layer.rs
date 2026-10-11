use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, SyncSender, TrySendError},
    },
    thread::JoinHandle,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde_json::{Map, Value};
use tracing::{
    Event, Id, Subscriber,
    field::{Field, Visit},
    span::{Attributes, Record as SpanValues},
};
use tracing_subscriber::{
    Layer, layer::Context, prelude::*, registry::LookupSpan, registry::Registry,
};

use crate::record::{Kind, Record};

const CHANNEL_CAPACITY: usize = 16_384;

/// Per-span state kept in the registry's extensions until the span closes.
struct SpanState {
    started: Instant,
    ts_us: u64,
    fields: Map<String, Value>,
}

enum Msg {
    Record(Box<Record>),
    /// Flush the file, then acknowledge on the given channel.
    Flush(SyncSender<()>),
}

/// A `tracing` layer that serializes spans and events to JSONL on a background thread.
pub struct JsonlLayer {
    service: String,
    tx: SyncSender<Msg>,
    dropped: Arc<AtomicU64>,
    /// If set, records from other targets are kept only at WARN and above.
    own_targets: Option<Vec<String>>,
}

/// Flushes and closes the trace file when dropped. Keep it alive for the life of the process.
pub struct TraceGuard {
    tx: SyncSender<Msg>,
    // The writer is a daemon thread: the global layer keeps a sender alive for the life of the
    // process, so it never sees a disconnect. Shutdown is an explicit flush instead.
    _writer: JoinHandle<()>,
    dropped: Arc<AtomicU64>,
    pub path: PathBuf,
}

impl TraceGuard {
    /// Records discarded because the writer could not keep up.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Blocks until everything recorded so far is on disk (bounded at 2 s).
    pub fn flush(&self) {
        let (ack_tx, ack_rx) = mpsc::sync_channel(1);
        if self.tx.send(Msg::Flush(ack_tx)).is_ok() {
            let _ = ack_rx.recv_timeout(Duration::from_secs(2));
        }
    }
}

impl Drop for TraceGuard {
    fn drop(&mut self) {
        self.flush();
    }
}

impl JsonlLayer {
    /// Creates the layer plus its guard, writing to `<dir>/<session>.jsonl`.
    pub fn new(service: &str, dir: &Path, session: &str) -> anyhow::Result<(Self, TraceGuard)> {
        fs::create_dir_all(dir)?;
        let path = dir.join(format!("{session}.jsonl"));
        let file = File::options().create(true).append(true).open(&path)?;
        let (tx, rx) = mpsc::sync_channel(CHANNEL_CAPACITY);
        let dropped = Arc::new(AtomicU64::new(0));
        let writer = std::thread::Builder::new()
            .name("grace-trace-writer".into())
            .spawn(move || write_loop(rx, BufWriter::new(file)))?;
        let layer = Self {
            service: service.to_owned(),
            tx: tx.clone(),
            dropped: dropped.clone(),
            own_targets: None,
        };
        let guard = TraceGuard {
            tx,
            _writer: writer,
            dropped,
            path,
        };
        Ok((layer, guard))
    }

    /// Keeps everything from targets starting with these prefixes, and only warnings and errors
    /// from the rest (dependencies such as ONNX Runtime log thousands of TRACE lines a second).
    pub fn own_targets(mut self, prefixes: &[&str]) -> Self {
        self.own_targets = Some(prefixes.iter().map(|p| (*p).to_owned()).collect());
        self
    }

    fn send(&self, record: Record) {
        match self.tx.try_send(Msg::Record(Box::new(record))) {
            Ok(()) | Err(TrySendError::Disconnected(_)) => {}
            Err(TrySendError::Full(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Installs a global subscriber that writes this process's trace to `dir`.
///
/// The session name is `<service>-<unix-millis>`, so files sort by start time.
pub fn init(service: &str, dir: &Path) -> anyhow::Result<TraceGuard> {
    let session = format!("{service}-{}", now_us() / 1000);
    let (layer, guard) = JsonlLayer::new(service, dir, &session)?;
    let layer = layer.own_targets(&["grace", "aec_lab", "demo_turns"]);
    Registry::default().with(layer).try_init()?;
    Ok(guard)
}

fn write_loop(rx: Receiver<Msg>, mut out: BufWriter<File>) {
    // Flush whenever the channel goes idle so the debug viewer sees records within ~100 ms,
    // without paying a syscall per record under load.
    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(Msg::Record(record)) => {
                if let Ok(line) = serde_json::to_string(&record) {
                    let _ = writeln!(out, "{line}");
                }
            }
            Ok(Msg::Flush(ack)) => {
                let _ = out.flush();
                let _ = ack.send(());
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let _ = out.flush();
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    let _ = out.flush();
}

fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

impl<S> Layer<S> for JsonlLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn enabled(&self, meta: &tracing::Metadata<'_>, _ctx: Context<'_, S>) -> bool {
        match &self.own_targets {
            None => true,
            Some(own) => {
                own.iter().any(|p| meta.target().starts_with(p.as_str()))
                    || *meta.level() <= tracing::Level::WARN
            }
        }
    }

    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        let mut fields = Map::new();
        attrs.record(&mut FieldVisitor(&mut fields));
        span.extensions_mut().insert(SpanState {
            started: Instant::now(),
            ts_us: now_us(),
            fields,
        });
    }

    fn on_record(&self, id: &Id, values: &SpanValues<'_>, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        if let Some(state) = span.extensions_mut().get_mut::<SpanState>() {
            values.record(&mut FieldVisitor(&mut state.fields));
        }
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let mut fields = Map::new();
        event.record(&mut FieldVisitor(&mut fields));
        let meta = event.metadata();
        let parent = ctx.event_span(event);
        let turn = parent.as_ref().and_then(|span| {
            span.scope()
                .find_map(|s| turn_of(s.extensions().get::<SpanState>()))
        });
        let name = fields
            .remove("message")
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_else(|| meta.name().to_owned());
        self.send(Record {
            kind: Kind::Event,
            ts_us: now_us(),
            dur_us: None,
            service: self.service.clone(),
            name,
            target: meta.target().to_owned(),
            level: meta.level().to_string(),
            span_id: None,
            parent_id: parent.map(|s| s.id().into_u64()),
            turn,
            fields,
        });
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else { return };
        let turn = span
            .scope()
            .find_map(|s| turn_of(s.extensions().get::<SpanState>()));
        let meta = span.metadata();
        let parent_id = span.parent().map(|p| p.id().into_u64());
        let mut extensions = span.extensions_mut();
        let Some(mut state) = extensions.remove::<SpanState>() else {
            return;
        };
        // `backdate_us`: the span really began this long before it was created (e.g. speech
        // the VAD only confirms after a few frames). Shift the start back, extend the duration.
        let backdate = state
            .fields
            .remove("backdate_us")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        // `trim_us`: the span really ended this long before it was closed.
        let trim = state
            .fields
            .remove("trim_us")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        self.send(Record {
            kind: Kind::Span,
            ts_us: state.ts_us.saturating_sub(backdate),
            dur_us: Some(
                (state.started.elapsed().as_micros() as u64 + backdate).saturating_sub(trim),
            ),
            service: self.service.clone(),
            name: meta.name().to_owned(),
            target: meta.target().to_owned(),
            level: meta.level().to_string(),
            span_id: Some(id.into_u64()),
            parent_id,
            turn,
            fields: state.fields,
        });
    }
}

fn turn_of(state: Option<&SpanState>) -> Option<String> {
    let value = state?.fields.get("turn")?;
    Some(match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    })
}

struct FieldVisitor<'a>(&'a mut Map<String, Value>);

impl Visit for FieldVisitor<'_> {
    fn record_f64(&mut self, field: &Field, value: f64) {
        self.0.insert(field.name().into(), Value::from(value));
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        self.0.insert(field.name().into(), Value::from(value));
    }
    fn record_u64(&mut self, field: &Field, value: u64) {
        self.0.insert(field.name().into(), Value::from(value));
    }
    fn record_bool(&mut self, field: &Field, value: bool) {
        self.0.insert(field.name().into(), Value::from(value));
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().into(), Value::from(value));
    }
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0
            .insert(field.name().into(), Value::from(format!("{value:?}")));
    }
}
