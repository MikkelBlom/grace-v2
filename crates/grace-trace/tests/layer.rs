use std::{thread::sleep, time::Duration};

use grace_trace::{JsonlLayer, Kind, read_session};
use tracing::{info, info_span};
use tracing_subscriber::{Registry, prelude::*};

#[test]
fn spans_and_events_inherit_turn_and_record_durations() {
    let dir = tempfile::tempdir().unwrap();
    let (layer, guard) = JsonlLayer::new("test", dir.path(), "s1").unwrap();
    let subscriber = Registry::default().with(layer);

    tracing::subscriber::with_default(subscriber, || {
        let turn = info_span!("turn", turn = "t-1");
        let _turn = turn.enter();
        {
            let stt = info_span!("stt", model = "roest-v3");
            let _stt = stt.enter();
            sleep(Duration::from_millis(5));
            info!(words = 3, "transcript ready");
        }
        info!("outside any stage");
    });
    guard.flush();

    let records = read_session(&guard.path).unwrap();
    let stt = records.iter().find(|r| r.name == "stt").expect("stt span");
    assert_eq!(stt.kind, Kind::Span);
    assert_eq!(stt.turn.as_deref(), Some("t-1"));
    assert_eq!(stt.fields["model"], "roest-v3");
    assert!(stt.dur_us.unwrap() >= 5_000, "duration {:?}", stt.dur_us);

    let event = records
        .iter()
        .find(|r| r.name == "transcript ready")
        .unwrap();
    assert_eq!(event.kind, Kind::Event);
    assert_eq!(event.turn.as_deref(), Some("t-1"));
    assert_eq!(event.parent_id, stt.span_id);
    assert_eq!(event.fields["words"], 3);

    let turn = records.iter().find(|r| r.name == "turn").unwrap();
    assert_eq!(stt.parent_id, turn.span_id);
    assert!(turn.dur_us >= stt.dur_us);
}

#[test]
fn records_without_a_turn_have_none() {
    let dir = tempfile::tempdir().unwrap();
    let (layer, guard) = JsonlLayer::new("test", dir.path(), "s2").unwrap();
    tracing::subscriber::with_default(Registry::default().with(layer), || {
        info!("boot");
    });
    guard.flush();
    let records = read_session(&guard.path).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].turn, None);
}

#[test]
fn torn_last_line_is_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("torn.jsonl");
    std::fs::write(
        &path,
        "{\"kind\":\"event\",\"ts_us\":1,\"service\":\"s\",\"name\":\"ok\",\"target\":\"t\",\"level\":\"INFO\"}\n{\"kind\":\"ev",
    )
    .unwrap();
    let records = read_session(&path).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].name, "ok");
}

#[test]
fn backdate_shifts_start_and_extends_duration() {
    let dir = tempfile::tempdir().unwrap();
    let (layer, guard) = JsonlLayer::new("test", dir.path(), "s3").unwrap();
    tracing::subscriber::with_default(Registry::default().with(layer), || {
        let span = info_span!("stage", stage = "vad.speech", backdate_us = 250_000u64);
        drop(span);
    });
    guard.flush();
    let r = &read_session(&guard.path).unwrap()[0];
    assert!(r.dur_us.unwrap() >= 250_000);
    assert!(!r.fields.contains_key("backdate_us"));
}
