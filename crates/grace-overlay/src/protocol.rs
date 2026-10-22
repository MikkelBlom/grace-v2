//! The control protocol: newline-delimited JSON on 127.0.0.1:7815, one command per line,
//! one reply line per command (`{"ok":true}` or `{"ok":false,"error":"..."}`).
//!
//!   {"glow": true|false}                                   edge glow on / off (fades)
//!   {"box": {"x":..,"y":..,"w":..,"h":..,"label":"..","ms":4000}}  focus box, times out
//!   {"clear": true}                                        removes every box (not the glow)
//!   {"ping": true}                                         is it running?
//!   {"quit": true}                                         closes the overlay
//!
//! Box coordinates are absolute virtual-desktop pixels (physical, per-monitor DPI aware): the
//! same space as a full-desktop screenshot, offset by the virtual screen's top-left corner.

use serde_json::Value;

pub const DEFAULT_PORT: u16 = 7815;
pub const DEFAULT_BOX_MS: u64 = 4000;
pub const MIN_BOX_MS: u64 = 200;
pub const MAX_BOX_MS: u64 = 60_000;
const MAX_LABEL_CHARS: usize = 80;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Rect {
    pub const fn new(x: i32, y: i32, w: i32, h: i32) -> Self {
        Self { x, y, w, h }
    }
    pub fn right(&self) -> i32 {
        self.x + self.w
    }
    pub fn bottom(&self) -> i32 {
        self.y + self.h
    }
    pub fn intersects(&self, o: &Rect) -> bool {
        self.x < o.right() && o.x < self.right() && self.y < o.bottom() && o.y < self.bottom()
    }
    pub fn offset(&self, dx: i32, dy: i32) -> Rect {
        Rect::new(self.x + dx, self.y + dy, self.w, self.h)
    }
    /// Grown by `d` on every side.
    pub fn inflate(&self, d: i32) -> Rect {
        Rect::new(self.x - d, self.y - d, self.w + 2 * d, self.h + 2 * d)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoxSpec {
    pub rect: Rect,
    pub label: Option<String>,
    pub ms: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Glow(bool),
    Box(BoxSpec),
    Clear,
    Ping,
    Quit,
}

fn int(v: &Value, key: &str) -> Result<i32, String> {
    let n = v
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("box needs a number {key}"))?;
    if !n.is_finite() || n.abs() > 1e6 {
        return Err(format!("box {key} out of range"));
    }
    Ok(n.round() as i32)
}

pub fn parse(line: &str) -> Result<Command, String> {
    let v: Value = serde_json::from_str(line.trim()).map_err(|e| format!("not JSON: {e}"))?;
    let obj = v.as_object().ok_or("expected a JSON object")?;
    if let Some(g) = obj.get("glow") {
        return g
            .as_bool()
            .map(Command::Glow)
            .ok_or_else(|| "glow must be true or false".into());
    }
    if let Some(b) = obj.get("box") {
        let rect = Rect::new(int(b, "x")?, int(b, "y")?, int(b, "w")?, int(b, "h")?);
        if rect.w <= 0 || rect.h <= 0 {
            return Err("box w and h must be positive".into());
        }
        let label = b
            .get("label")
            .and_then(Value::as_str)
            .map(|s| s.trim().chars().take(MAX_LABEL_CHARS).collect::<String>())
            .filter(|s| !s.is_empty());
        let ms = b
            .get("ms")
            .and_then(Value::as_f64)
            .map_or(DEFAULT_BOX_MS, |m| m.max(0.0) as u64);
        return Ok(Command::Box(BoxSpec {
            rect,
            label,
            ms: ms.clamp(MIN_BOX_MS, MAX_BOX_MS),
        }));
    }
    for (key, cmd) in [
        ("clear", Command::Clear),
        ("ping", Command::Ping),
        ("quit", Command::Quit),
    ] {
        if obj.get(key).and_then(Value::as_bool) == Some(true) {
            return Ok(cmd);
        }
    }
    Err("unknown command (glow, box, clear, ping, quit)".into())
}

pub fn reply(result: &Result<Command, String>) -> String {
    match result {
        Ok(_) => r#"{"ok":true}"#.to_string(),
        Err(e) => serde_json::json!({"ok": false, "error": e}).to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_glow() {
        assert_eq!(parse(r#"{"glow": true}"#), Ok(Command::Glow(true)));
        assert_eq!(parse("{\"glow\":false}\r\n"), Ok(Command::Glow(false)));
        assert!(parse(r#"{"glow": "yes"}"#).is_err());
    }

    #[test]
    fn parses_box_with_defaults_and_limits() {
        let c = parse(r#"{"box":{"x":10.4,"y":-20,"w":300,"h":40,"label":"  Gem  "}}"#).unwrap();
        assert_eq!(
            c,
            Command::Box(BoxSpec {
                rect: Rect::new(10, -20, 300, 40),
                label: Some("Gem".into()),
                ms: DEFAULT_BOX_MS
            })
        );
        let Command::Box(b) =
            parse(r#"{"box":{"x":0,"y":0,"w":5,"h":5,"ms":10,"label":""}}"#).unwrap()
        else {
            panic!()
        };
        assert_eq!((b.ms, b.label), (MIN_BOX_MS, None));
        let Command::Box(b) = parse(r#"{"box":{"x":0,"y":0,"w":5,"h":5,"ms":1e9}}"#).unwrap()
        else {
            panic!()
        };
        assert_eq!(b.ms, MAX_BOX_MS);
        let long = "æ".repeat(200);
        let Command::Box(b) = parse(&format!(
            r#"{{"box":{{"x":0,"y":0,"w":5,"h":5,"label":"{long}"}}}}"#
        ))
        .unwrap() else {
            panic!()
        };
        assert_eq!(b.label.unwrap().chars().count(), 80);
    }

    #[test]
    fn rejects_bad_boxes() {
        assert!(parse(r#"{"box":{"x":0,"y":0,"w":0,"h":5}}"#).is_err());
        assert!(parse(r#"{"box":{"x":0,"y":0,"w":5}}"#).is_err());
        assert!(parse(r#"{"box":{"x":"a","y":0,"w":5,"h":5}}"#).is_err());
        assert!(parse(r#"{"box":{"x":1e9,"y":0,"w":5,"h":5}}"#).is_err());
    }

    #[test]
    fn parses_the_rest_and_replies() {
        assert_eq!(parse(r#"{"clear":true}"#), Ok(Command::Clear));
        assert_eq!(parse(r#"{"ping":true}"#), Ok(Command::Ping));
        assert_eq!(parse(r#"{"quit":true}"#), Ok(Command::Quit));
        assert!(parse(r#"{"clear":false}"#).is_err());
        assert!(parse("not json").is_err());
        assert!(parse("[1,2]").is_err());
        assert_eq!(reply(&Ok(Command::Ping)), r#"{"ok":true}"#);
        let r: Value = serde_json::from_str(&reply(&parse("{}"))).unwrap();
        assert_eq!(r["ok"], false);
        assert!(r["error"].as_str().unwrap().contains("unknown"));
    }

    #[test]
    fn rect_math() {
        let a = Rect::new(0, 0, 10, 10);
        assert!(a.intersects(&Rect::new(9, 9, 5, 5)));
        assert!(!a.intersects(&Rect::new(10, 0, 5, 5)));
        assert_eq!(a.offset(-3, 4), Rect::new(-3, 4, 10, 10));
        assert_eq!(a.inflate(2), Rect::new(-2, -2, 14, 14));
    }
}
