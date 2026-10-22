//! grace-overlay [--port 7815] [--capturable]
//!
//! Listens on 127.0.0.1:<port> for newline JSON ({"glow":true}, {"box":{...}}, {"clear":true});
//! see `protocol`. By default the overlay is excluded from screen capture, so Grace's own
//! screenshots never see her glow or boxes; --capturable turns that off for verification.

fn main() {
    let mut port = grace_overlay::protocol::DEFAULT_PORT;
    let mut capturable = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--port" => {
                port = args
                    .next()
                    .and_then(|p| p.parse().ok())
                    .expect("--port needs a number")
            }
            "--capturable" => capturable = true,
            _ => {
                eprintln!("usage: grace-overlay [--port 7815] [--capturable]");
                std::process::exit(2);
            }
        }
    }
    #[cfg(windows)]
    if let Err(e) = grace_overlay::win::run(port, capturable) {
        eprintln!("grace-overlay: {e}");
        std::process::exit(1);
    }
    #[cfg(not(windows))]
    {
        let _ = (port, capturable);
        eprintln!("grace-overlay runs on Windows only");
        std::process::exit(1);
    }
}
