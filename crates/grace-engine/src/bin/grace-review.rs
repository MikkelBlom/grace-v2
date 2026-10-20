//! `grace-review`: the queue of replies Mikkel flagged, and the style rules Grace learned
//! (ADR 0017). Meant for a Claude Code session with him: go through the open entries, turn the
//! good ones into `fixtures/chat/scenarios` + examples in `config/prompts`, then mark them done.
//!
//!   grace-review list [--all]          open entries (or all), one line each
//!   grace-review show <id>             one entry with its context
//!   grace-review export [--all]        open entries as Markdown, for a review session
//!   grace-review done <id> [--note ..] turned into a scenario/example
//!   grace-review dismiss <id> [--note ..]
//!   grace-review rules                 current style rules with the reply behind each
//!   grace-review drop-rule <id>        retire a rule by hand
//!
//! Reads `data/memory.db` (local, gitignored) unless `--db` says otherwise.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use grace::memory::{DIM, EMBEDDER, local_time};
use grace_memory::{Memory, Review};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "data/memory.db")]
    db: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    List {
        #[arg(long)]
        all: bool,
    },
    Show {
        id: i64,
    },
    Export {
        #[arg(long)]
        all: bool,
    },
    Done {
        id: i64,
        #[arg(long)]
        note: Option<String>,
    },
    Dismiss {
        id: i64,
        #[arg(long)]
        note: Option<String>,
    },
    Rules,
    DropRule {
        id: i64,
    },
}

fn when(t: f64) -> String {
    local_time(t).format("%Y-%m-%d %H:%M").to_string()
}

fn one_line(s: &str, n: usize) -> String {
    let flat = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= n {
        flat
    } else {
        format!("{}…", flat.chars().take(n).collect::<String>())
    }
}

fn markdown(r: &Review) -> String {
    let mut out = format!(
        "## Entry {} ({}, {}, {})\n\n",
        r.id,
        r.reason,
        when(r.at),
        r.status
    );
    if !r.context.is_empty() {
        out.push_str("Context:\n\n");
        for (role, text) in &r.context {
            let who = if role == "user" { "Mikkel" } else { "Grace" };
            out.push_str(&format!("> **{who}:** {text}\n>\n"));
        }
        out.push('\n');
    }
    if let Some(reply) = &r.reply {
        out.push_str(&format!("Reply he reacted to: {reply}\n\n"));
    }
    out.push_str(&format!("Mikkel: {}\n\n", r.feedback));
    if let Some(rule) = r.rule_id {
        out.push_str(&format!("Rule learned: #{rule}\n\n"));
    }
    if let Some(note) = &r.note {
        out.push_str(&format!("Note: {note}\n\n"));
    }
    out
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        args.db.exists(),
        "{} not found (run from the repo root, or pass --db)",
        args.db.display()
    );
    let db = Memory::open(&args.db, EMBEDDER, DIM)?;
    let now = grace::memory::system_clock()();
    match args.cmd {
        Cmd::List { all } => {
            let rows = db.reviews((!all).then_some("open"))?;
            if rows.is_empty() {
                println!("(nothing to review)");
            }
            for r in rows {
                println!(
                    "{:>4}  {}  {:<9} {:<9} {}  <-  {}",
                    r.id,
                    when(r.at),
                    r.reason,
                    r.status,
                    one_line(&r.feedback, 60),
                    one_line(r.reply.as_deref().unwrap_or("-"), 60)
                );
            }
        }
        Cmd::Show { id } => {
            let r = db
                .reviews(None)?
                .into_iter()
                .find(|r| r.id == id)
                .ok_or_else(|| anyhow::anyhow!("no entry {id}"))?;
            print!("{}", markdown(&r));
        }
        Cmd::Export { all } => {
            let rows = db.reviews((!all).then_some("open"))?;
            println!("# Grace review queue ({} entries)\n", rows.len());
            for r in &rows {
                print!("{}", markdown(r));
            }
            let rules = db.style_rules()?;
            println!("## Current style rules ({})\n", rules.len());
            for r in rules {
                println!("- #{} {} (he said: \"{}\")", r.id, r.text, r.said);
            }
        }
        Cmd::Done { id, note } => db.close_review(id, "done", note.as_deref(), now)?,
        Cmd::Dismiss { id, note } => db.close_review(id, "dismissed", note.as_deref(), now)?,
        Cmd::Rules => {
            let rules = db.style_rules()?;
            if rules.is_empty() {
                println!("(no style rules)");
            }
            for r in rules {
                println!("#{} {}  [{}]", r.id, r.text, when(r.created_at));
                println!("     he said: \"{}\"", r.said);
                if let Some(e) = r.example_reply {
                    println!("     about:   {}", one_line(&e, 100));
                }
            }
        }
        Cmd::DropRule { id } => db.invalidate_style_rule(id, now)?,
    }
    Ok(())
}
