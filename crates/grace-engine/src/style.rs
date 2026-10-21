//! How Mikkel wants Grace to talk, learned from his feedback (ADR 0017).
//!
//! After each exchange a small classifier (LLM slot 2, background) reads Grace's previous reply,
//! what Mikkel said about it, and the current rules, and proposes rule changes. Code checks
//! them the way memory does: a rule must quote Mikkel's words, only shown rules can be changed,
//! rules are short, and there are at most [`MAX_RULES`] (the oldest retires). Rules are rendered
//! by code into their own block of the system message; Grace never edits her core prompt.
//!
//! Exchanges where he called a reply bad, or that changed a rule, go to the review queue so a
//! person can turn them into eval scenarios and style examples offline.

use std::sync::Mutex;

use grace_llm::client::{ChatRequest, LlmClient};
use grace_memory::{Memory, StyleRule, Transcript};
use serde_json::{Value, json};
use tracing::{info, warn};

use crate::memory::supported_by;

/// Rules shown at once. More than this and they pile up into overdone behaviour.
pub const MAX_RULES: usize = 10;
/// A rule is one specific instruction, not an essay or a second system prompt.
pub const MAX_RULE_CHARS: usize = 200;
/// Words that make up a request for quiet right now. Evidence made only of these can't create
/// a rule: v1 of the bench turned "hold lige kæft" after a long reply into a lasting "be concise"
/// rule in 3 of 3 seeds, the same mistake memory once made with "hold din kæft".
const HUSH: &[&str] = &[
    "hold", "din", "lige", "kæft", "kaeft", "stille", "ti", "vent", "nu", "lidt", "et", "øjeblik",
    "giv", "mig", "så", "bare", "shh", "tys", "okay", "ok",
];

fn only_hush(evidence: &str) -> bool {
    let words: Vec<String> = evidence
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect();
    !words.is_empty() && words.iter().all(|w| HUSH.contains(&w.as_str()))
}

/// Turns of context kept with a review entry.
pub const REVIEW_CONTEXT: usize = 6;

/// The system-message block for the current rules ("" when there are none).
pub fn render_rules(rules: &[StyleRule]) -> String {
    if rules.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "How Mikkel wants you to talk. He taught you these rules himself: follow them in every \
         reply, over the general style above. If two conflict, the later one wins. If he asks what \
         rules you have, say them briefly in your own words; he can tell you to drop one.\n",
    );
    for (i, r) in rules.iter().enumerate() {
        out.push_str(&format!(
            "{}. {} (he said: \"{}\")\n",
            i + 1,
            r.text,
            r.said
        ));
    }
    out
}

/// Restates the rules on the user's current message, for this turn only (like recall): the
/// cached prefix is untouched, and the rules sit next to the words she is answering.
pub fn with_reminder(user: &str, rules: &[StyleRule]) -> String {
    if rules.is_empty() {
        return user.to_owned();
    }
    let list: Vec<&str> = rules.iter().map(|r| r.text.as_str()).collect();
    format!(
        "{user}

[How Mikkel wants you to talk: {}]",
        list.join(" ")
    )
}

/// What one exchange changed.
#[derive(Debug, Default, PartialEq)]
pub struct Outcome {
    /// He commented on how Grace talks.
    pub feedback: bool,
    /// He said her previous reply was bad.
    pub complaint: bool,
    pub added: Vec<i64>,
    /// (old, new)
    pub updated: Vec<(i64, i64)>,
    pub removed: Vec<i64>,
    /// Retired by the cap.
    pub retired: Vec<i64>,
    /// Proposed ops code refused, with why.
    pub refused: Vec<String>,
}

impl Outcome {
    pub fn changed(&self) -> bool {
        !(self.added.is_empty() && self.updated.is_empty() && self.removed.is_empty())
    }
}

const STYLE_SYSTEM: &str = "You keep the rules for how Grace, Mikkel's voice assistant, talks to him. You see Grace's previous reply, what Mikkel said next (spoken Danish, machine-transcribed), Grace's answer to that, and the current rules with their ids.
Decide:
- 'feedback': true when Mikkel comments on HOW Grace talks or on her previous reply: length, tone, humour, questions, wording, formality, how chatty she is. Positive or negative. Taking a rule back or adding an exception to one is feedback too. Asking about a topic, correcting a fact, or giving her a task is not feedback.
- 'complaint': true when he says her previous reply was bad, wrong, annoying, too long, off, or not what he wanted.
- 'ops' on the rules:
  'add' (id 0) when he asks Grace to change how she talks in general ('giv mig kortere svar', 'stop med at stille så mange spørgsmål', 'du må gerne være lidt mere fræk'), or praises something and wants more of it ('det var sjovt, mere af det').
  'update' (an existing id) when his feedback changes, sharpens or contradicts that rule: the update replaces it, so its text is the whole new rule, keeping every part of the old one he didn't take back. An exception to a rule ('but for programming, long answers are fine') is an update that writes the exception into that rule, not a new rule. Never leave two rules that conflict or cover the same thing.
  'remove' (an existing id) when he takes a rule back ('du må godt stille spørgsmål igen', 'glem reglen om ...'). To drop every rule, remove each id.
  A request for silence or a pause ('hold kæft', 'stille', 'vent') is never a rule, even right after a long reply: he wants quiet now, not shorter answers forever.
  Rules are only about the manner of Grace's speech: length, tone, humour, questions, phrasing. Never about facts or accuracy, what she should tell him and when (weather, calendar), units, names or nicknames: those belong to her memory, not to rules.
  A rule must apply to many future replies on any topic. If it only makes sense for this conversation, this topic or this one task, it is not a rule.
  Not a rule, though 'complaint' can still be true: saying she didn't listen, misheard or misunderstood him ('hørte du ikke efter', 'det forstår jeg ikke'); a task with a length or format for this one answer ('en historie på mindst 200 ord'); correcting what she assumed about one topic ('det er ikke en komedie'); and how she continues after he interrupted her ('fortsæt hvor du slap', 'du startede forfra'), which the system handles, not her manner.
  Not a rule: requests for this moment or this one reply only ('sig det igen', 'skru op', 'vent', 'hold kæft', 'stille', 'snak lidt langsommere' during one answer), what he wants to talk about, facts about his life, and asking which rules Grace has.
Each op's 'about' names what it concerns: length, tone, humour, questions, phrasing or formality (her manner), or listening (she missed or misunderstood what he said), resuming (how she continues after he interrupted her), one_task (a request for this one answer, like a story of a given length), one_topic (a correction about one subject), facts, other. Only manner ops become rules, so pick the honest category. For 'update' and 'remove', 'about' is the category of the rule being changed and 'id' is that rule's id from the list.
Each rule: one specific instruction to Grace in English, imperative, concrete enough to check in a single reply, at most 25 words ('Keep replies to one or two short sentences unless Mikkel asks for detail.', never 'Be better.'). Base it on what he actually objected to or liked in the previous reply. 'evidence' copies Mikkel's words exactly.
Only Mikkel's words are a source. Never add or change a rule because Grace's reply broke a rule or looks wrong to you: if he didn't comment on it, there is nothing to do. A rule that already covers his feedback needs no new rule.
Most exchanges contain no feedback: then feedback false, complaint false and an empty list.";

/// What a proposed rule is about. The classifier must pick one; code keeps only [`MANNER`].
/// Live run 5: Gemma turned "hørte du ikke efter", "en historie på mindst 100 ord", "det er ikke
/// en comedy" and complaints about resuming after an interruption into rules (5 of 5 false), and
/// prompt wording alone cut that only from 7 to 5 of 24 turns in grace-style-bench 08.
const ABOUT: &[&str] = &[
    "length",
    "tone",
    "humour",
    "questions",
    "phrasing",
    "formality",
    "listening",
    "resuming",
    "one_task",
    "one_topic",
    "facts",
    "other",
];
const MANNER: &[&str] = &[
    "length",
    "tone",
    "humour",
    "questions",
    "phrasing",
    "formality",
];

fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "feedback": { "type": "boolean" },
            "complaint": { "type": "boolean" },
            "ops": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": { "type": "string", "enum": ["add", "update", "remove"] },
                        "id": { "type": "integer" },
                        "about": { "type": "string", "enum": ABOUT },
                        "evidence": { "type": "string" },
                        "rule": { "type": "string" }
                    },
                    "required": ["op", "id", "about", "evidence", "rule"]
                }
            }
        },
        "required": ["feedback", "complaint", "ops"]
    })
}

fn request(rules: &[StyleRule], previous: Option<&str>, user: &str, reply: &str) -> ChatRequest {
    let rules_text = if rules.is_empty() {
        "(none)".to_owned()
    } else {
        rules
            .iter()
            .map(|r| format!("[{}] {}", r.id, r.text))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let prompt = format!(
        "Grace's previous reply:\n{}\n\nMikkel then said:\n{user}\n\nGrace answered:\n{reply}\n\n\
         Current rules:\n{rules_text}",
        previous.unwrap_or("(none: this was the start of the conversation)")
    );
    let mut req = ChatRequest::new(vec![
        json!({ "role": "system", "content": STYLE_SYSTEM }),
        json!({ "role": "user", "content": prompt }),
    ]);
    req.max_tokens = 300;
    req.temperature = 0.2;
    req
}

/// Checks and applies the classifier's proposal. Pure apart from `db`, so it is unit-tested.
pub fn apply(
    db: &mut Memory,
    out: &Value,
    user: &str,
    previous: Option<&str>,
    user_turn: Option<i64>,
    now: f64,
) -> anyhow::Result<Outcome> {
    let mut o = Outcome {
        feedback: out["feedback"].as_bool().unwrap_or(false),
        complaint: out["complaint"].as_bool().unwrap_or(false),
        ..Outcome::default()
    };
    let shown: Vec<i64> = db.style_rules()?.iter().map(|r| r.id).collect();
    for op in out["ops"].as_array().into_iter().flatten() {
        let id = op["id"].as_i64().unwrap_or(0);
        // An update of an id it wasn't shown is still his feedback: keep it as a new rule.
        let name = match op["op"].as_str().unwrap_or("") {
            "update" if !shown.contains(&id) => "add",
            n => n,
        };
        // A new or changed rule without feedback is the classifier policing Grace on its own
        // (grace-style-bench v1: a neutral "jeg så en ret god film" added a second questions
        // rule in 3 of 3 seeds). Dropping a shown rule needs only his quoted words (v2 lost
        // "du må godt stille spørgsmål igen" when every op needed the flag).
        if matches!(name, "add" | "update") && !o.feedback {
            o.refused.push(format!("{name} without feedback"));
            continue;
        }
        let rule = op["rule"].as_str().unwrap_or("").trim();
        let evidence = op["evidence"].as_str().unwrap_or("").trim();
        // Same provenance rule as memory: Mikkel's own words, or nothing.
        if !supported_by(evidence, user) {
            o.refused.push(format!(
                "{name} {id}: evidence not in Mikkel's words: {evidence}"
            ));
            continue;
        }
        let about = op["about"].as_str().unwrap_or("");
        if matches!(name, "add" | "update") && !MANNER.contains(&about) {
            o.refused.push(format!(
                "{name} {id}: about {about:?}, not how she talks: {rule}"
            ));
            continue;
        }
        if matches!(name, "add" | "update") && only_hush(evidence) {
            o.refused.push(format!(
                "{name} {id}: a request for quiet now, not a rule: {evidence}"
            ));
            continue;
        }
        if matches!(name, "add" | "update")
            && (rule.is_empty() || rule.chars().count() > MAX_RULE_CHARS)
        {
            o.refused.push(format!(
                "{name} {id}: rule empty or too long ({} chars)",
                rule.len()
            ));
            continue;
        }
        if name == "remove" && !shown.contains(&id) {
            o.refused.push(format!("{name} {id}: not a current rule"));
            continue;
        }
        if o.removed.contains(&id) || o.updated.iter().any(|(old, _)| *old == id) {
            continue;
        }
        match name {
            "add" => o
                .added
                .push(db.add_style_rule(rule, evidence, previous, user_turn, now)?),
            "update" => {
                let new = db.supersede_style_rule(id, rule, evidence, previous, user_turn, now)?;
                o.updated.push((id, new));
            }
            "remove" => {
                db.invalidate_style_rule(id, now)?;
                o.removed.push(id);
            }
            other => o.refused.push(format!("unknown op {other}")),
        }
    }
    // Over the cap the oldest rule retires: newer feedback wins.
    let current = db.style_rules()?;
    for r in current.iter().take(current.len().saturating_sub(MAX_RULES)) {
        db.invalidate_style_rule(r.id, now)?;
        o.retired.push(r.id);
    }
    Ok(o)
}

/// Where an exchange came from, for the review queue.
pub struct Exchange<'a> {
    pub session: Option<i64>,
    pub user_turn: Option<i64>,
    pub user: &'a str,
    pub reply: &'a str,
    /// Grace's reply before `user`: what the feedback is about.
    pub previous: Option<&'a str>,
    /// Turns before `user`, oldest first.
    pub context: &'a Transcript,
}

/// Classifies one exchange, applies rule changes and queues it for review when it matters.
pub async fn learn(
    llm: &LlmClient,
    db: &Mutex<Memory>,
    ex: &Exchange<'_>,
    now: f64,
) -> anyhow::Result<Outcome> {
    let rules = db.lock().expect("memory poisoned").style_rules()?;
    let out = llm
        .complete_json(&request(&rules, ex.previous, ex.user, ex.reply), &schema())
        .await?;
    let mut db = db.lock().expect("memory poisoned");
    let o = apply(&mut db, &out, ex.user, ex.previous, ex.user_turn, now)?;
    for r in &o.refused {
        info!(refused = %r, "style: proposal refused");
    }
    for (id, text) in db
        .style_rules()?
        .iter()
        .filter(|r| o.added.contains(&r.id) || o.updated.iter().any(|u| u.1 == r.id))
        .map(|r| (r.id, r.text.clone()))
    {
        info!(rule = id, text = %text, "style: rule learned");
    }
    for id in o.removed.iter().chain(&o.retired) {
        info!(rule = id, "style: rule dropped");
    }
    if o.complaint || o.changed() {
        let reason = if o.complaint { "complaint" } else { "rule" };
        let rule = o.added.first().copied().or(o.updated.first().map(|u| u.1));
        if let Err(e) = db.add_review(
            ex.session,
            ex.user_turn,
            reason,
            ex.user,
            ex.previous,
            ex.context,
            rule,
            now,
        ) {
            warn!(error = %e, "style: queueing for review failed");
        }
    }
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> Memory {
        Memory::open_in_memory("test", 8).unwrap()
    }

    const SAID: &str = "kan du ikke give mig kortere svar det der var alt for langt";

    #[test]
    fn rule_needs_mikkels_words_and_a_short_text() {
        let mut m = db();
        let s = m.start_session(0.0).unwrap();
        let t = m.add_turn(s, "user", SAID, 0.5).unwrap();
        let out = json!({ "feedback": true, "complaint": true, "ops": [
            { "op": "add", "id": 0, "about": "length", "evidence": "kortere svar", "rule": "Keep replies to one or two sentences." },
            { "op": "add", "id": 0, "about": "length", "evidence": "be funnier", "rule": "Be funny." },
            { "op": "add", "id": 0, "about": "length", "evidence": "alt for langt", "rule": "x".repeat(300) },
        ]});
        let o = apply(&mut m, &out, SAID, Some("A very long reply."), Some(t), 1.0).unwrap();
        assert_eq!(o.added.len(), 1);
        assert_eq!(o.refused.len(), 2, "{:?}", o.refused);
        let r = &m.style_rules().unwrap()[0];
        assert_eq!(r.said, "kortere svar");
        assert_eq!(r.example_reply.as_deref(), Some("A very long reply."));
        assert_eq!(r.source_turn, Some(t));
    }

    #[test]
    fn only_shown_rules_change_and_update_replaces() {
        let mut m = db();
        let a = m
            .add_style_rule("Keep replies short.", "kort", None, None, 0.0)
            .unwrap();
        let out = json!({ "feedback": true, "complaint": false, "ops": [
            { "op": "update", "id": a, "about": "length", "evidence": "kortere svar", "rule": "Keep replies to one sentence." },
            { "op": "remove", "id": 99, "evidence": "kortere svar", "rule": "" },
            { "op": "remove", "id": a, "evidence": "kortere svar", "rule": "" },
        ]});
        let o = apply(&mut m, &out, SAID, None, None, 1.0).unwrap();
        assert_eq!(o.updated.len(), 1);
        assert!(
            o.removed.is_empty(),
            "a rule already replaced this time is left alone"
        );
        assert_eq!(o.refused.len(), 1);
        let rules = m.style_rules().unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].text, "Keep replies to one sentence.");
    }

    #[test]
    fn only_manner_becomes_a_rule() {
        let mut m = db();
        let said = "hørte du ikke efter og fortæl en historie på mindst 100 ord";
        let out = json!({ "feedback": true, "complaint": true, "ops": [
            { "op": "add", "id": 0, "about": "listening", "evidence": "hørte du ikke efter", "rule": "Listen carefully." },
            { "op": "add", "id": 0, "about": "one_task", "evidence": "mindst 100 ord", "rule": "Write long stories." },
            { "op": "add", "id": 0, "evidence": "hørte du ikke efter", "rule": "No category given." },
        ]});
        let o = apply(&mut m, &out, said, None, None, 1.0).unwrap();
        assert!(o.added.is_empty());
        assert_eq!(o.refused.len(), 3, "{:?}", o.refused);
        assert!(o.complaint, "still a complaint for the review queue");
    }

    #[test]
    fn remove_by_voice() {
        let mut m = db();
        let a = m
            .add_style_rule("Ask fewer questions.", "færre spørgsmål", None, None, 0.0)
            .unwrap();
        let said = "du må godt stille spørgsmål igen";
        let out = json!({ "feedback": true, "complaint": false, "ops": [
            { "op": "remove", "id": a, "evidence": "du må godt stille spørgsmål igen", "rule": "" },
        ]});
        let o = apply(&mut m, &out, said, None, None, 1.0).unwrap();
        assert_eq!(o.removed, vec![a]);
        assert!(m.style_rules().unwrap().is_empty());
    }

    #[test]
    fn cap_retires_the_oldest() {
        let mut m = db();
        for i in 0..MAX_RULES {
            m.add_style_rule(&format!("Rule {i}."), "kort", None, None, i as f64)
                .unwrap();
        }
        let first = m.style_rules().unwrap()[0].id;
        let out = json!({ "feedback": true, "complaint": false, "ops": [
            { "op": "add", "id": 0, "about": "length", "evidence": "kortere svar", "rule": "Newest rule." },
        ]});
        let o = apply(&mut m, &out, SAID, None, None, 100.0).unwrap();
        assert_eq!(o.retired, vec![first]);
        let rules = m.style_rules().unwrap();
        assert_eq!(rules.len(), MAX_RULES);
        assert_eq!(rules.last().unwrap().text, "Newest rule.");
    }

    #[test]
    fn hush_is_not_a_rule() {
        assert!(only_hush("hold lige kæft"));
        assert!(only_hush("Vent lige et øjeblik."));
        assert!(!only_hush("hold dine svar korte"));
        let mut m = db();
        let out = json!({ "feedback": true, "complaint": true, "ops": [
            { "op": "add", "id": 0, "about": "length", "evidence": "hold lige kæft", "rule": "Keep replies concise." },
        ]});
        let o = apply(&mut m, &out, "hold lige kæft", None, None, 1.0).unwrap();
        assert!(!o.changed() && o.refused.len() == 1);
    }

    #[test]
    fn ops_without_feedback_are_ignored() {
        let mut m = db();
        let out = json!({ "feedback": false, "complaint": false, "ops": [
            { "op": "add", "id": 0, "about": "length", "evidence": "kortere svar", "rule": "Keep replies short." },
        ]});
        let o = apply(&mut m, &out, SAID, None, None, 1.0).unwrap();
        assert!(!o.changed());
        assert_eq!(o.refused, vec!["add without feedback".to_owned()]);
        assert!(m.style_rules().unwrap().is_empty());
        let a = m
            .add_style_rule("Ask fewer questions.", "færre", None, None, 2.0)
            .unwrap();
        let out = json!({ "feedback": false, "complaint": false, "ops": [
            { "op": "remove", "id": a, "evidence": "kortere svar", "rule": "" },
        ]});
        let o = apply(&mut m, &out, SAID, None, None, 3.0).unwrap();
        assert_eq!(
            o.removed,
            vec![a],
            "dropping a shown rule needs only his words"
        );
    }

    #[test]
    fn update_of_an_unknown_id_becomes_an_add() {
        let mut m = db();
        let out = json!({ "feedback": true, "complaint": false, "ops": [
            { "op": "update", "id": 0, "about": "length", "evidence": "kortere svar", "rule": "Keep replies short." },
        ]});
        let o = apply(&mut m, &out, SAID, None, None, 1.0).unwrap();
        assert_eq!(o.added.len(), 1);
    }

    #[test]
    fn no_feedback_changes_nothing() {
        let mut m = db();
        let out = json!({ "feedback": false, "complaint": false, "ops": [] });
        let o = apply(&mut m, &out, "hvad er klokken", None, None, 1.0).unwrap();
        assert_eq!(o, Outcome::default());
    }

    #[test]
    fn reminder_only_when_there_are_rules() {
        assert_eq!(with_reminder("hej", &[]), "hej");
        let m = db();
        m.add_style_rule("Keep replies short.", "kort", None, None, 0.0)
            .unwrap();
        let r = with_reminder("hej", &m.style_rules().unwrap());
        assert_eq!(
            r,
            "hej

[How Mikkel wants you to talk: Keep replies short.]"
        );
    }

    #[test]
    fn block_lists_rules_in_order_with_his_words() {
        assert_eq!(render_rules(&[]), "");
        let m = db();
        m.add_style_rule("Keep replies short.", "kortere svar", None, None, 0.0)
            .unwrap();
        m.add_style_rule("Use more dry humour.", "mere tør humor", None, None, 1.0)
            .unwrap();
        let b = render_rules(&m.style_rules().unwrap());
        assert!(b.starts_with("How Mikkel wants you to talk."));
        assert!(b.contains("1. Keep replies short. (he said: \"kortere svar\")"));
        assert!(b.find("1. Keep").unwrap() < b.find("2. Use").unwrap());
    }
}
