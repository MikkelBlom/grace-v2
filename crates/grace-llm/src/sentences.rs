//! Cuts a streamed reply into speakable pieces as soon as each one is complete.
//!
//! First audio waits for the first piece, so the splitter is eager: it breaks at sentence
//! ends, and for the *first* piece only it also breaks at a clause boundary (comma, colon,
//! dash) once enough text has arrived. After that it waits for full sentences, which sound
//! more natural. It avoids breaking inside numbers ("3.5"), common abbreviations ("e.g.",
//! "fx.") and runs of punctuation ("...").

const ABBREVIATIONS: &[&str] = &[
    "e.g.", "i.e.", "etc.", "vs.", "mr.", "mrs.", "ms.", "dr.", "st.", "no.", "fx.", "dvs.",
    "osv.", "bl.a.", "ca.", "evt.", "nr.", "kl.", "jf.",
];

#[derive(Debug, Clone)]
pub struct SentenceSplitter {
    buf: String,
    emitted_any: bool,
    /// Minimum characters before the first piece may end at a clause boundary.
    pub first_clause_min: usize,
    /// Pieces shorter than this are held and merged into the next one ("Ok." alone is fine,
    /// but "1." is not).
    pub min_len: usize,
}

impl Default for SentenceSplitter {
    fn default() -> Self {
        Self {
            buf: String::new(),
            emitted_any: false,
            first_clause_min: 40,
            min_len: 2,
        }
    }
}

impl SentenceSplitter {
    /// Feeds streamed text; returns every piece completed by it.
    pub fn push(&mut self, text: &str) -> Vec<String> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        while let Some(end) = self.find_break() {
            let piece = self.buf[..end].trim().to_owned();
            self.buf.drain(..end);
            if !piece.is_empty() {
                self.emitted_any = true;
                out.push(piece);
            }
        }
        out
    }

    /// Returns whatever is left when the stream ends.
    pub fn finish(&mut self) -> Option<String> {
        let rest = std::mem::take(&mut self.buf);
        let rest = rest.trim();
        (!rest.is_empty()).then(|| rest.to_owned())
    }

    /// Byte index just past the first break point, if one is complete.
    fn find_break(&self) -> Option<usize> {
        let chars: Vec<(usize, char)> = self.buf.char_indices().collect();
        for (k, &(i, c)) in chars.iter().enumerate() {
            // A break needs to see the following character to know it's real.
            let &(next_i, next) = chars.get(k + 1)?;
            let end = next_i;
            let len = self.buf[..end].trim().chars().count();
            if c == '\n' && len >= self.min_len {
                return Some(end);
            }
            let sentence_end = matches!(c, '.' | '!' | '?' | '…');
            let clause_end = !self.emitted_any
                && matches!(c, ',' | ':' | ';' | '—' | '–')
                && len >= self.first_clause_min;
            if !(sentence_end || clause_end) || !next.is_whitespace() || len < self.min_len {
                continue;
            }
            if c == '.' && self.is_abbreviation_or_number(i) {
                continue;
            }
            return Some(end);
        }
        None
    }

    fn is_abbreviation_or_number(&self, dot: usize) -> bool {
        let before = &self.buf[..=dot];
        let word = before
            .rsplit(|c: char| c.is_whitespace())
            .next()
            .unwrap_or("")
            .to_lowercase();
        if ABBREVIATIONS.iter().any(|a| word.ends_with(a)) {
            return true;
        }
        // "kl. 3." / "1." list numbers: a bare number followed by a dot is not a sentence end
        // unless the piece already has words before it.
        let digits = word.trim_end_matches('.');
        !digits.is_empty()
            && digits.chars().all(|c| c.is_ascii_digit())
            && before.trim().len() == word.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split_stream(tokens: &[&str]) -> Vec<String> {
        let mut s = SentenceSplitter::default();
        let mut out: Vec<String> = tokens.iter().flat_map(|t| s.push(t)).collect();
        out.extend(s.finish());
        out
    }

    #[test]
    fn splits_sentences_as_they_complete() {
        let mut s = SentenceSplitter::default();
        assert!(s.push("Sure. I can").len() == 1);
        assert_eq!(s.push(" do that! And"), vec!["I can do that!"]);
        assert_eq!(s.finish().as_deref(), Some("And"));
    }

    #[test]
    fn needs_whitespace_after_the_dot() {
        let mut s = SentenceSplitter::default();
        assert!(s.push("It costs 3.").is_empty());
        assert!(s.push("5 kroner").is_empty());
        assert_eq!(s.push(". Ok"), vec!["It costs 3.5 kroner."]);
    }

    #[test]
    fn keeps_abbreviations_together() {
        assert_eq!(
            split_stream(&["Bring tools, e.g. a hammer. Done."]),
            vec!["Bring tools, e.g. a hammer.", "Done."]
        );
        assert_eq!(
            split_stream(&["Mødet er kl. 14 i dag. Ok."]),
            vec!["Mødet er kl. 14 i dag.", "Ok."]
        );
    }

    #[test]
    fn first_piece_may_end_at_a_long_clause() {
        let out = split_stream(&[
            "The weather tomorrow in Aarhus looks mostly cloudy, with some rain later, and wind. Bye.",
        ]);
        assert_eq!(
            out,
            vec![
                "The weather tomorrow in Aarhus looks mostly cloudy,",
                "with some rain later, and wind.",
                "Bye."
            ]
        );
    }

    #[test]
    fn short_first_clause_does_not_split() {
        assert_eq!(split_stream(&["Yes, sure. Ok."]), vec!["Yes, sure.", "Ok."]);
    }

    #[test]
    fn newline_breaks() {
        assert_eq!(
            split_stream(&["Line one\nLine two"]),
            vec!["Line one", "Line two"]
        );
    }

    #[test]
    fn ellipsis_runs_stay_together() {
        assert_eq!(
            split_stream(&["Hmm... let me see. Ok."]),
            vec!["Hmm...", "let me see.", "Ok."]
        );
    }

    #[test]
    fn handles_multibyte_characters() {
        assert_eq!(
            split_stream(&["Så er det gjort. Æblet!"]),
            vec!["Så er det gjort.", "Æblet!"]
        );
    }
}
