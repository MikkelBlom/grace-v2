//! Accuracy and latency metrics for bake-offs.

/// Lowercases and replaces everything that isn't a letter or digit with a space, so WER
/// compares words rather than punctuation or casing. Unicode-aware: æ, ø, å survive.
pub fn normalize(text: &str) -> String {
    let mapped: String = text
        .chars()
        .flat_map(char::to_lowercase)
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    mapped.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn edit_distance<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for (i, x) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let substitute = prev[j] + usize::from(x != y);
            cur[j + 1] = substitute.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Errors and reference length, so corpus-level rates can be summed across clips
/// (averaging per-clip rates would overweight short clips).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ErrorCount {
    pub errors: usize,
    pub reference_len: usize,
}

impl ErrorCount {
    pub fn rate(self) -> f64 {
        if self.reference_len == 0 {
            return if self.errors == 0 { 0.0 } else { 1.0 };
        }
        self.errors as f64 / self.reference_len as f64
    }

    pub fn add(self, other: Self) -> Self {
        Self {
            errors: self.errors + other.errors,
            reference_len: self.reference_len + other.reference_len,
        }
    }
}

pub fn word_errors(reference: &str, hypothesis: &str) -> ErrorCount {
    let r = normalize(reference);
    let h = normalize(hypothesis);
    let r: Vec<&str> = r.split_whitespace().collect();
    let h: Vec<&str> = h.split_whitespace().collect();
    ErrorCount {
        errors: edit_distance(&r, &h),
        reference_len: r.len(),
    }
}

pub fn char_errors(reference: &str, hypothesis: &str) -> ErrorCount {
    let r: Vec<char> = normalize(reference).chars().collect();
    let h: Vec<char> = normalize(hypothesis).chars().collect();
    ErrorCount {
        errors: edit_distance(&r, &h),
        reference_len: r.len(),
    }
}

/// Nearest-rank percentile of `values` (0–100). Returns 0 for an empty slice.
pub fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_keeps_danish_letters_and_drops_punctuation() {
        assert_eq!(normalize("Hej, Grace! Hvad så?"), "hej grace hvad så");
        assert_eq!(normalize("Ærø  –  Øresund."), "ærø øresund");
    }

    #[test]
    fn wer_counts_substitution_insertion_deletion() {
        assert_eq!(word_errors("en to tre", "en to tre").errors, 0);
        assert_eq!(word_errors("en to tre", "en fire tre").errors, 1);
        assert_eq!(word_errors("en to tre", "en to").errors, 1);
        assert_eq!(word_errors("en to", "en to tre").errors, 1);
        let e = word_errors("en to tre fire", "to tre");
        assert_eq!((e.errors, e.reference_len), (2, 4));
        assert!((e.rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn wer_ignores_case_and_punctuation() {
        assert_eq!(word_errors("Hej Grace.", "hej, grace").errors, 0);
    }

    #[test]
    fn cer_is_character_level() {
        let e = char_errors("hund", "hånd");
        assert_eq!((e.errors, e.reference_len), (1, 4));
    }

    #[test]
    fn corpus_rate_sums_before_dividing() {
        let total =
            word_errors("a", "b").add(word_errors("a b c d e f g h i", "a b c d e f g h i"));
        assert!((total.rate() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn empty_reference_rate() {
        assert_eq!(word_errors("", "").rate(), 0.0);
        assert_eq!(word_errors("", "noise").rate(), 1.0);
    }

    #[test]
    fn percentile_nearest_rank() {
        let v = [5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(percentile(&v, 50.0), 3.0);
        assert_eq!(percentile(&v, 95.0), 5.0);
        assert_eq!(percentile(&v, 0.0), 1.0);
        assert_eq!(percentile(&[], 50.0), 0.0);
    }
}
