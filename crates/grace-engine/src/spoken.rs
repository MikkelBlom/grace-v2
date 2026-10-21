//! Text as Grace should say it. Pocket TTS reads digits badly: live run 7 heard "22-year-old"
//! as "20 20" (issue #37 for symbols). Numbers become English words before synthesis; the text
//! kept in the conversation stays as written.

use std::sync::LazyLock;

use regex::{Captures, Regex};

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

fn below_thousand(n: u64) -> String {
    let (h, rest) = (n / 100, n % 100);
    let mut parts = Vec::new();
    if h > 0 {
        parts.push(format!("{} hundred", ONES[h as usize]));
    }
    if rest > 0 || h == 0 {
        let words = if rest < 20 {
            ONES[rest as usize].to_owned()
        } else if rest % 10 == 0 {
            TENS[(rest / 10) as usize].to_owned()
        } else {
            format!(
                "{}-{}",
                TENS[(rest / 10) as usize],
                ONES[(rest % 10) as usize]
            )
        };
        if h > 0 {
            parts.push(format!("and {words}"));
        } else {
            parts.push(words);
        }
    }
    parts.join(" ")
}

/// 0 ..= 999 999 999 999 in English words.
pub fn integer(n: u64) -> String {
    if n < 1000 {
        return below_thousand(n);
    }
    let mut parts = Vec::new();
    for (scale, name) in [
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1_000, "thousand"),
    ] {
        if !(n / scale).is_multiple_of(1000) {
            parts.push(format!("{} {name}", below_thousand(n / scale % 1000)));
        }
    }
    let rest = n % 1000;
    if rest > 0 {
        // "one thousand and five", "two thousand three hundred"
        if rest < 100 {
            parts.push(format!("and {}", below_thousand(rest)));
        } else {
            parts.push(below_thousand(rest));
        }
    }
    parts.join(" ")
}

/// A year as people say it: 2026 -> "twenty twenty-six", 1906 -> "nineteen oh six".
fn year(n: u64) -> String {
    let (hi, lo) = (n / 100, n % 100);
    if n % 1000 < 10 && n >= 2000 {
        return integer(n); // 2000, 2005: "two thousand and five"
    }
    let lo_words = match lo {
        0 => "hundred".to_owned(),
        1..=9 => format!("oh {}", ONES[lo as usize]),
        _ => below_thousand(lo),
    };
    format!("{} {lo_words}", below_thousand(hi))
}

static NUMBER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?x)
        (?P<neg>-)?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<frac>\d+))?(?P<pct>\s?%)?
    ",
    )
    .expect("number regex")
});

/// Every number in `text` written out in words.
pub fn numbers_to_words(text: &str) -> String {
    NUMBER
        .replace_all(text, |c: &Captures| {
            let int = c["int"].replace(',', "");
            let Ok(n) = int.parse::<u64>() else {
                return c[0].to_owned();
            };
            if n > 999_999_999_999 {
                return c[0].to_owned();
            }
            // A dash right before a number is a range or a hyphen ("5-10", "COVID-19"), not minus.
            let neg = c.name("neg").is_some()
                && c.get(0).is_some_and(|m| {
                    m.start() == 0 || !text[..m.start()].ends_with(|ch: char| ch.is_alphanumeric())
                });
            let plain_year = c.name("frac").is_none()
                && c.name("pct").is_none()
                && !int.contains(',')
                && int.len() == 4
                && (1100..=2099).contains(&n);
            let mut words = if plain_year { year(n) } else { integer(n) };
            if let Some(frac) = c.name("frac") {
                let digits: Vec<&str> = frac
                    .as_str()
                    .chars()
                    .map(|d| ONES[d.to_digit(10).unwrap_or(0) as usize])
                    .collect();
                words = format!("{words} point {}", digits.join(" "));
            }
            if c.name("pct").is_some() {
                words.push_str(" percent");
            }
            if c.name("neg").is_some() {
                if neg {
                    words = format!("minus {words}");
                } else {
                    words = format!("-{words}");
                }
            }
            words
        })
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_read_like_a_person() {
        assert_eq!(
            numbers_to_words("A 22-year-old was found guilty."),
            "A twenty-two-year-old was found guilty."
        );
        assert_eq!(
            numbers_to_words("17 * 23 = 391"),
            "seventeen * twenty-three = three hundred and ninety-one"
        );
        assert_eq!(
            numbers_to_words("The result is 33,367.7."),
            "The result is thirty-three thousand three hundred and sixty-seven point seven."
        );
        assert_eq!(
            numbers_to_words("In 2026 and 1906."),
            "In twenty twenty-six and nineteen oh six."
        );
        assert_eq!(numbers_to_words("95% chance"), "ninety-five percent chance");
        assert_eq!(
            numbers_to_words("between 10 and 13 degrees"),
            "between ten and thirteen degrees"
        );
        assert_eq!(
            numbers_to_words("7,474,356"),
            "seven million four hundred and seventy-four thousand three hundred and fifty-six"
        );
        assert_eq!(numbers_to_words("2005"), "two thousand and five");
        assert_eq!(numbers_to_words("-5 degrees"), "minus five degrees");
        assert_eq!(
            numbers_to_words("season 3 episode 8"),
            "season three episode eight"
        );
    }
}
