//! A typed, renderable [`Summary`] for every model / test result in this crate.
//!
//! Every estimator and hypothesis test in `tpt-stat` implements [`Summarize`],
//! which produces a `Summary` — a small structured document of sections
//! (key/value blocks, tables, free text). `Summary` implements `Display` with a
//! Unicode box-drawing renderer, and [`Summary::to_markdown`] emits the same
//! content as GitHub-flavoured Markdown so front-ends such as `tpt-lab` can
//! either print it verbatim or restyle the structured form.

use std::fmt;

/// Horizontal alignment of a table column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    Right,
    Center,
}

/// The content of one [`Section`].
#[derive(Debug, Clone)]
pub enum SectionBody {
    /// Aligned `key: value` pairs (packed two-per-row when there are many).
    KeyValues(Vec<(String, String)>),
    /// A header + rows table.
    Table {
        headers: Vec<String>,
        aligns: Vec<Align>,
        rows: Vec<Vec<String>>,
    },
    /// Free-form text (rendered line by line).
    Text(String),
}

/// One titled block of a [`Summary`].
#[derive(Debug, Clone)]
pub struct Section {
    pub title: Option<String>,
    pub body: SectionBody,
}

impl Section {
    pub fn key_values(title: impl Into<String>, kv: Vec<(String, String)>) -> Self {
        Section {
            title: Some(title.into()),
            body: SectionBody::KeyValues(kv),
        }
    }
    pub fn untitled_key_values(kv: Vec<(String, String)>) -> Self {
        Section {
            title: None,
            body: SectionBody::KeyValues(kv),
        }
    }
    pub fn table(
        title: impl Into<String>,
        headers: Vec<String>,
        aligns: Vec<Align>,
        rows: Vec<Vec<String>>,
    ) -> Self {
        Section {
            title: Some(title.into()),
            body: SectionBody::Table {
                headers,
                aligns,
                rows,
            },
        }
    }
    pub fn text(title: impl Into<String>, text: impl Into<String>) -> Self {
        Section {
            title: Some(title.into()),
            body: SectionBody::Text(text.into()),
        }
    }
}

/// A structured, printable result summary.
#[derive(Debug, Clone)]
pub struct Summary {
    pub title: String,
    pub subtitle: Option<String>,
    pub sections: Vec<Section>,
    pub notes: Vec<String>,
}

impl Summary {
    pub fn new(title: impl Into<String>) -> Self {
        Summary {
            title: title.into(),
            subtitle: None,
            sections: Vec::new(),
            notes: Vec::new(),
        }
    }

    pub fn subtitle(mut self, s: impl Into<String>) -> Self {
        self.subtitle = Some(s.into());
        self
    }

    pub fn section(mut self, s: Section) -> Self {
        self.sections.push(s);
        self
    }

    pub fn note(mut self, n: impl Into<String>) -> Self {
        self.notes.push(n.into());
        self
    }

    /// Render the summary as GitHub-flavoured Markdown.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("### {}\n", self.title));
        if let Some(s) = &self.subtitle {
            out.push_str(&format!("_{s}_\n"));
        }
        for sec in &self.sections {
            if let Some(t) = &sec.title {
                out.push_str(&format!("\n**{t}**\n\n"));
            } else {
                out.push('\n');
            }
            match &sec.body {
                SectionBody::KeyValues(kv) => {
                    for (k, v) in kv {
                        out.push_str(&format!("- {k}: {v}\n"));
                    }
                }
                SectionBody::Table {
                    headers,
                    aligns,
                    rows,
                } => {
                    out.push_str(&format!("| {} |\n", headers.join(" | ")));
                    let seps: Vec<String> = headers
                        .iter()
                        .enumerate()
                        .map(|(i, _)| match aligns.get(i).copied().unwrap_or(Align::Right) {
                            Align::Left => ":---".to_string(),
                            Align::Right => "---:".to_string(),
                            Align::Center => ":---:".to_string(),
                        })
                        .collect();
                    out.push_str(&format!("| {} |\n", seps.join(" | ")));
                    for r in rows {
                        out.push_str(&format!("| {} |\n", r.join(" | ")));
                    }
                }
                SectionBody::Text(t) => {
                    for line in t.lines() {
                        out.push_str(line);
                        out.push('\n');
                    }
                }
            }
        }
        if !self.notes.is_empty() {
            out.push('\n');
            for n in &self.notes {
                out.push_str(&format!("> {n}\n"));
            }
        }
        out
    }
}

/// Anything that can describe itself as a [`Summary`].
pub trait Summarize {
    /// Build a structured summary of this result.
    fn summary(&self) -> Summary;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

const MIN_WIDTH: usize = 46;

fn w(s: &str) -> usize {
    s.chars().count()
}

fn pad_right(s: &str, width: usize) -> String {
    let len = w(s);
    if len >= width {
        s.to_string()
    } else {
        format!("{}{}", s, " ".repeat(width - len))
    }
}

fn pad_left(s: &str, width: usize) -> String {
    let len = w(s);
    if len >= width {
        s.to_string()
    } else {
        format!("{}{}", " ".repeat(width - len), s)
    }
}

fn center(s: &str, width: usize) -> String {
    let len = w(s);
    if len >= width {
        return s.to_string();
    }
    let left = (width - len) / 2;
    let right = width - len - left;
    format!("{}{}{}", " ".repeat(left), s, " ".repeat(right))
}

/// A rendered block: either a run of content lines or a horizontal rule.
enum Line {
    Content(String),
    Rule,
}

fn kv_lines(kv: &[(String, String)]) -> Vec<String> {
    if kv.is_empty() {
        return Vec::new();
    }
    let kw = kv.iter().map(|(k, _)| w(k)).max().unwrap_or(0);
    let vw = kv.iter().map(|(_, v)| w(v)).max().unwrap_or(0);
    let cell = |(k, v): &(String, String)| format!("{}: {}", pad_right(k, kw), pad_left(v, vw));
    if kv.len() >= 4 {
        // Two pairs per row for compactness.
        let mut lines = Vec::new();
        let mut i = 0;
        while i < kv.len() {
            if i + 1 < kv.len() {
                lines.push(format!("{}   {}", cell(&kv[i]), cell(&kv[i + 1])));
            } else {
                lines.push(cell(&kv[i]));
            }
            i += 2;
        }
        lines
    } else {
        kv.iter().map(cell).collect()
    }
}

fn table_lines(headers: &[String], aligns: &[Align], rows: &[Vec<String>]) -> Vec<String> {
    let ncol = headers.len();
    let mut widths: Vec<usize> = headers.iter().map(|h| w(h)).collect();
    for r in rows {
        for (i, c) in r.iter().enumerate().take(ncol) {
            widths[i] = widths[i].max(w(c));
        }
    }
    let render = |cells: &[String], header: bool| -> String {
        let mut parts = Vec::with_capacity(ncol);
        for i in 0..ncol {
            let empty = String::new();
            let cell = cells.get(i).unwrap_or(&empty);
            let a = if header {
                match aligns.get(i).copied().unwrap_or(Align::Right) {
                    Align::Left => Align::Left,
                    _ => Align::Right,
                }
            } else {
                aligns.get(i).copied().unwrap_or(Align::Right)
            };
            parts.push(match a {
                Align::Left => pad_right(cell, widths[i]),
                Align::Right => pad_left(cell, widths[i]),
                Align::Center => center(cell, widths[i]),
            });
        }
        parts.join("  ")
    };
    let mut out = vec![render(headers, true)];
    let total: usize = widths.iter().sum::<usize>() + 2 * ncol.saturating_sub(1);
    out.push("─".repeat(total));
    for r in rows {
        out.push(render(r, false));
    }
    out
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut lines: Vec<Line> = Vec::new();

        for (i, sec) in self.sections.iter().enumerate() {
            if i > 0 {
                lines.push(Line::Rule);
            }
            if let Some(t) = &sec.title {
                lines.push(Line::Content(t.clone()));
            }
            match &sec.body {
                SectionBody::KeyValues(kv) => {
                    for l in kv_lines(kv) {
                        lines.push(Line::Content(l));
                    }
                }
                SectionBody::Table {
                    headers,
                    aligns,
                    rows,
                } => {
                    for l in table_lines(headers, aligns, rows) {
                        lines.push(Line::Content(l));
                    }
                }
                SectionBody::Text(t) => {
                    for l in t.lines() {
                        lines.push(Line::Content(l.to_string()));
                    }
                }
            }
        }

        if !self.notes.is_empty() {
            lines.push(Line::Rule);
            for n in &self.notes {
                lines.push(Line::Content(format!("* {n}")));
            }
        }

        let mut width = w(&self.title).max(MIN_WIDTH);
        if let Some(s) = &self.subtitle {
            width = width.max(w(s));
        }
        for l in &lines {
            if let Line::Content(c) = l {
                width = width.max(w(c));
            }
        }

        writeln!(f, "┌─{}─┐", "─".repeat(width))?;
        writeln!(f, "│ {} │", center(&self.title, width))?;
        if let Some(s) = &self.subtitle {
            writeln!(f, "│ {} │", center(s, width))?;
        }
        writeln!(f, "╞═{}═╡", "═".repeat(width))?;
        for l in &lines {
            match l {
                Line::Content(c) => writeln!(f, "│ {} │", pad_right(c, width))?,
                Line::Rule => writeln!(f, "├─{}─┤", "─".repeat(width))?,
            }
        }
        write!(f, "└─{}─┘", "─".repeat(width))
    }
}

// ---------------------------------------------------------------------------
// Number formatting helpers shared by all result types
// ---------------------------------------------------------------------------

/// Format a float with `d` significant-ish decimals, falling back to
/// scientific notation for very large/small magnitudes.
pub fn fmt_f(x: f64, d: usize) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let a = x.abs();
    if a != 0.0 && (a < 1e-4 || a >= 1e7) {
        format!("{x:.*e}", d)
    } else {
        format!("{x:.*}", d)
    }
}

/// Format a p-value the way statistical software does.
pub fn fmt_p(p: f64) -> String {
    if p.is_nan() {
        "NaN".into()
    } else if p < 1e-4 {
        "<0.0001".into()
    } else {
        format!("{p:.4}")
    }
}

/// Conventional significance stars for a p-value.
pub fn stars(p: f64) -> &'static str {
    if p < 0.001 {
        "***"
    } else if p < 0.01 {
        "**"
    } else if p < 0.05 {
        "*"
    } else if p < 0.1 {
        "."
    } else {
        ""
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_box_with_aligned_columns() {
        let s = Summary::new("Test Summary")
            .subtitle("subtitle here")
            .section(Section::key_values(
                "Fit",
                vec![
                    ("n".into(), "100".into()),
                    ("R²".into(), "0.9412".into()),
                    ("AIC".into(), "12.5".into()),
                    ("BIC".into(), "20.1".into()),
                ],
            ))
            .section(Section::table(
                "Coefficients",
                vec!["term".into(), "est".into()],
                vec![Align::Left, Align::Right],
                vec![
                    vec!["intercept".into(), "1.0".into()],
                    vec!["x".into(), "-2.25".into()],
                ],
            ))
            .note("a note");
        let rendered = s.to_string();
        let lines: Vec<&str> = rendered.lines().collect();
        // All rendered lines must have identical display width.
        let width = w(lines[0]);
        for l in &lines {
            assert_eq!(w(l), width, "line {l:?} has inconsistent width");
        }
        assert!(rendered.contains("Test Summary"));
        assert!(rendered.contains("intercept"));
        let md = s.to_markdown();
        assert!(md.contains("| term | est |"));
    }

    #[test]
    fn number_formatting() {
        assert_eq!(fmt_f(1.23456, 3), "1.235");
        assert_eq!(fmt_p(0.5), "0.5000");
        assert_eq!(fmt_p(1e-9), "<0.0001");
        assert_eq!(stars(0.02), "*");
        assert_eq!(stars(0.5), "");
    }
}
