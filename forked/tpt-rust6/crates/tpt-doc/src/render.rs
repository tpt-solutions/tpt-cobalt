//! HTML, LaTeX and plain-text renderers.

use std::fmt::Write as _;

use crate::ast::{Block, DocTable, Document, Equation, Section};

/// Escape text for HTML body/attribute context.
pub fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

/// Escape text for a LaTeX paragraph.
pub fn escape_latex(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\textbackslash{}"),
            '~' => out.push_str("\\textasciitilde{}"),
            '^' => out.push_str("\\textasciicircum{}"),
            '&' | '%' | '$' | '#' | '_' | '{' | '}' => {
                out.push('\\');
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------- HTML

impl Document {
    /// Render a standalone HTML document. Equations are emitted as
    /// `<span class="math">\(...\)</span>`, which KaTeX/MathJax auto-render
    /// picks up; the LaTeX source is preserved verbatim (HTML-escaped).
    pub fn to_html(&self) -> String {
        let mut s = String::new();
        let title = self.title.as_deref().unwrap_or("Untitled");
        s.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
        let _ = writeln!(s, "<title>{}</title>", escape_html(title));
        s.push_str("</head>\n<body>\n");
        if let Some(t) = &self.title {
            let _ = writeln!(s, "<h1 class=\"title\">{}</h1>", escape_html(t));
        }
        if !self.authors.is_empty() {
            let authors: Vec<String> = self.authors.iter().map(|a| escape_html(a)).collect();
            let _ = writeln!(s, "<p class=\"authors\">{}</p>", authors.join(", "));
        }
        if let Some(d) = &self.date {
            let _ = writeln!(s, "<p class=\"date\">{}</p>", escape_html(d));
        }
        if let Some(a) = &self.summary {
            let _ = writeln!(
                s,
                "<div class=\"abstract\">\n<h2>Abstract</h2>\n<p>{}</p>\n</div>",
                escape_html(a)
            );
        }
        for b in &self.blocks {
            html_block(b, 2, &mut s);
        }
        if !self.bibliography.is_empty() {
            s.push_str("<h2 class=\"references\">References</h2>\n<ol class=\"references\">\n");
            for e in self.bibliography.entries() {
                let _ = writeln!(
                    s,
                    "<li id=\"bib-{}\">{}</li>",
                    escape_html(&e.key),
                    escape_html(&e.format())
                );
            }
            s.push_str("</ol>\n");
        }
        s.push_str("</body>\n</html>\n");
        s
    }
}

/// Render one block with the *same* HTML writer the full-page renderer uses,
/// returning just that fragment. Used by the EPUB backend to build per-chapter
/// bodies without duplicating any markup logic.
#[cfg(feature = "epub")]
pub(crate) fn html_block_fragment(b: &Block, level: usize) -> String {
    let mut s = String::new();
    html_block(b, level, &mut s);
    s
}

fn html_block(b: &Block, level: usize, s: &mut String) {
    match b {
        Block::Section(sec) => html_section(sec, level, s),
        Block::Paragraph(p) => {
            let _ = writeln!(s, "<p>{}</p>", escape_html(p));
        }
        Block::Citation(k) => {
            let _ = writeln!(
                s,
                "<p class=\"cite\"><a href=\"#bib-{0}\">[{0}]</a></p>",
                escape_html(k)
            );
        }
        Block::Reference(l) => {
            let _ = writeln!(
                s,
                "<p class=\"ref\"><a href=\"#{0}\">{0}</a></p>",
                escape_html(l)
            );
        }
        Block::Equation(e) => html_equation(e, s),
        Block::Table(t) => html_table(t, s),
    }
}

fn html_section(sec: &Section, level: usize, s: &mut String) {
    let h = level.min(6);
    match &sec.label {
        Some(l) => {
            let _ = writeln!(s, "<section id=\"{}\">", escape_html(l));
        }
        None => s.push_str("<section>\n"),
    }
    let _ = writeln!(s, "<h{h}>{}</h{h}>", escape_html(&sec.title));
    for b in &sec.blocks {
        html_block(b, level + 1, s);
    }
    s.push_str("</section>\n");
}

fn html_equation(e: &Equation, s: &mut String) {
    let id = match &e.label {
        Some(l) => format!(" id=\"{}\"", escape_html(l)),
        None => String::new(),
    };
    let _ = writeln!(
        s,
        "<div class=\"equation\"{id}><span class=\"math\">\\[{}\\]</span></div>",
        escape_html(&e.latex)
    );
}

fn html_table(t: &DocTable, s: &mut String) {
    match &t.label {
        Some(l) => {
            let _ = writeln!(s, "<table id=\"{}\">", escape_html(l));
        }
        None => s.push_str("<table>\n"),
    }
    if let Some(c) = &t.caption {
        let _ = writeln!(s, "<caption>{}</caption>", escape_html(c));
    }
    if !t.headers.is_empty() {
        s.push_str("<thead>\n<tr>");
        for h in &t.headers {
            let _ = write!(s, "<th>{}</th>", escape_html(h));
        }
        s.push_str("</tr>\n</thead>\n");
    }
    s.push_str("<tbody>\n");
    for row in &t.rows {
        s.push_str("<tr>");
        for cell in row {
            let _ = write!(s, "<td>{}</td>", escape_html(cell));
        }
        s.push_str("</tr>\n");
    }
    s.push_str("</tbody>\n</table>\n");
}

// ---------------------------------------------------------------- LaTeX

impl Document {
    /// Render a compilable LaTeX article.
    pub fn to_latex(&self) -> String {
        let mut s = String::new();
        s.push_str("\\documentclass{article}\n");
        s.push_str("\\usepackage{amsmath}\n\\usepackage{booktabs}\n\\usepackage{hyperref}\n");
        if let Some(t) = &self.title {
            let _ = writeln!(s, "\\title{{{}}}", escape_latex(t));
        }
        if !self.authors.is_empty() {
            let authors: Vec<String> = self.authors.iter().map(|a| escape_latex(a)).collect();
            let _ = writeln!(s, "\\author{{{}}}", authors.join(" \\and "));
        }
        if let Some(d) = &self.date {
            let _ = writeln!(s, "\\date{{{}}}", escape_latex(d));
        }
        s.push_str("\\begin{document}\n");
        if self.title.is_some() {
            s.push_str("\\maketitle\n");
        }
        if let Some(a) = &self.summary {
            let _ = writeln!(
                s,
                "\\begin{{abstract}}\n{}\n\\end{{abstract}}",
                escape_latex(a)
            );
        }
        for b in &self.blocks {
            latex_block(b, 0, &mut s);
        }
        if let Some(f) = self.bib_files.first() {
            let stem = f.strip_suffix(".bib").unwrap_or(f);
            s.push_str("\\bibliographystyle{plain}\n");
            let _ = writeln!(s, "\\bibliography{{{stem}}}");
        } else if !self.bibliography.is_empty() {
            let _ = writeln!(
                s,
                "\\begin{{thebibliography}}{{{}}}",
                self.bibliography.len()
            );
            for e in self.bibliography.entries() {
                let _ = writeln!(s, "\\bibitem{{{}}} {}", e.key, escape_latex(&e.format()));
            }
            s.push_str("\\end{thebibliography}\n");
        }
        s.push_str("\\end{document}\n");
        s
    }
}

const LEVELS: [&str; 5] = [
    "section",
    "subsection",
    "subsubsection",
    "paragraph",
    "subparagraph",
];

fn latex_block(b: &Block, depth: usize, s: &mut String) {
    match b {
        Block::Section(sec) => {
            let cmd = LEVELS[depth.min(LEVELS.len() - 1)];
            let _ = writeln!(s, "\\{cmd}{{{}}}", escape_latex(&sec.title));
            if let Some(l) = &sec.label {
                let _ = writeln!(s, "\\label{{{l}}}");
            }
            for b in &sec.blocks {
                latex_block(b, depth + 1, s);
            }
        }
        Block::Paragraph(p) => {
            let _ = writeln!(s, "{}\n", escape_latex(p));
        }
        Block::Citation(k) => {
            let _ = writeln!(s, "\\cite{{{k}}}");
        }
        Block::Reference(l) => {
            let _ = writeln!(s, "\\ref{{{l}}}");
        }
        Block::Equation(e) => {
            s.push_str("\\begin{equation}\n");
            let _ = writeln!(s, "{}", e.latex);
            if let Some(l) = &e.label {
                let _ = writeln!(s, "\\label{{{l}}}");
            }
            s.push_str("\\end{equation}\n");
        }
        Block::Table(t) => latex_table(t, s),
    }
}

fn latex_table(t: &DocTable, s: &mut String) {
    let cols = t
        .headers
        .len()
        .max(t.rows.iter().map(Vec::len).max().unwrap_or(0));
    s.push_str("\\begin{table}[h]\n\\centering\n");
    let _ = writeln!(s, "\\begin{{tabular}}{{{}}}", "l".repeat(cols.max(1)));
    s.push_str("\\toprule\n");
    if !t.headers.is_empty() {
        let hs: Vec<String> = t.headers.iter().map(|h| escape_latex(h)).collect();
        let _ = writeln!(s, "{} \\\\", hs.join(" & "));
        s.push_str("\\midrule\n");
    }
    for row in &t.rows {
        let cs: Vec<String> = row.iter().map(|c| escape_latex(c)).collect();
        let _ = writeln!(s, "{} \\\\", cs.join(" & "));
    }
    s.push_str("\\bottomrule\n\\end{tabular}\n");
    if let Some(c) = &t.caption {
        let _ = writeln!(s, "\\caption{{{}}}", escape_latex(c));
    }
    if let Some(l) = &t.label {
        let _ = writeln!(s, "\\label{{{l}}}");
    }
    s.push_str("\\end{table}\n");
}

// ---------------------------------------------------------------- plain text

impl Document {
    /// Render to plain text lines (also the layout source for the PDF backend).
    pub fn to_text_lines(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(t) = &self.title {
            out.push(t.clone());
        }
        if !self.authors.is_empty() {
            out.push(self.authors.join(", "));
        }
        if let Some(d) = &self.date {
            out.push(d.clone());
        }
        if let Some(a) = &self.summary {
            out.push(String::new());
            out.push("Abstract".to_string());
            out.extend(wrap(a, 90));
        }
        for b in &self.blocks {
            text_block(b, 0, &mut out);
        }
        if !self.bibliography.is_empty() {
            out.push(String::new());
            out.push("References".to_string());
            for e in self.bibliography.entries() {
                out.extend(wrap(&format!("[{}] {}", e.key, e.format()), 90));
            }
        }
        out
    }

    /// Render to a plain-text string.
    pub fn to_text(&self) -> String {
        let mut s = self.to_text_lines().join("\n");
        s.push('\n');
        s
    }
}

fn text_block(b: &Block, depth: usize, out: &mut Vec<String>) {
    let indent = "  ".repeat(depth);
    match b {
        Block::Section(sec) => {
            out.push(String::new());
            out.push(format!("{indent}{}", sec.title));
            for b in &sec.blocks {
                text_block(b, depth + 1, out);
            }
        }
        Block::Paragraph(p) => out.extend(wrap(p, 90).into_iter().map(|l| format!("{indent}{l}"))),
        Block::Citation(k) => out.push(format!("{indent}[{k}]")),
        Block::Reference(l) => out.push(format!("{indent}(see {l})")),
        Block::Equation(e) => out.push(format!("{indent}    {}", e.latex)),
        Block::Table(t) => {
            if !t.headers.is_empty() {
                out.push(format!("{indent}{}", t.headers.join(" | ")));
            }
            for r in &t.rows {
                out.push(format!("{indent}{}", r.join(" | ")));
            }
        }
    }
}

fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        if !cur.is_empty() && cur.chars().count() + 1 + word.chars().count() > width {
            lines.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}
