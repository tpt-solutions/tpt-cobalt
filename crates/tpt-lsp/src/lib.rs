//! # tpt-lsp — the TPT Script language server (Phase 6)
//!
//! Tensor-aware LSP for TPT Script, superseding the forked
//! `tpt-gpu-script-lsp` for this language (same supersession pattern as the
//! REPL: the fork served a different scripting language; the Cobalt server
//! is built on `tpt-lang` directly).
//!
//! The analysis layer ([`analyze`], [`completions_at`], [`hover_at`]) is
//! pure and unit-tested; `server.rs` is a thin tower-lsp adapter over stdio.
//!
//! State awareness: completions and hover interpret the document *lines
//! above the cursor* in a fresh interpreter, so results reflect the real
//! globals — variable names, function signatures, and module members — at
//! that point in the file. Diagnostics come from the same static checker
//! the language uses at runtime (`tpt_lang::check`): unit mismatches and
//! impossible matmul shapes are compile errors before execution. Position
//! mapping is coarse for now (diagnostics carry the message but not a
//! precise source span — the AST does not track offsets yet; documented in
//! the roadmap).

use std::sync::{Arc, Mutex};

use tpt_lang::Value;
use tpt_lang::Interpreter;

/// A diagnostic in LSP shape (line/character positions, zero-based).
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostic {
    pub line: usize,
    pub col_start: usize,
    pub col_end: usize,
    /// 1 = error, 2 = warning, 3 = information.
    pub severity: u8,
    pub message: String,
    pub source: &'static str,
}

/// Static analysis of a whole document: syntax + unit/shape checks.
pub fn analyze(src: &str) -> Vec<Diagnostic> {
    let to_diag = |severity: u8, message: String| Diagnostic {
        line: 0,
        col_start: 0,
        col_end: 1,
        severity,
        message,
        source: "tpt",
    };
    let mut out = Vec::new();
    match tpt_lang::check::check_program(src) {
        Ok(()) => {}
        Err(e) => out.push(to_diag(1, format!("{e}"))),
    }
    // a syntax error in the interpreter surfaces the same problem from the
    // checker path above in most cases, but catch what the checker is
    // gradual about (it never reports syntax, so try a parse-only pass)
    let toks = match tpt_lang::interp::lex(src) {
        Ok(t) => t,
        Err(e) => {
            out.push(to_diag(1, format!("{}: {}", e.kind, e.message)));
            return out;
        }
    };
    let mut parser = tpt_lang::interp::Parser::new(&toks);
    if let Err(e) = parser.parse_block() {
        out.push(to_diag(1, format!("{}: {}", e.kind, e.message)));
    }
    out
}

/// Interpreter state from the lines strictly above `line` (the completion /
/// hover context), in a fresh side-effect-free interpreter.
fn state_above(src: &str, line: usize) -> Interpreter {
    let mut interp = Interpreter::new();
    let above: Vec<&str> = src.lines().take(line).collect();
    if !above.is_empty() {
        let _ = interp.run(&above.join("\n"));
    }
    interp
}

/// The identifier (or dotted path) under the cursor, returned as
/// `(start_col, word)` — expanded in both directions so a mid-word cursor
/// resolves the whole symbol.
fn ident_at(src: &str, line: usize, col: usize) -> Option<(usize, String)> {
    let row = src.lines().nth(line)?;
    let bytes: Vec<char> = row.chars().collect();
    let col = col.min(bytes.len());
    let is_word = |c: char| c.is_alphanumeric() || c == '_' || c == '.';
    let mut start = col.min(bytes.len());
    while start > 0 && is_word(bytes[start - 1]) {
        start -= 1;
    }
    let mut end = col.min(bytes.len());
    while end < bytes.len() && is_word(bytes[end]) {
        end += 1;
    }
    if start == end {
        return None;
    }
    let word: String = bytes[start..end].iter().collect();
    // trim trailing dots ("geom." hovering mid-path)
    let trimmed = word.trim_end_matches('.');
    let start_col = start + word.chars().count() - trimmed.chars().count();
    Some((start_col, trimmed.to_string()))
}

/// A rendered hover.
#[derive(Debug, Clone, PartialEq)]
pub struct Hover {
    pub line: usize,
    pub col_start: usize,
    pub col_end: usize,
    pub contents: String,
}

/// Resolve a (possibly dotted) path: `geom.area` looks up member `area` of
/// the module bound to `geom`.
fn resolve(interp: &Interpreter, path: &str) -> Option<Value> {
    match path.rsplit_once('.') {
        Some((base, member)) => match interp.get(base)? {
            Value::Module(m) => m.get(member),
            _ => None,
        },
        None => interp.get(path),
    }
}

/// Hover information for the symbol under the cursor: variable values with
/// tensor shapes, function signatures, module member listings.
pub fn hover_at(src: &str, line: usize, col: usize) -> Option<Hover> {
    let (start_col, path) = ident_at(src, line, col)?;
    let mut interp = state_above(src, line);
    let contents = match resolve(&interp, &path)? {
        Value::Function(f) => f.signature(),
        Value::Module(m) => {
            let members = m.member_names().join(", ");
            format!("<module {}> {{ {} }}", m.name, members)
        }
        Value::Tensor(t) => format!("tensor {:?} (f64)", t.shape()),
        Value::Unit(u) => format!("{} {}", u.value, u.dim),
        v => format!("{} = {}", v.type_name(), v),
    };
    Some(Hover {
        line,
        col_start: start_col,
        col_end: col,
        contents,
    })
}

/// Tab completions at the cursor: globals + keywords + natives from the
/// interpreted prefix, or module members for dotted prefixes (`geom.ar`).
pub fn completions_at(src: &str, line: usize, prefix: &str) -> Vec<String> {
    let interp = state_above(src, line);
    let (base, member) = match prefix.rsplit_once('.') {
        Some((b, m)) => (b, m),
        None => ("", prefix),
    };
    let mut names: Vec<String> = if base.is_empty() {
        let mut n: Vec<String> = interp.names();
        n.extend(
            [
                "let", "def", "fn", "if", "else", "while", "return", "print", "assert",
                "module", "true", "false", "nil", "None", "not",
            ]
            .iter()
            .map(|s| s.to_string()),
        );
        n
    } else {
        match interp.get(base) {
            Some(Value::Module(m)) => m.member_names(),
            _ => Vec::new(),
        }
    };
    names.retain(|n| n.starts_with(member));
    names.sort();
    names.dedup();
    names
}

pub mod server;

/// Shared document handle (kept minimal; the analysis layer is stateless
/// per request).
pub type SharedDoc = Arc<Mutex<String>>;

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = "\
let theta = 0.5
module geom {
    let area = 1
    def circle(r) {
        return 3.14 * r * r
    }
}
let c = geom.circle(theta)
";

    #[test]
    fn diagnostics_report_unit_and_shape_errors() {
        let diags = analyze(
            "let d = 3.0 m\nlet t = 5.0 s\nlet v = d + t\n",
        );
        assert_eq!(diags.len(), 1);
        assert!(diags[0].message.contains("m"), "{:?}", diags);

        let diags = analyze(
            "let a = ones([2, 3])\nlet b = ones([4, 5])\nlet c = matmul(a, b)\n",
        );
        assert_eq!(diags.len(), 1);
        assert!(diags[0].message.contains("matmul"), "{:?}", diags);

        assert!(analyze("let x = 1 + 2\n").is_empty());
        assert!(!analyze("let = ").is_empty(), "syntax errors are reported");
    }

    #[test]
    fn completions_reflect_the_interpreted_prefix() {
        // line 7: everything above is visible
        let cs = completions_at(DOC, 7, "th");
        assert_eq!(cs, vec!["theta".to_string()]);
        let cs = completions_at(DOC, 7, "geom.ar");
        assert_eq!(cs, vec!["area".to_string()]);
        let cs = completions_at(DOC, 2, "wh");
        assert!(cs.contains(&"while".to_string()));
        // natives exist on an empty document
        let cs = completions_at("", 0, "mat");
        assert!(cs.contains(&"matmul".to_string()), "{cs:?}");
        // nothing for an unknown module
        assert!(completions_at(DOC, 7, "nope.x").is_empty());
    }

    #[test]
    fn hover_shows_values_signatures_and_modules() {
        // DOC line 7 is `let c = geom.circle(theta)`
        // cursor anywhere on `theta` (cols 20..24)
        let h = hover_at(DOC, 7, 22).unwrap();
        assert_eq!(h.contents, "num = 0.5");

        // cursor inside `geom.circle` resolves the dotted module path
        let h = hover_at(DOC, 7, 17).unwrap();
        assert!(h.contents.contains("<function circle(r)>"), "{}", h.contents);

        // a dotted path always resolves to its member: `geom.circle` shows
        // the function, at any cursor position within the path
        assert!(hover_at(DOC, 7, 11).unwrap().contents.contains("circle(r)"));

        // a standalone module binding lists its members
        let doc = "module m2 {
    let z = 1
}
m2";
        let h = hover_at(doc, 3, 1).unwrap();
        assert!(h.contents.starts_with("<module m2>"), "{}", h.contents);
        assert!(h.contents.contains("z"));

        let doc = "let t = ones([2, 2])
let s = t
s";
        let h = hover_at(doc, 2, 0).unwrap();
        assert_eq!(h.contents, "tensor [2, 2] (f64)");

        // a keyword is not a binding: no hover
        assert!(hover_at(DOC, 7, 2).is_none(), "`let` has no hover");
    }

    #[test]
    fn hover_after_an_error_line_still_uses_earlier_state() {
        // line 1 has a runtime error; line 1's `x` still resolves from line 0
        let doc = "let x = 3\nlet y = 1 / 0\nx";
        let h = hover_at(doc, 2, 0).unwrap();
        assert_eq!(h.contents, "num = 3");
    }
}
