//! Notebook source analysis for the LSP server (feature `lsp`).
//!
//! Parses Rust source that builds a [`crate::Notebook`] and extracts the cells,
//! their declared dependencies, and the references between them. This is the
//! analysis the language server uses for autocomplete, go-to-definition,
//! hover and diagnostics — it works on the same `set` / `set_expr` /
//! `cell!` surface the runtime uses, so the LSP stays in sync with the actual
//! notebook semantics.

use proc_macro2::TokenTree;
use syn::visit::Visit;
use syn::{Block, Expr, ExprArray, ExprLit, ExprMethodCall, Lit};

/// A cell discovered in the source: its name, declared dependencies, the recorded
/// expression text (for derived cells), and the 1-based source line it starts on.
#[derive(Debug, Clone, PartialEq)]
pub struct CellSym {
    pub name: String,
    pub deps: Vec<String>,
    pub expr: Option<String>,
    pub line: usize,
}

/// The result of analysing one notebook source file.
#[derive(Debug, Clone, Default)]
pub struct NotebookAnalysis {
    pub cells: Vec<CellSym>,
    pub refs: Vec<(String, usize)>,
}

impl NotebookAnalysis {
    /// Parse `src` into a [`NotebookAnalysis`]. Parse failures yield an empty
    /// analysis (the LSP will simply offer no symbols rather than crashing).
    pub fn from_source(src: &str) -> Self {
        let mut collector = Collector::default();
        if let Ok(file) = syn::parse_file(src) {
            collector.visit_file(&file);
        } else if let Ok(block) = syn::parse_str::<Block>(src) {
            collector.visit_block(&block);
        }
        collector.into_analysis()
    }

    /// Cell names, suitable for completion.
    pub fn cell_names(&self) -> Vec<&str> {
        self.cells.iter().map(|c| c.name.as_str()).collect()
    }

    /// 0-based line of the definition of `name`, if known.
    pub fn definition_line(&self, name: &str) -> Option<usize> {
        self.cells
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.line.saturating_sub(1))
    }

    /// A human-readable description for hover, or `None` if unknown.
    pub fn hover(&self, name: &str) -> Option<String> {
        let cell = self.cells.iter().find(|c| c.name == name)?;
        let mut out = format!("cell `{}`", cell.name);
        if !cell.deps.is_empty() {
            out.push_str(&format!("\ndepends on: {}", cell.deps.join(", ")));
        }
        if let Some(e) = &cell.expr {
            out.push_str(&format!("\nexpr: {e}"));
        }
        Some(out)
    }

    /// Diagnostics: references to undefined cells, and cells whose declared
    /// dependency is undefined. Each entry is `(0-based line, message)`.
    pub fn diagnostics(&self) -> Vec<(usize, String)> {
        let defined: std::collections::BTreeSet<&str> =
            self.cells.iter().map(|c| c.name.as_str()).collect();
        let mut out = Vec::new();
        for (name, line) in &self.refs {
            if !defined.contains(name.as_str()) {
                out.push((
                    line.saturating_sub(1),
                    format!("reference to undefined cell `{name}`"),
                ));
            }
        }
        for cell in &self.cells {
            for dep in &cell.deps {
                if !defined.contains(dep.as_str()) {
                    out.push((
                        cell.line.saturating_sub(1),
                        format!("cell `{}` depends on undefined cell `{dep}`", cell.name),
                    ));
                }
            }
        }
        out
    }
}

#[derive(Default)]
struct Collector {
    cells: Vec<CellSym>,
    refs: Vec<(String, usize)>,
}

impl Collector {
    fn into_analysis(self) -> NotebookAnalysis {
        NotebookAnalysis {
            cells: self.cells,
            refs: self.refs,
        }
    }

    /// Extract the string-literal first argument of a method/array, if present.
    fn str_arg(args: &syn::punctuated::Punctuated<Expr, syn::Token![,]>, idx: usize) -> Option<String> {
        args.get(idx).and_then(lit_string)
    }
}

impl<'ast> Visit<'ast> for Collector {
    fn visit_expr_method_call(&mut self, m: &'ast ExprMethodCall) {
        let name = m.method.to_string();
        match name.as_str() {
            "set" | "set_expr" => {
                if let Some(cell) = Self::str_arg(&m.args, 0) {
                    let mut deps = Vec::new();
                    let mut expr = None;
                    if name == "set_expr" {
                        expr = Self::str_arg(&m.args, 1);
                        if let Some(dep_array) = m.args.get(2) {
                            deps = array_string_literals(unwrap_reference(dep_array));
                        }
                    }
                    self.cells.push(CellSym {
                        name: cell,
                        deps,
                        expr,
                        line: m.method.span().start().line,
                    });
                }
            }
            "get" | "try_get" => {
                if let Some(r) = Self::str_arg(&m.args, 0) {
                    self.refs.push((r, m.method.span().start().line));
                }
            }
            _ => {}
        }
        syn::visit::visit_expr_method_call(self, m);
    }

    fn visit_expr_macro(&mut self, m: &'ast syn::ExprMacro) {
        if m.mac.path.is_ident("cell") {
            if let Some((name, deps, line)) = extract_cell_macro(&m.mac.tokens) {
                self.cells.push(CellSym {
                    name,
                    deps,
                    expr: None,
                    line,
                });
            }
        }
        syn::visit::visit_expr_macro(self, m);
    }
}

/// If `e` is `&[..]`, peel the reference to reach the array.
fn unwrap_reference(e: &Expr) -> &Expr {
    if let Expr::Reference(r) = e {
        &r.expr
    } else {
        e
    }
}

/// Collect string literals from an array expression like `["a", "b"]`.
fn array_string_literals(e: &Expr) -> Vec<String> {
    let mut out = Vec::new();
    if let Expr::Array(ExprArray { elems, .. }) = e {
        for el in elems {
            if let Some(s) = lit_string(el) {
                out.push(s);
            }
        }
    }
    out
}

/// Get the inner string of a string-literal expression.
fn lit_string(e: &Expr) -> Option<String> {
    if let Expr::Lit(ExprLit {
        lit: Lit::Str(s), ..
    }) = e
    {
        Some(s.value())
    } else {
        None
    }
}

/// Extract the cell name (and any `[dep]` list) from a `cell!(..)` token stream.
///
/// Recognised forms:
/// * `cell!(nb, NAME = VALUE)`
/// * `cell!(nb, NAME = [A, B] "expr", closure)`
fn extract_cell_macro(tokens: &proc_macro2::TokenStream) -> Option<(String, Vec<String>, usize)> {
    let mut line = 0usize;
    let mut tokens: Vec<TokenTree> = tokens.clone().into_iter().collect();
    // Drop the leading `nb ,` (receiver + comma) if present.
    if matches!(tokens.first(), Some(TokenTree::Ident(_))) {
        // first ident is the receiver `nb`; drop it and a following comma
        tokens.remove(0);
        if matches!(tokens.first(), Some(TokenTree::Punct(p)) if p.as_char() == ',') {
            tokens.remove(0);
        }
    }
    let mut name = None;
    let mut deps = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        match &tokens[i] {
            TokenTree::Ident(id) => {
                // The cell name is the identifier immediately followed by `=`.
                if let Some(TokenTree::Punct(p)) = tokens.get(i + 1) {
                    if p.as_char() == '=' {
                        name = Some(id.to_string());
                    }
                }
            }
            TokenTree::Group(g) if g.delimiter() == proc_macro2::Delimiter::Bracket => {
                // A `[A, B]` dependency group preceding the `=`.
                for inner in g.stream() {
                    if let TokenTree::Ident(d) = inner {
                        deps.push(d.to_string());
                    }
                }
            }
            _ => {}
        }
        if let TokenTree::Ident(id) = &tokens[i] {
            line = id.span().start().line;
        }
        i += 1;
    }
    name.map(|n| (n, deps, line))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_set_and_set_expr_cells() {
        let src = r#"
            let mut nb = Notebook::new();
            nb.set("A", 1i64);
            nb.set_expr("B", "A + 1", &["A"], |nb| Ok(nb.try_get::<i64>("A")? + 1));
            let _ = nb.try_get::<i64>("A");
        "#;
        let a = NotebookAnalysis::from_source(src);
        assert_eq!(a.cells.len(), 2);
        assert_eq!(a.cells[0].name, "A");
        assert_eq!(a.cells[1].name, "B");
        assert_eq!(a.cells[1].deps, vec!["A".to_string()]);
        assert_eq!(a.cells[1].expr.as_deref(), Some("A + 1"));
        // The reference to "A" resolves, so no diagnostics.
        assert!(a.diagnostics().is_empty());
    }

    #[test]
    fn flags_undefined_reference() {
        let src = r#"
            nb.set("A", 1i64);
            let _ = nb.try_get::<i64>("Missing");
        "#;
        let a = NotebookAnalysis::from_source(src);
        let diags = a.diagnostics();
        assert_eq!(diags.len(), 1);
        assert!(diags[0].1.contains("Missing"));
    }

    #[test]
    fn extracts_cell_macro() {
        let src = r#"
            cell!(nb, A = 2i64);
            cell!(nb, B = [A] "A * 3", |nb| Ok(nb.try_get::<i64>("A")? * 3));
        "#;
        let a = NotebookAnalysis::from_source(src);
        let names = a.cell_names();
        assert!(names.contains(&"A"));
        assert!(names.contains(&"B"));
        let b = a.cells.iter().find(|c| c.name == "B").unwrap();
        assert_eq!(b.deps, vec!["A".to_string()]);
    }
}
