//! # Notebook kernel (Phase 6, first slice)
//!
//! The notebook engine over the eager interpreter: stateful cells with
//! Jupyter-style execution counters, **rich display** (MIME-typed bundles —
//! tensors render as HTML tables alongside `text/plain`), tab **autocomplete**
//! aware of globals, keywords, and module members, and error isolation (a
//! failing cell never kills the kernel).
//!
//! Divergence from the roadmap text, documented: the **Jupyter wire protocol**
//! (ZeroMQ shell/IOPub channels) is future work — this slice is the in-process
//! kernel those transports plug into. Everything else in the line item (cell
//! state, rich display, autocomplete) is here.

use crate::interp::{Interpreter, InterpreterError};
use crate::value::Value;

/// One notebook cell: source plus its execution state.
#[derive(Debug, Clone)]
pub struct Cell {
    pub source: String,
    /// Execution count of the last successful-or-failed run (`None` = never
    /// run). Counts are kernel-global and monotonic, like Jupyter's `[n]`.
    pub exec_count: Option<usize>,
    /// Output of the last run, if any.
    pub last_output: Option<CellOutput>,
}

/// A MIME-typed rich-display payload.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplayData {
    pub mime: String,
    pub data: String,
}

/// Everything one cell run produced.
#[derive(Debug, Clone)]
pub struct CellOutput {
    pub exec_count: usize,
    /// Captured `print` output.
    pub stdout: String,
    /// Value of the last expression statement (the `Out[n]` value).
    pub result: Option<Value>,
    /// Rich renderings of the result (`text/plain` always present when
    /// `result` is `Some`).
    pub display: Vec<DisplayData>,
    /// The cell's error, if it failed (kernel keeps running).
    pub error: Option<InterpreterError>,
}

impl CellOutput {
    /// Convenience lookup of one MIME type.
    pub fn get(&self, mime: &str) -> Option<&str> {
        self.display
            .iter()
            .find(|d| d.mime == mime)
            .map(|d| d.data.as_str())
    }
}

/// The in-process notebook kernel.
#[derive(Default)]
pub struct Notebook {
    interp: Interpreter,
    cells: Vec<Cell>,
    exec_counter: usize,
}

impl Notebook {
    pub fn new() -> Self {
        Notebook::default()
    }

    /// The underlying interpreter (seed globals / models before running).
    pub fn interpreter(&mut self) -> &mut Interpreter {
        &mut self.interp
    }

    /// Append a cell; returns its index.
    pub fn add_cell(&mut self, source: impl Into<String>) -> usize {
        self.cells.push(Cell {
            source: source.into(),
            exec_count: None,
            last_output: None,
        });
        self.cells.len() - 1
    }

    /// Number of cells.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Whether the notebook has no cells.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Run cell `idx`, updating its state and the kernel-wide counter.
    pub fn run_cell(&mut self, idx: usize) -> &CellOutput {
        assert!(idx < self.cells.len(), "cell index out of range");
        let src = self.cells[idx].source.clone();
        // `run` clears the capture buffer itself, so the post-run snapshot
        // is exactly this cell's stdout
        let run = self.interp.run(&src);
        let stdout = self.interp.output().to_string();
        self.exec_counter += 1;
        let exec_count = self.exec_counter;
        let output = match run {
            Ok(result) => {
                let mut display = Vec::new();
                if let Some(v) = &result {
                    display.push(DisplayData {
                        mime: "text/plain".into(),
                        data: v.to_string(),
                    });
                    if let Some(html) = display_html(v) {
                        display.push(DisplayData {
                            mime: "text/html".into(),
                            data: html,
                        });
                    }
                }
                CellOutput {
                    exec_count,
                    stdout,
                    result,
                    display,
                    error: None,
                }
            }
            Err(e) => CellOutput {
                exec_count,
                stdout,
                result: None,
                display: Vec::new(),
                error: Some(e),
            },
        };
        let cell = &mut self.cells[idx];
        cell.exec_count = Some(exec_count);
        cell.last_output = Some(output);
        cell.last_output.as_ref().unwrap()
    }

    /// Tab completions for `prefix` at the kernel's current state:
    /// globals + natives + language keywords, or — when the prefix is dotted
    /// (`geom.ar`) — the members of the named module. Sorted.
    pub fn completions(&self, prefix: &str) -> Vec<String> {
        let (base, member) = match prefix.rsplit_once('.') {
            Some((b, m)) => (b, m),
            None => ("", prefix),
        };
        let mut names: Vec<String> = if base.is_empty() {
            let mut n: Vec<String> = self.interp.names();
            n.extend(KEYWORDS.iter().map(|s| s.to_string()));
            n
        } else {
            match self.interp.get(base) {
                Some(Value::Module(m)) => m.member_names(),
                _ => Vec::new(),
            }
        };
        names.retain(|n| n.starts_with(member));
        names.sort();
        names.dedup();
        names
    }

    /// Access a cell's last output.
    pub fn cell_output(&self, idx: usize) -> Option<&CellOutput> {
        self.cells[idx].last_output.as_ref()
    }

    /// Execution count of a cell (Jupyter's `[n]`), if it has run.
    pub fn cell_exec_count(&self, idx: usize) -> Option<usize> {
        self.cells[idx].exec_count
    }
}

const KEYWORDS: &[&str] = &[
    "let", "def", "fn", "if", "else", "while", "return", "print", "assert",
    "module", "true", "false", "nil", "None", "not",
];

/// HTML rich display for values where it adds something over text: small
/// tensors become tables, dicts become key/value tables.
fn display_html(v: &Value) -> Option<String> {
    match v {
        Value::Tensor(t) if t.numel() <= 64 => {
            let data = t.to_vec::<f64>().ok()?;
            let shape = t.shape();
            let mut html = String::from("<table class=\"tpt-tensor\">");
            html.push_str(&format!(
                "<caption>Tensor{:?}</caption>",
                shape
            ));
            if shape.len() == 1 {
                html.push_str("<tr>");
                for x in &data {
                    html.push_str(&format!("<td>{x}</td>"));
                }
                html.push_str("</tr>");
            } else if shape.len() == 2 {
                let (rows, cols) = (shape[0], shape[1]);
                for r in 0..rows {
                    html.push_str("<tr>");
                    for c in 0..cols {
                        html.push_str(&format!("<td>{}</td>", data[r * cols + c]));
                    }
                    html.push_str("</tr>");
                }
            } else {
                for x in &data {
                    html.push_str(&format!("<tr><td>{x}</td></tr>"));
                }
            }
            html.push_str("</table>");
            Some(html)
        }
        Value::Dict(d) => {
            let map = d.lock().unwrap();
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut html = String::from("<table class=\"tpt-dict\">");
            for k in keys {
                let plain = map[k].to_string();
                // keep cell payloads text-safe
                let plain = plain.replace('<', "&lt;").replace('>', "&gt;");
                html.push_str(&format!("<tr><th>{k}</th><td>{plain}</td></tr>"));
            }
            html.push_str("</table>");
            Some(html)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;

    #[test]
    fn cells_share_state_and_count_executions() {
        let mut nb = Notebook::new();
        let c0 = nb.add_cell("let x = 6");
        let c1 = nb.add_cell("x * 7");
        let out0 = nb.run_cell(c0);
        assert_eq!(out0.exec_count, 1);
        let out1 = nb.run_cell(c1);
        assert_eq!(out1.exec_count, 2);
        assert_eq!(out1.result, Some(Value::Num(42.0)));
        assert_eq!(nb.cell_exec_count(c1), Some(2));
    }

    #[test]
    fn a_failing_cell_does_not_kill_the_kernel() {
        let mut nb = Notebook::new();
        let bad = nb.add_cell("1 / 0");
        let out = nb.run_cell(bad);
        assert!(out.error.is_some());
        assert_eq!(out.error.as_ref().unwrap().kind, "ZeroDivisionError");
        // the counter still advanced, and the kernel keeps working
        let good = nb.add_cell("2 + 3");
        let out = nb.run_cell(good);
        assert_eq!(out.result, Some(Value::Num(5.0)));
        assert_eq!(out.exec_count, 2);
    }

    #[test]
    fn tensor_results_get_rich_html_display() {
        let mut nb = Notebook::new();
        let c = nb.add_cell("ones([2, 3])");
        let out = nb.run_cell(c);
        assert!(out.result.is_some());
        let plain = out.get("text/plain").unwrap();
        assert!(plain.contains("Tensor"), "{plain}");
        let html = out.get("text/html").unwrap();
        assert!(html.contains("<table"), "{html}");
        assert!(html.contains("<caption>Tensor[2, 3]</caption>"));
        // all six elements present
        assert!(html.contains("1"));
        assert_eq!(html.matches("<td>").count(), 6);
    }

    #[test]
    fn print_output_is_captured_per_cell() {
        let mut nb = Notebook::new();
        let a = nb.add_cell("print \"hello notebook\"\n7");
        let b = nb.add_cell("8");
        let out_a = nb.run_cell(a);
        assert_eq!(out_a.stdout, "hello notebook\n");
        let out_b = nb.run_cell(b);
        assert_eq!(out_b.stdout, "", "cells own their stdout slice");
    }

    #[test]
    fn re_running_a_cell_updates_its_count() {
        let mut nb = Notebook::new();
        let c = nb.add_cell("1 + 1");
        assert_eq!(nb.run_cell(c).exec_count, 1);
        assert_eq!(nb.run_cell(c).exec_count, 2);
        assert_eq!(nb.cell_exec_count(c), Some(2));
    }

    #[test]
    fn completions_cover_globals_keywords_and_module_members() {
        let mut nb = Notebook::new();
        nb.add_cell("let theta = 0.5\nmodule geom { let area = 1 }");
        nb.run_cell(0);
        let cs = nb.completions("th");
        assert!(cs.contains(&"theta".to_string()), "{cs:?}");
        let cs = nb.completions("mat");
        assert!(cs.contains(&"matmul".to_string()), "{cs:?}");
        let cs = nb.completions("wh");
        assert!(cs.contains(&"while".to_string()), "{cs:?}");
        // dotted prefix completes into the module
        let cs = nb.completions("geom.ar");
        assert_eq!(cs, vec!["area".to_string()]);
        // unknown base module completes nothing
        assert!(nb.completions("nope.x").is_empty());
    }
}
