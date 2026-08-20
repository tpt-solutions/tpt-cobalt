//! The `.tpt` runner: panic-to-traceback bridging plus a tiny built-in
//! interpreter used by [`run_script`].
//!
//! ## Scope (be honest)
//!
//! The production `tpt script foo.tpt` binary wraps the source in a generated
//! `fn main` (adding `use tpt_script::prelude::*;` and implicit `?`) and hands
//! it to `rustc`. **This crate does not shell out to a compiler.** Instead:
//!
//! * [`run`] is the real runtime half: it executes any closure, catches
//!   unwinding panics, and converts them into a [`ScriptError`] traceback that
//!   carries the panic message and source location. That is what makes
//!   "exceptions" from [`crate::table`] chaining recoverable.
//! * [`run_script`] additionally *executes source text* — but only the small,
//!   documented statement language below, not arbitrary Rust:
//!
//! ```text
//! # comment
//! let x = 2 + 3 * 4       # f64 arithmetic over let-bound variables
//! print x                 # or: print "literal text"
//! assert x > 10           # failing asserts panic -> caught -> Err
//! panic "boom"            # explicit raise
//! ```
//!
//! Anything else is reported as a `SyntaxError` with the offending line, so a
//! caller never gets a silent no-op.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Mutex, OnceLock};

use crate::error::{Res, ScriptError};

thread_local! {
    static LAST_LOC: RefCell<Option<String>> = const { RefCell::new(None) };
}

thread_local! {
    /// When `Some`, `print` output is redirected here instead of stdout. Used by
    /// the in-browser "Try TPT" playground, where `std::io::set_print` is not
    /// available on `wasm32-unknown-unknown`.
    static CAPTURE: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Run `f` while capturing any `print` output into a returned `String`.
///
/// On native builds the captured text is also written to stdout, so the CLI
/// keeps its existing behaviour; on wasm the returned string is the only
/// observable output.
pub fn capture<F, T>(f: F) -> (T, String)
where
    F: FnOnce() -> T,
{
    CAPTURE.with(|c| *c.borrow_mut() = Some(String::new()));
    let result = f();
    let out = CAPTURE.with(|c| c.borrow_mut().take()).unwrap_or_default();
    #[cfg(not(target_arch = "wasm32"))]
    if !out.is_empty() {
        print!("{out}");
    }
    (result, out)
}

/// Emit one `print` line: redirect to the capture sink when active, else stdout.
fn emit_print(value: String) {
    let redirected = CAPTURE.with(|c| {
        if let Some(buf) = c.borrow_mut().as_mut() {
            buf.push_str(&value);
            buf.push('\n');
            true
        } else {
            false
        }
    });
    if !redirected {
        println!("{value}");
    }
}

fn hook_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Run a script body, converting panics ("exceptions") into [`ScriptError`].
///
/// Implicit error handling: script helpers raise instead of returning
/// `Result`, and any `?` inside `f` still short-circuits normally because the
/// closure itself returns [`Res`].
pub fn run<T>(f: impl FnOnce() -> Res<T>) -> Res<T> {
    let guard = hook_lock().lock().unwrap_or_else(|e| e.into_inner());
    let prev = panic::take_hook();
    panic::set_hook(Box::new(|info| {
        let loc = info.location().map(|l| {
            format!(
                "File \"{}\", line {}, col {}",
                l.file(),
                l.line(),
                l.column()
            )
        });
        LAST_LOC.with(|c| *c.borrow_mut() = loc);
    }));
    let caught = panic::catch_unwind(AssertUnwindSafe(f));
    panic::set_hook(prev);
    drop(guard);

    match caught {
        Ok(r) => r,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            let loc = LAST_LOC.with(|c| c.borrow_mut().take());
            // A raised ScriptError already renders its own traceback.
            let (kind, msg) = match msg.strip_prefix("Traceback (most recent call last):") {
                Some(rest) => (
                    "ScriptError",
                    rest.lines().last().unwrap_or(&msg).trim().to_string(),
                ),
                None => ("PanicError", msg),
            };
            let mut e = ScriptError::new(kind, msg);
            if let Some(l) = loc {
                e = e.frame(l);
            }
            Err(e)
        }
    }
}

/// Execute `.tpt` source text (see the module docs for the supported subset).
///
/// Returns a rendered traceback on failure instead of aborting the process.
pub fn run_script(src: &str) -> Result<(), String> {
    let owned = src.to_string();
    run(move || exec(&owned)).map_err(|e| e.traceback())
}

/// Same as [`run_script`] but keeps the structured error.
pub fn run_script_err(src: &str) -> Res<()> {
    let owned = src.to_string();
    run(move || exec(&owned))
}

fn exec(src: &str) -> Res<()> {
    let mut vars: BTreeMap<String, f64> = BTreeMap::new();
    for (i, raw) in src.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let frame = format!("File \"<tpt script>\", line {}\n    {}", i + 1, line);
        step(line, &mut vars).map_err(|e| e.frame(frame))?;
    }
    Ok(())
}

fn step(line: &str, vars: &mut BTreeMap<String, f64>) -> Res<()> {
    if let Some(rest) = line.strip_prefix("let ") {
        let (name, expr) = rest.split_once('=').ok_or_else(syntax(line))?;
        let name = name.trim();
        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(ScriptError::new(
                "SyntaxError",
                format!("invalid binding name '{name}'"),
            ));
        }
        let v = eval(expr, vars)?;
        vars.insert(name.to_string(), v);
    } else if let Some(rest) = line.strip_prefix("print ") {
        match literal(rest.trim()) {
            Some(s) => emit_print(s.to_string()),
            None => emit_print(format!("{}", eval(rest, vars)?)),
        }
    } else if let Some(rest) = line.strip_prefix("assert ") {
        let (l, op, r) = split_cmp(rest).ok_or_else(syntax(line))?;
        let (a, b) = (eval(l, vars)?, eval(r, vars)?);
        let ok = match op {
            ">" => a > b,
            "<" => a < b,
            ">=" => a >= b,
            "<=" => a <= b,
            "==" => (a - b).abs() < 1e-12,
            _ => (a - b).abs() >= 1e-12,
        };
        // A failing assert is a genuine panic, caught by `run`.
        assert!(ok, "assertion failed: {} ({a} {op} {b})", rest.trim());
    } else if let Some(rest) = line
        .strip_prefix("panic!(")
        .and_then(|r| r.strip_suffix(')'))
        .or_else(|| line.strip_prefix("panic "))
    {
        let msg = literal(rest.trim()).unwrap_or_else(|| rest.trim().to_string());
        panic!("{msg}");
    } else {
        return Err(syntax(line)());
    }
    Ok(())
}

fn syntax(line: &str) -> impl Fn() -> ScriptError + '_ {
    move || {
        ScriptError::new(
            "SyntaxError",
            format!("cannot parse statement: `{}`", line.trim()),
        )
    }
}

fn literal(s: &str) -> Option<String> {
    s.strip_prefix('"')
        .and_then(|r| r.strip_suffix('"'))
        .map(|r| r.to_string())
}

fn split_cmp(s: &str) -> Option<(&str, &'static str, &str)> {
    for op in [">=", "<=", "==", "!=", ">", "<"] {
        if let Some(p) = s.find(op) {
            return Some((&s[..p], op, &s[p + op.len()..]));
        }
    }
    None
}

// --- micro expression evaluator: + - * / ( ) idents numbers ------------------

fn tokenize(s: &str) -> Res<Vec<String>> {
    let mut out = Vec::new();
    let cs: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || c == '.' {
            let start = i;
            while i < cs.len() && (cs[i].is_ascii_digit() || cs[i] == '.') {
                i += 1;
            }
            out.push(cs[start..i].iter().collect());
        } else if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < cs.len() && (cs[i].is_alphanumeric() || cs[i] == '_') {
                i += 1;
            }
            out.push(cs[start..i].iter().collect());
        } else if "+-*/()".contains(c) {
            out.push(c.to_string());
            i += 1;
        } else {
            return Err(ScriptError::new(
                "SyntaxError",
                format!("unexpected character '{c}'"),
            ));
        }
    }
    Ok(out)
}

fn eval(expr: &str, vars: &BTreeMap<String, f64>) -> Res<f64> {
    let toks = tokenize(expr)?;
    if toks.is_empty() {
        return Err(ScriptError::new("SyntaxError", "empty expression"));
    }
    let mut p = 0;
    let v = sum(&toks, &mut p, vars)?;
    if p != toks.len() {
        return Err(ScriptError::new(
            "SyntaxError",
            format!("trailing tokens in `{}`", expr.trim()),
        ));
    }
    Ok(v)
}

fn sum(t: &[String], p: &mut usize, v: &BTreeMap<String, f64>) -> Res<f64> {
    let mut acc = product(t, p, v)?;
    while *p < t.len() && (t[*p] == "+" || t[*p] == "-") {
        let op = t[*p].clone();
        *p += 1;
        let rhs = product(t, p, v)?;
        acc = if op == "+" { acc + rhs } else { acc - rhs };
    }
    Ok(acc)
}

fn product(t: &[String], p: &mut usize, v: &BTreeMap<String, f64>) -> Res<f64> {
    let mut acc = atom(t, p, v)?;
    while *p < t.len() && (t[*p] == "*" || t[*p] == "/") {
        let op = t[*p].clone();
        *p += 1;
        let rhs = atom(t, p, v)?;
        acc = if op == "*" { acc * rhs } else { acc / rhs };
    }
    Ok(acc)
}

fn atom(t: &[String], p: &mut usize, v: &BTreeMap<String, f64>) -> Res<f64> {
    let tok = t
        .get(*p)
        .ok_or_else(|| ScriptError::new("SyntaxError", "unexpected end of expression"))?
        .clone();
    *p += 1;
    if tok == "-" {
        return Ok(-atom(t, p, v)?);
    }
    if tok == "(" {
        let inner = sum(t, p, v)?;
        if t.get(*p).map(|s| s.as_str()) != Some(")") {
            return Err(ScriptError::new("SyntaxError", "missing ')'"));
        }
        *p += 1;
        return Ok(inner);
    }
    if let Ok(n) = tok.parse::<f64>() {
        return Ok(n);
    }
    v.get(&tok)
        .copied()
        .ok_or_else(|| ScriptError::new("NameError", format!("name '{tok}' is not defined")))
}
