//! A minimal, dependency-light Language Server Protocol server for tpt-lab
//! notebooks (feature `lsp`).
//!
//! Speaks JSON-RPC over stdio (the LSP wire format) directly, so it needs no
//! external LSP framework. It keeps a [`crate::analysis::NotebookAnalysis`] for
//! every open document and answers `completion`, `definition`, `hover` and
//! publishes `diagnostics` whenever a document changes.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};

use serde_json::{json, Value};

use crate::analysis::NotebookAnalysis;

/// Entry point for the `tpt-lab-lsp` binary.
pub fn run_lsp() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut out = stdout.lock();

    let mut docs: HashMap<String, (String, NotebookAnalysis)> = HashMap::new();

    loop {
        let Some(msg) = read_message(&mut reader) else {
            return;
        };
        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let id = msg.get("id").cloned();

        if id.is_some() {
            let result = handle_request(method, &msg, &docs);
            let resp = json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": result,
            });
            if write_message(&mut out, &resp).is_err() {
                return;
            }
        } else {
            let exit = handle_notification(method, &msg, &mut docs, &mut out);
            if exit {
                return;
            }
        }
    }
}

/// Read one `Content-Length` framed JSON-RPC message from `reader`.
fn read_message<R: BufRead>(reader: &mut R) -> Option<Value> {
    let mut content_length: Option<usize> = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 {
            return None;
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some(rest) = line.strip_prefix("Content-Length:") {
            content_length = rest.trim().parse().ok();
        }
    }
    let len = content_length?;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).ok()?;
    serde_json::from_slice(&buf).ok()
}

/// Write one JSON-RPC message, framed with a `Content-Length` header.
fn write_message<W: Write>(out: &mut W, value: &Value) -> std::io::Result<()> {
    let bytes = serde_json::to_vec(value)?;
    write!(out, "Content-Length: {}\r\n\r\n", bytes.len())?;
    out.write_all(&bytes)?;
    out.flush()
}

/// Handle a request (has an `id`). Returns the `result` payload.
fn handle_request(method: &str, msg: &Value, docs: &HashMap<String, (String, NotebookAnalysis)>) -> Value {
    match method {
        "initialize" => json!({
            "capabilities": {
                "textDocumentSync": 1,
                "completionProvider": { "triggerCharacters": ["."] },
                "definitionProvider": true,
                "hoverProvider": true
            }
        }),
        "textDocument/completion" => {
            let uri = msg["params"]["textDocument"]["uri"].as_str().unwrap_or("");
            let empty = NotebookAnalysis::default();
            let analysis = docs.get(uri).map(|(_, a)| a).unwrap_or(&empty);
            let items: Vec<Value> = analysis
                .cell_names()
                .into_iter()
                .map(|name| {
                    json!({
                        "label": name,
                        "kind": 6, // Variable
                        "detail": "tpt-lab cell"
                    })
                })
                .collect();
            json!({ "isIncomplete": false, "items": items })
        }
        "textDocument/definition" => {
            if let Some((line, character)) = position_of(msg) {
                let uri = msg["params"]["textDocument"]["uri"].as_str().unwrap_or("");
                if let Some((text, analysis)) = docs.get(uri) {
                    if let Some(word) = word_at(&line_text(text, line), character) {
                        if let Some(def_line) = analysis.definition_line(&word) {
                            return json!({
                                "uri": uri,
                                "range": {
                                    "start": { "line": def_line, "character": 0 },
                                    "end": { "line": def_line, "character": 0 }
                                }
                            });
                        }
                    }
                }
            }
            Value::Null
        }
        "textDocument/hover" => {
            if let Some((line, character)) = position_of(msg) {
                let uri = msg["params"]["textDocument"]["uri"].as_str().unwrap_or("");
                if let Some((text, analysis)) = docs.get(uri) {
                    if let Some(word) = word_at(&line_text(text, line), character) {
                        if let Some(info) = analysis.hover(&word) {
                            return json!({ "contents": info });
                        }
                    }
                }
            }
            Value::Null
        }
        _ => Value::Null,
    }
}

/// Handle a notification (no `id`). Returns `true` if the process should exit.
fn handle_notification<W: Write>(
    method: &str,
    msg: &Value,
    docs: &mut HashMap<String, (String, NotebookAnalysis)>,
    out: &mut W,
) -> bool {
    match method {
        "textDocument/didOpen" | "textDocument/didChange" => {
            let uri = msg["params"]["textDocument"]["uri"].as_str().unwrap_or("").to_string();
            let text = if method == "textDocument/didOpen" {
                msg["params"]["textDocument"]["text"].as_str().unwrap_or("").to_string()
            } else {
                // didChange: assume full-sync, last content change holds the new text.
                msg["params"]["contentChanges"]
                    .as_array()
                    .and_then(|c| c.last())
                    .and_then(|c| c["text"].as_str())
                    .unwrap_or("")
                    .to_string()
            };
            let analysis = NotebookAnalysis::from_source(&text);
            publish_diagnostics(out, &uri, &analysis);
            docs.insert(uri, (text, analysis));
        }
        "exit" => return true,
        _ => {}
    }
    false
}

/// Emit `textDocument/publishDiagnostics` for the current analysis.
fn publish_diagnostics<W: Write>(out: &mut W, uri: &str, analysis: &NotebookAnalysis) {
    let diagnostics: Vec<Value> = analysis
        .diagnostics()
        .into_iter()
        .map(|(line, message)| {
            json!({
                "range": {
                    "start": { "line": line, "character": 0 },
                    "end": { "line": line, "character": 0 }
                },
                "severity": 1,
                "source": "tpt-lab",
                "message": message
            })
        })
        .collect();
    let note = json!({
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": { "uri": uri, "diagnostics": diagnostics }
    });
    let _ = write_message(out, &note);
}

/// Extract the `(line, character)` from a request's `params.position`.
fn position_of(msg: &Value) -> Option<(usize, usize)> {
    let pos = &msg["params"]["position"];
    let line = pos["line"].as_u64()? as usize;
    let character = pos["character"].as_u64()? as usize;
    Some((line, character))
}

/// The source line at `line` (0-based), or empty string.
fn line_text(src: &str, line: usize) -> String {
    src.lines().nth(line).unwrap_or("").to_string()
}

/// The identifier in `line` spanning `col` (0-based), if any.
fn word_at(line: &str, col: usize) -> Option<String> {
    let bytes = line.as_bytes();
    if col >= bytes.len() {
        return None;
    }
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    if !is_ident(bytes[col]) {
        return None;
    }
    let mut start = col;
    let mut end = col;
    while start > 0 && is_ident(bytes[start - 1]) {
        start -= 1;
    }
    while end < bytes.len() && is_ident(bytes[end]) {
        end += 1;
    }
    Some(line[start..end].to_string())
}
