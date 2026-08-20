use tower_lsp::lsp_types::*;
use tpt_gpu_script_core::{compile_str, tokenize, Program, Token};

use crate::completion::CompletionDatabase;

/// In-memory representation of an open TPT Script document.
pub struct DocumentStore {
    pub uri: Url,
    pub source: String,
    pub version: i32,
    pub tokens: Vec<Token>,
    pub ast: Option<Program>,
}

impl DocumentStore {
    pub fn new(uri: Url, text: String, version: i32) -> Self {
        let tokens = tokenize(&text).unwrap_or_default();
        let ast = compile_str(&text).ok();
        Self {
            uri,
            source: text,
            version,
            tokens,
            ast,
        }
    }

    pub fn update_content(&mut self, version: i32, changes: Vec<TextDocumentContentChangeEvent>) {
        if let Some(last) = changes.last() {
            self.source = last.text.clone();
        }
        self.version = version;
        self.tokens = tokenize(&self.source).unwrap_or_default();
        self.ast = compile_str(&self.source).ok();
    }

    pub fn completion_db(&self) -> CompletionDatabase {
        CompletionDatabase::from_tokens(&self.tokens)
    }

    /// Return the token whose byte range contains `pos`, if any.
    pub fn token_at(&self, pos: Position) -> Option<&Token> {
        let offset = position_to_offset(&self.source, pos);
        self.tokens
            .iter()
            .find(|t| t.span.start <= offset && offset < t.span.end)
    }
}

/// Format the whole document, replacing it with a single full-range edit.
pub fn format_document(doc: &DocumentStore, _options: &FormattingOptions) -> Option<Vec<TextEdit>> {
    let formatted = tpt_gpu_script_format::format(&doc.source).ok()?;
    if formatted == doc.source {
        return Some(Vec::new());
    }
    let end = offset_to_position(&doc.source, doc.source.len());
    Some(vec![TextEdit {
        range: Range {
            start: Position {
                line: 0,
                character: 0,
            },
            end,
        },
        new_text: formatted,
    }])
}

fn position_to_offset(source: &str, pos: Position) -> usize {
    let mut offset = 0;
    for (i, line) in source.lines().enumerate() {
        if i as u32 == pos.line {
            return offset + (pos.character as usize).min(line.len());
        }
        offset += line.len() + 1;
    }
    source.len()
}

fn offset_to_position(source: &str, offset: usize) -> Position {
    let mut line = 0u32;
    let mut col = 0u32;
    for ch in source.chars().take(offset) {
        if ch == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }
    Position {
        line,
        character: col,
    }
}
