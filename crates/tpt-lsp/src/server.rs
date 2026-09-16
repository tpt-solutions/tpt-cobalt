//! Thin tower-lsp adapter over the pure analysis layer in `lib.rs`.

use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

use crate::{analyze, completions_at, hover_at};

#[derive(Debug)]
struct TptLanguageServer {
    client: Client,
    doc: std::sync::Mutex<String>,
}

impl TptLanguageServer {
    fn new(client: Client) -> Self {
        TptLanguageServer {
            client,
            doc: std::sync::Mutex::new(String::new()),
        }
    }

    async fn publish_diagnostics(&self, uri: &Url, text: &str) {
        let diags = analyze(text)
            .into_iter()
            .map(|d| Diagnostic {
                range: Range {
                    start: Position::new(d.line as u32, d.col_start as u32),
                    end: Position::new(d.line as u32, d.col_end as u32),
                },
                severity: Some(match d.severity {
                    1 => DiagnosticSeverity::ERROR,
                    2 => DiagnosticSeverity::WARNING,
                    _ => DiagnosticSeverity::INFORMATION,
                }),
                source: Some(d.source.to_string()),
                message: d.message,
                ..Default::default()
            })
            .collect();
        self.client
            .publish_diagnostics(uri.clone(), diags, None)
            .await;
    }

    fn text(&self) -> String {
        self.doc.lock().unwrap().clone()
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for TptLanguageServer {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec![".".to_string()]),
                    resolve_provider: Some(false),
                    ..Default::default()
                }),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "tpt-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "tpt-lsp initialized")
            .await;
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        *self.doc.lock().unwrap() = params.text_document.text.clone();
        self.publish_diagnostics(&params.text_document.uri, &params.text_document.text)
            .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        if let Some(last) = params.content_changes.last() {
            *self.doc.lock().unwrap() = last.text.clone();
            self.publish_diagnostics(&params.text_document.uri, &last.text)
                .await;
        }
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let text = self.text();
        let pos = params.text_document_position_params.position;
        Ok(
            hover_at(&text, pos.line as usize, pos.character as usize).map(|h| Hover {
                contents: HoverContents::Markup(MarkupContent {
                    kind: MarkupKind::PlainText,
                    value: h.contents,
                }),
                range: Some(Range {
                    start: Position::new(h.line as u32, h.col_start as u32),
                    end: Position::new(h.line as u32, h.col_end as u32),
                }),
            }),
        )
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let text = self.text();
        let pos = params.text_document_position.position;
        let line_no = pos.line as usize;
        let prefix = text
            .lines()
            .nth(line_no)
            .map(|row| {
                let chars: Vec<char> = row.chars().collect();
                let col = pos.character as usize;
                let is_word = |c: char| c.is_alphanumeric() || c == '_' || c == '.';
                let mut start = col.min(chars.len());
                while start > 0 && is_word(chars[start - 1]) {
                    start -= 1;
                }
                chars[start..col.min(chars.len())]
                    .iter()
                    .collect::<String>()
            })
            .unwrap_or_default();
        let items = completions_at(&text, line_no, &prefix)
            .into_iter()
            .map(|label| CompletionItem {
                label,
                kind: Some(CompletionItemKind::VARIABLE),
                ..Default::default()
            })
            .collect();
        Ok(Some(CompletionResponse::Array(items)))
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }
}

/// Run the LSP over stdio (the `tpt-lsp` binary entry).
pub async fn run_stdio() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let (service, socket) = LspService::new(TptLanguageServer::new);
    Server::new(stdin, stdout, socket).serve(service).await;
}
