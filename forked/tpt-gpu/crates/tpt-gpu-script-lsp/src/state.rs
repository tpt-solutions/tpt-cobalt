use crate::document::DocumentStore;
use dashmap::DashMap;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, ClientSocket, LanguageServer, LspService};

pub struct TptLspService {
    client: Client,
    documents: DashMap<Url, DocumentStore>,
}

impl TptLspService {
    pub fn new() -> (LspService<Self>, ClientSocket) {
        LspService::new(|client| TptLspService {
            client,
            documents: DashMap::new(),
        })
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for TptLspService {
    async fn initialize(&self, _params: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Options(
                    TextDocumentSyncOptions {
                        open_close: Some(true),
                        change: Some(TextDocumentSyncKind::FULL),
                        ..Default::default()
                    },
                )),
                completion_provider: Some(CompletionOptions {
                    resolve_provider: Some(true),
                    trigger_characters: Some(vec![
                        String::from("."),
                        String::from("@"),
                        String::from(":"),
                    ]),
                    ..Default::default()
                }),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                document_symbol_provider: Some(OneOf::Left(true)),
                definition_provider: Some(OneOf::Left(true)),
                document_formatting_provider: Some(OneOf::Left(true)),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: String::from("tpt-gpu-lsp"),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn initialized(&self, _params: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "tpt-gpu-lsp initialized")
            .await;
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        let store = DocumentStore::new(
            params.text_document.uri,
            params.text_document.text,
            params.text_document.version,
        );
        self.documents.insert(uri.clone(), store);
        self.publish_diagnostics(&uri).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        if let Some(mut entry) = self.documents.get_mut(&uri) {
            entry.update_content(params.text_document.version, params.content_changes);
        }
        self.publish_diagnostics(&uri).await;
    }

    async fn did_save(&self, params: DidSaveTextDocumentParams) {
        self.publish_diagnostics(&params.text_document.uri).await;
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        self.documents.remove(&params.text_document.uri);
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = params.text_document_position.text_document.uri;
        if let Some(entry) = self.documents.get(&uri) {
            Ok(crate::completion::provide_completions(
                &entry,
                params.text_document_position.position,
            ))
        } else {
            Ok(None)
        }
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri;
        if let Some(entry) = self.documents.get(&uri) {
            Ok(crate::hover::provide_hover(
                &entry,
                params.text_document_position_params.position,
            ))
        } else {
            Ok(None)
        }
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = params.text_document.uri;
        if let Some(entry) = self.documents.get(&uri) {
            Ok(crate::symbols::provide_symbols(&entry))
        } else {
            Ok(None)
        }
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = params.text_document_position_params.text_document.uri;
        if let Some(entry) = self.documents.get(&uri) {
            Ok(crate::definition::goto_definition(
                &entry,
                params.text_document_position_params.position,
            ))
        } else {
            Ok(None)
        }
    }

    async fn formatting(&self, params: DocumentFormattingParams) -> Result<Option<Vec<TextEdit>>> {
        let uri = params.text_document.uri;
        if let Some(entry) = self.documents.get(&uri) {
            Ok(crate::document::format_document(&entry, &params.options))
        } else {
            Ok(None)
        }
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }
}

impl TptLspService {
    async fn publish_diagnostics(&self, uri: &Url) {
        if let Some(entry) = self.documents.get(uri) {
            let diagnostics = crate::diagnostics::compute_diagnostics(&entry);
            self.client
                .publish_diagnostics(uri.clone(), diagnostics, None)
                .await;
        }
    }
}
