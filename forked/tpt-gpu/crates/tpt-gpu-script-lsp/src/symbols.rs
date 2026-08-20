use crate::document::DocumentStore;
use tower_lsp::lsp_types::*;

#[allow(deprecated)]
pub fn provide_symbols(doc: &DocumentStore) -> Option<DocumentSymbolResponse> {
    let program = doc.ast.as_ref()?;
    let mut symbols = Vec::new();
    for item in &program.items {
        match item {
            tpt_gpu_script_core::ast::Item::Function(func) => {
                let start = Position {
                    line: func.span.line - 1,
                    character: func.span.col - 1,
                };
                let end_line = func.body.span.line - 1;
                let end_col = func.body.span.col;
                symbols.push(SymbolInformation {
                    name: func.name.clone(),
                    kind: SymbolKind::FUNCTION,
                    tags: None,
                    deprecated: None,
                    container_name: None,
                    location: Location {
                        uri: doc.uri.clone(),
                        range: Range {
                            start,
                            end: Position {
                                line: end_line,
                                character: end_col,
                            },
                        },
                    },
                });
            }
            tpt_gpu_script_core::ast::Item::TypeAlias(ty) => {
                let start = Position {
                    line: ty.span.line - 1,
                    character: ty.span.col - 1,
                };
                symbols.push(SymbolInformation {
                    name: ty.name.clone(),
                    kind: SymbolKind::CLASS,
                    tags: None,
                    deprecated: None,
                    container_name: None,
                    location: Location {
                        uri: doc.uri.clone(),
                        range: Range {
                            start,
                            end: Position {
                                line: ty.span.line - 1,
                                character: ty.span.col,
                            },
                        },
                    },
                });
            }
            tpt_gpu_script_core::ast::Item::Import(imp) => {
                let path = imp.path.join("::");
                let start = Position {
                    line: imp.span.line - 1,
                    character: imp.span.col - 1,
                };
                symbols.push(SymbolInformation {
                    name: format!("import {}", path),
                    kind: SymbolKind::NAMESPACE,
                    tags: None,
                    deprecated: None,
                    container_name: None,
                    location: Location {
                        uri: doc.uri.clone(),
                        range: Range {
                            start,
                            end: Position {
                                line: imp.span.line - 1,
                                character: imp.span.col,
                            },
                        },
                    },
                });
            }
        }
    }
    if symbols.is_empty() {
        None
    } else {
        Some(DocumentSymbolResponse::Flat(symbols))
    }
}
