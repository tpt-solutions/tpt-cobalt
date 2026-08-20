use tower_lsp::lsp_types::*;
use tpt_gpu_script_core::TokenKind;

use crate::document::DocumentStore;

/// Go-to-definition: resolve function/type references to their definitions.
pub fn goto_definition(doc: &DocumentStore, pos: Position) -> Option<GotoDefinitionResponse> {
    let program = doc.ast.as_ref()?;
    let offset = position_to_offset(&doc.source, pos);

    // Find the token at the cursor position
    let token = doc
        .tokens
        .iter()
        .find(|t| t.span.start <= offset && offset < t.span.end)?;

    let name = match &token.kind {
        TokenKind::Ident(n) => n.as_str(),
        _ => return None,
    };

    // Search for a matching function definition
    for item in &program.items {
        if let tpt_gpu_script_core::ast::Item::Function(func) = item {
            if func.name == name {
                return Some(GotoDefinitionResponse::Scalar(Location {
                    uri: doc.uri.clone(),
                    range: Range {
                        start: Position {
                            line: func.span.line - 1,
                            character: func.span.col - 1,
                        },
                        end: Position {
                            line: func.span.line - 1,
                            character: func.span.col + name.len() as u32,
                        },
                    },
                }));
            }
        }
        if let tpt_gpu_script_core::ast::Item::TypeAlias(ty) = item {
            if ty.name == name {
                return Some(GotoDefinitionResponse::Scalar(Location {
                    uri: doc.uri.clone(),
                    range: Range {
                        start: Position {
                            line: ty.span.line - 1,
                            character: ty.span.col - 1,
                        },
                        end: Position {
                            line: ty.span.line - 1,
                            character: ty.span.col + name.len() as u32,
                        },
                    },
                }));
            }
        }
    }

    None
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
