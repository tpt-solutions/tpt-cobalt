use tower_lsp::lsp_types::*;
use tpt_gpu_script_core::TokenKind;

/// A database of completable items extracted from the current document.
pub struct CompletionDatabase {
    pub keywords: Vec<String>,
    pub builtins: Vec<String>,
    pub annotations: Vec<String>,
    pub identifiers: Vec<String>,
}

impl CompletionDatabase {
    pub fn from_tokens(tokens: &[tpt_gpu_script_core::Token]) -> Self {
        let mut identifiers = Vec::new();
        for token in tokens {
            if let TokenKind::Ident(name) = &token.kind {
                if !identifiers.contains(name) {
                    identifiers.push(name.clone());
                }
            }
        }
        Self {
            keywords: vec![
                "fn", "let", "return", "if", "else", "for", "while", "break", "continue", "import",
                "in", "type",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            builtins: vec![
                "tpt.zeros",
                "tpt.ones",
                "tpt.empty",
                "tpt.full",
                "tpt.random",
                "tpt.relu",
                "tpt.gelu",
                "tpt.sigmoid",
                "tpt.tanh",
                "tpt.matmul",
                "tpt.gemm",
                "tpt.attention",
                "tpt.softmax",
                "tpt.cross_entropy",
                "tpt.mse",
                "tpt.reshape",
                "tpt.transpose",
                "tpt.concat",
                "tpt.conv2d",
                "tpt.pool2d",
                "tpt.no_grad",
                "tpt.shape",
                "tpt.dtype",
                "Tensor",
                "Model",
                "DataLoader",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            annotations: vec![
                "@doc",
                "@input",
                "@output",
                "@example",
                "@constraint",
                "@complexity",
                "@memory",
                "@flops",
                "@differentiable",
                "@gradient_checkpoint",
                "@requires_gpu",
                "@requires_tensor_cores",
                "@min_vram_gb",
                "@supports_distributed",
                "@max_batch_size",
                "@preferred_dtype",
                "@gpu_optimized",
                "@distributed",
                "@deploy",
                "@async_exec",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            identifiers,
        }
    }
}

/// Provide completions at the given position.
pub fn provide_completions(
    doc: &crate::document::DocumentStore,
    pos: Position,
) -> Option<CompletionResponse> {
    let db = doc.completion_db();
    let offset = position_to_offset(&doc.source, pos);

    let mut items = Vec::new();

    // If we're inside an annotation context (after @), suggest annotations
    let prev_token = find_prev_token(&doc.tokens, offset);
    let is_in_annotation = matches!(prev_token, Some(t) if matches!(t.kind, TokenKind::At));
    let is_after_dot = matches!(prev_token, Some(t) if matches!(t.kind, TokenKind::Dot));

    if is_in_annotation {
        // Only suggest annotations
        for ann in &db.annotations {
            items.push(CompletionItem {
                label: ann.clone(),
                kind: Some(CompletionItemKind::PROPERTY),
                detail: Some(format!("TPT Script annotation {}", ann)),
                insert_text: Some(
                    if ann == "@constraint"
                        || ann == "@doc"
                        || ann == "@input"
                        || ann == "@output"
                        || ann == "@example"
                    {
                        format!("{}(\"\")", ann)
                    } else if ann == "@requires_gpu"
                        || ann == "@differentiable"
                        || ann == "@gpu_optimized"
                        || ann == "@async_exec"
                    {
                        format!("{}(true)", ann)
                    } else if ann == "@min_vram_gb" || ann == "@max_batch_size" {
                        format!("{}(8)", ann)
                    } else if ann == "@distributed" {
                        format!("{}(strategy=\"fsdp\", devices=8)", ann)
                    } else if ann == "@deploy" {
                        format!("{}(target=\"cloud\", optimize=true)", ann)
                    } else {
                        ann.clone()
                    },
                ),
                ..Default::default()
            });
        }
    } else if is_after_dot {
        // Suggest tpt.* builtins
        for builtin in &db.builtins {
            if builtin.starts_with("tpt.") {
                items.push(CompletionItem {
                    label: builtin.clone(),
                    kind: Some(CompletionItemKind::FUNCTION),
                    detail: Some(format!("TPT builtin {}", builtin)),
                    ..Default::default()
                });
            }
        }
    } else {
        // General context: suggest keywords, identifiers, and builtins
        for kw in &db.keywords {
            items.push(CompletionItem {
                label: kw.clone(),
                kind: Some(CompletionItemKind::KEYWORD),
                ..Default::default()
            });
        }
        for ident in &db.identifiers {
            items.push(CompletionItem {
                label: ident.clone(),
                kind: Some(CompletionItemKind::VARIABLE),
                ..Default::default()
            });
        }
        for builtin in &db.builtins {
            items.push(CompletionItem {
                label: builtin.clone(),
                kind: Some(CompletionItemKind::FUNCTION),
                detail: Some(format!("TPT builtin {}", builtin)),
                ..Default::default()
            });
        }
    }

    if items.is_empty() {
        None
    } else {
        Some(CompletionResponse::Array(items))
    }
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

fn find_prev_token(
    tokens: &[tpt_gpu_script_core::Token],
    offset: usize,
) -> Option<&tpt_gpu_script_core::Token> {
    tokens
        .iter()
        .filter(|t| t.span.end <= offset)
        .max_by_key(|t| t.span.end)
}
