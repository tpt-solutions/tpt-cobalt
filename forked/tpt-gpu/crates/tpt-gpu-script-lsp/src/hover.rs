use crate::document::DocumentStore;
use tower_lsp::lsp_types::*;
use tpt_gpu_script_core::TokenKind;

pub fn provide_hover(doc: &DocumentStore, pos: Position) -> Option<Hover> {
    let token = doc.token_at(pos)?;
    match &token.kind {
        TokenKind::Ident(name) => {
            if let Some(doc_str) = builtin_doc(name) {
                return Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: doc_str.to_string(),
                    }),
                    range: Some(Range {
                        start: offset_to_position(&doc.source, token.span.start),
                        end: offset_to_position(&doc.source, token.span.end),
                    }),
                });
            }
            if is_type_name(name) {
                return Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: format!("**Type**: `{}`", name),
                    }),
                    range: Some(Range {
                        start: offset_to_position(&doc.source, token.span.start),
                        end: offset_to_position(&doc.source, token.span.end),
                    }),
                });
            }
            None
        }
        TokenKind::KwFn => Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: String::from("`fn` - Declares a new function."),
            }),
            range: Some(Range {
                start: offset_to_position(&doc.source, token.span.start),
                end: offset_to_position(&doc.source, token.span.end),
            }),
        }),
        TokenKind::KwLet => Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: String::from("`let` - Declares an immutable variable binding."),
            }),
            range: Some(Range {
                start: offset_to_position(&doc.source, token.span.start),
                end: offset_to_position(&doc.source, token.span.end),
            }),
        }),
        _ => None,
    }
}

fn builtin_doc(name: &str) -> Option<&'static str> {
    match name {
        "tpt.zeros" => Some("zeros(shape, dtype=f32) -> Tensor - Creates a zero-filled tensor."),
        "tpt.ones" => Some("ones(shape, dtype=f32) -> Tensor - Creates a one-filled tensor."),
        "tpt.relu" => Some("relu(tensor) -> Tensor - Applies ReLU activation."),
        "tpt.gelu" => Some("gelu(tensor) -> Tensor - Applies GELU activation."),
        "tpt.matmul" => Some("matmul(a, b) -> Tensor - Matrix multiplication."),
        "tpt.attention" => {
            Some("attention(q, k, v, scale) -> Tensor - Scaled dot-product attention.")
        }
        "tpt.softmax" => Some("softmax(tensor, dim) -> Tensor - Softmax along dimension."),
        "tpt.cross_entropy" => Some("cross_entropy(pred, targets) -> Tensor - Cross-entropy loss."),
        "tpt.no_grad" => Some("no_grad { ... } - Disables gradient tracking."),
        "tpt.shape" => Some("shape(tensor) -> [index] - Returns tensor shape."),
        "Tensor" => Some("Tensor[dtype, dim0, ...] - Multi-dimensional GPU array."),
        "Model" => Some("Model - A trained model. Supports .forward(), .backward(), .step()."),
        "DataLoader" => Some("DataLoader - Iterable yielding training batches."),
        _ => None,
    }
}

fn is_type_name(name: &str) -> bool {
    matches!(
        name,
        "i8" | "i16"
            | "i32"
            | "i64"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "f16"
            | "bf16"
            | "f32"
            | "f64"
            | "bool"
            | "index"
            | "Tensor"
            | "Model"
            | "DataLoader"
            | "ComputeStream"
            | "Optimizer"
    )
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
