use crate::document::DocumentStore;
use tower_lsp::lsp_types::*;

fn lex_error_to_position(e: &tpt_gpu_script_core::LexError) -> (u32, u32) {
    match e {
        tpt_gpu_script_core::LexError::UnexpectedChar { line, col, .. } => (*line, *col),
        tpt_gpu_script_core::LexError::UnterminatedString { line, col, .. } => (*line, *col),
        tpt_gpu_script_core::LexError::UnterminatedBlockComment { line, col, .. } => (*line, *col),
        tpt_gpu_script_core::LexError::InvalidEscape { line, col, .. } => (*line, *col),
        tpt_gpu_script_core::LexError::InvalidNumber { line, col, .. } => (*line, *col),
    }
}

pub fn compute_diagnostics(doc: &DocumentStore) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    if let Err(e) = tpt_gpu_script_core::tokenize(&doc.source) {
        let (line, col) = lex_error_to_position(&e);
        diagnostics.push(Diagnostic {
            range: Range {
                start: Position {
                    line: line - 1,
                    character: col - 1,
                },
                end: Position {
                    line: line - 1,
                    character: col,
                },
            },
            severity: Some(DiagnosticSeverity::ERROR),
            code: Some(NumberOrString::String("LEX_ERROR".to_string())),
            source: Some("tpt-gpu-lsp".to_string()),
            message: e.to_string(),
            ..Default::default()
        });
        return diagnostics;
    }
    let program = match tpt_gpu_script_core::compile_str(&doc.source) {
        Err(e) => {
            let span = match &e {
                tpt_gpu_script_core::CompileError::Parse(pe) => match pe {
                    tpt_gpu_script_core::ParseError::Expected { span, .. }
                    | tpt_gpu_script_core::ParseError::ExpectedIdent { span, .. }
                    | tpt_gpu_script_core::ParseError::Unexpected { span, .. }
                    | tpt_gpu_script_core::ParseError::IntLitTooLarge { span } => span,
                },
                _ => {
                    diagnostics.push(Diagnostic {
                        severity: Some(DiagnosticSeverity::ERROR),
                        code: Some(NumberOrString::String("COMPILE_ERROR".to_string())),
                        source: Some("tpt-gpu-lsp".to_string()),
                        message: e.to_string(),
                        ..Default::default()
                    });
                    return diagnostics;
                }
            };
            diagnostics.push(Diagnostic {
                range: Range {
                    start: Position {
                        line: span.line - 1,
                        character: span.col - 1,
                    },
                    end: Position {
                        line: span.line - 1,
                        character: span.col,
                    },
                },
                severity: Some(DiagnosticSeverity::ERROR),
                code: Some(NumberOrString::String("PARSE_ERROR".to_string())),
                source: Some("tpt-gpu-lsp".to_string()),
                message: e.to_string(),
                ..Default::default()
            });
            return diagnostics;
        }
        Ok(p) => p,
    };
    let checker = tpt_gpu_script_core::type_check(&program);
    for err in &checker.errors {
        diagnostics.push(Diagnostic {
            range: Range {
                start: Position {
                    line: err.span.line - 1,
                    character: err.span.col - 1,
                },
                end: Position {
                    line: err.span.line - 1,
                    character: err.span.col,
                },
            },
            severity: Some(DiagnosticSeverity::ERROR),
            code: Some(NumberOrString::String(err.code.to_string())),
            source: Some("tpt-gpu-lsp".to_string()),
            message: err.message.clone(),
            ..Default::default()
        });
    }
    diagnostics
}
