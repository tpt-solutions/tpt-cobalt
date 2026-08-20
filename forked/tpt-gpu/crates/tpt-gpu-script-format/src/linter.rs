use serde::{Deserialize, Serialize};
use tpt_gpu_script_core;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintWarning {
    pub rule: String,
    pub message: String,
    pub line: u32,
    pub col: u32,
    pub severity: LintSeverity,
    pub fix: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LintSeverity {
    #[serde(rename = "error")]
    Error,
    #[serde(rename = "warning")]
    Warning,
    #[serde(rename = "info")]
    Info,
}

pub fn lint_source(source: &str) -> Vec<LintWarning> {
    let mut warnings = Vec::new();
    let tokens = match tpt_gpu_script_core::tokenize(source) {
        Ok(t) => t,
        Err(_) => return warnings,
    };
    warnings.extend(check_missing_doc(&tokens));
    warnings.extend(check_naming_conventions(&tokens));
    warnings.extend(check_line_length(source));
    warnings.extend(check_trailing_whitespace(source));
    warnings.extend(check_missing_returns(&tokens));
    warnings.extend(check_unnecessary_semicolons(&tokens));
    warnings
}

fn check_missing_doc(tokens: &[tpt_gpu_script_core::Token]) -> Vec<LintWarning> {
    let mut warnings = Vec::new();
    for (i, token) in tokens.iter().enumerate() {
        if matches!(token.kind, tpt_gpu_script_core::TokenKind::KwFn) {
            let has_doc = tokens[..i]
                .iter()
                .any(|t| matches!(t.kind, tpt_gpu_script_core::TokenKind::At));
            if !has_doc {
                warnings.push(LintWarning {
                    rule: "missing_doc".to_string(),
                    message: "Public function is missing @doc annotation".to_string(),
                    line: token.span.line,
                    col: token.span.col,
                    severity: LintSeverity::Warning,
                    fix: None,
                });
            }
        }
    }
    warnings
}

fn check_naming_conventions(tokens: &[tpt_gpu_script_core::Token]) -> Vec<LintWarning> {
    let mut warnings = Vec::new();
    for token in tokens {
        if let tpt_gpu_script_core::TokenKind::Ident(name) = &token.kind {
            if name.len() > 1
                && name.chars().any(|c| c.is_uppercase())
                && !is_pascal_case(name)
                && !is_known_tpt_name(name)
            {
                warnings.push(LintWarning {
                    rule: "naming_convention".to_string(),
                    message: format!("Identifier '{}' should use snake_case", name),
                    line: token.span.line,
                    col: token.span.col,
                    severity: LintSeverity::Info,
                    fix: None,
                });
            }
        }
    }
    warnings
}

fn check_line_length(source: &str) -> Vec<LintWarning> {
    let mut warnings = Vec::new();
    for (line_num, line) in source.lines().enumerate() {
        if line.len() > 100 {
            warnings.push(LintWarning {
                rule: "line_too_long".to_string(),
                message: format!("Line is {} characters (max 100)", line.len()),
                line: (line_num + 1) as u32,
                col: 101,
                severity: LintSeverity::Info,
                fix: None,
            });
        }
    }
    warnings
}

fn check_trailing_whitespace(source: &str) -> Vec<LintWarning> {
    let mut warnings = Vec::new();
    for (line_num, line) in source.lines().enumerate() {
        if line != line.trim_end() {
            warnings.push(LintWarning {
                rule: "trailing_whitespace".to_string(),
                message: "Line has trailing whitespace".to_string(),
                line: (line_num + 1) as u32,
                col: line.len() as u32,
                severity: LintSeverity::Info,
                fix: Some("Remove trailing whitespace".to_string()),
            });
        }
    }
    warnings
}

fn check_missing_returns(tokens: &[tpt_gpu_script_core::Token]) -> Vec<LintWarning> {
    let mut warnings = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if matches!(tokens[i].kind, tpt_gpu_script_core::TokenKind::KwFn) {
            let mut j = i + 1;
            while j < tokens.len()
                && !matches!(tokens[j].kind, tpt_gpu_script_core::TokenKind::Ident(_))
            {
                j += 1;
            }
            if j >= tokens.len() {
                i += 1;
                continue;
            }
            let func_name = match &tokens[j].kind {
                tpt_gpu_script_core::TokenKind::Ident(n) => n.clone(),
                _ => {
                    i += 1;
                    continue;
                }
            };
            j += 1;
            let mut has_return_type = false;
            while j < tokens.len()
                && !matches!(tokens[j].kind, tpt_gpu_script_core::TokenKind::LBrace)
            {
                if matches!(tokens[j].kind, tpt_gpu_script_core::TokenKind::Arrow) {
                    has_return_type = true;
                    break;
                }
                j += 1;
            }
            if has_return_type {
                let mut brace_depth = 0;
                let mut has_return = false;
                while j < tokens.len() {
                    match tokens[j].kind {
                        tpt_gpu_script_core::TokenKind::LBrace => brace_depth += 1,
                        tpt_gpu_script_core::TokenKind::RBrace => {
                            brace_depth -= 1;
                            if brace_depth == 0 {
                                break;
                            }
                        }
                        tpt_gpu_script_core::TokenKind::KwReturn if brace_depth > 0 => {
                            has_return = true;
                        }
                        _ => {}
                    }
                    j += 1;
                }
                if !has_return {
                    warnings.push(LintWarning {
                        rule: "missing_return".to_string(),
                        message: format!(
                            "Function '{}' has a return type but no return statement",
                            func_name
                        ),
                        line: tokens[i].span.line,
                        col: tokens[i].span.col,
                        severity: LintSeverity::Warning,
                        fix: None,
                    });
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
    warnings
}

fn check_unnecessary_semicolons(tokens: &[tpt_gpu_script_core::Token]) -> Vec<LintWarning> {
    let mut warnings = Vec::new();
    for i in 0..tokens.len().saturating_sub(1) {
        if matches!(tokens[i].kind, tpt_gpu_script_core::TokenKind::RBrace)
            && matches!(
                tokens[i + 1].kind,
                tpt_gpu_script_core::TokenKind::Semicolon
            )
        {
            warnings.push(LintWarning {
                rule: "unnecessary_semicolon".to_string(),
                message: "Unnecessary semicolon after closing brace".to_string(),
                line: tokens[i + 1].span.line,
                col: tokens[i + 1].span.col,
                severity: LintSeverity::Info,
                fix: Some("Remove the semicolon".to_string()),
            });
        }
    }
    warnings
}

fn is_pascal_case(name: &str) -> bool {
    !name.is_empty() && name.chars().next().unwrap().is_uppercase() && !name.contains('_')
}

fn is_known_tpt_name(name: &str) -> bool {
    matches!(
        name,
        "Tensor" | "Model" | "DataLoader" | "ComputeStream" | "Optimizer" | "Checkpoint" | "self"
    )
}

pub fn format_warnings(warnings: &[LintWarning]) -> String {
    let mut output = String::new();
    for w in warnings {
        let severity_str = match w.severity {
            LintSeverity::Error => "error",
            LintSeverity::Warning => "warning",
            LintSeverity::Info => "info",
        };
        output.push_str(&format!(
            "{} [{}] at line {}:col {}: {}\n",
            severity_str, w.rule, w.line, w.col, w.message
        ));
        if let Some(fix) = &w.fix {
            output.push_str(&format!("  fix: {}\n", fix));
        }
    }
    output
}
