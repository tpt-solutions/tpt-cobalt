pub mod ast;
pub mod codegen;
pub mod errors;
pub mod introspect;
pub mod lexer;
pub mod modules;
pub mod parser;
pub mod semantic;

pub use ast::Program;
pub use codegen::{emit, CodegenOutput};
pub use errors::{ErrorCode, ErrorContext, TptError, TypeError};
pub use introspect::DocFormat;
pub use lexer::{tokenize, LexError, Span, Token, TokenKind};
pub use modules::ProjectConfig;
pub use parser::{parse, ParseError};
pub use semantic::{type_check, TypeChecker};

/// Convenience: lex and parse a TPT Script source string in one call.
pub fn compile_str(source: &str) -> Result<Program, CompileError> {
    let tokens = tokenize(source)?;
    let program = parse(tokens)?;
    Ok(program)
}

/// Full pipeline: lex -> parse -> type-check -> codegen.
///
/// Returns `Err` if lexing or parsing fails.  Type errors are non-fatal
/// and are accessible via [`TypeChecker::errors`] on the returned checker.
pub fn compile_full(source: &str) -> Result<(TypeChecker, CodegenOutput), CompileError> {
    let program = compile_str(source)?;
    let checker = type_check(&program);
    let output = emit(&program);
    Ok((checker, output))
}

/// Full pipeline with module resolution and project configuration.
///
/// This is the recommended entry point for the TPT Script API.
/// It resolves imports against the standard library modules and applies
/// project-level configuration from `tpt.toml`.
pub fn compile_project(
    source: &str,
    config: &ProjectConfig,
) -> Result<(TypeChecker, CodegenOutput), CompileError> {
    let program = compile_str(source)?;
    let checker = type_check(&program);
    let mut output = emit(&program);

    // Inject module preamble based on resolved imports
    let preamble = modules::resolve_imports_preamble(&program, config);
    if !preamble.is_empty() {
        output.rust_source = format!("{}\n{}", preamble, output.rust_source);
    }

    Ok((checker, output))
}

#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error("lex error: {0}")]
    Lex(#[from] LexError),
    #[error("parse error: {0}")]
    Parse(#[from] ParseError),
}
