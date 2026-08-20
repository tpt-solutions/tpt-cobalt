pub mod formatter;
pub mod linter;

/// Format a TPT Script source string, returning the canonical formatted output.
pub fn format(source: &str) -> Result<String, tpt_gpu_script_core::CompileError> {
    formatter::format_source(source)
}

/// Lint a TPT Script source string, returning a list of lint warnings.
pub fn lint(source: &str) -> Vec<linter::LintWarning> {
    linter::lint_source(source)
}

/// Format and lint in one pass.
pub fn format_and_lint(
    source: &str,
) -> Result<(String, Vec<linter::LintWarning>), tpt_gpu_script_core::CompileError> {
    let formatted = format(source)?;
    let warnings = lint(&formatted);
    Ok((formatted, warnings))
}
