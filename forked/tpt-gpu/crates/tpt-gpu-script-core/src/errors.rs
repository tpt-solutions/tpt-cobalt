// ---------------------------------------------------------------------------
// Structured error system for the TPT Script compiler.
//
// TptError = code (enum) + message + span + context (k/v) + fix_code (snippet)
// ---------------------------------------------------------------------------

use std::fmt;

use crate::lexer::Span;

// ---------------------------------------------------------------------------
// ErrorCode — full taxonomy
// ---------------------------------------------------------------------------

/// Machine-readable error category for every diagnostic emitted by tptb.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    // --- Type system ---
    /// General type incompatibility not covered by a more specific code.
    TypeError,
    /// Tensor shape dimensions do not match.
    ShapeMismatch,
    /// Tensor element dtype does not match.
    DtypeMismatch,
    /// Function return type does not match the declared return type.
    ReturnTypeMismatch,
    /// Expression is not callable.
    NotCallable,
    /// Index expression is not an integer type.
    InvalidIndexType,

    // --- Name resolution ---
    /// Variable used before it was declared in scope.
    UndefinedVariable,
    /// Method or operation not found on the given type.
    UndefinedOperation,

    // --- Constraints ---
    /// An `@constraint` expression evaluated to `false` at compile time.
    ConstraintViolation,

    // --- Parse / syntax ---
    /// Could not parse a constraint or annotation expression.
    ParseError,

    // --- Arity ---
    /// Wrong number of arguments supplied to a function or operation.
    ArityError,

    // --- Annotations ---
    /// A required annotation is absent from a function declaration.
    MissingAnnotation,
}

impl ErrorCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TypeError => "TYPE_ERROR",
            Self::ShapeMismatch => "SHAPE_MISMATCH",
            Self::DtypeMismatch => "DTYPE_MISMATCH",
            Self::ReturnTypeMismatch => "RETURN_TYPE_MISMATCH",
            Self::NotCallable => "NOT_CALLABLE",
            Self::InvalidIndexType => "INVALID_INDEX_TYPE",
            Self::UndefinedVariable => "UNDEFINED_VARIABLE",
            Self::UndefinedOperation => "UNDEFINED_OPERATION",
            Self::ConstraintViolation => "CONSTRAINT_VIOLATION",
            Self::ParseError => "PARSE_ERROR",
            Self::ArityError => "ARITY_ERROR",
            Self::MissingAnnotation => "MISSING_ANNOTATION",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// ErrorContext — structured key-value pairs attached to each error
// ---------------------------------------------------------------------------

/// Structured diagnostic context carried alongside a [`TptError`].
///
/// Each field is a `(key, value)` string pair providing machine-readable
/// information the caller (IDE, AI agent, auto-fixer) can act on.
#[derive(Debug, Clone, Default)]
pub struct ErrorContext {
    fields: Vec<(String, String)>,
}

impl ErrorContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: add a context field.
    pub fn with(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.push((key.into(), value.into()));
        self
    }

    /// Look up a field value by key.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    pub fn fields(&self) -> &[(String, String)] {
        &self.fields
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Auto-fix suggestion engine
// ---------------------------------------------------------------------------

/// Given an error code and its context, produce a TPT Script snippet that
/// would resolve the error, or `None` if no automatic fix is available.
pub fn suggest_fix(code: &ErrorCode, ctx: &ErrorContext) -> Option<String> {
    match code {
        ErrorCode::DtypeMismatch => {
            let var = ctx.get("var_name").unwrap_or("x");
            let dtype = ctx.get("expected_dtype")?;
            Some(format!("tpt.cast({var}, dtype={dtype})"))
        }

        ErrorCode::ShapeMismatch => {
            let var = ctx.get("var_name").unwrap_or("x");
            let shape = ctx.get("expected_shape")?;
            Some(format!("tpt.reshape({var}, {shape})"))
        }

        ErrorCode::UndefinedVariable => {
            let name = ctx.get("name")?;
            Some(format!("let {name} = ..."))
        }

        ErrorCode::ReturnTypeMismatch => {
            // Only emit a fix when both the variable and target type are known.
            let var = ctx.get("var_name")?;
            let dtype = ctx
                .get("expected_dtype")
                .or_else(|| ctx.get("expected_type"))?;
            Some(format!("return tpt.cast({var}, dtype={dtype})"))
        }

        ErrorCode::TypeError => {
            // Suggest a cast when we know what type is expected.
            let dtype = ctx
                .get("expected_dtype")
                .or_else(|| ctx.get("expected_type"))?;
            let var = ctx.get("var_name").unwrap_or("x");
            Some(format!("tpt.cast({var}, dtype={dtype})"))
        }

        ErrorCode::ArityError => {
            let fn_name = ctx.get("fn_name")?;
            let expected = ctx.get("expected_arity")?;
            Some(format!("# {fn_name} expects {expected} argument(s)"))
        }

        _ => None,
    }
}

// ---------------------------------------------------------------------------
// TptError — canonical compiler error type
// ---------------------------------------------------------------------------

/// A structured compiler diagnostic emitted by the TPT Script compiler.
///
/// Fields:
/// - `code`       — machine-readable error category ([`ErrorCode`])
/// - `message`    — human-readable description
/// - `span`       — source location
/// - `context`    — structured key-value context (shapes, types, names, …)
/// - `fix_code`   — TPT Script snippet that resolves the error, if derivable
/// - `suggestion` — human-readable fix description
#[derive(Debug, Clone)]
pub struct TptError {
    pub code: ErrorCode,
    pub message: String,
    pub span: Span,
    pub context: ErrorContext,
    /// A runnable TPT Script snippet that would fix the error.
    pub fix_code: Option<String>,
    /// A plain-English suggestion for resolving the error.
    pub suggestion: Option<String>,
}

impl TptError {
    pub fn new(code: ErrorCode, message: impl Into<String>, span: Span) -> Self {
        Self {
            code,
            message: message.into(),
            span,
            context: ErrorContext::new(),
            fix_code: None,
            suggestion: None,
        }
    }

    /// Attach structured context and run the auto-fix engine.
    pub fn with_context(mut self, ctx: ErrorContext) -> Self {
        self.fix_code = suggest_fix(&self.code, &ctx);
        self.context = ctx;
        self
    }

    /// Attach a human-readable suggestion (shown alongside `fix_code`).
    pub fn with_suggestion(mut self, s: impl Into<String>) -> Self {
        self.suggestion = Some(s.into());
        self
    }

    /// Override the auto-generated fix snippet.
    pub fn with_fix_code(mut self, code: impl Into<String>) -> Self {
        self.fix_code = Some(code.into());
        self
    }
}

impl fmt::Display for TptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error[{}] at {}: {}", self.code, self.span, self.message)?;
        if !self.context.is_empty() {
            for (k, v) in self.context.fields() {
                write!(f, "\n    {k}: {v}")?;
            }
        }
        if let Some(fix) = &self.fix_code {
            write!(f, "\n  fix: {fix}")?;
        }
        if let Some(sug) = &self.suggestion {
            write!(f, "\n  suggestion: {sug}")?;
        }
        Ok(())
    }
}

/// Backward-compatible alias so existing call sites that name `TypeError` compile.
pub type TypeError = TptError;
