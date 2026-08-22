//! Script-level errors rendered as Python-style tracebacks.

use std::fmt;

/// Convenience alias: in script mode every fallible helper returns this, so
/// `?` composes without importing an error type.
pub type Res<T = ()> = Result<T, ScriptError>;

/// A script error carrying a Python-like `kind: message` plus traceback frames.
///
/// `Display` renders the full traceback, e.g.
///
/// ```text
/// Traceback (most recent call last):
///   File "<tpt script>", line 2
///     panic "boom"
/// PanicError: boom
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptError {
    /// Python-style exception name, e.g. `ColumnError`, `PanicError`.
    pub kind: String,
    pub message: String,
    /// Traceback frames, outermost first.
    pub frames: Vec<String>,
}

impl ScriptError {
    pub fn new(kind: &str, message: impl Into<String>) -> Self {
        Self {
            kind: kind.to_string(),
            message: message.into(),
            frames: Vec::new(),
        }
    }

    /// Push a traceback frame (source line, call site, ...).
    pub fn frame(mut self, frame: impl Into<String>) -> Self {
        self.frames.push(frame.into());
        self
    }

    /// Render the full Python-style traceback.
    pub fn traceback(&self) -> String {
        let mut s = String::from("Traceback (most recent call last):\n");
        if self.frames.is_empty() {
            s.push_str("  File \"<tpt script>\"\n");
        }
        for f in &self.frames {
            s.push_str("  ");
            s.push_str(f);
            s.push('\n');
        }
        s.push_str(&format!("{}: {}", self.kind, self.message));
        s
    }
}

impl fmt::Display for ScriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.traceback())
    }
}

impl std::error::Error for ScriptError {}

impl From<tpt_omni::OmniError> for ScriptError {
    fn from(e: tpt_omni::OmniError) -> Self {
        ScriptError::new("OmniError", e.to_string())
    }
}

impl From<tpt_io::IoError> for ScriptError {
    fn from(e: tpt_io::IoError) -> Self {
        ScriptError::new("IoError", e.to_string())
    }
}

impl From<tpt_columnar::error::ColumnarError> for ScriptError {
    fn from(e: tpt_columnar::error::ColumnarError) -> Self {
        ScriptError::new("ArrowError", e.to_string())
    }
}
