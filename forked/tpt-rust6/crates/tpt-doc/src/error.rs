use thiserror::Error;

/// Every way a document can be wrong.
///
/// `DocError` is `Clone` + `PartialEq` so that deferred errors (for example a
/// `.bib` file that could not be read while the `doc!` macro was building the
/// AST) can be stored inside the [`Document`](crate::Document) and replayed by
/// [`Document::validate`](crate::Document::validate).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DocError {
    /// `cite("key")` referred to a key with no bibliography entry.
    #[error("broken citation: no bibliography entry for key `{0}`")]
    BrokenCitation(String),
    /// `ref("label")` referred to a label that no section/equation/table defines.
    #[error("broken reference: no target labelled `{0}`")]
    BrokenReference(String),
    /// Two bibliography entries share one key.
    #[error("duplicate bibliography key `{0}`")]
    DuplicateKey(String),
    /// The BibTeX source could not be parsed.
    #[error("malformed bibtex at char {pos}: {msg}")]
    Bibtex { pos: usize, msg: String },
    /// A `bibliography("...")` file could not be read.
    #[error("bibliography file `{path}` could not be read: {msg}")]
    BibFile { path: String, msg: String },
    /// Data-driven content (e.g. an `tpt_omni::Table`) could not be rendered.
    #[error("data error: {0}")]
    Data(String),
    /// Writing a rendered document failed.
    #[error("io error: {0}")]
    Io(String),
    /// Building the EPUB container (zip archive) failed.
    #[error("epub error: {0}")]
    Epub(String),
    /// A renderer is not available in this build.
    #[error("unsupported: {0}")]
    Unsupported(String),
}
