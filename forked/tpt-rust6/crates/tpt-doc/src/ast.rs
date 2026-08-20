//! The document AST: [`Document`], [`Section`], [`Block`], [`Equation`], [`DocTable`].

use std::collections::BTreeSet;

use crate::bib::Bibliography;
use crate::error::DocError;

/// One block-level element of a document.
#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    /// A (possibly nested) section.
    Section(Section),
    /// Body text. Interpolation happens at construction time via `format!`.
    Paragraph(String),
    /// `cite("key")` — checked by [`Document::validate`].
    Citation(String),
    /// `ref("label")` — checked by [`Document::validate`].
    Reference(String),
    /// A display equation.
    Equation(Equation),
    /// A data table.
    Table(DocTable),
}

/// A titled, optionally labelled group of blocks.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Section {
    pub title: String,
    pub label: Option<String>,
    pub blocks: Vec<Block>,
}

impl Section {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            label: None,
            blocks: Vec::new(),
        }
    }
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
    pub fn set_label(&mut self, label: impl Into<String>) {
        self.label = Some(label.into());
    }
    pub fn push_block(&mut self, block: Block) {
        self.blocks.push(block);
    }
}

/// A display equation, carrying LaTeX plus (when built from `tpt_sym`) the
/// original symbolic expression so it can still be differentiated or evaluated.
#[derive(Debug, Clone, PartialEq)]
pub struct Equation {
    pub latex: String,
    pub label: Option<String>,
    pub expr: Option<tpt_sym::Expr>,
}

impl Equation {
    /// A raw LaTeX equation body.
    pub fn raw(latex: impl Into<String>) -> Self {
        Self {
            latex: latex.into(),
            label: None,
            expr: None,
        }
    }
    /// An equation backed by a symbolic expression.
    pub fn from_expr(expr: tpt_sym::Expr) -> Self {
        Self {
            latex: expr.to_latex(),
            label: None,
            expr: Some(expr),
        }
    }
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

impl From<tpt_sym::Expr> for Equation {
    fn from(e: tpt_sym::Expr) -> Self {
        Equation::from_expr(e)
    }
}
impl From<&tpt_sym::Expr> for Equation {
    fn from(e: &tpt_sym::Expr) -> Self {
        Equation::from_expr(e.clone())
    }
}
impl From<&str> for Equation {
    fn from(s: &str) -> Self {
        Equation::raw(s)
    }
}
impl From<String> for Equation {
    fn from(s: String) -> Self {
        Equation::raw(s)
    }
}

/// A data table: headers plus string cells, from literals or an
/// [`tpt_omni::Table`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DocTable {
    pub caption: Option<String>,
    pub label: Option<String>,
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    /// Error captured while loading data (surfaced by [`Document::validate`]).
    pub error: Option<String>,
}

impl DocTable {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn set_caption(&mut self, caption: impl Into<String>) {
        self.caption = Some(caption.into());
    }
    pub fn set_label(&mut self, label: impl Into<String>) {
        self.label = Some(label.into());
    }
    pub fn set_headers<I, S>(&mut self, headers: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.headers = headers.into_iter().map(Into::into).collect();
    }
    pub fn push_row<I, S>(&mut self, row: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.rows
            .push(row.into_iter().map(Into::into).collect::<Vec<String>>());
    }
    pub fn set_rows<I, R, S>(&mut self, rows: I)
    where
        I: IntoIterator<Item = R>,
        R: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.rows.clear();
        for r in rows {
            self.push_row(r);
        }
    }

    /// Stringify an Arrow-backed [`tpt_omni::Table`] into document cells.
    pub fn from_omni(table: &tpt_omni::Table) -> Result<Self, DocError> {
        let batch = table.batch();
        let mut out = DocTable::new();
        out.headers = table.column_names();
        for r in 0..batch.num_rows() {
            let mut row = Vec::with_capacity(batch.num_columns());
            for c in 0..batch.num_columns() {
                let cell = arrow::util::display::array_value_to_string(batch.column(c), r)
                    .map_err(|e| DocError::Data(e.to_string()))?;
                row.push(cell);
            }
            out.rows.push(row);
        }
        Ok(out)
    }

    /// Non-panicking `from_omni` for macro use: failures are recorded in
    /// [`DocTable::error`] and reported by [`Document::validate`].
    pub fn load_omni(&mut self, table: &tpt_omni::Table) {
        match DocTable::from_omni(table) {
            Ok(t) => {
                self.headers = t.headers;
                self.rows = t.rows;
            }
            Err(e) => self.error = Some(e.to_string()),
        }
    }
}

/// A whole document: metadata, blocks and bibliography.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Document {
    pub title: Option<String>,
    pub authors: Vec<String>,
    pub date: Option<String>,
    /// The abstract (`abstract` is a reserved word in Rust).
    pub summary: Option<String>,
    pub blocks: Vec<Block>,
    pub bibliography: Bibliography,
    /// Paths passed to `bibliography("...")`, used by `\bibliography{...}`.
    pub bib_files: Vec<String>,
    deferred: Vec<DocError>,
}

impl Document {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_title(&mut self, title: impl Into<String>) {
        self.title = Some(title.into());
    }
    pub fn set_date(&mut self, date: impl Into<String>) {
        self.date = Some(date.into());
    }
    pub fn set_abstract(&mut self, text: impl Into<String>) {
        self.summary = Some(text.into());
    }
    pub fn add_author(&mut self, author: impl Into<String>) {
        self.authors.push(author.into());
    }
    pub fn push_block(&mut self, block: Block) {
        self.blocks.push(block);
    }
    pub fn abstract_text(&self) -> Option<&str> {
        self.summary.as_deref()
    }

    /// Merge inline entries (from [`bib!`](crate::bib)) into the bibliography.
    pub fn merge_bibliography(&mut self, bib: Bibliography) {
        for e in bib.entries() {
            self.bibliography.push(e.clone());
        }
    }

    /// Register a `.bib` file and load it if it is readable. Read/parse
    /// failures are deferred to [`Document::validate`] so the `doc!` macro can
    /// stay infallible.
    pub fn add_bibliography(&mut self, path: impl Into<String>) {
        let path = path.into();
        match std::fs::read_to_string(&path) {
            Ok(src) => match Bibliography::parse(&src) {
                Ok(bib) => {
                    if let Err(e) = self.bibliography.try_extend(bib) {
                        self.deferred.push(e);
                    }
                }
                Err(e) => self.deferred.push(e),
            },
            Err(e) => self.deferred.push(DocError::BibFile {
                path: path.clone(),
                msg: e.to_string(),
            }),
        }
        self.bib_files.push(path);
    }

    /// Every label defined by a section, equation or table.
    pub fn labels(&self) -> BTreeSet<String> {
        fn walk(blocks: &[Block], out: &mut BTreeSet<String>) {
            for b in blocks {
                match b {
                    Block::Section(s) => {
                        if let Some(l) = &s.label {
                            out.insert(l.clone());
                        }
                        walk(&s.blocks, out);
                    }
                    Block::Equation(e) => {
                        if let Some(l) = &e.label {
                            out.insert(l.clone());
                        }
                    }
                    Block::Table(t) => {
                        if let Some(l) = &t.label {
                            out.insert(l.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
        let mut out = BTreeSet::new();
        walk(&self.blocks, &mut out);
        out
    }

    /// Every citation key used in the document, in order of appearance.
    pub fn citations(&self) -> Vec<String> {
        fn walk(blocks: &[Block], out: &mut Vec<String>) {
            for b in blocks {
                match b {
                    Block::Section(s) => walk(&s.blocks, out),
                    Block::Citation(k) => out.push(k.clone()),
                    _ => {}
                }
            }
        }
        let mut out = Vec::new();
        walk(&self.blocks, &mut out);
        out
    }

    /// All problems found, in a deterministic order.
    pub fn errors(&self) -> Vec<DocError> {
        let mut out = self.deferred.clone();
        if let Err(e) = self.bibliography.check_duplicates() {
            out.push(e);
        }
        for key in self.citations() {
            if !self.bibliography.contains(&key) {
                out.push(DocError::BrokenCitation(key));
            }
        }
        let labels = self.labels();
        fn walk(blocks: &[Block], labels: &BTreeSet<String>, out: &mut Vec<DocError>) {
            for b in blocks {
                match b {
                    Block::Section(s) => walk(&s.blocks, labels, out),
                    Block::Reference(l) => {
                        if !labels.contains(l) {
                            out.push(DocError::BrokenReference(l.clone()));
                        }
                    }
                    Block::Table(t) => {
                        if let Some(e) = &t.error {
                            out.push(DocError::Data(e.clone()));
                        }
                    }
                    _ => {}
                }
            }
        }
        walk(&self.blocks, &labels, &mut out);
        out
    }

    /// Reference/citation check: every `cite` key must exist in the
    /// bibliography, every `ref` label must have a target, and bibliography
    /// keys must be unique.
    ///
    /// A *compile-time* check of the whole document is future work (it needs a
    /// procedural macro); see [`validate_const`](crate::validate_const) for the
    /// `const`-evaluable subset.
    pub fn validate(&self) -> Result<(), DocError> {
        match self.errors().into_iter().next() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
