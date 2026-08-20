//! # tpt-doc — Type-checked documents (the LaTeX killer)
//!
//! Write papers as Rust. The [`doc!`] macro builds a [`Document`] AST out of a
//! small DSL, [`Document::validate`] checks that every `cite`/`ref` resolves,
//! and the renderers emit HTML, LaTeX or (behind the `pdf` feature) PDF.
//!
//! ```
//! use tpt_doc::prelude::*;
//!
//! let x = sym!(x);
//! let energy = x.clone() * x.clone();
//!
//! let paper = doc! {
//!     title: "A Very Short Paper",
//!     authors: ["Ada Lovelace", "Alan Turing"],
//!     date: "2026-01-01",
//!     abstract: { "We square {} things.", 1 },
//!     section("sec:intro" => "Introduction") {
//!         p("Squaring is useful, as shown by Knuth.");
//!         cite("knuth1984");
//!         equation! { "eq:sq" => energy.clone() };
//!         table! { headers: ["n", "n^2"], rows: [["2", "4"]] };
//!     },
//!     section("Discussion") {
//!         p("See the equation above.");
//!         ref("eq:sq");
//!     },
//!     bib! { @book { knuth1984, title = "The TeXbook", author = "D. E. Knuth", year = "1984" } },
//! };
//!
//! paper.validate().unwrap();
//! assert!(paper.to_html().contains("<span class=\"math\">"));
//! assert!(paper.to_latex().contains("\\begin{equation}"));
//! ```
//!
//! ## Scope
//!
//! * Validation is a **runtime** check ([`Document::validate`]). A true
//!   compile-time check of a whole document needs a procedural macro and is
//!   future work; [`validate_const`] covers the `const`-evaluable subset
//!   (static citation-key lists).
//! * `to_pdf` (feature `pdf`) is a **text-only** layout: paginated plain text
//!   in a built-in font, with LaTeX shown as source. HTML and LaTeX are the
//!   full-fidelity outputs.
//! * `to_epub`/`write_epub` (feature `epub`) package the HTML renderer's output
//!   as an EPUB 3 book: one XHTML chapter per top-level section, plus a
//!   navigation document, package document and container.
//! * Paragraph interpolation uses `format!`, so `p("x is {x}")` picks up the
//!   Rust 2021 implicit named-argument capture.

mod ast;
mod bib;
#[cfg(feature = "epub")]
mod epub;
mod error;
mod macros;
#[cfg(feature = "pdf")]
mod pdf;
#[cfg(feature = "harfbuzz")]
mod typography;
#[cfg(feature = "wasm")]
mod wasm;
mod render;

pub use ast::{Block, DocTable, Document, Equation, Section};
pub use bib::{BibEntry, Bibliography};
pub use error::DocError;
pub use render::{escape_html, escape_latex};

/// `const`-evaluable citation check over static key lists.
///
/// This is the compile-time-checkable subset of [`Document::validate`]: given
/// the citation keys a document uses and the keys its bibliography defines,
/// report whether all citations resolve. Useful in a `const` assertion:
///
/// ```
/// const CITES: &[&str] = &["knuth1984"];
/// const KEYS: &[&str] = &["knuth1984", "einstein1905"];
/// const _: () = assert!(tpt_doc::validate_const(CITES, KEYS));
/// ```
pub const fn validate_const(cites: &[&str], keys: &[&str]) -> bool {
    let mut i = 0;
    while i < cites.len() {
        let mut found = false;
        let mut j = 0;
        while j < keys.len() {
            if str_eq(cites[i], keys[j]) {
                found = true;
                break;
            }
            j += 1;
        }
        if !found {
            return false;
        }
        i += 1;
    }
    true
}

const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Everything needed to write a document.
pub mod prelude {
    pub use crate::{bib, doc, equation, table};
    pub use crate::{
        BibEntry, Bibliography, Block, DocError, DocTable, Document, Equation, Section,
    };
    pub use tpt_sym::{sym, Expr};
}
