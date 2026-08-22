//! Integration tests for `tpt-doc`.

use std::sync::Arc;

use tpt_columnar::array::{Float64Array, Int64Array, StringArray};
use tpt_columnar::datatypes::{DataType, Field, Schema};
use tpt_columnar::record_batch::RecordBatch;
use tpt_doc::prelude::*;
use tpt_omni::Table;

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// The headline test: a small paper renders to both HTML and LaTeX.
#[test]
fn doc_macro_renders_html_and_latex() {
    let x = sym!(x);
    let square = x.clone() * x.clone();
    let latex = square.to_latex();

    let paper = doc! {
        title: "Squares",
        authors: ["Ada Lovelace"],
        date: "2026-01-01",
        abstract: { "A study of {} idea.", 1 },
        section("Introduction") {
            p("Squaring is useful.");
            equation! { square.clone() };
            cite("knuth1984");
        },
        bib! { @book { knuth1984, title = "The TeXbook", author = "D. E. Knuth", year = "1984" } },
    };

    let html = paper.to_html();
    assert!(html.contains("<html"), "missing <html>: {html}");
    assert!(html.contains("<h1 class=\"title\">Squares</h1>"));
    assert!(html.contains("Ada Lovelace"));
    assert!(html.contains("A study of 1 idea."));
    assert!(html.contains("<span class=\"math\">"));
    assert!(html.contains(&latex), "equation latex `{latex}` missing");
    assert!(html.contains("<li id=\"bib-knuth1984\">"));

    let tex = paper.to_latex();
    assert!(tex.contains("\\documentclass{article}"));
    assert!(tex.contains("\\section{Introduction}"));
    assert!(tex.contains("\\begin{equation}"));
    assert!(tex.contains(&latex));
    assert!(tex.contains("\\cite{knuth1984}"));
    assert!(tex.contains("\\bibitem{knuth1984}"));
    assert!(tex.contains("\\end{document}"));
}

#[test]
fn validate_passes_when_citations_resolve() {
    let paper = doc! {
        title: "Fine",
        section("S") { p("Body."); cite("a"); },
        bib! { @article { a, title = "A" } },
    };
    assert_eq!(paper.validate(), Ok(()));
    assert!(paper.errors().is_empty());
}

#[test]
fn validate_fails_on_missing_citation() {
    let paper = doc! {
        title: "Broken",
        section("S") { p("Body."); cite("missing"); },
        bib! { @article { present, title = "P" } },
    };
    let err = paper.validate().unwrap_err();
    assert_eq!(err, DocError::BrokenCitation("missing".into()));
    assert!(err.to_string().contains("missing"));
}

#[test]
fn validate_checks_labels_and_refs() {
    let ok = doc! {
        section("sec:intro" => "Intro") { p("hi"); },
        section("Other") { ref("sec:intro"); },
    };
    assert!(ok.validate().is_ok());
    assert!(ok.labels().contains("sec:intro"));

    let bad = doc! { section("S") { ref("nope"); } };
    assert_eq!(
        bad.validate(),
        Err(DocError::BrokenReference("nope".into()))
    );
}

#[test]
fn bibtex_parse_reads_entries() {
    let src = std::fs::read_to_string(fixture("refs.bib")).unwrap();
    let bib = Bibliography::parse(&src).unwrap();
    assert_eq!(bib.len(), 2);
    let e = bib.get("einstein1905").unwrap();
    assert_eq!(e.kind, "article");
    assert_eq!(e.author(), Some("A. Einstein"));
    assert_eq!(e.year(), Some("1905"));
    assert_eq!(e.field("journal"), Some("Annalen der Physik"));
    // Quoted values and nested braces survive; bare values parse.
    let k = bib.get("knuth1984").unwrap();
    assert_eq!(k.title(), Some("The {TeX}book"));
    assert_eq!(k.year(), Some("1984"));
    assert!(bib.check_duplicates().is_ok());
}

#[test]
fn bibtex_parse_detects_duplicate_key() {
    let src = r#"
        @article{x, title = {First}}
        @article{x, title = {Second}}
    "#;
    assert_eq!(
        Bibliography::parse(src),
        Err(DocError::DuplicateKey("x".into()))
    );
}

#[test]
fn validate_detects_duplicate_inline_keys() {
    let paper = doc! {
        bib! { @article { dup, title = "One" }, @book { dup, title = "Two" } },
    };
    assert_eq!(paper.validate(), Err(DocError::DuplicateKey("dup".into())));
}

#[test]
fn bibliography_file_loads_and_missing_file_is_reported() {
    let paper = doc! {
        title: "From file",
        section("S") { cite("einstein1905"); },
        bibliography(fixture("refs.bib")),
    };
    assert!(paper.validate().is_ok(), "{:?}", paper.errors());
    assert!(paper.to_latex().contains("\\bibliography{"));
    assert!(paper.to_html().contains("Annalen der Physik"));

    let broken = doc! { bibliography("does-not-exist.bib") };
    assert!(matches!(broken.validate(), Err(DocError::BibFile { .. })));
}

#[test]
fn table_macro_renders_rows() {
    let paper = doc! {
        title: "Data",
        section("Results") {
            table! {
                caption: "Squares",
                label: "tab:sq",
                headers: ["n", "n^2"],
                rows: [["1", "1"], ["2", "4"], ["3", "9"]],
            };
        },
    };
    let html = paper.to_html();
    assert!(html.contains("<caption>Squares</caption>"));
    assert!(html.contains("<th>n</th><th>n^2</th>"));
    assert!(html.contains("<td>2</td><td>4</td>"));
    assert!(html.contains("<td>3</td><td>9</td>"));

    let tex = paper.to_latex();
    assert!(tex.contains("\\begin{tabular}{ll}"));
    assert!(tex.contains("3 & 9 \\\\"));
    assert!(tex.contains("\\label{tab:sq}"));
}

#[test]
fn table_accepts_runtime_values_and_omni_tables() {
    let headers: Vec<String> = vec!["city".into(), "pop".into(), "score".into()];
    let rows: Vec<Vec<String>> = vec![vec!["Oslo".into(), "1".into(), "2.5".into()]];
    let runtime = doc! { table! { headers: headers.clone(), rows: rows.clone() } };
    assert!(runtime.to_html().contains("<td>Oslo</td>"));

    let schema = Arc::new(Schema::new(vec![
        Field::new("city", DataType::Utf8, false),
        Field::new("pop", DataType::Int64, false),
        Field::new("score", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["Oslo", "Bergen"])),
            Arc::new(Int64Array::from(vec![700_000, 280_000])),
            Arc::new(Float64Array::from(vec![1.5, 2.25])),
        ],
    )
    .unwrap();
    let omni = Table::new(batch);

    let paper = doc! { table! { caption: "Cities", from: &omni } };
    assert!(paper.validate().is_ok(), "{:?}", paper.errors());
    let html = paper.to_html();
    assert!(html.contains("<th>city</th><th>pop</th><th>score</th>"));
    assert!(html.contains("<td>Bergen</td>"));
    assert!(html.contains("280000"));
    assert!(html.contains("2.25"));
}

#[test]
fn equation_forms_and_symbolic_derivative() {
    let x = sym!(x);
    let f = x.clone() * x.clone();
    let paper = doc! {
        section("Calc") {
            equation! { "eq:f" => f.clone() };
            equation! { raw "E = mc^2" };
            equation! { f.diff("x") };
            ref("eq:f");
        },
    };
    assert!(paper.validate().is_ok());
    let tex = paper.to_latex();
    assert!(tex.contains("\\label{eq:f}"));
    assert!(tex.contains("E = mc^2"));
    assert!(tex.contains(&f.diff("x").to_latex()));
    assert!(tex.contains("\\ref{eq:f}"));
    assert_eq!(tex.matches("\\begin{equation}").count(), 3);
}

#[test]
fn nested_sections_and_escaping() {
    let paper = doc! {
        title: "Escapes & <tags>",
        section("Outer") {
            p("100% of a & b < c");
            section("Inner") { p("deep"); };
        },
    };
    let html = paper.to_html();
    assert!(html.contains("<h1 class=\"title\">Escapes &amp; &lt;tags&gt;</h1>"));
    assert!(html.contains("100% of a &amp; b &lt; c"));
    assert!(html.contains("<h2>Outer</h2>"));
    assert!(html.contains("<h3>Inner</h3>"));

    let tex = paper.to_latex();
    assert!(tex.contains("100\\% of a \\& b < c"));
    assert!(tex.contains("\\section{Outer}"));
    assert!(tex.contains("\\subsection{Inner}"));
}

#[test]
fn paragraph_interpolation_and_plain_text() {
    let n = 42;
    let paper = doc! {
        title: "Interp",
        section("S") { p("the answer is {n}"); text(1.5_f64 + 1.0); },
    };
    assert!(paper.to_html().contains("the answer is 42"));
    let text = paper.to_text();
    assert!(text.contains("the answer is 42"));
    assert!(text.contains("2.5"));
}

#[test]
#[allow(clippy::assertions_on_constants)]
fn validate_const_is_compile_time_evaluable() {
    const CITES: &[&str] = &["a", "b"];
    const KEYS: &[&str] = &["b", "a", "c"];
    const OK: bool = tpt_doc::validate_const(CITES, KEYS);
    const BAD: bool = tpt_doc::validate_const(&["z"], KEYS);
    assert!(OK);
    assert!(!BAD);
}

#[test]
fn document_can_be_built_without_the_macro() {
    let mut d = Document::new();
    d.set_title("Manual");
    let mut s = Section::new("S");
    s.push_block(Block::Paragraph("hi".into()));
    s.push_block(Block::Equation(Equation::raw("a+b").with_label("eq:1")));
    d.push_block(Block::Section(s));
    d.bibliography
        .push(BibEntry::new("misc", "k").with("title", "T"));
    d.push_block(Block::Citation("k".into()));
    assert!(d.validate().is_ok());
    assert_eq!(d.citations(), vec!["k".to_string()]);
}

/// Only runs with `--features pdf` (the text-only PDF backend).
#[cfg(feature = "pdf")]
#[test]
fn pdf_backend_emits_a_pdf() {
    let paper = doc! {
        title: "PDF",
        section("S") { p("hello"); equation! { sym!(x) }; },
    };
    let bytes = paper.to_pdf_bytes().unwrap();
    assert!(bytes.starts_with(b"%PDF"));

    let path = std::env::temp_dir().join("tpt_doc_test.pdf");
    paper.to_pdf(&path).unwrap();
    assert!(std::fs::metadata(&path).unwrap().len() > 0);
    let _ = std::fs::remove_file(&path);
}
