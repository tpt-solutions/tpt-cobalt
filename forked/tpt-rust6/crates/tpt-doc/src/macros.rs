//! The `doc!` DSL and its helper macros.
//!
//! `doc!` is a `macro_rules!` token muncher. Items are separated by `,` or `;`
//! (both accepted everywhere), and the recognised items are:
//!
//! | item | effect |
//! |------|--------|
//! | `title: <expr>,` / `date: <expr>,` | metadata |
//! | `authors: [<expr>, ...]` | metadata |
//! | `abstract: { "text {interp}" }` | abstract, `format!` syntax |
//! | `section("Name") { ... }` / `section("lbl" => "Name") { ... }` | nested section |
//! | `p("text {interp}")` | paragraph (`format!` syntax, implicit captures) |
//! | `text(<expr>)` | paragraph from any `Display` value |
//! | `cite("key", ...)` | citation(s) |
//! | `ref("label")` | cross-reference |
//! | `equation! { <tpt_sym expr> }` | display equation |
//! | `table! { ... }` | data table |
//! | `bibliography("refs.bib")` | load a `.bib` file |
//! | `bib! { @article { key, title = "..." } }` | inline entries |

/// Build a [`Document`](crate::Document). See the [module docs](self).
///
/// ```
/// use tpt_doc::prelude::*;
/// let x = sym!(x);
/// let d = doc! {
///     title: "On Squares",
///     authors: ["A. Turing"],
///     section("Result") {
///         p("The square of {x:?} is below.");
///         equation! { x.clone() * x.clone() };
///         cite("knuth1984");
///     },
///     bib! { @book { knuth1984, title = "The TeXbook", author = "D. Knuth" } },
/// };
/// assert!(d.validate().is_ok());
/// ```
#[macro_export]
macro_rules! doc {
    ($($item:tt)*) => {{
        #[allow(unused_mut)]
        let mut __tpt_doc = $crate::Document::new();
        $crate::__doc_items!(__tpt_doc ; $($item)* ,);
        __tpt_doc
    }};
}

/// Item muncher shared by the document body and every `section` body.
#[doc(hidden)]
#[macro_export]
macro_rules! __doc_items {
    // ---- separators / end -------------------------------------------------
    ($d:ident ; ) => {};
    ($d:ident ; , $($r:tt)*) => { $crate::__doc_items!($d ; $($r)*); };
    ($d:ident ; ; $($r:tt)*) => { $crate::__doc_items!($d ; $($r)*); };

    // ---- metadata ---------------------------------------------------------
    ($d:ident ; title : $v:expr , $($r:tt)*) => {
        $d.set_title($v); $crate::__doc_items!($d ; $($r)*);
    };
    ($d:ident ; title : $v:expr ; $($r:tt)*) => {
        $d.set_title($v); $crate::__doc_items!($d ; $($r)*);
    };
    ($d:ident ; date : $v:expr , $($r:tt)*) => {
        $d.set_date($v); $crate::__doc_items!($d ; $($r)*);
    };
    ($d:ident ; date : $v:expr ; $($r:tt)*) => {
        $d.set_date($v); $crate::__doc_items!($d ; $($r)*);
    };
    ($d:ident ; authors : [ $($a:expr),* $(,)? ] $($r:tt)*) => {
        $( $d.add_author($a); )* $crate::__doc_items!($d ; $($r)*);
    };
    ($d:ident ; abstract : { $($a:tt)* } $($r:tt)*) => {
        $d.set_abstract(format!($($a)*)); $crate::__doc_items!($d ; $($r)*);
    };
    ($d:ident ; abstract : $v:literal $($r:tt)*) => {
        $d.set_abstract($v); $crate::__doc_items!($d ; $($r)*);
    };

    // ---- bibliography -----------------------------------------------------
    ($d:ident ; bibliography ( $p:expr ) $($r:tt)*) => {
        $d.add_bibliography($p); $crate::__doc_items!($d ; $($r)*);
    };
    ($d:ident ; bib ! { $($b:tt)* } $($r:tt)*) => {
        $d.merge_bibliography($crate::bib! { $($b)* }); $crate::__doc_items!($d ; $($r)*);
    };

    // ---- blocks -----------------------------------------------------------
    ($d:ident ; section ( $l:tt => $n:expr ) { $($body:tt)* } $($r:tt)*) => {
        {
            #[allow(unused_mut)]
            let mut __tpt_sec = $crate::Section::new($n).with_label($l);
            $crate::__doc_items!(__tpt_sec ; $($body)* ,);
            $d.push_block($crate::Block::Section(__tpt_sec));
        }
        $crate::__doc_items!($d ; $($r)*);
    };
    ($d:ident ; section ( $n:expr ) { $($body:tt)* } $($r:tt)*) => {
        {
            #[allow(unused_mut)]
            let mut __tpt_sec = $crate::Section::new($n);
            $crate::__doc_items!(__tpt_sec ; $($body)* ,);
            $d.push_block($crate::Block::Section(__tpt_sec));
        }
        $crate::__doc_items!($d ; $($r)*);
    };
    ($d:ident ; p ( $($a:tt)* ) $($r:tt)*) => {
        $d.push_block($crate::Block::Paragraph(format!($($a)*)));
        $crate::__doc_items!($d ; $($r)*);
    };
    ($d:ident ; text ( $e:expr ) $($r:tt)*) => {
        $d.push_block($crate::Block::Paragraph(::std::string::ToString::to_string(&$e)));
        $crate::__doc_items!($d ; $($r)*);
    };
    ($d:ident ; cite ( $($k:expr),+ $(,)? ) $($r:tt)*) => {
        $( $d.push_block($crate::Block::Citation(::std::string::ToString::to_string(&$k))); )+
        $crate::__doc_items!($d ; $($r)*);
    };
    ($d:ident ; ref ( $l:expr ) $($r:tt)*) => {
        $d.push_block($crate::Block::Reference(::std::string::ToString::to_string(&$l)));
        $crate::__doc_items!($d ; $($r)*);
    };
    ($d:ident ; equation ! { $($e:tt)* } $($r:tt)*) => {
        $d.push_block($crate::equation! { $($e)* }); $crate::__doc_items!($d ; $($r)*);
    };
    ($d:ident ; table ! { $($t:tt)* } $($r:tt)*) => {
        $d.push_block($crate::table! { $($t)* }); $crate::__doc_items!($d ; $($r)*);
    };
}

/// A display equation, as a [`Block`](crate::Block).
///
/// * `equation! { expr }` — any value convertible into an
///   [`Equation`](crate::Equation) (notably `tpt_sym::Expr`).
/// * `equation! { "eq:label" => expr }` — labelled, so `ref("eq:label")` resolves.
/// * `equation! { raw "E = mc^2" }` — raw LaTeX.
///
/// ```
/// # use tpt_doc::prelude::*;
/// let e = equation! { "eq:x" => sym!(x) };
/// assert!(matches!(e, Block::Equation(_)));
/// ```
#[macro_export]
macro_rules! equation {
    (raw $s:expr) => {
        $crate::Block::Equation($crate::Equation::raw($s))
    };
    ($l:tt => $e:expr) => {
        $crate::Block::Equation($crate::Equation::from($e).with_label($l))
    };
    ($e:expr) => {
        $crate::Block::Equation($crate::Equation::from($e))
    };
}

/// A data table, as a [`Block`](crate::Block).
///
/// Fields: `caption:`, `label:`, `headers:`, `rows:`, `from:` (an
/// `&tpt_omni::Table`). `headers`/`rows` accept literal arrays or any runtime
/// iterable of `Into<String>` values.
///
/// ```
/// # use tpt_doc::prelude::*;
/// let t = table! {
///     caption: "Counts",
///     headers: ["n", "n^2"],
///     rows: [["1", "1"], ["2", "4"]],
/// };
/// assert!(matches!(t, Block::Table(_)));
/// ```
#[macro_export]
macro_rules! table {
    ($($t:tt)*) => {{
        #[allow(unused_mut)]
        let mut __tpt_tab = $crate::DocTable::new();
        $crate::__table_fields!(__tpt_tab ; $($t)* ,);
        $crate::Block::Table(__tpt_tab)
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __table_fields {
    ($t:ident ; ) => {};
    ($t:ident ; , $($r:tt)*) => { $crate::__table_fields!($t ; $($r)*); };
    ($t:ident ; ; $($r:tt)*) => { $crate::__table_fields!($t ; $($r)*); };
    ($t:ident ; caption : $v:expr , $($r:tt)*) => {
        $t.set_caption($v); $crate::__table_fields!($t ; $($r)*);
    };
    ($t:ident ; label : $v:expr , $($r:tt)*) => {
        $t.set_label($v); $crate::__table_fields!($t ; $($r)*);
    };
    ($t:ident ; headers : [ $($h:expr),* $(,)? ] $($r:tt)*) => {
        $t.set_headers([$($h),*]); $crate::__table_fields!($t ; $($r)*);
    };
    ($t:ident ; headers : $e:expr , $($r:tt)*) => {
        $t.set_headers($e); $crate::__table_fields!($t ; $($r)*);
    };
    ($t:ident ; rows : [ $( [ $($c:expr),* $(,)? ] ),* $(,)? ] $($r:tt)*) => {
        $( $t.push_row([$($c),*]); )* $crate::__table_fields!($t ; $($r)*);
    };
    ($t:ident ; rows : $e:expr , $($r:tt)*) => {
        $t.set_rows($e); $crate::__table_fields!($t ; $($r)*);
    };
    ($t:ident ; from : $e:expr , $($r:tt)*) => {
        $t.load_omni($e); $crate::__table_fields!($t ; $($r)*);
    };
}

/// Inline BibTeX entries, producing a [`Bibliography`](crate::Bibliography).
///
/// Keys are identifiers and values are string expressions (this keeps the
/// entries inside Rust's tokenizer); use [`Bibliography::parse`] for arbitrary
/// BibTeX text.
///
/// ```
/// # use tpt_doc::prelude::*;
/// let b = bib! {
///     @article { einstein1905, title = "Zur Elektrodynamik", year = "1905" },
///     @book { knuth1984, title = "The TeXbook" },
/// };
/// assert_eq!(b.len(), 2);
/// assert!(b.contains("knuth1984"));
/// ```
#[macro_export]
macro_rules! bib {
    ($( @ $kind:ident { $key:ident $(, $field:ident = $val:expr )* $(,)? } ),* $(,)?) => {{
        #[allow(unused_mut)]
        let mut __tpt_bib = $crate::Bibliography::new();
        $(
            __tpt_bib.push(
                $crate::BibEntry::new(stringify!($kind), stringify!($key))
                $( .with(stringify!($field), $val) )*
            );
        )*
        __tpt_bib
    }};
}
