//! EPUB 3 backend (feature `epub`).
//!
//! The EPUB is a thin wrapper around the existing HTML renderer: every
//! top-level [`Block::Section`] becomes one XHTML chapter rendered with the
//! same writer [`Document::to_html`] uses ([`html_block_fragment`]), plus an
//! optional front-matter page (title/authors/date/abstract) and an optional
//! references page. The package is a zip archive laid out as
//!
//! ```text
//! mimetype                 (stored, must be the first entry)
//! META-INF/container.xml
//! OEBPS/content.opf
//! OEBPS/nav.xhtml
//! OEBPS/title.xhtml        (only when there is front matter)
//! OEBPS/chapterN.xhtml     (one per top-level section)
//! OEBPS/references.xhtml   (only when the bibliography is non-empty)
//! ```
//!
//! Equations keep their `\[...\]` LaTeX source inside
//! `<span class="math">`; reading systems with a MathJax/KaTeX pipeline can
//! typeset them, others show the source.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::{Cursor, Write as _};

use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

use crate::ast::{Block, Document};
use crate::error::DocError;
use crate::render::{escape_html, html_block_fragment};

const MIMETYPE: &str = "application/epub+zip";
const XHTML_MEDIA: &str = "application/xhtml+xml";

/// One XHTML file in the spine.
struct Chapter {
    /// File name inside `OEBPS/`.
    file: String,
    /// Manifest/spine id.
    id: String,
    /// Title, used by the navigation document.
    title: String,
    /// HTML fragment that goes inside `<body>`.
    body: String,
}

impl Document {
    /// Render the document into EPUB 3 bytes.
    ///
    /// ```
    /// # use tpt_doc::prelude::*;
    /// let paper = doc! {
    ///     title: "Squares",
    ///     section("Intro") { p("Hello."); },
    /// };
    /// let bytes = paper.to_epub().unwrap();
    /// assert_eq!(&bytes[..4], b"PK\x03\x04");
    /// ```
    pub fn to_epub(&self) -> Result<Vec<u8>, DocError> {
        let chapters = self.epub_chapters();
        let uid = format!("urn:uuid:{}", stable_uuid(&self.uuid_seed(&chapters)));

        let mut cursor = Cursor::new(Vec::new());
        {
            let mut zip = ZipWriter::new(&mut cursor);
            let stored = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
            let deflated =
                SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);

            // The `mimetype` entry must come first and be uncompressed.
            zip.start_file("mimetype", stored).map_err(zip_err)?;
            zip.write_all(MIMETYPE.as_bytes()).map_err(io_err)?;

            for (name, contents) in [
                ("META-INF/container.xml", container_xml()),
                ("OEBPS/content.opf", self.content_opf(&chapters, &uid)),
                ("OEBPS/nav.xhtml", nav_xhtml(&chapters)),
            ] {
                zip.start_file(name, deflated).map_err(zip_err)?;
                zip.write_all(contents.as_bytes()).map_err(io_err)?;
            }

            for ch in &chapters {
                zip.start_file(format!("OEBPS/{}", ch.file), deflated)
                    .map_err(zip_err)?;
                zip.write_all(xhtml_page(&ch.title, &ch.body).as_bytes())
                    .map_err(io_err)?;
            }

            zip.finish().map_err(zip_err)?;
        }
        Ok(cursor.into_inner())
    }

    /// Render the document to an `.epub` file.
    pub fn write_epub(&self, path: impl AsRef<std::path::Path>) -> Result<(), DocError> {
        let bytes = self.to_epub()?;
        std::fs::write(path, bytes).map_err(io_err)
    }

    // ------------------------------------------------------------ chapters

    /// Split the document into spine documents, preserving block order.
    fn epub_chapters(&self) -> Vec<Chapter> {
        let mut chapters: Vec<Chapter> = Vec::new();

        // Front matter: title block + abstract.
        let mut front = String::new();
        if let Some(t) = &self.title {
            let _ = writeln!(front, "<h1 class=\"title\">{}</h1>", escape_html(t));
        }
        if !self.authors.is_empty() {
            let authors: Vec<String> = self.authors.iter().map(|a| escape_html(a)).collect();
            let _ = writeln!(front, "<p class=\"authors\">{}</p>", authors.join(", "));
        }
        if let Some(d) = &self.date {
            let _ = writeln!(front, "<p class=\"date\">{}</p>", escape_html(d));
        }
        if let Some(a) = &self.summary {
            let _ = writeln!(
                front,
                "<div class=\"abstract\">\n<h2>Abstract</h2>\n<p>{}</p>\n</div>",
                escape_html(a)
            );
        }
        if !front.is_empty() {
            chapters.push(Chapter {
                file: "title.xhtml".to_string(),
                id: "titlepage".to_string(),
                title: self.title.clone().unwrap_or_else(|| "Title".to_string()),
                body: front,
            });
        }

        // One chapter per top-level section; loose blocks stay with whatever
        // came before them (or open a "Front matter" chapter of their own).
        let mut n = 0usize;
        for b in &self.blocks {
            match b {
                Block::Section(sec) => {
                    n += 1;
                    chapters.push(Chapter {
                        file: format!("chapter{n}.xhtml"),
                        id: format!("chapter{n}"),
                        title: sec.title.clone(),
                        body: html_block_fragment(b, 1),
                    });
                }
                other => {
                    let frag = html_block_fragment(other, 1);
                    match chapters.last_mut() {
                        Some(c) => c.body.push_str(&frag),
                        None => chapters.push(Chapter {
                            file: "title.xhtml".to_string(),
                            id: "titlepage".to_string(),
                            title: "Front matter".to_string(),
                            body: frag,
                        }),
                    }
                }
            }
        }

        // References.
        if !self.bibliography.is_empty() {
            let mut body = String::new();
            body.push_str("<h1 class=\"references\">References</h1>\n<ol class=\"references\">\n");
            for e in self.bibliography.entries() {
                let _ = writeln!(
                    body,
                    "<li id=\"bib-{}\">{}</li>",
                    escape_html(&e.key),
                    escape_html(&e.format())
                );
            }
            body.push_str("</ol>\n");
            chapters.push(Chapter {
                file: "references.xhtml".to_string(),
                id: "references".to_string(),
                title: "References".to_string(),
                body,
            });
        }

        // `cite`/`ref` links are same-page anchors in the single-file HTML
        // output; across chapters they need the target file name.
        let mut targets: BTreeMap<String, String> = BTreeMap::new();
        for ch in &chapters {
            collect_ids(&ch.body, &ch.file, &mut targets);
        }
        for ch in &mut chapters {
            ch.body = rewrite_hrefs(&ch.body, &ch.file, &targets);
        }
        chapters
    }

    // ------------------------------------------------------------ packaging

    fn uuid_seed(&self, chapters: &[Chapter]) -> String {
        let mut seed = String::new();
        seed.push_str(self.title.as_deref().unwrap_or("Untitled"));
        for a in &self.authors {
            seed.push('\u{1f}');
            seed.push_str(a);
        }
        seed.push('\u{1f}');
        seed.push_str(self.date.as_deref().unwrap_or(""));
        for ch in chapters {
            seed.push('\u{1f}');
            seed.push_str(&ch.title);
        }
        seed
    }

    fn content_opf(&self, chapters: &[Chapter], uid: &str) -> String {
        let title = self.title.as_deref().unwrap_or("Untitled");
        let mut s = String::new();
        s.push_str("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
        s.push_str(
            "<package xmlns=\"http://www.idpf.org/2007/opf\" version=\"3.0\" \
             unique-identifier=\"bookid\" xml:lang=\"en\">\n",
        );
        s.push_str("<metadata xmlns:dc=\"http://purl.org/dc/elements/1.1/\">\n");
        let _ = writeln!(s, "<dc:identifier id=\"bookid\">{uid}</dc:identifier>");
        let _ = writeln!(s, "<dc:title>{}</dc:title>", escape_html(title));
        s.push_str("<dc:language>en</dc:language>\n");
        for a in &self.authors {
            let _ = writeln!(s, "<dc:creator>{}</dc:creator>", escape_html(a));
        }
        if let Some(d) = self.date.as_deref().filter(|d| is_iso_date(d)) {
            let _ = writeln!(s, "<dc:date>{}</dc:date>", escape_html(d));
        }
        let _ = writeln!(
            s,
            "<meta property=\"dcterms:modified\">{}</meta>",
            utc_now()
        );
        s.push_str("</metadata>\n<manifest>\n");
        s.push_str(
            "<item id=\"nav\" href=\"nav.xhtml\" media-type=\"application/xhtml+xml\" \
             properties=\"nav\"/>\n",
        );
        for ch in chapters {
            let _ = writeln!(
                s,
                "<item id=\"{}\" href=\"{}\" media-type=\"{XHTML_MEDIA}\"/>",
                escape_html(&ch.id),
                escape_html(&ch.file)
            );
        }
        s.push_str("</manifest>\n<spine>\n");
        for ch in chapters {
            let _ = writeln!(s, "<itemref idref=\"{}\"/>", escape_html(&ch.id));
        }
        s.push_str("</spine>\n</package>\n");
        s
    }
}

fn container_xml() -> String {
    "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n\
     <container version=\"1.0\" xmlns=\"urn:oasis:names:tc:opendocument:xmlns:container\">\n\
     <rootfiles>\n\
     <rootfile full-path=\"OEBPS/content.opf\" media-type=\"application/oebps-package+xml\"/>\n\
     </rootfiles>\n\
     </container>\n"
        .to_string()
}

fn nav_xhtml(chapters: &[Chapter]) -> String {
    let mut body = String::new();
    body.push_str("<nav epub:type=\"toc\" id=\"toc\">\n<h1>Contents</h1>\n<ol>\n");
    for ch in chapters {
        let _ = writeln!(
            body,
            "<li><a href=\"{}\">{}</a></li>",
            escape_html(&ch.file),
            escape_html(&ch.title)
        );
    }
    body.push_str("</ol>\n</nav>\n");
    xhtml_page("Contents", &body)
}

/// Wrap an HTML fragment in a minimal, well-formed XHTML document.
fn xhtml_page(title: &str, body: &str) -> String {
    let mut s = String::with_capacity(body.len() + 512);
    s.push_str("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
    s.push_str("<!DOCTYPE html>\n");
    s.push_str(
        "<html xmlns=\"http://www.w3.org/1999/xhtml\" \
         xmlns:epub=\"http://www.idpf.org/2007/ops\" xml:lang=\"en\" lang=\"en\">\n",
    );
    s.push_str("<head>\n<meta charset=\"utf-8\"/>\n");
    let _ = writeln!(s, "<title>{}</title>", escape_html(title));
    s.push_str("</head>\n<body>\n");
    s.push_str(body);
    if !body.ends_with('\n') {
        s.push('\n');
    }
    s.push_str("</body>\n</html>\n");
    s
}

/// Record every `id="..."` in `body` as living in `file`.
fn collect_ids(body: &str, file: &str, out: &mut BTreeMap<String, String>) {
    let mut rest = body;
    while let Some(p) = rest.find("id=\"") {
        rest = &rest[p + 4..];
        let Some(end) = rest.find('"') else { break };
        out.entry(rest[..end].to_string())
            .or_insert_with(|| file.to_string());
        rest = &rest[end + 1..];
    }
}

/// Turn `href="#frag"` into `href="other.xhtml#frag"` when the anchor lives in
/// a different chapter. Unknown fragments are left untouched.
fn rewrite_hrefs(body: &str, file: &str, targets: &BTreeMap<String, String>) -> String {
    const PAT: &str = "href=\"#";
    let mut out = String::with_capacity(body.len());
    let mut rest = body;
    while let Some(p) = rest.find(PAT) {
        out.push_str(&rest[..p]);
        rest = &rest[p + PAT.len()..];
        let Some(end) = rest.find('"') else {
            out.push_str(PAT);
            break;
        };
        let frag = &rest[..end];
        match targets.get(frag) {
            Some(target) if target != file => {
                let _ = write!(out, "href=\"{target}#{frag}\"");
            }
            _ => {
                let _ = write!(out, "href=\"#{frag}\"");
            }
        }
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    out
}

fn is_iso_date(d: &str) -> bool {
    let b = d.as_bytes();
    matches!(b.len(), 4 | 7 | 10)
        && b.iter().enumerate().all(|(i, c)| {
            if matches!(i, 4 | 7) {
                *c == b'-'
            } else {
                c.is_ascii_digit()
            }
        })
}

fn zip_err(e: zip::result::ZipError) -> DocError {
    DocError::Epub(e.to_string())
}

fn io_err(e: std::io::Error) -> DocError {
    DocError::Io(e.to_string())
}

// ------------------------------------------------------------------ time

/// `dcterms:modified` needs a `YYYY-MM-DDTHH:MM:SSZ` UTC timestamp; doing the
/// arithmetic here avoids pulling in a date/time dependency.
fn utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    let (y, m, d) = civil_from_days(days);
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Days since the Unix epoch to a proleptic-Gregorian date (Hinnant's algorithm).
fn civil_from_days(z: i64) -> (i64, u64, u64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

// ------------------------------------------------------------------ uuid

/// A deterministic RFC-4122-shaped identifier derived from the document, so
/// re-exporting the same document keeps the same `dc:identifier`.
fn stable_uuid(seed: &str) -> String {
    let hi = fnv1a64(seed.as_bytes());
    let lo = fnv1a64(&hi.to_le_bytes());
    let hi = (hi & 0xffff_ffff_ffff_0fff) | 0x0000_0000_0000_4000; // version 4
    let lo = (lo & 0x3fff_ffff_ffff_ffff) | 0x8000_0000_0000_0000; // variant 1
    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        hi >> 32,
        (hi >> 16) & 0xffff,
        hi & 0xffff,
        lo >> 48,
        lo & 0xffff_ffff_ffff
    )
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(all(test, feature = "epub"))]
mod tests {
    use crate::prelude::*;
    use std::io::Read as _;

    fn sample() -> Document {
        doc! {
            title: "A Very Short Paper",
            authors: ["Ada Lovelace"],
            date: "2026-01-01",
            abstract: { "We square {} things.", 1 },
            section("sec:intro" => "Introduction") {
                p("Squaring is useful, as shown by Knuth.");
                cite("knuth1984");
                equation! { "eq:sq" => "x^2" };
            },
            section("Discussion") {
                p("See the equation above.");
                ref("eq:sq");
            },
            bib! { @book { knuth1984, title = "The TeXbook", author = "D. E. Knuth", year = "1984" } },
        }
    }

    #[test]
    fn epub_is_a_zip_starting_with_a_stored_mimetype() {
        let bytes = sample().to_epub().unwrap();
        assert_eq!(&bytes[..4], b"PK\x03\x04", "missing zip local-file header");

        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();

        // `mimetype` must be the first entry, stored uncompressed.
        {
            let first = zip.by_index(0).unwrap();
            assert_eq!(first.name(), "mimetype");
            assert_eq!(first.compression(), zip::CompressionMethod::Stored);
        }
        let mut s = String::new();
        zip.by_name("mimetype")
            .unwrap()
            .read_to_string(&mut s)
            .unwrap();
        assert_eq!(s, "application/epub+zip");
    }

    #[test]
    fn epub_contains_the_expected_package_entries() {
        let bytes = sample().to_epub().unwrap();
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
        let names: Vec<String> = zip.file_names().map(str::to_string).collect();
        for want in [
            "mimetype",
            "META-INF/container.xml",
            "OEBPS/content.opf",
            "OEBPS/nav.xhtml",
            "OEBPS/title.xhtml",
            "OEBPS/chapter1.xhtml",
            "OEBPS/chapter2.xhtml",
            "OEBPS/references.xhtml",
        ] {
            assert!(
                names.contains(&want.to_string()),
                "missing {want}: {names:?}"
            );
        }

        let read = |zip: &mut zip::ZipArchive<std::io::Cursor<Vec<u8>>>, name: &str| {
            let mut s = String::new();
            zip.by_name(name).unwrap().read_to_string(&mut s).unwrap();
            s
        };

        let container = read(&mut zip, "META-INF/container.xml");
        assert!(container.contains("full-path=\"OEBPS/content.opf\""));

        let opf = read(&mut zip, "OEBPS/content.opf");
        assert!(opf.contains("<dc:title>A Very Short Paper</dc:title>"));
        assert!(opf.contains("<dc:creator>Ada Lovelace</dc:creator>"));
        assert!(opf.contains("properties=\"nav\""));
        assert!(opf.contains("<itemref idref=\"chapter1\"/>"));
        assert!(opf.contains("dcterms:modified"));

        let nav = read(&mut zip, "OEBPS/nav.xhtml");
        assert!(nav.contains("epub:type=\"toc\""));
        assert!(nav.contains("<a href=\"chapter1.xhtml\">Introduction</a>"));
        assert!(nav.contains("<a href=\"chapter2.xhtml\">Discussion</a>"));

        let ch1 = read(&mut zip, "OEBPS/chapter1.xhtml");
        assert!(ch1.starts_with("<?xml version=\"1.0\" encoding=\"utf-8\"?>"));
        assert!(ch1.contains("xmlns=\"http://www.w3.org/1999/xhtml\""));
        assert!(ch1.contains("<h1>Introduction</h1>"));
        assert!(ch1.contains("<span class=\"math\">"));
        // The citation anchor lives in references.xhtml.
        assert!(ch1.contains("href=\"references.xhtml#bib-knuth1984\""));

        // ...and the cross-chapter `ref("eq:sq")` points back at chapter 1.
        let ch2 = read(&mut zip, "OEBPS/chapter2.xhtml");
        assert!(ch2.contains("href=\"chapter1.xhtml#eq:sq\""));
    }

    #[test]
    fn to_html_and_to_latex_are_untouched() {
        let d = sample();
        assert!(d.to_html().starts_with("<!DOCTYPE html>"));
        assert!(d.to_latex().contains("\\begin{equation}"));
    }

    #[test]
    fn write_epub_round_trips_to_disk() {
        let path = std::env::temp_dir().join("tpt-doc-epub-test.epub");
        sample().write_epub(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(&bytes[..4], b"PK\x03\x04");
    }
}
