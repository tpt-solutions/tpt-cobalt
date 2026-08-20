//! Text-only PDF backend (feature `pdf`).
//!
//! Reduced scope: the document is laid out as paginated plain text
//! ([`Document::to_text_lines`]) in the built-in Helvetica font; equations are
//! printed as their LaTeX source. HTML/LaTeX remain the full-fidelity outputs.
//!
//! With the `harfbuzz` feature, [`Document::to_pdf_shaped`] embeds a TrueType/
//! OpenType font and uses HarfBuzz-shaped advances to lay out proportional,
//! kerned lines.

use printpdf::{BuiltinFont, Mm, PdfDocument};

use crate::ast::Document;
use crate::error::DocError;

const PAGE_W: f32 = 210.0;
const PAGE_H: f32 = 297.0;
const MARGIN: f32 = 20.0;
const LEADING: f32 = 6.0;
const FONT_SIZE: f32 = 11.0;
/// Millimetres -> points.
const MM_TO_PT: f32 = 2.834_645_7;

impl Document {
    /// Render the document into PDF bytes.
    pub fn to_pdf_bytes(&self) -> Result<Vec<u8>, DocError> {
        let title = self.title.clone().unwrap_or_else(|| "Untitled".to_string());
        let (doc, page, layer) = PdfDocument::new(&title, Mm(PAGE_W), Mm(PAGE_H), "Layer 1");
        let font = doc
            .add_builtin_font(BuiltinFont::Helvetica)
            .map_err(|e| DocError::Io(e.to_string()))?;

        let per_page = ((PAGE_H - 2.0 * MARGIN) / LEADING) as usize;
        let lines = self.to_text_lines();
        let mut current = doc.get_page(page).get_layer(layer);
        let mut y = PAGE_H - MARGIN;
        for (i, line) in lines.iter().enumerate() {
            if i > 0 && i % per_page == 0 {
                let (p, l) = doc.add_page(Mm(PAGE_W), Mm(PAGE_H), "Layer 1");
                current = doc.get_page(p).get_layer(l);
                y = PAGE_H - MARGIN;
            }
            current.use_text(line.as_str(), FONT_SIZE, Mm(MARGIN), Mm(y), &font);
            y -= LEADING;
        }
        doc.save_to_bytes().map_err(|e| DocError::Io(e.to_string()))
    }

    /// Render the document to a PDF file.
    pub fn to_pdf(&self, path: impl AsRef<std::path::Path>) -> Result<(), DocError> {
        let bytes = self.to_pdf_bytes()?;
        std::fs::write(path, bytes).map_err(|e| DocError::Io(e.to_string()))
    }

    /// Render the document to PDF using an embedded TrueType/OpenType font and
    /// HarfBuzz-shaped (proportional, kerned) line measurement for wrapping.
    ///
    /// Glyph *painting* still flows through `printpdf`'s own font backend, so
    /// advanced OpenType substitution is applied at the layout/measurement
    /// stage; the embedded font is what gets drawn.
    #[cfg(feature = "harfbuzz")]
    pub fn to_pdf_shaped(&self, font_ttf: &[u8]) -> Result<Vec<u8>, DocError> {
        let title = self.title.clone().unwrap_or_else(|| "Untitled".to_string());
        let (doc, page, layer) = PdfDocument::new(&title, Mm(PAGE_W), Mm(PAGE_H), "Layer 1");
        let font = doc
            .add_external_font(font_ttf)
            .map_err(|e| DocError::Io(e.to_string()))?;

        let usable_pt = (PAGE_W - 2.0 * MARGIN) * MM_TO_PT;
        let per_page = ((PAGE_H - 2.0 * MARGIN) / LEADING) as usize;

        let mut physical: Vec<String> = Vec::new();
        for line in self.to_text_lines() {
            physical.extend(wrap_shaped(&line, font_ttf, FONT_SIZE, usable_pt));
        }

        let mut current = doc.get_page(page).get_layer(layer);
        let mut y = PAGE_H - MARGIN;
        for (i, line) in physical.iter().enumerate() {
            if i > 0 && i % per_page == 0 {
                let (p, l) = doc.add_page(Mm(PAGE_W), Mm(PAGE_H), "Layer 1");
                current = doc.get_page(p).get_layer(l);
                y = PAGE_H - MARGIN;
            }
            current.use_text(line.as_str(), FONT_SIZE, Mm(MARGIN), Mm(y), &font);
            y -= LEADING;
        }
        doc.save_to_bytes().map_err(|e| DocError::Io(e.to_string()))
    }

    /// Render the document to a PDF file using an embedded font (see
    /// [`Document::to_pdf_shaped`]).
    #[cfg(feature = "harfbuzz")]
    pub fn to_pdf_shaped_file(
        &self,
        font_ttf: &[u8],
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), DocError> {
        let bytes = self.to_pdf_shaped(font_ttf)?;
        std::fs::write(path, bytes).map_err(|e| DocError::Io(e.to_string()))
    }
}

/// Re-flow `line` into lines that fit `usable_pt` points, measuring each
/// candidate with HarfBuzz-shaped advances.
#[cfg(feature = "harfbuzz")]
fn wrap_shaped(line: &str, font: &[u8], size: f32, usable_pt: f32) -> Vec<String> {
    let width = crate::typography::line_width_pt(line, font, size).unwrap_or(f32::MAX);
    if width <= usable_pt {
        return vec![line.to_string()];
    }
    let mut out = Vec::new();
    let mut cur = String::new();
    for word in line.split(' ') {
        let candidate = if cur.is_empty() {
            word.to_string()
        } else {
            format!("{cur} {word}")
        };
        let w = crate::typography::line_width_pt(&candidate, font, size).unwrap_or(f32::MAX);
        if w <= usable_pt || cur.is_empty() {
            cur = candidate;
        } else {
            out.push(std::mem::take(&mut cur));
            cur = word.to_string();
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}
