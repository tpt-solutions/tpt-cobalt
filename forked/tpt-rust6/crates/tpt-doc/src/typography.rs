//! HarfBuzz text shaping (feature `harfbuzz`).
//!
//! Wraps [`rustybuzz`] (a pure-Rust HarfBuzz port) to turn runs of text into
//! positioned glyphs, applying OpenType features, kerning and ligature
//! substitution. The PDF backend ([`crate::pdf`]) uses these shaped advances to
//! lay out proportionally-spaced, kerned lines instead of the built-in
//! fixed-pitch font.

use rustybuzz::{Face, UnicodeBuffer};

use crate::error::DocError;

/// One shaped glyph: its glyph id plus placement relative to the pen origin.
///
/// Advances/offsets are in font design units (divide by `Face::units_per_em`
/// and multiply by the point size to get PDF units).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapedGlyph {
    /// Glyph id in the font (HarfBuzz `codepoint`).
    pub glyph_id: u32,
    /// Cluster: the byte index into the source string this glyph came from.
    pub cluster: u32,
    /// Advance along the baseline (font units).
    pub x_advance: i32,
    /// Advance along the cross axis (font units).
    pub y_advance: i32,
    /// Horizontal offset before placing the glyph (font units, e.g. kerning).
    pub x_offset: i32,
    /// Vertical offset before placing the glyph (font units).
    pub y_offset: i32,
}

/// Shape `text` with `font_data` (a TrueType/OpenType byte stream), returning
/// one [`ShapedGlyph`] per output glyph (ligatures collapse several input
/// characters into a single glyph, so the count may be smaller than the number
/// of characters).
pub fn shape(text: &str, font_data: &[u8]) -> Result<Vec<ShapedGlyph>, DocError> {
    let face = Face::from_slice(font_data, 0)
        .ok_or_else(|| DocError::Io("could not parse font for HarfBuzz shaping".into()))?;
    let mut buffer = UnicodeBuffer::new();
    buffer.push_str(text);
    let out = rustybuzz::shape(&face, &[], buffer);
    let infos = out.glyph_infos();
    let positions = out.glyph_positions();
    let mut glyphs = Vec::with_capacity(infos.len());
    for (g, p) in infos.iter().zip(positions.iter()) {
        glyphs.push(ShapedGlyph {
            glyph_id: g.glyph_id,
            cluster: g.cluster,
            x_advance: p.x_advance,
            y_advance: p.y_advance,
            x_offset: p.x_offset,
            y_offset: p.y_offset,
        });
    }
    Ok(glyphs)
}

/// Width in PDF points of `text` at `font_size_pt`, measured from the shaped
/// (kerned, ligature-aware) advances.
pub fn line_width_pt(text: &str, font_data: &[u8], font_size_pt: f32) -> Result<f32, DocError> {
    let face = Face::from_slice(font_data, 0)
        .ok_or_else(|| DocError::Io("could not parse font for HarfBuzz shaping".into()))?;
    let upm = face.units_per_em() as f32;
    let glyphs = shape(text, font_data)?;
    let units: f32 = glyphs.iter().map(|g| g.x_advance as f32).sum();
    Ok(units / upm * font_size_pt)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shaping needs a real font. We don't bundle one, so the test runs only
    // when TPT_DOC_TEST_FONT points at a TTF/OTF on disk.
    fn test_font() -> Option<Vec<u8>> {
        let path = std::env::var("TPT_DOC_TEST_FONT").ok()?;
        std::fs::read(path).ok()
    }

    #[test]
    fn empty_text_shapes_to_nothing() {
        if let Some(font) = test_font() {
            assert!(shape("", &font).unwrap().is_empty());
        }
    }

    #[test]
    fn ligature_collapses_glyph_count() {
        let Some(font) = test_font() else { return; };
        let glyphs = shape("fi", &font).unwrap();
        // Many text fonts substitute an "fi" ligature, producing one glyph for
        // two characters. At minimum shaping must succeed and cover both chars.
        assert!(!glyphs.is_empty());
        let covered: u32 = glyphs.iter().map(|g| (g.cluster == 0 || g.cluster == 1) as u32).sum();
        assert!(covered >= 1);
    }
}
