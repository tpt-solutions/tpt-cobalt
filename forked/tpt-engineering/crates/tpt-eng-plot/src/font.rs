//! Self-contained 5x7 bitmap font used to draw axis labels, titles, and tick numbers without
//! relying on any external font backend (which would pull in copyleft font dependencies).

use plotters::coord::Shift;
use plotters::drawing::DrawingArea;
use plotters::prelude::*;

/// Glyph definitions as 7 rows of 5 columns (`#` = lit).
const GLYPHS: &[(&str, [&str; 7])] = &[
    (
        "A",
        [
            " ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #",
        ],
    ),
    (
        "B",
        [
            "#### ", "#   #", "#### ", "#   #", "#   #", "#   #", "#### ",
        ],
    ),
    (
        "C",
        [
            " ####", "#    ", "#    ", "#    ", "#    ", "#    ", " ####",
        ],
    ),
    (
        "D",
        [
            "#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### ",
        ],
    ),
    (
        "E",
        [
            "#####", "#    ", "###  ", "#    ", "#    ", "#    ", "#####",
        ],
    ),
    (
        "F",
        [
            "#####", "#    ", "###  ", "#    ", "#    ", "#    ", "#    ",
        ],
    ),
    (
        "G",
        [
            " ####", "#    ", "#    ", "#  ##", "#   #", "#   #", " ####",
        ],
    ),
    (
        "H",
        [
            "#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #",
        ],
    ),
    (
        "I",
        [
            " ####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", " ####",
        ],
    ),
    (
        "J",
        [
            "  ###", "   # ", "   # ", "   # ", "#  # ", "#  # ", " ##  ",
        ],
    ),
    (
        "K",
        [
            "#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #",
        ],
    ),
    (
        "L",
        [
            "#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####",
        ],
    ),
    (
        "M",
        [
            "#   #", "## ##", "# # #", "# # #", "#   #", "#   #", "#   #",
        ],
    ),
    (
        "N",
        [
            "#   #", "##  #", "# # #", "# # #", "#  ##", "#   #", "#   #",
        ],
    ),
    (
        "O",
        [
            " ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### ",
        ],
    ),
    (
        "P",
        [
            "#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    ",
        ],
    ),
    (
        "Q",
        [
            " ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #",
        ],
    ),
    (
        "R",
        [
            "#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #",
        ],
    ),
    (
        "S",
        [
            " ####", "#    ", " ### ", "    #", "    #", "#   #", " ####",
        ],
    ),
    (
        "T",
        [
            "#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ",
        ],
    ),
    (
        "U",
        [
            "#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### ",
        ],
    ),
    (
        "V",
        [
            "#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  ",
        ],
    ),
    (
        "W",
        [
            "#   #", "#   #", "#   #", "# # #", "# # #", "## ##", "#   #",
        ],
    ),
    (
        "X",
        [
            "#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #",
        ],
    ),
    (
        "Y",
        [
            "#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  ",
        ],
    ),
    (
        "Z",
        [
            "#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####",
        ],
    ),
    (
        "0",
        [
            " ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### ",
        ],
    ),
    (
        "1",
        [
            "  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### ",
        ],
    ),
    (
        "2",
        [
            " ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####",
        ],
    ),
    (
        "3",
        [
            "#### ", "    #", "  ## ", "    #", "    #", "#   #", " ### ",
        ],
    ),
    (
        "4",
        [
            "   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # ",
        ],
    ),
    (
        "5",
        [
            "#####", "#    ", "#### ", "    #", "    #", "#   #", " ### ",
        ],
    ),
    (
        "6",
        [
            " ### ", "#    ", "#    ", "#### ", "#   #", "#   #", " ### ",
        ],
    ),
    (
        "7",
        [
            "#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   ",
        ],
    ),
    (
        "8",
        [
            " ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### ",
        ],
    ),
    (
        "9",
        [
            " ### ", "#   #", "#   #", " ####", "    #", "    #", " ### ",
        ],
    ),
    (
        " ",
        [
            "     ", "     ", "     ", "     ", "     ", "     ", "     ",
        ],
    ),
    (
        ".",
        [
            "     ", "     ", "     ", "     ", "     ", " ##  ", " ##  ",
        ],
    ),
    (
        ",",
        [
            "     ", "     ", "     ", "     ", "     ", " ##  ", " #   ",
        ],
    ),
    (
        "-",
        [
            "     ", "     ", "     ", "#####", "     ", "     ", "     ",
        ],
    ),
    (
        "+",
        [
            "     ", "  #  ", "  #  ", "#####", "  #  ", "  #  ", "     ",
        ],
    ),
    (
        "=",
        [
            "     ", "     ", "#####", "     ", "#####", "     ", "     ",
        ],
    ),
    (
        "/",
        [
            "    #", "   # ", "   # ", "  #  ", " #   ", " #   ", "#    ",
        ],
    ),
    (
        "(",
        [
            "  ## ", " #   ", "#    ", "#    ", "#    ", " #   ", "  ## ",
        ],
    ),
    (
        ")",
        [
            " ##  ", "   # ", "    #", "    #", "    #", "   # ", " ##  ",
        ],
    ),
    (
        ":",
        [
            "     ", " ##  ", " ##  ", "     ", " ##  ", " ##  ", "     ",
        ],
    ),
    (
        "%",
        [
            "#   #", "#  # ", "  #  ", "     ", "  #  ", " #  #", "#   #",
        ],
    ),
    (
        "[",
        [
            " ### ", " #   ", " #   ", " #   ", " #   ", " #   ", " ### ",
        ],
    ),
    (
        "]",
        [
            " ### ", "   # ", "   # ", "   # ", "   # ", "   # ", " ### ",
        ],
    ),
    (
        "_",
        [
            "     ", "     ", "     ", "     ", "     ", "     ", "#####",
        ],
    ),
    (
        "*",
        [
            "     ", "# # #", " # # ", "#####", " # # ", "# # #", "     ",
        ],
    ),
    (
        "<",
        [
            "   # ", "  #  ", " #   ", "#    ", " #   ", "  #  ", "   # ",
        ],
    ),
    (
        ">",
        [
            " #   ", "  #  ", "   # ", "    #", "   # ", "  #  ", " #   ",
        ],
    ),
    (
        "?",
        [
            " ### ", "#   #", "   # ", "  #  ", "     ", "     ", "     ",
        ],
    ),
    (
        "!",
        [
            "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "     ", "  #  ",
        ],
    ),
    (
        "'",
        [
            "  #  ", "  #  ", "     ", "     ", "     ", "     ", "     ",
        ],
    ),
    (
        "#",
        [
            "# # #", "     ", "# # #", "     ", "# # #", "     ", "# # #",
        ],
    ),
    (
        "&",
        [
            " # # ", "# #  ", " #   ", "  # #", " # # ", "#  # ", " # # ",
        ],
    ),
    (
        "@",
        [
            " ### ", "#   #", "# ## ", "# # #", "# ## ", "#    ", " ### ",
        ],
    ),
    (
        "^",
        [
            "  #  ", " # # ", "#   #", "     ", "     ", "     ", "     ",
        ],
    ),
    (
        ";",
        [
            "     ", " ##  ", " ##  ", "     ", " ##  ", " ##  ", " #   ",
        ],
    ),
    (
        "$",
        [
            "  #  ", " ####", "# #  ", " ### ", "  # #", "#### ", "  #  ",
        ],
    ),
];

/// Map a character to its 7x5 bitmap (lowercase letters reuse uppercase).
pub fn glyph(c: char) -> [u8; 7] {
    let key = c.to_ascii_uppercase().to_string();
    let entry = GLYPHS.iter().find(|(k, _)| *k == key);
    match entry {
        Some((_, rows)) => {
            let mut out = [0u8; 7];
            for (r, row) in rows.iter().enumerate() {
                let mut byte = 0u8;
                for (col, ch) in row.chars().enumerate() {
                    if ch == '#' {
                        byte |= 1 << (4 - col);
                    }
                }
                out[r] = byte;
            }
            out
        }
        None => [0u8; 7],
    }
}

/// Draw `text` onto a drawing area using the bitmap font, with the top-left at `(origin_x, origin_y)`
/// in backend pixel coordinates. Each font pixel is rendered as a `scale` x `scale` block.
pub fn draw_text<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    origin_x: i32,
    origin_y: i32,
    scale: u32,
    color: RGBColor,
    text: &str,
) {
    let mut x = origin_x;
    for c in text.chars() {
        let g = glyph(c);
        for (row, _) in g.iter().enumerate() {
            for col in 0..5i32 {
                if (g[row] >> (4 - col)) & 1 == 1 {
                    let px = x + col * scale as i32;
                    let py = origin_y + row as i32 * scale as i32;
                    for dx in 0..scale as i32 {
                        for dy in 0..scale as i32 {
                            let _ = area.draw_pixel((px + dx, py + dy), &color);
                        }
                    }
                }
            }
        }
        x += (5 + 1) * scale as i32;
    }
}

/// Draw a title and axis labels (with tick numbers) onto a plot, using the bitmap font.
///
/// `to_pixel` maps a data coordinate to a backend pixel coordinate (e.g. `chart.backend_coord`).
/// When `x_tick_labels` is `Some`, named labels are placed at bar centres instead of numeric ticks.
#[allow(clippy::too_many_arguments)]
pub fn annotate_axes<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    to_pixel: impl Fn(f64, f64) -> (i32, i32),
    width: u32,
    height: u32,
    title: &str,
    x_label: &str,
    y_label: &str,
    x_range: (f64, f64),
    y_range: (f64, f64),
    x_tick_labels: Option<&[String]>,
) {
    let scale = 2u32;
    let char_w = 6 * scale as i32;
    let char_h = 7 * scale as i32;

    // Title (centered near the top).
    let title_w = title.chars().count() as i32 * char_w;
    let title_x = ((width as i32 - title_w) / 2).max(4);
    draw_text(area, title_x, 6, scale, BLACK, title);

    // X-axis label (centered under the axis).
    let (xa_px, _) = to_pixel((x_range.0 + x_range.1) / 2.0, y_range.0);
    let xlab_w = x_label.chars().count() as i32 * char_w;
    let xlab_x = (xa_px - xlab_w / 2).max(4);
    draw_text(
        area,
        xlab_x,
        (height as i32 - 28).max(4),
        scale,
        BLACK,
        x_label,
    );

    // Y-axis label (stacked vertically, near the top of the y axis).
    let (_, ya_py) = to_pixel(x_range.0, (y_range.0 + y_range.1) / 2.0);
    let ylab_block = y_label.chars().count() as i32 * char_h;
    let ylab_start = (ya_py - ylab_block / 2).max(4);
    for (i, c) in y_label.chars().enumerate() {
        draw_text(
            area,
            6,
            ylab_start + i as i32 * char_h,
            scale,
            BLACK,
            &c.to_string(),
        );
    }

    // X tick labels (named for bar charts, numeric otherwise).
    match x_tick_labels {
        Some(labels) => {
            let len = labels.len().max(1) as f64;
            for (i, lab) in labels.iter().enumerate() {
                let v = x_range.0 + (i as f64 + 0.5) * (x_range.1 - x_range.0) / len;
                let (px, _) = to_pixel(v, y_range.0);
                let lw = lab.chars().count() as i32 * char_w;
                draw_text(
                    area,
                    (px - lw / 2).max(2),
                    (height as i32 - 16).max(2),
                    scale,
                    BLACK,
                    lab,
                );
            }
        }
        None => {
            for i in 0..=4 {
                let v = x_range.0 + (x_range.1 - x_range.0) * (i as f64) / 4.0;
                let (px, _) = to_pixel(v, y_range.0);
                let label = format!("{v:.2}");
                let lw = label.chars().count() as i32 * char_w;
                draw_text(
                    area,
                    (px - lw / 2).max(2),
                    (height as i32 - 16).max(2),
                    scale,
                    BLACK,
                    &label,
                );
            }
        }
    }

    // Y tick numbers.
    for i in 0..=4 {
        let v = y_range.0 + (y_range.1 - y_range.0) * (i as f64) / 4.0;
        let (_, py) = to_pixel(x_range.0, v);
        let label = format!("{v:.2}");
        draw_text(area, 4, (py - (char_h / 2)).max(2), scale, BLACK, &label);
    }
}
