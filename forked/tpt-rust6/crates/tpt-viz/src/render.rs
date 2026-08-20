//! Software (CPU) renderer: turns a [`Plot`] into a self-contained SVG string.

use crate::frame::{Frame, MB, ML, MR, MT, TICKS};
use crate::geom::{ColorSpec, Heatmap, Histogram, Layer, Line, Quiver, Scatter};
use crate::lod::LodRender;
use crate::plot::Plot;
use crate::scale::{hex, Gradient, Linear, DEFAULT_COLOR};
use std::fmt::Write;

/// Render `plot` as a standalone SVG document.
pub fn render(plot: &Plot) -> String {
    let f = Frame::new(plot);
    let (w, h) = (plot.width, plot.height);

    let mut s = String::new();
    let _ = writeln!(
        s,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" \
         viewBox=\"0 0 {w} {h}\" font-family=\"sans-serif\">\n\
         <rect x=\"0\" y=\"0\" width=\"{w}\" height=\"{h}\" fill=\"white\"/>"
    );

    axes(&f, plot, &mut s);
    for layer in &plot.layers {
        match layer {
            Layer::Scatter(l) => scatter(l, &f, plot.gradient, &mut s),
            Layer::Line(l) => line(l, &f, &mut s),
            Layer::Histogram(l) => histogram(l, &f, &mut s),
            Layer::Heatmap(l) => heatmap(l, &f, plot.gradient, &mut s),
            Layer::Quiver(l) => quiver(l, &f, plot.gradient, &mut s),
        }
    }
    labels(plot, &mut s);

    s.push_str("</svg>\n");
    s
}

fn axes(f: &Frame, plot: &Plot, s: &mut String) {
    let (x0, y0) = (ML, MT);
    let (x1, y1) = (f.w - MR, f.h - MB);
    let _ = writeln!(
        s,
        "<rect x=\"{x0}\" y=\"{y0}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"#333\"/>",
        x1 - x0,
        y1 - y0
    );
    if !plot.axes {
        return;
    }
    for i in 0..=TICKS {
        let t = i as f64 / TICKS as f64;
        let xv = f.x.min + t * (f.x.max - f.x.min);
        let yv = f.y.min + t * (f.y.max - f.y.min);
        let (px, py) = (f.px(xv), f.py(yv));
        let _ = writeln!(
            s,
            "<line x1=\"{px:.2}\" y1=\"{y1}\" x2=\"{px:.2}\" y2=\"{}\" stroke=\"#999\"/>\n\
             <text x=\"{px:.2}\" y=\"{}\" text-anchor=\"middle\" font-size=\"10\">{}</text>\n\
             <line x1=\"{}\" y1=\"{py:.2}\" x2=\"{x0}\" y2=\"{py:.2}\" stroke=\"#999\"/>\n\
             <text x=\"{}\" y=\"{:.2}\" text-anchor=\"end\" font-size=\"10\">{}</text>",
            y1 + 4.0,
            y1 + 16.0,
            tick(xv),
            x0 - 4.0,
            x0 - 8.0,
            py + 3.0,
            tick(yv),
        );
    }
}

fn labels(plot: &Plot, s: &mut String) {
    if let Some(t) = &plot.title {
        let _ = writeln!(
            s,
            "<text x=\"{:.1}\" y=\"22\" text-anchor=\"middle\" font-size=\"16\">{}</text>",
            plot.width / 2.0,
            esc(t)
        );
    }
    if let Some(t) = &plot.xlabel {
        let _ = writeln!(
            s,
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"12\">{}</text>",
            plot.width / 2.0,
            plot.height - 12.0,
            esc(t)
        );
    }
    if let Some(t) = &plot.ylabel {
        let (x, y) = (16.0, plot.height / 2.0);
        let _ = writeln!(
            s,
            "<text x=\"{x:.1}\" y=\"{y:.1}\" text-anchor=\"middle\" font-size=\"12\" \
             transform=\"rotate(-90 {x:.1} {y:.1})\">{}</text>",
            esc(t)
        );
    }
}

fn scatter(l: &Scatter, f: &Frame, g: Gradient, s: &mut String) {
    match l.lod().render() {
        LodRender::Points(_) => {
            let dom = match &l.color {
                ColorSpec::Values(v) => Some(Linear::fit(v.iter().copied())),
                ColorSpec::Solid(_) => None,
            };
            for (i, (&x, &y)) in l.x.iter().zip(l.y.iter()).enumerate() {
                let fill = match (&l.color, &dom) {
                    (ColorSpec::Values(v), Some(d)) => g.hex(d.normalize(v[i])),
                    (ColorSpec::Solid(c), _) => hex(*c),
                    _ => hex(DEFAULT_COLOR),
                };
                let _ = writeln!(
                    s,
                    "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"{}\" fill=\"{fill}\" fill-opacity=\"{}\"/>",
                    f.px(x),
                    f.py(y),
                    l.size,
                    l.opacity
                );
            }
        }
        // Too many points to draw individually: paint the density grid.
        LodRender::Cells(cells) => {
            let max = cells.iter().map(|c| c.4).max().unwrap_or(1).max(1) as f64;
            for (x0, x1, y0, y1, n) in cells {
                // sqrt keeps sparse cells visible against dense ones.
                let fill = g.hex((n as f64 / max).sqrt());
                cell(f, (x0, x1, y0, y1), &fill, l.opacity.max(0.6), s);
            }
        }
    }
}

fn line(l: &Line, f: &Frame, s: &mut String) {
    if l.x.is_empty() {
        return;
    }
    let pts: Vec<String> =
        l.x.iter()
            .zip(l.y.iter())
            .map(|(&x, &y)| format!("{:.2},{:.2}", f.px(x), f.py(y)))
            .collect();
    let _ = writeln!(
        s,
        "<polyline points=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{}\" \
         stroke-opacity=\"{}\"/>",
        pts.join(" "),
        hex(l.stroke),
        l.width,
        l.opacity
    );
}

fn histogram(l: &Histogram, f: &Frame, s: &mut String) {
    let (edges, counts) = l.bin_edges_counts();
    let fill = hex(l.fill);
    for (e, &n) in edges.windows(2).zip(counts.iter()) {
        let (x, y, w, h) = f.rect((e[0], e[1], 0.0, n as f64));
        let _ = writeln!(
            s,
            "<rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{w:.2}\" height=\"{h:.2}\" fill=\"{fill}\" \
             fill-opacity=\"{}\" stroke=\"white\" stroke-width=\"0.5\"/>",
            l.opacity
        );
    }
}

fn heatmap(l: &Heatmap, f: &Frame, g: Gradient, s: &mut String) {
    let dom = l.grid.domain();
    let (nx, ny) = (l.grid.nx, l.grid.ny);
    let cw = (l.x_extent.1 - l.x_extent.0) / nx as f64;
    let ch = (l.y_extent.1 - l.y_extent.0) / ny as f64;
    for j in 0..ny {
        for i in 0..nx {
            let mut t = dom.normalize(l.grid.get(i, j));
            if let Some(k) = l.levels {
                // Quantize into `k` filled contour bands.
                t = ((t * k as f64).floor() / (k.max(2) - 1) as f64).min(1.0);
            }
            let x0 = l.x_extent.0 + i as f64 * cw;
            let y0 = l.y_extent.0 + j as f64 * ch;
            cell(f, (x0, x0 + cw, y0, y0 + ch), &g.hex(t), l.opacity, s);
        }
    }
}

/// Draw a vector-field layer as arrows from `(x,y)` along `scale*(u,v)`.
fn quiver(l: &Quiver, f: &Frame, g: Gradient, s: &mut String) {
    let dom = match &l.color {
        ColorSpec::Values(v) => Some(Linear::fit(v.iter().copied())),
        ColorSpec::Solid(_) => None,
    };
    let arrow = l.width.max(0.5) + 2.0;
    for i in 0..l.x.len() {
        let (x, y) = (l.x[i], l.y[i]);
        let (tx, ty) = (x + l.u[i] * l.scale, y + l.v[i] * l.scale);
        let (x0, y0) = (f.px(x), f.py(y));
        let (x1, y1) = (f.px(tx), f.py(ty));
        let stroke = match (&l.color, &dom) {
            (ColorSpec::Values(m), Some(d)) => g.hex(d.normalize(m[i])),
            (ColorSpec::Solid(c), _) => hex(*c),
            _ => hex(DEFAULT_COLOR),
        };
        // Arrow shaft plus a small head.
        let (dx, dy) = ((x1 - x0), (y1 - y0));
        let len = (dx * dx + dy * dy).sqrt();
        let (hx, hy) = if len > 1e-6 {
            (dx / len, dy / len)
        } else {
            (0.0, 0.0)
        };
        let (px, py) = (-hy, hx); // perpendicular
        let (bx, by) = (x1 - hx * arrow, y1 - hy * arrow);
        let _ = writeln!(
            s,
            "<line x1=\"{x0:.2}\" y1=\"{y0:.2}\" x2=\"{x1:.2}\" y2=\"{y1:.2}\" \
             stroke=\"{stroke}\" stroke-width=\"{}\" stroke-opacity=\"{}\"/>",
            l.width,
            l.opacity
        );
        if len > arrow {
            let _ = writeln!(
                s,
                "<polyline points=\"{x1:.2},{y1:.2} {:.2},{:.2} {:.2},{:.2}\" \
                 fill=\"{stroke}\" stroke=\"{stroke}\" stroke-width=\"{}\"/>",
                bx + px * arrow * 0.5,
                by + py * arrow * 0.5,
                bx - px * arrow * 0.5,
                by - py * arrow * 0.5,
                l.width
            );
        }
    }
}

/// Emit one filled cell for a data-space box `(x0, x1, y0, y1)`.
fn cell(f: &Frame, b: (f64, f64, f64, f64), fill: &str, op: f64, s: &mut String) {
    let (x, y, w, h) = f.rect(b);
    let _ = writeln!(
        s,
        "<rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{w:.2}\" height=\"{h:.2}\" fill=\"{fill}\" \
         fill-opacity=\"{op}\"/>"
    );
}

fn tick(v: f64) -> String {
    let s = format!("{v:.3}");
    let t = s.trim_end_matches('0').trim_end_matches('.');
    if t.is_empty() || t == "-" {
        "0".into()
    } else {
        t.into()
    }
}

/// Escape XML-significant characters in user-supplied text.
fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Flat `[x0, y0, x1, y1, ...]` pixel positions and `[r, g, b, a, ...]` colors
/// for a scatter layer, ready to be uploaded as GPU vertex buffers.
#[cfg(feature = "gpu")]
pub fn scatter_buffers(l: &Scatter, plot: &Plot) -> (Vec<f32>, Vec<f32>) {
    let f = Frame::new(plot);
    let dom = match &l.color {
        ColorSpec::Values(v) => Some(Linear::fit(v.iter().copied())),
        ColorSpec::Solid(_) => None,
    };
    let mut pos = Vec::with_capacity(l.x.len() * 2);
    let mut col = Vec::with_capacity(l.x.len() * 4);
    for (i, (&x, &y)) in l.x.iter().zip(l.y.iter()).enumerate() {
        pos.push(f.px(x) as f32);
        pos.push(f.py(y) as f32);
        let c: crate::scale::Rgb = match (&l.color, &dom) {
            (ColorSpec::Values(v), Some(d)) => plot.gradient.color(d.normalize(v[i])),
            (ColorSpec::Solid(c), _) => *c,
            _ => DEFAULT_COLOR,
        };
        col.extend_from_slice(&[
            c.0 as f32 / 255.0,
            c.1 as f32 / 255.0,
            c.2 as f32 / 255.0,
            l.opacity as f32,
        ]);
    }
    (pos, col)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_trims_trailing_zeros() {
        assert_eq!(tick(1.0), "1");
        assert_eq!(tick(0.5), "0.5");
        assert_eq!(tick(0.0), "0");
    }

    #[test]
    fn esc_escapes_xml() {
        assert_eq!(esc("a < b & \"c\""), "a &lt; b &amp; &quot;c&quot;");
    }
}
