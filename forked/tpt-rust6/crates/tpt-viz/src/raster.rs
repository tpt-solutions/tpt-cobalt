//! Software (CPU) raster backend: turns a [`Plot`] into an RGBA pixel buffer and
//! encodes it as a PNG — no external crates, so the default build stays
//! dependency-free and wasm-clean (the entry points are `cfg`-ed off on wasm).
//!
//! The PNG uses a zlib *stored* (uncompressed) stream plus the mandatory
//! Adler-32 checksum, which is a fully valid, if slightly larger, PNG.

use crate::frame::{Frame, MB, ML, MR, MT, TICKS};
use crate::geom::{ColorSpec, Heatmap, Histogram, Layer, Line, Quiver, Scatter};
use crate::lod::LodRender;
use crate::plot::Plot;
use crate::scale::{Gradient, Linear, Rgb, DEFAULT_COLOR};

/// An RGBA pixel buffer, row-major.
pub(crate) struct Raster {
    w: usize,
    h: usize,
    buf: Vec<u8>,
}

impl Raster {
    fn new(w: usize, h: usize) -> Self {
        Raster {
            w,
            h,
            buf: vec![255; w * h * 4],
        }
    }

    #[inline]
    fn put(&mut self, x: i64, y: i64, c: (u8, u8, u8, u8)) {
        if x < 0 || y < 0 || x >= self.w as i64 || y >= self.h as i64 {
            return;
        }
        let i = (y as usize * self.w + x as usize) * 4;
        let (sr, sg, sb, sa) = (c.0 as f64, c.1 as f64, c.2 as f64, c.3 as f64 / 255.0);
        let (dr, dg, db, da) = (
            self.buf[i] as f64,
            self.buf[i + 1] as f64,
            self.buf[i + 2] as f64,
            self.buf[i + 3] as f64 / 255.0,
        );
        let oa = sa + da * (1.0 - sa);
        if oa <= 0.0 {
            return;
        }
        self.buf[i] = (sr * sa + dr * da * (1.0 - sa)) as u8;
        self.buf[i + 1] = (sg * sa + dg * da * (1.0 - sa)) as u8;
        self.buf[i + 2] = (sb * sa + db * da * (1.0 - sa)) as u8;
        self.buf[i + 3] = (oa * 255.0) as u8;
    }

    fn fill_rect(&mut self, x: f64, y: f64, w: f64, h: f64, c: (u8, u8, u8, u8)) {
        let x0 = x.floor() as i64;
        let y0 = y.floor() as i64;
        let x1 = (x + w).ceil() as i64;
        let y1 = (y + h).ceil() as i64;
        for py in y0..y1 {
            for px in x0..x1 {
                self.put(px, py, c);
            }
        }
    }

    fn stroke_line(&mut self, mut x0: i64, mut y0: i64, x1: i64, y1: i64, c: (u8, u8, u8, u8)) {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            self.put(x0, y0, c);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
    }

    fn fill_circle(&mut self, cx: f64, cy: f64, r: f64, c: (u8, u8, u8, u8)) {
        let rr = r * r;
        let x0 = (cx - r).floor() as i64;
        let y0 = (cy - r).floor() as i64;
        let x1 = (cx + r).ceil() as i64;
        let y1 = (cy + r).ceil() as i64;
        for py in y0..y1 {
            for px in x0..x1 {
                let ddx = px as f64 - cx;
                let ddy = py as f64 - cy;
                if ddx * ddx + ddy * ddy <= rr {
                    self.put(px, py, c);
                }
            }
        }
    }
}

/// Render `plot` to an RGBA pixel buffer (`width` x `height`).
pub(crate) fn render_raster(plot: &Plot) -> Vec<u8> {
    let (w, h) = (plot.width as usize, plot.height as usize);
    let mut r = Raster::new(w, h);
    let f = Frame::new(plot);

    // Axis border.
    let (ax0, ay0) = (ML as i64, MT as i64);
    let (ax1, ay1) = (f.w as i64 - MR as i64, f.h as i64 - MB as i64);
    let axis = (0x33, 0x33, 0x33, 255);
    r.stroke_line(ax0, ay0, ax1, ay0, axis);
    r.stroke_line(ax0, ay1, ax1, ay1, axis);
    r.stroke_line(ax0, ay0, ax0, ay1, axis);
    r.stroke_line(ax1, ay0, ax1, ay1, axis);

    if plot.axes {
        let tick = (0x99, 0x99, 0x99, 255);
        for i in 0..=TICKS {
            let t = i as f64 / TICKS as f64;
            let xv = f.x.min + t * (f.x.max - f.x.min);
            let yv = f.y.min + t * (f.y.max - f.y.min);
            let (px, py) = (f.px(xv) as i64, f.py(yv) as i64);
            r.stroke_line(px, ay1, px, ay1 + 4, tick);
            r.stroke_line(ax0, py, ax0 - 4, py, tick);
        }
    }

    for layer in &plot.layers {
        match layer {
            Layer::Scatter(l) => scatter(l, &f, plot.gradient, &mut r),
            Layer::Line(l) => line(l, &f, &mut r),
            Layer::Histogram(l) => histogram(l, &f, &mut r),
            Layer::Heatmap(l) => heatmap(l, &f, plot.gradient, &mut r),
            Layer::Quiver(l) => quiver(l, &f, plot.gradient, &mut r),
        }
    }
    r.buf
}

fn color_of(c: ColorSpec, dom: Option<Linear>, idx: usize, g: Gradient, solid: Rgb) -> (u8, u8, u8, u8) {
    let rgb = match (&c, dom) {
        (ColorSpec::Values(v), Some(d)) => g.color(d.normalize(v[idx])),
        (ColorSpec::Solid(s), _) => *s,
        _ => solid,
    };
    (rgb.0, rgb.1, rgb.2, 255)
}

fn scatter(l: &Scatter, f: &Frame, g: Gradient, r: &mut Raster) {
    let dom = match &l.color {
        ColorSpec::Values(v) => Some(Linear::fit(v.iter().copied())),
        ColorSpec::Solid(_) => None,
    };
    match l.lod().render() {
        LodRender::Points(_) => {
            for i in 0..l.x.len() {
                let (cx, cy) = (f.px(l.x[i]), f.py(l.y[i]));
                let (cr, cg, cb, _) = color_of(l.color.clone(), dom, i, g, DEFAULT_COLOR);
                r.fill_circle(cx, cy, l.size, (cr, cg, cb, (l.opacity * 255.0) as u8));
            }
        }
        LodRender::Cells(cells) => {
            let max = cells.iter().map(|c| c.4).max().unwrap_or(1).max(1) as f64;
            for (x0, x1, y0, y1, n) in cells {
                let (cr, cg, cb) = g.color((n as f64 / max).sqrt());
                r.fill_rect(
                    f.px(x0),
                    f.py(y1),
                    f.px(x1) - f.px(x0),
                    f.py(y0) - f.py(y1),
                    (cr, cg, cb, (l.opacity.max(0.6) * 255.0) as u8),
                );
            }
        }
    }
}

fn line(l: &Line, f: &Frame, r: &mut Raster) {
    for w in l.x.windows(2).zip(l.y.windows(2)) {
        let (xa, xb) = (w.0[0], w.0[1]);
        let (ya, yb) = (w.1[0], w.1[1]);
        let (cx, cy, cb) = (l.stroke.0, l.stroke.1, l.stroke.2);
        r.stroke_line(
            f.px(xa) as i64,
            f.py(ya) as i64,
            f.px(xb) as i64,
            f.py(yb) as i64,
            (cx, cy, cb, (l.opacity * 255.0) as u8),
        );
    }
}

fn histogram(l: &Histogram, f: &Frame, r: &mut Raster) {
    let (edges, counts) = l.bin_edges_counts();
    let (cr, cg, cb) = (l.fill.0, l.fill.1, l.fill.2);
    for (e, &n) in edges.windows(2).zip(counts.iter()) {
        let (x, y, w, h) = f.rect((e[0], e[1], 0.0, n as f64));
        r.fill_rect(x, y, w, h, (cr, cg, cb, (l.opacity * 255.0) as u8));
    }
}

fn heatmap(l: &Heatmap, f: &Frame, g: Gradient, r: &mut Raster) {
    let dom = l.grid.domain();
    let (nx, ny) = (l.grid.nx, l.grid.ny);
    let cw = (l.x_extent.1 - l.x_extent.0) / nx as f64;
    let ch = (l.y_extent.1 - l.y_extent.0) / ny as f64;
    for j in 0..ny {
        for i in 0..nx {
            let mut t = dom.normalize(l.grid.get(i, j));
            if let Some(k) = l.levels {
                t = ((t * k as f64).floor() / (k.max(2) - 1) as f64).min(1.0);
            }
            let (cr, cg, cb) = g.color(t);
            let x0 = l.x_extent.0 + i as f64 * cw;
            let y0 = l.y_extent.0 + j as f64 * ch;
            let (px, py, pw, ph) = f.rect((x0, x0 + cw, y0, y0 + ch));
            r.fill_rect(px, py, pw, ph, (cr, cg, cb, (l.opacity * 255.0) as u8));
        }
    }
}

fn quiver(l: &Quiver, f: &Frame, g: Gradient, r: &mut Raster) {
    let dom = match &l.color {
        ColorSpec::Values(v) => Some(Linear::fit(v.iter().copied())),
        ColorSpec::Solid(_) => None,
    };
    let head = l.width.max(0.5) + 2.0;
    for i in 0..l.x.len() {
        let (x, y) = (l.x[i], l.y[i]);
        let (tx, ty) = (x + l.u[i] * l.scale, y + l.v[i] * l.scale);
        let (x0, y0) = (f.px(x) as i64, f.py(y) as i64);
        let (x1, y1) = (f.px(tx) as i64, f.py(ty) as i64);
        let (cr, cg, cb, _) = color_of(l.color.clone(), dom, i, g, DEFAULT_COLOR);
        let (dx, dy) = ((x1 - x0) as f64, (y1 - y0) as f64);
        let len = (dx * dx + dy * dy).sqrt();
        let (hx, hy) = if len > 1e-6 { (dx / len, dy / len) } else { (0.0, 0.0) };
        r.stroke_line(x0, y0, x1, y1, (cr, cg, cb, (l.opacity * 255.0) as u8));
        if len > head {
            let (bx, by) = (x1 as f64 - hx * head, y1 as f64 - hy * head);
            let (px, py) = (-hy, hx);
            for k in 0..=head as i64 {
                let ax = bx + px * head * 0.5 * (k as f64 / head);
                let ay = by + py * head * 0.5 * (k as f64 / head);
                r.put(ax as i64, ay as i64, (cr, cg, cb, (l.opacity * 255.0) as u8));
            }
            for k in 0..=head as i64 {
                let ax = bx - px * head * 0.5 * (k as f64 / head);
                let ay = by - py * head * 0.5 * (k as f64 / head);
                r.put(ax as i64, ay as i64, (cr, cg, cb, (l.opacity * 255.0) as u8));
            }
        }
    }
}

/// Encode an RGBA buffer as a PNG (zlib *stored* stream + Adler-32).
pub(crate) fn encode_png(w: usize, h: usize, rgba: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

    // IHDR
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit, RGBA, no interlace
    chunk(&mut out, b"IHDR", &ihdr);

    // IDAT: zlib stored stream of raw scanlines (each prefixed with filter byte 0).
    let stride = w * 4;
    let mut raw = Vec::with_capacity((stride + 1) * h);
    for y in 0..h {
        raw.push(0);
        raw.extend_from_slice(&rgba[y * stride..(y + 1) * stride]);
    }
    let mut idat = Vec::new();
    idat.push(0x78);
    idat.push(0x01); // zlib: deflate, stored
    let adler = adler32(&raw);
    // store blocks of at most 65535 bytes
    let mut off = 0;
    while off < raw.len() {
        let last = off + 65535 >= raw.len();
        let n = (raw.len() - off).min(65535);
        idat.push(if last { 0x01 } else { 0x00 }); // BFINAL | BTYPE=00
        idat.extend_from_slice(&(n as u16).to_le_bytes());
        idat.extend_from_slice(&(!n as u16).to_le_bytes());
        idat.extend_from_slice(&raw[off..off + n]);
        off += n;
    }
    idat.extend_from_slice(&adler.to_be_bytes());
    chunk(&mut out, b"IDAT", &idat);

    // IEND
    chunk(&mut out, b"IEND", &[]);
    out
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &d in data {
        a = (a + d as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}
