//! Standalone 3D plotting: project `(x, y, z)` points to 2D with a simple
//! isometric camera and render as SVG. This is the "3D volume / surface"
//! deliverable of the grammar-of-graphics roadmap. It is a self-contained
//! renderer (no GPU, no external crates) so it stays wasm-clean, exactly like
//! the rest of `tpt-viz`.

use crate::scale::{hex, Gradient, Linear};

/// A 3D point cloud.
#[derive(Clone, Debug)]
pub struct Scatter3D {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub size: f64,
    pub opacity: f64,
}

impl Scatter3D {
    /// Build from three equal-length coordinate sequences.
    pub fn new(x: impl AsRef<[f64]>, y: impl AsRef<[f64]>, z: impl AsRef<[f64]>) -> Self {
        let (x, y, z) = (
            x.as_ref().to_vec(),
            y.as_ref().to_vec(),
            z.as_ref().to_vec(),
        );
        assert_eq!(x.len(), y.len(), "Scatter3D: x/y length mismatch");
        assert_eq!(x.len(), z.len(), "Scatter3D: x/z length mismatch");
        Self {
            x,
            y,
            z,
            size: 3.0,
            opacity: 1.0,
        }
    }

    /// Marker radius in pixels.
    pub fn size(mut self, px: f64) -> Self {
        self.size = px.max(0.0);
        self
    }
    /// Marker opacity.
    pub fn opacity(mut self, alpha: f64) -> Self {
        self.opacity = alpha.clamp(0.0, 1.0);
        self
    }

    fn points(&self) -> impl Iterator<Item = [f64; 3]> + '_ {
        (0..self.x.len()).map(move |i| [self.x[i], self.y[i], self.z[i]])
    }
}

/// A height surface `z = f(x, y)` sampled on a regular `(nx, ny)` grid. `z` is
/// row-major with index `i * ny + j` (x index `i`, y index `j`).
#[derive(Clone, Debug)]
pub struct Surface3D {
    pub nx: usize,
    pub ny: usize,
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub z: Vec<f64>,
}

impl Surface3D {
    /// Build directly from grid coordinates and a row-major `z` of length `nx*ny`.
    pub fn new(nx: usize, ny: usize, xs: Vec<f64>, ys: Vec<f64>, z: Vec<f64>) -> Self {
        assert_eq!(xs.len(), nx, "Surface3D: xs length");
        assert_eq!(ys.len(), ny, "Surface3D: ys length");
        assert_eq!(z.len(), nx * ny, "Surface3D: z length must be nx*ny");
        Self { nx, ny, xs, ys, z }
    }

    /// Build on a unit grid `[0,1] x [0,1]` from a function of `(x, y)`.
    pub fn from_fn(nx: usize, ny: usize, f: impl Fn(f64, f64) -> f64) -> Self {
        let xs: Vec<f64> = (0..nx).map(|i| i as f64 / (nx - 1).max(1) as f64).collect();
        let ys: Vec<f64> = (0..ny).map(|j| j as f64 / (ny - 1).max(1) as f64).collect();
        let mut z = Vec::with_capacity(nx * ny);
        for &xv in &xs {
            for &yv in &ys {
                z.push(f(xv, yv));
            }
        }
        Self { nx, ny, xs, ys, z }
    }

    fn corner(&self, i: usize, j: usize) -> [f64; 3] {
        [self.xs[i], self.ys[j], self.z[i * self.ny + j]]
    }
}

/// A layer accepted by [`Plot3D::layer`].
#[derive(Clone, Debug)]
pub enum Layer3D {
    /// A 3D point cloud.
    Scatter(Scatter3D),
    /// A height surface.
    Surface(Surface3D),
}

impl From<Scatter3D> for Layer3D {
    fn from(s: Scatter3D) -> Self {
        Layer3D::Scatter(s)
    }
}
impl From<Surface3D> for Layer3D {
    fn from(s: Surface3D) -> Self {
        Layer3D::Surface(s)
    }
}

/// A declarative 3D plot specification rendered to SVG via isometric projection.
#[derive(Clone, Debug)]
pub struct Plot3D {
    pub(crate) layers: Vec<Layer3D>,
    pub(crate) title: Option<String>,
    pub(crate) width: f64,
    pub(crate) height: f64,
    pub(crate) azimuth: f64,
    pub(crate) elevation: f64,
}

impl Default for Plot3D {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            title: None,
            width: 640.0,
            height: 480.0,
            azimuth: 0.6,
            elevation: 0.5,
        }
    }
}

impl Plot3D {
    /// Create an empty 3D plot.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a 3D layer (scatter or surface).
    pub fn layer(mut self, layer: impl Into<Layer3D>) -> Self {
        self.layers.push(layer.into());
        self
    }

    /// Camera azimuth (rotation about the vertical axis), radians.
    pub fn azimuth(mut self, a: f64) -> Self {
        self.azimuth = a;
        self
    }
    /// Camera elevation (tilt), radians.
    pub fn elevation(mut self, e: f64) -> Self {
        self.elevation = e;
        self
    }
    /// Plot title.
    pub fn title(mut self, t: &str) -> Self {
        self.title = Some(t.to_string());
        self
    }
    /// Canvas size in pixels.
    pub fn size(mut self, width: f64, height: f64) -> Self {
        self.width = width.max(1.0);
        self.height = height.max(1.0);
        self
    }

    /// Orthographic isometric projection of a centered, unit-scaled point.
    fn project(&self, p: [f64; 3]) -> (f64, f64, f64) {
        let (sa, ca) = self.azimuth.sin_cos();
        let x1 = ca * p[0] - sa * p[1];
        let y1 = sa * p[0] + ca * p[1];
        let z1 = p[2];
        let (se, ce) = self.elevation.sin_cos();
        let y2 = ce * y1 - se * z1;
        let z2 = se * y1 + ce * z1;
        (x1, y2, z2)
    }

    /// Render to a self-contained SVG document.
    pub fn to_svg(&self) -> String {
        // Global bounds across all layers for shared normalization.
        let (mut xmin, mut xmax) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut zmin, mut zmax) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut acc = |v: [f64; 3]| {
            xmin = xmin.min(v[0]);
            xmax = xmax.max(v[0]);
            ymin = ymin.min(v[1]);
            ymax = ymax.max(v[1]);
            zmin = zmin.min(v[2]);
            zmax = zmax.max(v[2]);
        };
        for l in &self.layers {
            match l {
                Layer3D::Scatter(s) => s.points().for_each(&mut acc),
                Layer3D::Surface(s) => {
                    for i in 0..s.nx {
                        for j in 0..s.ny {
                            acc(s.corner(i, j));
                        }
                    }
                }
            }
        }
        let sx = Linear::fit([xmin, xmax]);
        let sy = Linear::fit([ymin, ymax]);
        let sz = Linear::fit([zmin, zmax]);
        let norm = |lx: &Linear, v: f64| lx.normalize(v) - 0.5;

        let m = 40.0f64;
        let pw = self.width - 2.0 * m;
        let ph = self.height - 2.0 * m;
        let to_screen = |proj: (f64, f64, f64)| -> (f64, f64) {
            let px = m + (proj.0 + 0.5) * pw;
            let py = m + (0.5 - proj.1) * ph;
            (px, py)
        };

        let mut out = String::new();
        out.push_str(&format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\">\n",
            self.width as u32, self.height as u32
        ));
        out.push_str(&format!(
            "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#ffffff\"/>\n",
            self.width as u32, self.height as u32
        ));
        if let Some(t) = &self.title {
            out.push_str(&format!(
                "<text x=\"{}\" y=\"24\" font-family=\"sans-serif\" font-size=\"16\">{t}</text>\n",
                (self.width / 2.0) as u32
            ));
        }

        // Painter's algorithm: collect primitives with depth, draw far (small
        // rotated-z) first.
        struct Prim {
            depth: f64,
            svg: String,
        }
        let mut prims: Vec<Prim> = Vec::new();

        for l in &self.layers {
            match l {
                Layer3D::Scatter(s) => {
                    for p in s.points() {
                        let proj =
                            self.project([norm(&sx, p[0]), norm(&sy, p[1]), norm(&sz, p[2])]);
                        let (px, py) = to_screen(proj);
                        prims.push(Prim {
                            depth: proj.2,
                            svg: format!(
                                "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{:.1}\" fill=\"#1f77b4\" opacity=\"{:.2}\"/>\n",
                                px, py, s.size, s.opacity
                            ),
                        });
                    }
                }
                Layer3D::Surface(s) => {
                    for i in 0..s.nx.saturating_sub(1) {
                        for j in 0..s.ny.saturating_sub(1) {
                            let c = [
                                s.corner(i, j),
                                s.corner(i + 1, j),
                                s.corner(i + 1, j + 1),
                                s.corner(i, j + 1),
                            ];
                            let proj: Vec<(f64, f64, f64)> = c
                                .iter()
                                .map(|p| {
                                    self.project([
                                        norm(&sx, p[0]),
                                        norm(&sy, p[1]),
                                        norm(&sz, p[2]),
                                    ])
                                })
                                .collect();
                            let screen: Vec<(f64, f64)> =
                                proj.iter().map(|p| to_screen(*p)).collect();
                            let pts = screen
                                .iter()
                                .map(|(x, y)| format!("{x:.1},{y:.1}"))
                                .collect::<Vec<_>>()
                                .join(" ");
                            let avg_z = (c[0][2] + c[1][2] + c[2][2] + c[3][2]) / 4.0;
                            let zn = sz.normalize(avg_z).clamp(0.0, 1.0);
                            let fill = hex(Gradient::Viridis.color(zn));
                            let depth = proj.iter().map(|p| p.2).sum::<f64>() / 4.0;
                            prims.push(Prim {
                                depth,
                                svg: format!(
                                    "<polygon points=\"{}\" fill=\"{}\" stroke=\"#333\" stroke-width=\"0.3\" opacity=\"0.9\"/>\n",
                                    pts, fill
                                ),
                            });
                        }
                    }
                }
            }
        }

        prims.sort_by(|a, b| {
            a.depth
                .partial_cmp(&b.depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for p in prims {
            out.push_str(&p.svg);
        }
        out.push_str("</svg>\n");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter3d_renders_circles() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let svg = Plot3D::new()
            .layer(Scatter3D::new(x.clone(), x.clone(), x.clone()))
            .title("cloud")
            .to_svg();
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("<circle"));
        assert!(svg.contains("cloud"));
    }

    #[test]
    fn surface3d_renders_polygons() {
        let s = Surface3D::from_fn(12, 12, |x, y| (x - 0.5).powi(2) + (y - 0.5).powi(2));
        let svg = Plot3D::new().layer(s).to_svg();
        assert!(svg.contains("<polygon"));
    }
}
