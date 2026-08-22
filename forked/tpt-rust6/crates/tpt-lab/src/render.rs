//! Rich rendering of cell values.
//!
//! [`Render`] is implemented for the scalar types, [`tpt_omni::Table`]
//! (Markdown) and [`tpt_omni::Tensor<f64>`] (text grid, or an SVG heatmap with
//! `--features viz`). Because cells are type-erased, [`render_value`] performs
//! a downcast chain over the known types; anything else falls back to a short
//! `<type-name>` placeholder. Register a renderer for your own type with
//! `Notebook::set_rendered`.

use tpt_columnar::display::{ArrayFormatter, FormatOptions};
use tpt_omni::{Table, Tensor};

use crate::value::Value;
use crate::Notebook;

/// Renders a value to a display string (Markdown / text grid / SVG).
pub trait Render {
    /// Render `self`. The notebook is passed so a renderer can consult other
    /// cells (e.g. a cell holding display options).
    fn render(&self, notebook: &Notebook) -> String;
}

macro_rules! render_via_display {
    ($($t:ty),* $(,)?) => {$(
        impl Render for $t {
            fn render(&self, _notebook: &Notebook) -> String {
                self.to_string()
            }
        }
    )*};
}
render_via_display!(
    i32,
    i64,
    u32,
    u64,
    usize,
    f32,
    f64,
    bool,
    String,
    &'static str,
    char
);

impl<T: Render> Render for Vec<T> {
    fn render(&self, notebook: &Notebook) -> String {
        let items: Vec<String> = self.iter().map(|v| v.render(notebook)).collect();
        format!("[{}]", items.join(", "))
    }
}

/// Markdown table, truncated to [`MAX_ROWS`] rows.
pub const MAX_ROWS: usize = 20;
/// Maximum tensor rows / columns printed by the text-grid renderer.
pub const MAX_GRID: usize = 10;

impl Render for Table {
    fn render(&self, _notebook: &Notebook) -> String {
        let batch = self.batch();
        let names = self.column_names();
        let mut out = format!(
            "**Table** — {} rows x {} cols\n\n",
            self.num_rows(),
            self.num_cols()
        );
        out.push_str(&format!("| {} |\n", names.join(" | ")));
        out.push_str(&format!(
            "|{}|\n",
            names.iter().map(|_| " --- ").collect::<Vec<_>>().join("|")
        ));

        let opts = FormatOptions::default();
        let fmts: Vec<Option<ArrayFormatter<'_>>> = (0..batch.num_columns())
            .map(|i| ArrayFormatter::try_new(batch.column(i), &opts).ok())
            .collect();
        let shown = self.num_rows().min(MAX_ROWS);
        for r in 0..shown {
            let cells: Vec<String> = fmts
                .iter()
                .map(|f| match f {
                    Some(f) => f.value(r).to_string(),
                    None => "?".to_string(),
                })
                .collect();
            out.push_str(&format!("| {} |\n", cells.join(" | ")));
        }
        if self.num_rows() > shown {
            out.push_str(&format!("\n_… {} more rows_\n", self.num_rows() - shown));
        }
        out
    }
}

impl Render for Tensor<f64> {
    fn render(&self, _notebook: &Notebook) -> String {
        #[cfg(feature = "viz")]
        {
            if let Some(svg) = heatmap_svg(self) {
                return svg;
            }
        }
        text_grid(self)
    }
}

/// `rows x cols` view of a tensor: the last axis is the column axis and all
/// leading axes are flattened into rows.
fn grid_shape(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (1, shape[0]),
        n => (shape[..n - 1].iter().product(), shape[n - 1]),
    }
}

fn text_grid(t: &Tensor<f64>) -> String {
    let shape = t.shape();
    let data = t.to_vec();
    let (rows, cols) = grid_shape(shape);
    let mut out = format!("Tensor<f64> shape={shape:?}\n");
    let (rshown, cshown) = (rows.min(MAX_GRID), cols.min(MAX_GRID));
    for r in 0..rshown {
        let mut line = String::from("[");
        for c in 0..cshown {
            match data.get(r * cols + c) {
                Some(v) => line.push_str(&format!("{v:>10.4}")),
                None => line.push_str(&format!("{:>10}", "-")),
            }
        }
        if cols > cshown {
            line.push_str(&format!("{:>10}", "…"));
        }
        line.push_str(" ]\n");
        out.push_str(&line);
    }
    if rows > rshown {
        out.push_str(&format!("… {} more rows\n", rows - rshown));
    }
    out
}

/// SVG heatmap of a 2-D (or flattenable) tensor via `tpt-viz`.
#[cfg(feature = "viz")]
fn heatmap_svg(t: &Tensor<f64>) -> Option<String> {
    use tpt_viz::prelude::*;
    let (rows, cols) = grid_shape(t.shape());
    let data = t.to_vec();
    if rows == 0 || cols == 0 || data.len() != rows * cols {
        return None;
    }
    Some(
        Plot::new()
            .layer(Heatmap::new(Grid2D::new(cols, rows, data)))
            .scale_color(Gradient::Viridis)
            .title(&format!("Tensor<f64> {:?}", t.shape()))
            .to_svg(),
    )
}

#[cfg(feature = "viz")]
impl Render for tpt_viz::Plot {
    fn render(&self, _notebook: &Notebook) -> String {
        self.to_svg()
    }
}

/// Render a type-erased value by downcasting through the known types.
pub fn render_value(value: &Value, notebook: &Notebook) -> String {
    macro_rules! try_render {
        ($($t:ty),* $(,)?) => {$(
            if let Some(v) = value.downcast_ref::<$t>() {
                return v.render(notebook);
            }
        )*};
    }
    try_render!(
        i32,
        i64,
        u32,
        u64,
        usize,
        f32,
        f64,
        bool,
        String,
        &'static str,
        char,
        Vec<i64>,
        Vec<f64>,
        Vec<String>,
        Table,
        Tensor<f64>,
    );
    #[cfg(feature = "viz")]
    try_render!(tpt_viz::Plot);
    format!("<{}>", value.type_name())
}
