//! Integration tests for the public `tpt-viz` API.

use tpt_viz::prelude::*;
use tpt_viz::{lod_scatter_bins, DensityGrid, Grid2D};

fn ramp(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64).collect()
}

#[test]
fn scatter_plot_renders_circles() {
    let x = ramp(10);
    let y: Vec<f64> = x.iter().map(|v| v * 1.5).collect();

    let svg = Plot::new()
        .layer(Scatter::new(&x, &y).color(&y).size(4.0).opacity(0.75))
        .scale_color(Gradient::Viridis)
        .title("ten points")
        .xlabel("x")
        .ylabel("y")
        .to_svg();

    assert!(svg.contains("<svg"), "missing <svg root");
    assert!(svg.contains("<circle"), "missing <circle marks");
    assert_eq!(svg.matches("<circle").count(), 10);
    assert!(svg.contains("</svg>"));
    assert!(svg.contains("ten points"));
    // Color aesthetic maps through the gradient, so endpoints differ.
    assert!(svg.contains(&Gradient::Viridis.hex(0.0)));
    assert!(svg.contains(&Gradient::Viridis.hex(1.0)));
}

#[test]
fn histogram_renders_rects() {
    let values: Vec<f64> = (0..100).map(|i| (i % 10) as f64).collect();
    let svg = Plot::new()
        .layer(Histogram::new(&values, 10))
        .title("hist")
        .to_svg();

    assert!(svg.contains("<svg"));
    assert!(svg.contains("<rect"));
    // 10 bars + background + axis frame.
    assert!(svg.matches("<rect").count() >= 12);
}

#[test]
fn line_renders_a_polyline() {
    let x = ramp(5);
    let svg = Plot::new().layer(Line::new(&x, &x).width(2.0)).to_svg();
    assert!(svg.contains("<polyline"));
    assert!(svg.contains("stroke-width=\"2\""));
}

#[test]
fn heatmap_and_contour_render_cell_grids() {
    let grid = Grid2D::new(4, 3, (0..12).map(|i| i as f64).collect::<Vec<_>>());
    let heat = Plot::new().layer(Heatmap::new(grid.clone())).to_svg();
    // 12 cells + background + frame.
    assert_eq!(heat.matches("<rect").count(), 14);

    let contour = Plot::new()
        .layer(Contour::new(grid).levels(3).extent(0.0, 1.0, 0.0, 1.0))
        .scale_color(Gradient::Magma)
        .to_svg();
    assert_eq!(contour.matches("<rect").count(), 14);
    // Banding collapses 12 distinct values into at most 3 distinct fills.
    let fills: std::collections::BTreeSet<&str> =
        contour.split("fill=\"#").skip(1).map(|s| &s[..6]).collect();
    assert!(fills.len() <= 3, "expected <=3 bands, got {fills:?}");
}

#[test]
fn lod_aggregates_above_threshold_and_preserves_counts() {
    // 1000 samples spread over a 10x10 lattice, so many points share a cell.
    let points: Vec<(f64, f64)> = (0..1000)
        .map(|i| ((i % 10) as f64, ((i / 10) % 10) as f64))
        .collect();

    let lod = lod_scatter(&points, 100);
    assert!(lod.is_aggregated(), "1000 > 100 must aggregate");
    let grid = lod.grid().expect("aggregated variant exposes its grid");

    // Every input point is preserved exactly once across the bins.
    assert_eq!(grid.total() as usize, 1000);
    assert_eq!(grid.counts().iter().sum::<u32>(), 1000);
    assert_eq!(lod.total_points(), 1000);

    // Aggregation reduces the number of rendered elements.
    assert!(lod.element_count() < points.len());
    assert_eq!(lod.element_count(), grid.occupied());
    assert_eq!(grid.cells().len(), grid.occupied());

    // ...and the raw path is untouched below the threshold.
    let raw = lod_scatter(&points[..50], 100);
    assert!(!raw.is_aggregated());
    assert_eq!(raw.total_points(), 50);
    assert_eq!(raw.element_count(), 50);
}

#[test]
fn lod_bin_counts_are_exact() {
    // 4 points, one per quadrant of a 2x2 grid, plus a duplicate in cell (0,0).
    let pts = [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
    let g = DensityGrid::build(&pts, 2, 2);
    assert_eq!(g.count(0, 0), 2);
    assert_eq!(g.count(1, 0), 1);
    assert_eq!(g.count(0, 1), 1);
    assert_eq!(g.count(1, 1), 1);
    assert_eq!(g.total(), 5);

    let lod = lod_scatter_bins(&pts, 1, 2);
    assert_eq!(lod.grid().unwrap().total(), 5);
}

#[test]
fn dense_scatter_layer_renders_rects_instead_of_circles() {
    let points: Vec<f64> = (0..1000).map(|i| (i % 37) as f64).collect();
    let svg = Plot::new()
        .layer(
            Scatter::new(&points, &points)
                .lod_threshold(100)
                .lod_bins(16),
        )
        .to_svg();

    assert!(!svg.contains("<circle"), "dense layer must aggregate");
    assert!(svg.contains("<rect"));
    assert!(svg.matches("<rect").count() < 1000);
}

#[test]
fn gradients_have_distinct_endpoints() {
    assert_ne!(Gradient::Viridis.color(0.0), Gradient::Viridis.color(1.0));
    assert_eq!(Gradient::Viridis.color(0.0), (68, 1, 84));
    assert_eq!(Gradient::Viridis.color(1.0), (253, 231, 37));
    assert_ne!(
        Gradient::Grayscale.color(0.0),
        Gradient::Grayscale.color(1.0)
    );

    let flat: Rgb = (10, 20, 30);
    let svg = Plot::new()
        .layer(Scatter::new([0.0, 1.0], [0.0, 1.0]).color(flat))
        .to_svg();
    assert!(svg.contains("#0a141e"));
}

#[test]
fn plot_displays_and_saves_svg() {
    let p = Plot::new().layer(Scatter::new([0.0, 1.0], [1.0, 0.0]));
    // Display / render_to_svg let tpt-lab embed output directly.
    assert_eq!(format!("{p}"), p.to_svg());
    assert_eq!(p.render_to_svg(), p.to_svg());

    let path = std::env::temp_dir().join("tpt-viz-test-plot.svg");
    p.save_svg(&path).expect("save_svg");
    let written = std::fs::read_to_string(&path).unwrap();
    assert_eq!(written, p.to_svg());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn escapes_xml_in_labels() {
    let svg = Plot::new()
        .layer(Line::new([0.0, 1.0], [0.0, 1.0]))
        .title("a < b & \"c\"")
        .to_svg();
    assert!(svg.contains("a &lt; b &amp; &quot;c&quot;"));
    assert!(!svg.contains("a < b"));
}

// ---------------------------------------------------------------------------
// tpt-omni integration (feature `omni`, enabled for dev/test builds).
// ---------------------------------------------------------------------------

#[cfg(feature = "omni")]
mod omni {
    use super::*;
    use std::sync::Arc;
    use tpt_omni::ndarray::{ArrayD, IxDyn};
    use tpt_omni::{OmniFrame, Tensor};
    use tpt_viz::omni::{columns_xy, table_xy};

    fn frame() -> OmniFrame {
        let x: Arc<dyn arrow::array::Array> =
            Arc::new(arrow::array::Float64Array::from(vec![0.0, 1.0, 2.0]));
        let y: Arc<dyn arrow::array::Array> =
            Arc::new(arrow::array::Float64Array::from(vec![3.0, 4.0, 5.0]));
        OmniFrame::from_columns(vec![("x".into(), x), ("y".into(), y)]).unwrap()
    }

    #[test]
    fn scatter_from_two_column_tensor() {
        let t = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
        );
        let s = Scatter::new_tensor(&t).unwrap();
        assert_eq!(s.points(), vec![(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]);
        assert!(Plot::new().layer(s).to_svg().contains("<circle"));

        let bad = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0; 3]).unwrap());
        assert!(Scatter::new_tensor(&bad).is_err());
    }

    #[test]
    fn columns_pulled_from_frame_and_table_by_name() {
        let f = frame();
        assert_eq!(
            columns_xy(&f, "x", "y").unwrap(),
            (vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0])
        );
        assert_eq!(
            table_xy(&f.as_table(), "x", "y").unwrap(),
            (vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0])
        );
        assert!(columns_xy(&f, "x", "nope").is_err());

        let svg = Plot::new()
            .layer(Scatter::from_frame(&f, "x", "y").unwrap())
            .layer(Line::from_frame(&f, "x", "y").unwrap())
            .to_svg();
        assert!(svg.contains("<circle") && svg.contains("<polyline"));
    }

    #[test]
    fn histogram_and_heatmap_from_tensors() {
        let t = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[6]), ramp(6)).unwrap());
        let (_, counts) = Histogram::from_tensor(&t, 3).bin_edges_counts();
        assert_eq!(counts.iter().sum::<u32>(), 6);

        let g = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[2, 3]), ramp(6)).unwrap());
        let heat = Heatmap::from_tensor(&g).unwrap();
        assert!(Plot::new().layer(heat).to_svg().contains("<rect"));

        let h = Histogram::from_frame(&frame(), "x", 3).unwrap();
        assert_eq!(h.bin_edges_counts().1.iter().sum::<u32>(), 3);
    }
}
