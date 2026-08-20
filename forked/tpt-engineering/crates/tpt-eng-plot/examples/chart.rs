//! tpt-eng-plot: a richer example combining a force/displacement XY plot and a
//! custom polygon section drawing, rendered to PNG and SVG.
//!
//! Run with: `cargo run --example chart -p tpt-eng-plot`

use tpt_eng_plot::{Drawing, SectionDrawing, SectionShape, XyPlot, XySeries};

fn main() {
    let dir = std::env::temp_dir();

    // Force vs. displacement: a stiffening (nonlinear) curve plus a linear reference.
    let mut nonlinear = Vec::new();
    let mut linear = Vec::new();
    for i in 0..=20 {
        let d = i as f64 * 0.1;
        let f = 10.0 * d + 2.0 * d * d; // stiffening spring
        nonlinear.push((d, f));
        linear.push((d, 10.0 * d));
    }
    let n_series =
        XySeries::new("nonlinear k(x)=10+2x", nonlinear).with_color(plotters::prelude::RED);
    let l_series = XySeries::new("linear k=10", linear).with_color(plotters::prelude::BLUE);

    let plot = XyPlot::new("Load–displacement response")
        .with_x_label("displacement (mm)")
        .with_y_label("force (N)")
        .with_x_range(0.0, 2.0)
        .with_y_range(0.0, 40.0)
        .with_markers(false)
        .with_series(n_series)
        .with_series(l_series);

    let png = dir.join("tpt_plot_chart.png");
    let svg = dir.join("tpt_plot_chart.svg");
    plot.save_png(&png).unwrap();
    plot.save_svg(&svg).unwrap();
    println!(
        "load–displacement plot: {} / {}",
        png.display(),
        svg.display()
    );

    // A custom I-beam polygon cross-section.
    let i_beam = SectionShape::Polygon {
        points: vec![
            (-50.0, -60.0),
            (50.0, -60.0),
            (50.0, -40.0),
            (20.0, -40.0),
            (20.0, 40.0),
            (50.0, 40.0),
            (50.0, 60.0),
            (-50.0, 60.0),
            (-50.0, 40.0),
            (-20.0, 40.0),
            (-20.0, -40.0),
            (-50.0, -40.0),
        ],
    };
    let section = SectionDrawing::new("I-beam cross-section")
        .with_labels("x (mm)", "y (mm)")
        .with_shape(i_beam);
    let sec_png = dir.join("tpt_plot_chart_section.png");
    let sec_svg = dir.join("tpt_plot_chart_section.svg");
    section.save_png(&sec_png).unwrap();
    section.save_svg(&sec_svg).unwrap();
    println!(
        "I-beam section: {} / {}",
        sec_png.display(),
        sec_svg.display()
    );
}
