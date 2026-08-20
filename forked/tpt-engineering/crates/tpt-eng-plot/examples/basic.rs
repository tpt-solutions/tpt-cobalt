//! tpt-eng-plot basic usage: render an XY plot, a result chart, and a section drawing.
//!
//! Outputs are written to the temp directory as PNG and SVG (no GUI is opened).
//!
//! Run with: `cargo run --example basic -p tpt-eng-plot`

use tpt_eng_plot::{Drawing, ResultChart, SectionDrawing, SectionShape, XyPlot, XySeries};

fn main() {
    let dir = std::env::temp_dir();

    // Multi-series XY plot.
    let sine = XySeries::new(
        "sin",
        (0..100)
            .map(|i| {
                let x = i as f64 / 10.0;
                (x, x.sin())
            })
            .collect(),
    );
    let cosine = XySeries::new(
        "cos",
        (0..100)
            .map(|i| {
                let x = i as f64 / 10.0;
                (x, x.cos())
            })
            .collect(),
    )
    .with_color(plotters::prelude::BLUE);
    let xy = XyPlot::new("Trigonometry")
        .with_x_label("x")
        .with_y_label("y")
        .with_series(sine)
        .with_series(cosine);
    let xy_png = dir.join("tpt_plot_basic_xy.png");
    let xy_svg = dir.join("tpt_plot_basic_xy.svg");
    xy.save_png(&xy_png).unwrap();
    xy.save_svg(&xy_svg).unwrap();
    println!(
        "XY plot written: {} ({} bytes) / {} ({} bytes)",
        xy_png.display(),
        std::fs::metadata(&xy_png).unwrap().len(),
        xy_svg.display(),
        std::fs::metadata(&xy_svg).unwrap().len()
    );

    // Result bar chart.
    let chart = ResultChart::new(
        "Reactions",
        vec![
            ("R_A".into(), 12.5),
            ("R_B".into(), 7.5),
            ("M_max".into(), 18.0),
        ],
    )
    .with_unit("kN");
    let chart_png = dir.join("tpt_plot_basic_chart.png");
    let chart_svg = dir.join("tpt_plot_basic_chart.svg");
    chart.save_png(&chart_png).unwrap();
    chart.save_svg(&chart_svg).unwrap();
    println!(
        "Result chart written: {} / {}",
        chart_png.display(),
        chart_svg.display()
    );

    // Section drawing.
    let section = SectionDrawing::new("I-section")
        .with_labels("x (mm)", "y (mm)")
        .with_shape(SectionShape::Rectangle {
            width: 100.0,
            height: 20.0,
        })
        .with_shape(SectionShape::Circle { radius: 15.0 });
    let sec_png = dir.join("tpt_plot_basic_section.png");
    let sec_svg = dir.join("tpt_plot_basic_section.svg");
    section.save_png(&sec_png).unwrap();
    section.save_svg(&sec_svg).unwrap();
    println!(
        "Section drawing written: {} / {}",
        sec_png.display(),
        sec_svg.display()
    );
}
