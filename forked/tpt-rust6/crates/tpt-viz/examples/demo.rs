//! Renders a small multi-layer plot to `tpt-viz-demo.svg`.
//! Run with: `cargo run -p tpt-viz --example demo`

use tpt_viz::prelude::*;

fn main() -> std::io::Result<()> {
    let x: Vec<f64> = (0..200).map(|i| i as f64 * 0.05).collect();
    let y: Vec<f64> = x.iter().map(|v| v.sin()).collect();

    let plot = Plot::new()
        .layer(Line::new(&x, &y).width(2.0))
        .layer(Scatter::new(&x, &y).color(&y).size(2.5).opacity(0.7))
        .scale_color(Gradient::Viridis)
        .title("sin(x)")
        .xlabel("x")
        .ylabel("sin(x)");

    plot.save_svg("tpt-viz-demo.svg")?;
    println!("wrote tpt-viz-demo.svg ({} bytes)", plot.to_svg().len());
    Ok(())
}
