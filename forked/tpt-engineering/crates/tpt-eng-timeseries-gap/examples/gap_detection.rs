//! Richer example: repair a gappy stream on a uniform grid using hold, linear,
//! and zero-order fill strategies and compare them.

use tpt_eng_timeseries_core::{Sample, Series, Timestamp};
use tpt_eng_timeseries_gap::{Strategy, fill_gaps, interpolate_at};

fn main() {
    // Sampled signal with a large dropout between t=2 and t=8.
    let raw = Series::from_samples([
        Sample::new(Timestamp::from_seconds(0.0), 0.0),
        Sample::new(Timestamp::from_seconds(2.0), 4.0),
        Sample::new(Timestamp::from_seconds(8.0), 10.0),
        Sample::new(Timestamp::from_seconds(10.0), 12.0),
    ]);

    let grid: Vec<f64> = (0..=10).map(|i| i as f64).collect();

    let hold = fill_gaps(&raw, &grid, Strategy::Hold);
    let linear = fill_gaps(&raw, &grid, Strategy::Linear);
    let zero = fill_gaps(&raw, &grid, Strategy::Zero);

    println!("  t | hold | linear | zero");
    for (((&t, h), lin), z) in grid
        .iter()
        .zip(hold.iter())
        .zip(linear.iter())
        .zip(zero.iter())
    {
        println!(
            "{:3.0} | {:4.2} | {:6.2} | {:4.2}",
            t, h.value, lin.value, z.value
        );
    }

    println!("\ninterpolate_at t=5.0 (inside the gap):");
    println!(
        "  hold   = {:.3}",
        interpolate_at(&raw, 5.0, Strategy::Hold)
    );
    println!(
        "  linear = {:.3}",
        interpolate_at(&raw, 5.0, Strategy::Linear)
    );
    println!(
        "  zero   = {:.3}",
        interpolate_at(&raw, 5.0, Strategy::Zero)
    );

    assert!((interpolate_at(&raw, 5.0, Strategy::Linear) - 7.0).abs() < 1e-9);
    assert!((interpolate_at(&raw, 5.0, Strategy::Hold) - 4.0).abs() < 1e-9);
    println!("timeseries-gap gap_detection example passed");
}
