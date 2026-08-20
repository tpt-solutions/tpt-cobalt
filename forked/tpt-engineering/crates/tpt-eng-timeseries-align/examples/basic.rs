//! Basic example: resample a single irregular stream onto a uniform grid.

use tpt_eng_timeseries_align::align_to_grid;
use tpt_eng_timeseries_core::{Sample, Series, Timestamp};

fn main() {
    let src = Series::from_samples([
        Sample::new(Timestamp::from_seconds(0.0), 0.0),
        Sample::new(Timestamp::from_seconds(2.0), 10.0),
        Sample::new(Timestamp::from_seconds(5.0), 25.0),
    ]);

    let grid = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let aligned = align_to_grid(&src, &grid);

    println!("  t | aligned");
    for (t, v) in grid.iter().zip(aligned.iter()) {
        println!("{:5.1} | {:6.3}", t, v);
    }

    // Linear interpolation at t=1.0 between (0,0) and (2,10) -> 5.0.
    assert!((aligned[1] - 5.0).abs() < 1e-9, "midpoint should be 5");
    println!("timeseries-align basic example passed");
}
