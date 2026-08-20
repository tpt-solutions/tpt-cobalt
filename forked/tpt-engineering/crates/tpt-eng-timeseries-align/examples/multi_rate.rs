//! Richer example: align several multi-rate streams onto one common grid and
//! wrap a repaired stream back into a Series.

use tpt_eng_timeseries_align::{align_streams, series_from_grid, uniform_grid};
use tpt_eng_timeseries_core::{Sample, Series, Timestamp};

fn main() {
    // Slow 1 Hz-ish, fast 2 Hz-ish, and a sparse 0.5 Hz-ish stream.
    let slow = Series::from_samples([
        Sample::new(Timestamp::from_seconds(0.0), 0.0),
        Sample::new(Timestamp::from_seconds(1.0), 10.0),
        Sample::new(Timestamp::from_seconds(2.0), 20.0),
    ]);
    let fast = Series::from_samples([
        Sample::new(Timestamp::from_seconds(0.0), 0.0),
        Sample::new(Timestamp::from_seconds(0.5), 5.0),
        Sample::new(Timestamp::from_seconds(1.0), 11.0),
        Sample::new(Timestamp::from_seconds(1.5), 16.0),
        Sample::new(Timestamp::from_seconds(2.0), 22.0),
    ]);
    let sparse = Series::from_samples([
        Sample::new(Timestamp::from_seconds(0.0), 100.0),
        Sample::new(Timestamp::from_seconds(2.0), 60.0),
    ]);

    let grid = uniform_grid(0.0, 2.0, 9); // 0.0 .. 2.0 step 0.25
    let aligned = align_streams(&[slow, fast, sparse], &grid);

    println!("    t | slow | fast | sparse");
    for i in 0..grid.len() {
        println!(
            "{:5.2} | {:5.2} | {:5.2} | {:6.2}",
            grid[i], aligned[0][i], aligned[1][i], aligned[2][i]
        );
    }

    // Re-wrap the slow aligned stream into a Series and check round-trip length.
    let repaired = series_from_grid(&grid, aligned[0].clone());
    assert_eq!(repaired.len(), grid.len());
    println!("timeseries-align multi_rate example passed");
}
