//! Runnable example: align irregular multi-rate sensor streams onto one grid.

use tpt_eng_timeseries::align::{align_streams, uniform_grid};
use tpt_eng_timeseries::core::{Sample, Series, Timestamp};

fn main() {
    // A 1 Hz-ish stream and a faster 2 Hz-ish stream, both irregular.
    let slow = Series::from_samples(vec![
        Sample::new(Timestamp::from_seconds(0.0), 10.0),
        Sample::new(Timestamp::from_seconds(1.0), 12.0),
        Sample::new(Timestamp::from_seconds(3.0), 9.0),
    ]);
    let fast = Series::from_samples(vec![
        Sample::new(Timestamp::from_seconds(0.0), 1.0),
        Sample::new(Timestamp::from_seconds(0.5), 2.0),
        Sample::new(Timestamp::from_seconds(2.5), 3.0),
    ]);

    let grid = uniform_grid(0.0, 3.0, 7);
    let aligned = align_streams(&[slow, fast], &grid);

    // `align_streams` returns one aligned vector per stream; each inner vector
    // has one value per grid point.
    println!("  t | slow | fast");
    for i in 0..grid.len() {
        println!(
            "{:4.1} | {:4.1} | {:4.1}",
            grid[i], aligned[0][i], aligned[1][i]
        );
    }
    assert_eq!(aligned.len(), 2);
    assert_eq!(aligned[0].len(), grid.len());
    println!("timeseries align example passed");
}
