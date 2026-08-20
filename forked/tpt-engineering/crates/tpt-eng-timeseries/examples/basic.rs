//! Basic example: use the umbrella crate to build series and align streams.

use tpt_eng_timeseries::align::align_streams;
use tpt_eng_timeseries::core::{Sample, Series, Timestamp};

fn main() {
    let a = Series::from_samples([
        Sample::new(Timestamp::from_seconds(0.0), 1.0),
        Sample::new(Timestamp::from_seconds(1.0), 2.0),
    ]);
    let b = Series::from_samples([
        Sample::new(Timestamp::from_seconds(0.0), 10.0),
        Sample::new(Timestamp::from_seconds(0.5), 15.0),
        Sample::new(Timestamp::from_seconds(1.0), 20.0),
    ]);

    let grid = [0.0, 0.5, 1.0];
    let aligned = align_streams(&[a, b], &grid);

    println!("    t |  a |  b");
    for i in 0..grid.len() {
        println!(
            "{:5.1} | {:.2} | {:.2}",
            grid[i], aligned[0][i], aligned[1][i]
        );
    }

    assert_eq!(aligned[1], vec![10.0, 15.0, 20.0]);
    println!("timeseries basic example passed");
}
