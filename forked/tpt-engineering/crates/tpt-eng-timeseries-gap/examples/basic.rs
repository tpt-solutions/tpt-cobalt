//! Basic example: detect gaps and check staleness in a sensor stream.

use tpt_eng_timeseries_core::{Sample, Series, Timestamp};
use tpt_eng_timeseries_gap::{detect_gaps, is_stale};

fn main() {
    let s = Series::from_samples([
        Sample::new(Timestamp::from_seconds(0.0), 1.0),
        Sample::new(Timestamp::from_seconds(1.0), 2.0),
        // 8 s dropout here (> 2 s maximum allowed spacing)
        Sample::new(Timestamp::from_seconds(9.0), 3.0),
        Sample::new(Timestamp::from_seconds(10.0), 4.0),
    ]);

    let gaps = detect_gaps(&s, 2.0);
    println!("detected {} gap(s):", gaps.len());
    for g in &gaps {
        println!(
            "  gap from {:.1}s to {:.1}s (span {:.1}s)",
            g.start,
            g.end,
            g.end - g.start
        );
    }

    println!("stale @ 10.5s (max age 2s)? {}", is_stale(&s, 10.5, 2.0));
    println!("stale @ 50.0s (max age 2s)? {}", is_stale(&s, 50.0, 2.0));

    assert_eq!(gaps.len(), 1);
    assert!(is_stale(&s, 50.0, 2.0));
    assert!(!is_stale(&s, 10.5, 2.0));
    println!("timeseries-gap basic example passed");
}
