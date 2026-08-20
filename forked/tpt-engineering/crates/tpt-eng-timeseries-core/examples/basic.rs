//! Basic example: build and inspect a time-series using the core types.

use tpt_eng_timeseries_core::{Sample, Series, Timestamp};

fn main() {
    let mut s = Series::new();
    for (t, v) in [(0.0, 1.0), (1.0, 4.0), (2.5, 9.0), (4.0, 16.0)] {
        s.push(Sample::new(Timestamp::from_seconds(t), v));
    }

    println!("len       = {}", s.len());
    println!("sorted?   = {}", s.is_sorted());
    println!("duration  = {:.3} s", s.duration().unwrap());
    println!(
        "first     = {:.3} @ {:.3}s",
        s.first().unwrap().value,
        s.first().unwrap().t.as_seconds()
    );
    println!(
        "last      = {:.3} @ {:.3}s",
        s.last().unwrap().value,
        s.last().unwrap().t.as_seconds()
    );

    println!("\n  t |  v");
    for sample in s.iter() {
        println!("{:5.2} | {:5.3}", sample.t.as_seconds(), sample.value);
    }

    let mean = s.iter().map(|x| x.value).sum::<f64>() / s.len() as f64;
    println!("\nmean value = {:.3}", mean);

    assert!(s.is_sorted());
    assert!((s.duration().unwrap() - 4.0).abs() < 1e-9);
    println!("timeseries-core basic example passed");
}
