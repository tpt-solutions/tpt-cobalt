//! Richer example: construct series with iterators and compute simple
//! statistics over the samples.

use tpt_eng_timeseries_core::{Sample, Series, Timestamp};

fn main() {
    // Build a series from an iterator of (time, value) pairs via FromIterator.
    let s: Series<f64> = (0..=10)
        .map(|i| {
            let t = (i as f64) * 0.5;
            Sample::new(Timestamp::from_seconds(t), (t - 2.0).powi(2))
        })
        .collect();

    let vals: Vec<f64> = s.iter().map(|x| x.value).collect();
    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;

    println!("samples    = {}", s.len());
    println!(
        "t span     = {:.3} .. {:.3} s",
        s.first().unwrap().t.as_seconds(),
        s.last().unwrap().t.as_seconds()
    );
    println!("value min  = {:.3}", min);
    println!("value max  = {:.3}", max);
    println!("value mean = {:.3}", mean);

    // as_slice gives direct access to the underlying samples for custom folds.
    let peak = s
        .as_slice()
        .iter()
        .max_by(|a, b| a.value.partial_cmp(&b.value).unwrap())
        .unwrap();
    println!(
        "peak value = {:.3} @ {:.3} s",
        peak.value,
        peak.t.as_seconds()
    );

    assert!(s.is_sorted());
    assert!(max - min > 0.0);
    println!("timeseries-core series example passed");
}
