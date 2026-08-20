//! Statistical analysis utilities for benchmark measurements
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSummary {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub p5: f64,
    pub p25: f64,
    pub p75: f64,
    pub p95: f64,
    pub p99: f64,
    pub iqr: f64,
    pub cv: f64,
    pub ci95_lower: f64,
    pub ci95_upper: f64,
    pub outliers_removed: usize,
}

pub fn compute_statistics(data: &[f64]) -> StatisticalSummary {
    if data.is_empty() {
        return StatisticalSummary {
            count: 0,
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            p5: 0.0,
            p25: 0.0,
            p75: 0.0,
            p95: 0.0,
            p99: 0.0,
            iqr: 0.0,
            cv: 0.0,
            ci95_lower: 0.0,
            ci95_upper: 0.0,
            outliers_removed: 0,
        };
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let count = sorted.len();
    let mean = sorted.iter().sum::<f64>() / count as f64;
    let variance = if count > 1 {
        sorted.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (count as f64 - 1.0)
    } else {
        0.0
    };
    let std_dev = variance.sqrt();
    let median = if count.is_multiple_of(2) {
        (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
    } else {
        sorted[count / 2]
    };
    let percentile = |p: f64| -> f64 {
        if sorted.len() == 1 {
            return sorted[0];
        }
        let idx = (p / 100.0) * (sorted.len() - 1) as f64;
        let lower = idx.floor() as usize;
        let upper = idx.ceil() as usize;
        if lower == upper {
            sorted[lower]
        } else {
            sorted[lower] + (sorted[upper] - sorted[lower]) * (idx - lower as f64)
        }
    };
    let p5 = percentile(5.0);
    let p25 = percentile(25.0);
    let p75 = percentile(75.0);
    let p95 = percentile(95.0);
    let p99 = percentile(99.0);
    let iqr = p75 - p25;
    let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };
    let z = 1.96;
    let ci_margin = if count > 1 {
        z * (std_dev / (count as f64).sqrt())
    } else {
        0.0
    };
    StatisticalSummary {
        count,
        mean,
        median,
        std_dev,
        min: sorted[0],
        max: sorted[count - 1],
        p5,
        p25,
        p75,
        p95,
        p99,
        iqr,
        cv,
        ci95_lower: mean - ci_margin,
        ci95_upper: mean + ci_margin,
        outliers_removed: 0,
    }
}

pub fn remove_outliers(data: &[f64], multiplier: f64) -> (Vec<f64>, usize) {
    if data.len() < 4 {
        return (data.to_vec(), 0);
    }
    let stats = compute_statistics(data);
    let lower_bound = stats.p25 - multiplier * stats.iqr;
    let upper_bound = stats.p75 + multiplier * stats.iqr;
    let filtered: Vec<f64> = data
        .iter()
        .filter(|&&x| x >= lower_bound && x <= upper_bound)
        .cloned()
        .collect();
    let removed = data.len() - filtered.len();
    (filtered, removed)
}
