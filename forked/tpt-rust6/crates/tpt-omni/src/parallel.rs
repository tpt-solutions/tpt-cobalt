use rayon::prelude::*;

use crate::frame::OmniFrame;

/// Parallel reductions over raw slices (Rayon-backed).
pub fn par_sum_f64(values: &[f64]) -> f64 {
    values.par_iter().copied().sum()
}

pub fn par_mean_f64(values: &[f64]) -> f64 {
    let s: f64 = values.par_iter().copied().sum();
    s / values.len() as f64
}

pub fn par_std_f64(values: &[f64]) -> f64 {
    let m = par_mean_f64(values);
    let n = values.len() as f64;
    (values.par_iter().map(|&v| (v - m) * (v - m)).sum::<f64>() / n).sqrt()
}

/// Parallel summary statistics of a float column inside an `OmniFrame`.
pub fn column_stats(frame: &OmniFrame, name: &str) -> Option<(f64, f64, f64)> {
    let arr = frame.column(name).ok()?;
    let v = crate::tensor::values_of::<f64>(&arr).ok()?;
    Some((par_sum_f64(v), par_mean_f64(v), par_std_f64(v)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Float64Array};
    use std::sync::Arc;

    #[test]
    fn par_sum_matches_expected() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(par_sum_f64(&v), 10.0);
    }

    #[test]
    fn par_mean_matches_expected() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(par_mean_f64(&v), 2.5);
    }

    #[test]
    fn par_std_takes_sqrt_of_variance() {
        // Wikipedia's canonical population-stddev example: variance is 4, std is 2.
        let v = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((par_std_f64(&v) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn column_stats_computes_sum_mean_std() {
        let arr: ArrayRef = Arc::new(Float64Array::from(vec![
            2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0,
        ]));
        let frame = OmniFrame::from_columns(vec![("v".to_string(), arr)]).unwrap();
        let (sum, mean, std) = column_stats(&frame, "v").unwrap();
        assert!((sum - 40.0).abs() < 1e-9);
        assert!((mean - 5.0).abs() < 1e-9);
        assert!((std - 2.0).abs() < 1e-9);
    }

    #[test]
    fn column_stats_missing_column_returns_none() {
        let arr: ArrayRef = Arc::new(Float64Array::from(vec![1.0]));
        let frame = OmniFrame::from_columns(vec![("v".to_string(), arr)]).unwrap();
        assert!(column_stats(&frame, "missing").is_none());
    }
}
