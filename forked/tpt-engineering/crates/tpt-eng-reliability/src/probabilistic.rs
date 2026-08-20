//! Probabilistic design helpers (normal-integration reliability).

use tpt_math_stats::{ContinuousCDF, Normal};

use crate::ReliabilityError;

/// Standard-normal CDF `Phi(z)`.
///
/// # Panics
/// This function does not panic in practice: the standard normal `N(0, 1)` is
/// always constructible, and the `.expect` only guards an unreachable case.
pub fn standard_normal_cdf(z: f64) -> f64 {
    Normal::new(0.0, 1.0)
        .expect("standard normal always valid")
        .cdf(z)
}

/// Inverse standard-normal CDF `Phi^{-1}(p)` via Acklam's rational
/// approximation (relative error < 1.15e-9).
#[allow(clippy::excessive_precision)]
pub fn inv_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ];
    let b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    let c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ];
    let d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// Inverse standard-normal CDF `Phi^{-1}(p)` (the quantile function).
///
/// # Errors
/// Returns [`ReliabilityError::InvalidParameter`] for `p` outside `(0, 1)`.
pub fn z_for_reliability(p: f64) -> Result<f64, ReliabilityError> {
    if !(p > 0.0 && p < 1.0) {
        return Err(ReliabilityError::InvalidParameter(
            "probability must be in (0, 1)".into(),
        ));
    }
    Ok(inv_normal_cdf(p))
}

/// Hasofer–Lind reliability index for a single normal variable and a limit:
/// `beta = (limit - mu) / sigma`.
pub fn reliability_index(mu: f64, sigma: f64, limit: f64) -> f64 {
    if sigma == 0.0 {
        return if limit > mu {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
    }
    (limit - mu) / sigma
}

/// Probability that a normal variable `X ~ N(mu, sigma)` is below `limit`.
pub fn prob_failure_below(mu: f64, sigma: f64, limit: f64) -> f64 {
    if sigma <= 0.0 {
        return if mu < limit { 1.0 } else { 0.0 };
    }
    standard_normal_cdf((limit - mu) / sigma)
}

/// Probability that a normal variable `X ~ N(mu, sigma)` is above `limit`.
pub fn prob_failure_above(mu: f64, sigma: f64, limit: f64) -> f64 {
    1.0 - prob_failure_below(mu, sigma, limit)
}

/// Probability that `X ~ N(mu, sigma)` falls within `[low, high]`.
pub fn prob_within(mu: f64, sigma: f64, low: f64, high: f64) -> f64 {
    prob_failure_below(mu, sigma, high) - prob_failure_below(mu, sigma, low)
}

/// Reliability of a strength `S ~ N(mu_s, sigma_s)` surviving a stress
/// `L ~ N(mu_l, sigma_l)`: `P(S > L) = Phi((mu_s - mu_l) / sqrt(sigma_s^2 + sigma_l^2))`.
pub fn reliability_strength_vs_stress(mu_s: f64, sigma_s: f64, mu_l: f64, sigma_l: f64) -> f64 {
    let sigma_d = (sigma_s * sigma_s + sigma_l * sigma_l).sqrt();
    if sigma_d == 0.0 {
        return if mu_s > mu_l { 1.0 } else { 0.0 };
    }
    standard_normal_cdf((mu_s - mu_l) / sigma_d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn normal_cdf_symmetry() {
        assert_relative_eq!(standard_normal_cdf(0.0), 0.5, epsilon = 1e-9);
        assert_relative_eq!(
            standard_normal_cdf(1.0) + standard_normal_cdf(-1.0),
            1.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn prob_failure_below_mean() {
        // For a standard normal, P(X < 0) = 0.5.
        assert_relative_eq!(prob_failure_below(0.0, 1.0, 0.0), 0.5, epsilon = 1e-9);
    }

    #[test]
    fn reliability_strength_above_stress() {
        // Strong, low-variability strength vs. stress -> high reliability.
        let r = reliability_strength_vs_stress(100.0, 5.0, 50.0, 5.0);
        assert!(r > 0.999, "got {r}");
        // Equal means and equal variances -> 0.5.
        let r2 = reliability_strength_vs_stress(50.0, 5.0, 50.0, 5.0);
        assert_relative_eq!(r2, 0.5, epsilon = 1e-9);
    }

    #[test]
    fn z_quantile_roundtrip() {
        let z = z_for_reliability(0.9772).unwrap();
        assert_relative_eq!(z, 2.0, epsilon = 0.01);
    }
}
