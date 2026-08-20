//! Life-distribution models (Weibull and exponential).

use crate::ReliabilityError;

/// Weibull reliability `R(t) = exp(-(t/eta)^beta)` for `t >= 0`.
///
/// # Errors
/// Returns [`ReliabilityError::InvalidParameter`] for non-positive `eta`/`beta`.
pub fn weibull_reliability(t: f64, eta: f64, beta: f64) -> Result<f64, ReliabilityError> {
    if eta <= 0.0 || beta <= 0.0 {
        return Err(ReliabilityError::InvalidParameter(
            "eta and beta must be positive".into(),
        ));
    }
    if t <= 0.0 {
        return Ok(1.0);
    }
    Ok((-(t / eta).powf(beta)).exp())
}

/// Weibull probability density `f(t)`.
///
/// # Errors
/// Returns [`ReliabilityError::InvalidParameter`] for non-positive `eta`/`beta`.
pub fn weibull_pdf(t: f64, eta: f64, beta: f64) -> Result<f64, ReliabilityError> {
    if eta <= 0.0 || beta <= 0.0 {
        return Err(ReliabilityError::InvalidParameter(
            "eta and beta must be positive".into(),
        ));
    }
    if t <= 0.0 {
        return Ok(0.0);
    }
    let x = t / eta;
    Ok((beta / eta) * x.powf(beta - 1.0) * (-x.powf(beta)).exp())
}

/// Weibull hazard (failure-rate) function `h(t) = (beta/eta) (t/eta)^(beta-1)`.
///
/// # Errors
/// Returns [`ReliabilityError::InvalidParameter`] for non-positive `eta`/`beta`.
pub fn weibull_failure_rate(t: f64, eta: f64, beta: f64) -> Result<f64, ReliabilityError> {
    if eta <= 0.0 || beta <= 0.0 {
        return Err(ReliabilityError::InvalidParameter(
            "eta and beta must be positive".into(),
        ));
    }
    if t <= 0.0 {
        return Ok(0.0);
    }
    let x = t / eta;
    Ok((beta / eta) * x.powf(beta - 1.0))
}

/// Mean life of a Weibull distribution: `eta * Gamma(1 + 1/beta)`.
///
/// # Errors
/// Returns [`ReliabilityError::InvalidParameter`] for non-positive `eta`/`beta`.
pub fn weibull_mean(eta: f64, beta: f64) -> Result<f64, ReliabilityError> {
    if eta <= 0.0 || beta <= 0.0 {
        return Err(ReliabilityError::InvalidParameter(
            "eta and beta must be positive".into(),
        ));
    }
    Ok(eta * tpt_math_stats::gamma(1.0 + 1.0 / beta))
}

/// B-life: time by which a fraction `b` (in percent, e.g. 10 for B10) of the
/// population has failed. `R = 1 - b/100`, so `t = eta * (-ln(R))^(1/beta)`.
///
/// # Errors
/// Returns [`ReliabilityError::InvalidParameter`] for `b` outside `(0, 100)`.
pub fn weibull_b_life(b: f64, eta: f64, beta: f64) -> Result<f64, ReliabilityError> {
    if !(b > 0.0 && b < 100.0) {
        return Err(ReliabilityError::InvalidParameter(
            "b must be in (0, 100)".into(),
        ));
    }
    let r = 1.0 - b / 100.0;
    Ok(eta * (-r.ln()).powf(1.0 / beta))
}

/// Exponential reliability `R(t) = exp(-lambda t)` for `t >= 0`.
///
/// # Errors
/// Returns [`ReliabilityError::InvalidParameter`] for non-positive `lambda`.
pub fn exponential_reliability(t: f64, lambda: f64) -> Result<f64, ReliabilityError> {
    if lambda <= 0.0 {
        return Err(ReliabilityError::InvalidParameter(
            "lambda must be positive".into(),
        ));
    }
    if t <= 0.0 {
        return Ok(1.0);
    }
    Ok((-lambda * t).exp())
}

/// Mean life of an exponential distribution: `1 / lambda`.
///
/// # Errors
/// Returns [`ReliabilityError::InvalidParameter`] for non-positive `lambda`.
pub fn exponential_mean(lambda: f64) -> Result<f64, ReliabilityError> {
    if lambda <= 0.0 {
        return Err(ReliabilityError::InvalidParameter(
            "lambda must be positive".into(),
        ));
    }
    Ok(1.0 / lambda)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn weibull_reliability_monotone() {
        let r0 = weibull_reliability(0.0, 100.0, 2.0).unwrap();
        let r1 = weibull_reliability(50.0, 100.0, 2.0).unwrap();
        let r2 = weibull_reliability(100.0, 100.0, 2.0).unwrap();
        assert_relative_eq!(r0, 1.0);
        assert!(r1 < 1.0 && r1 > r2);
        assert_relative_eq!(r2, (-1.0f64).exp(), epsilon = 1e-9);
    }

    #[test]
    fn weibull_b10_life() {
        // For eta=100, beta=2, B10 unreliability 0.1 -> t = 100 * (-ln 0.9)^0.5.
        let t = weibull_b_life(10.0, 100.0, 2.0).unwrap();
        let expected = 100.0 * (-(0.9f64).ln()).sqrt();
        assert_relative_eq!(t, expected, epsilon = 1e-9);
        // At that time, unreliability should be ~0.1.
        let r = weibull_reliability(t, 100.0, 2.0).unwrap();
        assert_relative_eq!(r, 0.9, epsilon = 1e-6);
    }

    #[test]
    fn exponential_mean_and_reliability() {
        let m = exponential_mean(0.001).unwrap();
        assert_relative_eq!(m, 1000.0, epsilon = 1e-9);
        let r = exponential_reliability(1000.0, 0.001).unwrap();
        assert_relative_eq!(r, (-1.0f64).exp(), epsilon = 1e-9);
    }
}
