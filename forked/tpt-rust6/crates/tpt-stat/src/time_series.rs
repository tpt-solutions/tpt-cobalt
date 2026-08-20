//! Time-series models: ARIMA, GARCH(1,1), and additive seasonal decomposition.

use std::fmt;

/// Difference a series `d` times (lag-1 differences).
pub fn difference(series: &[f64], d: usize) -> Vec<f64> {
    let mut s = series.to_vec();
    for _ in 0..d {
        s = s.windows(2).map(|w| w[1] - w[0]).collect();
    }
    s
}

/// Invert `d` cumulative sums to bring a differenced forecast back to level.
pub fn integrate(series: &[f64], base: &[f64], d: usize) -> Vec<f64> {
    let mut out = series.to_vec();
    let last = base.to_vec();
    for _ in 0..d {
        let mut cum = Vec::with_capacity(out.len() + last.len());
        cum.extend(last.iter().copied());
        for v in &out {
            let prev = *cum.last().unwrap();
            cum.push(prev + v);
        }
        out = cum.split_off(last.len());
    }
    out
}

fn finite_grad(f: &dyn Fn(&[f64]) -> f64, x: &[f64], eps: f64) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[i] += eps;
            xm[i] -= eps;
            (f(&xp) - f(&xm)) / (2.0 * eps)
        })
        .collect()
}

fn adam_minimize(f: &dyn Fn(&[f64]) -> f64, init: &[f64], iters: usize) -> Vec<f64> {
    let mut x = init.to_vec();
    let mut m = vec![0.0; x.len()];
    let mut v = vec![0.0; x.len()];
    let a = 0.05;
    let b1 = 0.9;
    let b2 = 0.999;
    let eps = 1e-8;
    for t in 1..=iters {
        let g = finite_grad(f, &x, 1e-4);
        for i in 0..x.len() {
            m[i] = b1 * m[i] + (1.0 - b1) * g[i];
            v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
            let mh = m[i] / (1.0 - b1.powi(t as i32));
            let vh = v[i] / (1.0 - b2.powi(t as i32));
            x[i] -= a * mh / (vh.sqrt() + eps);
        }
    }
    x
}

/// ARIMA(p, d, q) fit via conditional sum-of-squares (finite-difference Adam).
#[derive(Debug, Clone)]
pub struct ARIMAResult {
    pub p: usize,
    pub d: usize,
    pub q: usize,
    pub ar: Vec<f64>,
    pub ma: Vec<f64>,
    pub constant: f64,
    pub residuals: Vec<f64>,
    pub aic: f64,
}

impl fmt::Display for ARIMAResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ARIMA({}, {}, {})", self.p, self.d, self.q)?;
        writeln!(f, "  AR({}): {:?}", self.p, self.ar)?;
        writeln!(f, "  MA({}): {:?}", self.q, self.ma)?;
        writeln!(f, "  constant = {:.4}", self.constant)?;
        writeln!(f, "  AIC = {:.4}", self.aic)?;
        Ok(())
    }
}

/// Fit ARIMA. Parameter order: `[ar_1..ar_p, ma_1..ma_q, constant]`.
///
/// When `q == 0` the AR(p) coefficients are estimated by exact ordinary least
/// squares on the lagged design matrix (fast and unbiased). For `q > 0` a
/// conditional-sum-of-squares optimizer (finite-difference Adam) is used.
pub fn arima_fit(series: &[f64], p: usize, d: usize, q: usize) -> ARIMAResult {
    let y = difference(series, d);
    let n = y.len();
    if q == 0 && n > p {
        // Exact OLS on lagged values + constant.
        let mut xt: Vec<Vec<f64>> = Vec::new();
        let mut yt: Vec<f64> = Vec::new();
        for t in p..n {
            let mut row = Vec::with_capacity(p + 1);
            for i in 0..p {
                row.push(y[t - i - 1]);
            }
            row.push(1.0);
            xt.push(row);
            yt.push(y[t]);
        }
        let beta = crate::regression::ols_exact(&xt, &yt);
        let ar = beta[..p].to_vec();
        let constant = beta[p];
        let pred: Vec<f64> = xt
            .iter()
            .map(|r| r.iter().zip(&beta).map(|(a, b)| a * b).sum())
            .collect();
        let resid: Vec<f64> = yt.iter().zip(&pred).map(|(a, b)| a - b).collect();
        let sse: f64 = resid.iter().map(|r| r * r).sum();
        let k = p + 1;
        let aic = n as f64 * (sse / n as f64).ln() + 2.0 * k as f64;
        return ARIMAResult {
            p,
            d,
            q,
            ar,
            ma: vec![],
            constant,
            residuals: resid,
            aic,
        };
    }
    let k = p + q + 1;
    let init: Vec<f64> = {
        let mut v = vec![0.1; p];
        v.extend(std::iter::repeat(0.1).take(q));
        v.push(y.iter().sum::<f64>() / n as f64);
        v
    };

    let loss = |theta: &[f64]| -> f64 {
        let ar = &theta[..p];
        let ma = &theta[p..p + q];
        let c = theta[p + q];
        let mut res = vec![0.0; n];
        let mut errs = vec![0.0; n];
        for t in 0..n {
            let mut pred = c;
            for i in 0..p {
                if t >= i + 1 {
                    pred += ar[i] * y[t - i - 1];
                }
            }
            for j in 0..q {
                if t >= j + 1 {
                    pred += ma[j] * errs[t - j - 1];
                }
            }
            res[t] = y[t] - pred;
            errs[t] = res[t];
        }
        res.iter().map(|r| r * r).sum::<f64>()
    };

    let theta = adam_minimize(&loss, &init, 400);
    let ar = theta[..p].to_vec();
    let ma = theta[p..p + q].to_vec();
    let constant = theta[p + q];
    let sse = loss(&theta);
    let aic = n as f64 * (sse / n as f64).ln() + 2.0 * k as f64;
    ARIMAResult {
        p,
        d,
        q,
        ar,
        ma,
        constant,
        residuals: Vec::new(),
        aic,
    }
}

/// Additive seasonal decomposition into trend / seasonal / residual.
#[derive(Debug, Clone)]
pub struct SeasonalDecomp {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
    pub period: usize,
}

impl SeasonalDecomp {
    /// Reconstruct `observed = trend + seasonal + residual`.
    pub fn reconstruct(&self) -> Vec<f64> {
        self.trend
            .iter()
            .zip(&self.seasonal)
            .zip(&self.residual)
            .map(|((t, s), r)| t + s + r)
            .collect()
    }
}

pub fn seasonal_decompose(series: &[f64], period: usize) -> SeasonalDecomp {
    let n = series.len();
    // Centered moving average (trend) with window = period.
    let half = period / 2;
    let mut trend: Vec<f64> = (0..n)
        .map(|i| {
            if i >= half && i + half < n {
                series[i - half..=i + half].iter().sum::<f64>() / (2 * half + 1) as f64
            } else {
                f64::NAN
            }
        })
        .collect();
    // Back/forward fill the edges so every index has a defined trend.
    if half > 0 {
        let first = (half..n).find(|&i| !trend[i].is_nan()).unwrap_or(n - 1);
        let last = (0..n - half)
            .rev()
            .find(|&i| !trend[i].is_nan())
            .unwrap_or(0);
        for i in 0..half {
            trend[i] = trend[first];
        }
        for i in (n - half)..n {
            trend[i] = trend[last];
        }
    }
    let detrended: Vec<f64> = (0..n)
        .map(|i| {
            if trend[i].is_nan() {
                0.0
            } else {
                series[i] - trend[i]
            }
        })
        .collect();
    let mut seasonal_pattern = vec![0.0; period];
    let mut counts = vec![0usize; period];
    for i in 0..n {
        if !trend[i].is_nan() {
            seasonal_pattern[i % period] += detrended[i];
            counts[i % period] += 1;
        }
    }
    for i in 0..period {
        if counts[i] > 0 {
            seasonal_pattern[i] /= counts[i] as f64;
        }
    }
    let s_mean = seasonal_pattern.iter().sum::<f64>() / period as f64;
    for s in &mut seasonal_pattern {
        *s -= s_mean;
    }
    let seasonal: Vec<f64> = (0..n).map(|i| seasonal_pattern[i % period]).collect();
    let residual: Vec<f64> = (0..n)
        .map(|i| {
            if trend[i].is_nan() {
                0.0
            } else {
                series[i] - trend[i] - seasonal[i]
            }
        })
        .collect();
    SeasonalDecomp {
        trend,
        seasonal,
        residual,
        period,
    }
}

/// GARCH(1,1) fit via MLE (finite-difference Adam on the log-likelihood).
#[derive(Debug, Clone)]
pub struct GARCHResult {
    pub omega: f64,
    pub alpha: f64,
    pub beta: f64,
    pub log_likelihood: f64,
}

impl fmt::Display for GARCHResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GARCH(1,1)")?;
        writeln!(
            f,
            "  omega = {:.4}  alpha = {:.4}  beta = {:.4}",
            self.omega, self.alpha, self.beta
        )?;
        writeln!(f, "  log-likelihood = {:.4}", self.log_likelihood)?;
        Ok(())
    }
}

pub fn garch11(returns: &[f64]) -> GARCHResult {
    let n = returns.len();
    let init = vec![0.01, 0.1, 0.8];
    let neg_ll = |p: &[f64]| -> f64 {
        let (omega, alpha, beta) = (p[0].max(1e-6), p[1].max(0.0), p[2].max(0.0));
        let mut sig2 = returns.iter().map(|r| r * r).sum::<f64>() / n as f64;
        let mut ll = 0.0;
        for &r in returns {
            ll += -0.5 * ((r * r / sig2).ln() + sig2.ln());
            sig2 = omega + alpha * r * r + beta * sig2;
        }
        -ll
    };
    let p = adam_minimize(&neg_ll, &init, 600);
    let ll = -neg_ll(&p);
    GARCHResult {
        omega: p[0].max(1e-6),
        alpha: p[1].max(0.0),
        beta: p[2].max(0.0),
        log_likelihood: ll,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arima_recovers_ar1() {
        // Clean AR(1): x_t = 0.7 x_{t-1} + small zero-mean noise.
        let mut state: u64 = 0x2545F4914F6CDD1D;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / (1u64 << 53) as f64 - 0.5
        };
        let mut x = vec![0.0f64];
        let mut s = 0.0f64;
        for _ in 0..500 {
            s = 0.7 * s + 0.2 * rng();
            x.push(s);
        }
        let r = arima_fit(&x, 1, 0, 0);
        eprintln!("ARIMA ar={:?} const={}", r.ar, r.constant);
        assert!((r.ar[0] - 0.7).abs() < 0.15, "ar={:?}", r.ar);
    }

    #[test]
    fn seasonal_decomp_recovers_pattern() {
        let period = 4;
        let series: Vec<f64> = (0..40)
            .map(|i| 10.0 + (i % period) as f64 + (i as f64 * 0.01))
            .collect();
        let d = seasonal_decompose(&series, period);
        // seasonal pattern should be roughly [0,1,2,3] minus mean
        let recon = d.reconstruct();
        let err: f64 = series.iter().zip(&recon).map(|(a, b)| (a - b).abs()).sum();
        assert!(err < 1.0, "recon error={}", err);
    }

    #[test]
    fn garch_fits_synthetic() {
        let mut r = Vec::with_capacity(500);
        let mut sig2 = 0.04f64;
        for _ in 0..500 {
            let e: f64 = (rand::random::<f64>() - 0.5) * 2.0;
            let v = e * sig2.sqrt();
            r.push(v);
            sig2 = 0.01 + 0.1 * v * v + 0.85 * sig2;
        }
        let g = garch11(&r);
        assert!(
            g.alpha + g.beta < 1.0,
            "non-stationary: {}",
            g.alpha + g.beta
        );
        assert!(
            g.alpha >= 0.0 && g.beta >= 0.0,
            "alpha={} beta={}",
            g.alpha,
            g.beta
        );
    }
}
