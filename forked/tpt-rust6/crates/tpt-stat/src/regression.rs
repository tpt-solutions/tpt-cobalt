//! Regression models with automatically computed standard errors and
//! confidence intervals.

use crate::dist::Distribution;
use crate::dist::{NormalDist, StudentTDist};
use crate::tests::TestResult;
use std::fmt;

/// Solve `A x = b` for a square `n x n` matrix via Gaussian elimination with
/// partial pivoting.
fn solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .zip(b)
        .map(|(row, bi)| {
            let mut r = row.clone();
            r.push(*bi);
            r
        })
        .collect();
    for i in 0..n {
        let mut piv = i;
        for r in i + 1..n {
            if m[r][i].abs() > m[piv][i].abs() {
                piv = r;
            }
        }
        m.swap(i, piv);
        let diag = m[i][i];
        for c in i..=n {
            m[i][c] /= diag;
        }
        for r in 0..n {
            if r != i {
                let f = m[r][i];
                for c in i..=n {
                    m[r][c] -= f * m[i][c];
                }
            }
        }
    }
    (0..n).map(|i| m[i][n]).collect()
}

fn inverse(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = m.len();
    let mut a: Vec<Vec<f64>> = m.to_vec();
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        inv[i][i] = 1.0;
    }
    for i in 0..n {
        let mut piv = i;
        for r in i + 1..n {
            if a[r][i].abs() > a[piv][i].abs() {
                piv = r;
            }
        }
        a.swap(i, piv);
        inv.swap(i, piv);
        let d = a[i][i];
        for c in 0..n {
            a[i][c] /= d;
            inv[i][c] /= d;
        }
        for r in 0..n {
            if r != i {
                let f = a[r][i];
                for c in 0..n {
                    a[r][c] -= f * a[i][c];
                    inv[r][c] -= f * inv[i][c];
                }
            }
        }
    }
    inv
}

fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    let k = b.len();
    let mut out = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            let mut s = 0.0;
            for l in 0..k {
                s += a[i][l] * b[l][j];
            }
            out[i][j] = s;
        }
    }
    out
}

fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut t = vec![vec![0.0; n]; m];
    for i in 0..n {
        for j in 0..m {
            t[j][i] = a[i][j];
        }
    }
    t
}

fn t_quantile(p: f64, df: f64) -> f64 {
    // bisection on the CDF
    let d = StudentTDist { nu: df };
    let (mut lo, mut hi) = (-100.0, 100.0);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if d.cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Ordinary least squares result with full inference.
#[derive(Debug, Clone)]
pub struct OLSResult {
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub t_stats: Vec<f64>,
    pub p_values: Vec<f64>,
    pub ci_lower: Vec<f64>,
    pub ci_upper: Vec<f64>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub residual_std: f64,
    pub n: usize,
    pub k: usize,
}

impl fmt::Display for OLSResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "OLS Regression Results")?;
        writeln!(
            f,
            "  R² = {:.4}  (adj {:.4})",
            self.r_squared, self.adj_r_squared
        )?;
        writeln!(
            f,
            "  residual sd = {:.4}  n = {}",
            self.residual_std, self.n
        )?;
        writeln!(
            f,
            "  {:>4} {:>10} {:>10} {:>8} {:>8} {:>9} {:>9}",
            "coef", "estimate", "std_err", "t", "p", "2.5%", "97.5%"
        )?;
        for i in 0..self.k {
            writeln!(
                f,
                "  {:>4} {:>10.4} {:>10.4} {:>8.3} {:>8.3} {:>9.4} {:>9.4}",
                i,
                self.coefficients[i],
                self.std_errors[i],
                self.t_stats[i],
                self.p_values[i],
                self.ci_lower[i],
                self.ci_upper[i]
            )?;
        }
        Ok(())
    }
}

/// Fit OLS and return just the coefficients (used internally by ARIMA).
pub fn ols_exact(x: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let xt = transpose(x);
    let xtx = matmul(&xt, x);
    let xty: Vec<f64> = xt
        .iter()
        .map(|row| row.iter().zip(y).map(|(a, b)| a * b).sum())
        .collect();
    solve(&xtx, &xty)
}

/// Fit OLS `y = X beta + e` where `x` is `n x k` (rows of observations) and a
/// constant is NOT automatically added.
pub fn ols(x: &[Vec<f64>], y: &[f64]) -> OLSResult {
    let xt = transpose(x);
    let xtx = matmul(&xt, x);
    let xty: Vec<f64> = xt
        .iter()
        .map(|row| row.iter().zip(y).map(|(a, b)| a * b).sum())
        .collect();
    let beta = solve(&xtx, &xty);
    let n = y.len();
    let k = beta.len();
    let pred: Vec<f64> = x
        .iter()
        .map(|row| row.iter().zip(&beta).map(|(a, b)| a * b).sum())
        .collect();
    let resid: Vec<f64> = y.iter().zip(&pred).map(|(a, b)| a - b).collect();
    let rss: f64 = resid.iter().map(|r| r * r).sum();
    let ybar = y.iter().sum::<f64>() / n as f64;
    let tss: f64 = y.iter().map(|v| (v - ybar).powi(2)).sum();
    let r2 = 1.0 - rss / tss;
    let adj = 1.0 - (1.0 - r2) * (n as f64 - 1.0) / (n as f64 - k as f64);
    let sigma2 = rss / (n as f64 - k as f64);
    let xtx_inv = inverse(&xtx);
    let se: Vec<f64> = (0..k).map(|i| (sigma2 * xtx_inv[i][i]).sqrt()).collect();
    let tcrit = t_quantile(0.975, (n - k) as f64);
    let mut t_stats = vec![0.0; k];
    let mut p = vec![0.0; k];
    let mut lo = vec![0.0; k];
    let mut hi = vec![0.0; k];
    for i in 0..k {
        t_stats[i] = beta[i] / se[i];
        p[i] = 2.0 * (1.0 - StudentTDist { nu: (n - k) as f64 }.cdf(t_stats[i].abs()));
        lo[i] = beta[i] - tcrit * se[i];
        hi[i] = beta[i] + tcrit * se[i];
    }
    OLSResult {
        coefficients: beta,
        std_errors: se,
        t_stats,
        p_values: p,
        ci_lower: lo,
        ci_upper: hi,
        r_squared: r2,
        adj_r_squared: adj,
        residual_std: sigma2.sqrt(),
        n,
        k,
    }
}

/// Fit logistic regression via gradient ascent on the log-likelihood.
pub fn logistic(x: &[Vec<f64>], y: &[f64]) -> OLSResult {
    let n = y.len();
    let k = x[0].len();
    let mut beta = vec![0.0; k];
    let lr = 0.1;
    for _ in 0..800 {
        let p: Vec<f64> = x
            .iter()
            .map(|row| {
                let z: f64 = row.iter().zip(&beta).map(|(a, b)| a * b).sum();
                1.0 / (1.0 + (-z).exp())
            })
            .collect();
        let mut g = vec![0.0; k];
        for (row, (yi, pi)) in x.iter().zip(y.iter().zip(&p)) {
            let d = yi - pi;
            for (gi, a) in g.iter_mut().zip(row) {
                *gi += d * a;
            }
        }
        let mut max = 0.0f64;
        for i in 0..k {
            let step = lr * g[i] / n as f64;
            beta[i] += step;
            max = max.max(step.abs());
        }
        if max < 1e-7 {
            break;
        }
    }
    let p: Vec<f64> = x
        .iter()
        .map(|row| {
            let z: f64 = row.iter().zip(&beta).map(|(a, b)| a * b).sum();
            1.0 / (1.0 + (-z).exp())
        })
        .collect();
    let xt = transpose(x);
    let w: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let wi = p[i] * (1.0 - p[i]);
            (0..n).map(|j| if i == j { wi } else { 0.0 }).collect()
        })
        .collect();
    let xtw = matmul(&xt, &w);
    let xtwx = matmul(&xtw, x);
    let xtwx_inv = inverse(&xtwx);
    let ll_null = y
        .iter()
        .map(|yi| {
            let p0 = y.iter().sum::<f64>() / n as f64;
            yi * p0.ln() + (1.0 - yi) * (1.0 - p0).ln()
        })
        .sum::<f64>();
    let ll_mod = y
        .iter()
        .zip(&p)
        .map(|(yi, pi)| yi * pi.ln() + (1.0 - yi) * (1.0 - pi).ln())
        .sum::<f64>();
    let r2 = 1.0 - ll_mod / ll_null;
    let se: Vec<f64> = (0..k).map(|i| xtwx_inv[i][i].sqrt()).collect();
    let tcrit = t_quantile(0.975, (n - k) as f64);
    let mut lo = vec![0.0; k];
    let mut hi = vec![0.0; k];
    for i in 0..k {
        lo[i] = beta[i] - tcrit * se[i];
        hi[i] = beta[i] + tcrit * se[i];
    }
    OLSResult {
        coefficients: beta.clone(),
        std_errors: se.clone(),
        t_stats: beta.iter().zip(&se).map(|(b, s)| b / s).collect(),
        p_values: beta
            .iter()
            .zip(&se)
            .map(|(b, s)| {
                2.0 * (1.0
                    - NormalDist {
                        mu: 0.0,
                        sigma: 1.0,
                    }
                    .cdf((b / s).abs()))
            })
            .collect(),
        ci_lower: lo,
        ci_upper: hi,
        r_squared: r2,
        adj_r_squared: r2,
        residual_std: f64::NAN,
        n,
        k,
    }
}

/// Fit Poisson regression via gradient ascent on the log-likelihood.
pub fn poisson(x: &[Vec<f64>], y: &[f64]) -> OLSResult {
    let n = y.len();
    let k = x[0].len();
    let mut beta = vec![0.0; k];
    let lr = 0.05;
    for _ in 0..800 {
        let mu: Vec<f64> = x
            .iter()
            .map(|row| {
                let eta: f64 = row.iter().zip(&beta).map(|(a, b)| a * b).sum();
                eta.exp()
            })
            .collect();
        let mut g = vec![0.0; k];
        for (row, (yi, mi)) in x.iter().zip(y.iter().zip(&mu)) {
            let d = yi - mi;
            for (gi, a) in g.iter_mut().zip(row) {
                *gi += d * a;
            }
        }
        let mut max = 0.0f64;
        for i in 0..k {
            let step = lr * g[i] / n as f64;
            beta[i] += step;
            max = max.max(step.abs());
        }
        if max < 1e-7 {
            break;
        }
    }
    let mu: Vec<f64> = x
        .iter()
        .map(|row| {
            let eta: f64 = row.iter().zip(&beta).map(|(a, b)| a * b).sum();
            eta.exp()
        })
        .collect();
    let xt = transpose(x);
    let w: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { mu[i] } else { 0.0 }).collect())
        .collect();
    let xtw = matmul(&xt, &w);
    let xtwx = matmul(&xtw, x);
    let xtwx_inv = inverse(&xtwx);
    let se: Vec<f64> = (0..k).map(|i| xtwx_inv[i][i].sqrt()).collect();
    let tcrit = t_quantile(0.975, (n - k) as f64);
    let mut lo = vec![0.0; k];
    let mut hi = vec![0.0; k];
    for i in 0..k {
        lo[i] = beta[i] - tcrit * se[i];
        hi[i] = beta[i] + tcrit * se[i];
    }
    OLSResult {
        coefficients: beta.clone(),
        std_errors: se.clone(),
        t_stats: beta.iter().zip(&se).map(|(b, s)| b / s).collect(),
        p_values: beta
            .iter()
            .zip(&se)
            .map(|(b, s)| {
                2.0 * (1.0
                    - NormalDist {
                        mu: 0.0,
                        sigma: 1.0,
                    }
                    .cdf((b / s).abs()))
            })
            .collect(),
        ci_lower: lo,
        ci_upper: hi,
        r_squared: f64::NAN,
        adj_r_squared: f64::NAN,
        residual_std: f64::NAN,
        n,
        k,
    }
}

/// Wrap a regression result in the generic [`TestResult`]-compatible form used
/// by the rest of the stack (exposes the model F-equivalent summary).
pub fn ols_as_summary(r: &OLSResult) -> TestResult {
    TestResult {
        name: "OLS",
        statistic: r.r_squared,
        p_value: f64::NAN,
        df: Some((r.k as f64, (r.n - r.k) as f64)),
        extra: vec![("residual_sd".into(), r.residual_std)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ols_recovers_coefficients() {
        // y = 2*x0 + 3*x1 + 1
        let x: Vec<Vec<f64>> = (0..100)
            .map(|i| {
                let a = (i % 7) as f64;
                let b = (i % 5) as f64;
                vec![a, b, 1.0]
            })
            .collect();
        let y: Vec<f64> = x.iter().map(|r| 2.0 * r[0] + 3.0 * r[1] + 1.0).collect();
        let r = ols(&x, &y);
        assert!((r.coefficients[0] - 2.0).abs() < 1e-6);
        assert!((r.coefficients[1] - 3.0).abs() < 1e-6);
        assert!((r.coefficients[2] - 1.0).abs() < 1e-6);
        assert!(r.r_squared > 0.999);
    }

    #[test]
    fn logistic_separates() {
        let x: Vec<Vec<f64>> = (0..100)
            .map(|i| {
                let a = (i % 7) as f64;
                let b = (i % 5) as f64;
                vec![a, b, 1.0]
            })
            .collect();
        let y: Vec<f64> = x
            .iter()
            .map(|r| {
                if 2.0 * r[0] + 3.0 * r[1] + 1.0 > 12.0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        // add a constant already included; run logistic
        let r = logistic(&x, &y);
        let p: Vec<f64> = x
            .iter()
            .map(|row| {
                let z: f64 = row.iter().zip(&r.coefficients).map(|(a, b)| a * b).sum();
                1.0 / (1.0 + (-z).exp())
            })
            .collect();
        let acc = y
            .iter()
            .zip(p.iter())
            .filter(|(yi, pi)| (*yi - *pi).abs() < 0.5)
            .count();
        assert!(acc > 90, "accuracy={}", acc);
    }
}
