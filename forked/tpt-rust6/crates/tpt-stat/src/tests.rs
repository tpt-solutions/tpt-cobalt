//! Classical hypothesis tests returning a typed [`TestResult`].

use crate::dist::Distribution;
use crate::dist::{ChiSquaredDist, NormalDist, StudentTDist};
use crate::special;
use std::fmt;

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}
fn var(xs: &[f64]) -> f64 {
    let m = mean(xs);
    xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (xs.len() as f64 - 1.0)
}

/// The outcome of a statistical test.
#[derive(Debug, Clone)]
pub struct TestResult {
    pub name: &'static str,
    pub statistic: f64,
    pub p_value: f64,
    pub df: Option<(f64, f64)>,
    pub extra: Vec<(String, f64)>,
}

impl fmt::Display for TestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.name)?;
        writeln!(f, "  statistic = {:.4}", self.statistic)?;
        if let Some((a, b)) = self.df {
            writeln!(f, "  df        = ({:.1}, {:.1})", a, b)?;
        }
        writeln!(f, "  p_value   = {:.4}", self.p_value)?;
        for (k, v) in &self.extra {
            writeln!(f, "  {} = {:.4}", k, v)?;
        }
        Ok(())
    }
}

/// One-sample t-test of `mean(xs) == mu0`.
pub fn one_sample_ttest(xs: &[f64], mu0: f64) -> TestResult {
    let n = xs.len() as f64;
    let m = mean(xs);
    let s = var(xs).sqrt();
    let t = (m - mu0) / (s / n.sqrt());
    let df = n - 1.0;
    let p = 2.0 * (1.0 - StudentTDist { nu: df }.cdf(t.abs()));
    TestResult {
        name: "One-sample t-test",
        statistic: t,
        p_value: p,
        df: Some((df, f64::NAN)),
        extra: vec![("mean".into(), m), ("sd".into(), s)],
    }
}

/// Welch's independent two-sample t-test (unequal variance).
pub fn independent_ttest(x: &[f64], y: &[f64]) -> TestResult {
    let nx = x.len() as f64;
    let ny = y.len() as f64;
    let mx = mean(x);
    let my = mean(y);
    let vx = var(x);
    let vy = var(y);
    let se = (vx / nx + vy / ny).sqrt();
    let t = (mx - my) / se;
    let df = (vx / nx + vy / ny).powi(2)
        / ((vx / nx).powi(2) / (nx - 1.0) + (vy / ny).powi(2) / (ny - 1.0));
    let p = 2.0 * (1.0 - StudentTDist { nu: df }.cdf(t.abs()));
    TestResult {
        name: "Welch's t-test",
        statistic: t,
        p_value: p,
        df: Some((df, f64::NAN)),
        extra: vec![("mean_x".into(), mx), ("mean_y".into(), my)],
    }
}

/// Paired t-test on `x - y`.
pub fn paired_ttest(x: &[f64], y: &[f64]) -> TestResult {
    let d: Vec<f64> = x.iter().zip(y).map(|(a, b)| a - b).collect();
    one_sample_ttest(&d, 0.0)
}

/// One-way ANOVA across groups.
pub fn anova_oneway(groups: &[Vec<f64>]) -> TestResult {
    let k = groups.len() as f64;
    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    let n_total = n_total as f64;
    let grand = mean(&groups.iter().flatten().copied().collect::<Vec<_>>());
    let mut ss_between = 0.0;
    let mut ss_within = 0.0;
    for g in groups {
        let m = mean(g);
        ss_between += g.len() as f64 * (m - grand).powi(2);
        let v = var(g);
        ss_within += (g.len() as f64 - 1.0) * v;
    }
    let df_b = k - 1.0;
    let df_w = n_total - k;
    let f_stat = (ss_between / df_b) / (ss_within / df_w);
    // F CDF via regularized incomplete beta.
    let p = 1.0
        - special::betai(
            df_b / 2.0,
            df_w / 2.0,
            df_b * f_stat / (df_b * f_stat + df_w),
        );
    TestResult {
        name: "One-way ANOVA",
        statistic: f_stat,
        p_value: p,
        df: Some((df_b, df_w)),
        extra: vec![],
    }
}

/// Chi-squared goodness-of-fit / contingency test. Provide observed counts and
/// expected counts (same length). `df` is provided by the caller.
pub fn chi_squared(observed: &[f64], expected: &[f64], df: f64) -> TestResult {
    let stat: f64 = observed
        .iter()
        .zip(expected)
        .map(|(o, e)| (o - e) * (o - e) / e)
        .sum();
    let p = 1.0 - ChiSquaredDist { k: df }.cdf(stat);
    TestResult {
        name: "Chi-squared test",
        statistic: stat,
        p_value: p,
        df: Some((df, f64::NAN)),
        extra: vec![],
    }
}

/// Kolmogorov-Smirnov: one-sample against a reference CDF closure.
pub fn ks_test_one_sample<F>(xs: &[f64], cdf: F) -> TestResult
where
    F: Fn(f64) -> f64,
{
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len() as f64;
    let mut d = 0.0f64;
    for (i, &x) in sorted.iter().enumerate() {
        let i = (i + 1) as f64;
        let fo = i / n;
        let fc = cdf(x);
        d = d.max((fo - fc).abs()).max((fc - (i - 1.0) / n).abs());
    }
    let p = ks_pvalue(d, n);
    TestResult {
        name: "Kolmogorov-Smirnov (one-sample)",
        statistic: d,
        p_value: p,
        df: None,
        extra: vec![],
    }
}

/// Kolmogorov-Smirnov two-sample test.
pub fn ks_test_two_sample(x: &[f64], y: &[f64]) -> TestResult {
    let mut a = x.to_vec();
    let mut b = y.to_vec();
    a.sort_by(|p, q| p.total_cmp(q));
    b.sort_by(|p, q| p.total_cmp(q));
    let n = a.len() as f64;
    let m = b.len() as f64;
    let mut d = 0.0f64;
    let (mut ia, mut ib) = (0usize, 0usize);
    while ia < a.len() && ib < b.len() {
        let fa = ia as f64 / n;
        let fb = ib as f64 / m;
        d = d.max((fa - fb).abs());
        if a[ia] < b[ib] {
            ia += 1;
        } else {
            ib += 1;
        }
    }
    let p = ks_pvalue(d, (n * m / (n + m)).sqrt());
    TestResult {
        name: "Kolmogorov-Smirnov (two-sample)",
        statistic: d,
        p_value: p,
        df: None,
        extra: vec![],
    }
}

fn ks_pvalue(d: f64, n: f64) -> f64 {
    let lambda = (n.sqrt() + 0.12 + 0.11 / n.sqrt()) * d;
    let mut s = 0.0;
    for j in 1..=100 {
        let j = j as f64;
        s += (-2.0 * j * j * lambda * lambda).exp() * if j as i32 % 2 == 1 { 1.0 } else { -1.0 };
    }
    2.0 * s
}

/// Mann-Whitney U test (normal approximation with tie correction).
pub fn mann_whitney_u(x: &[f64], y: &[f64]) -> TestResult {
    let mut combined: Vec<(f64, u8)> = x
        .iter()
        .map(|v| (*v, 0))
        .chain(y.iter().map(|v| (*v, 1)))
        .collect();
    combined.sort_by(|a, b| a.0.total_cmp(&b.0));
    let n = x.len() as f64;
    let m = y.len() as f64;
    // average ranks with tie handling
    let mut ranks = vec![0.0; combined.len()];
    let mut i = 0;
    while i < combined.len() {
        let mut j = i;
        while j + 1 < combined.len() && (combined[j + 1].0 - combined[i].0).abs() < 1e-12 {
            j += 1;
        }
        let avg = (i + 1 + j + 1) as f64 / 2.0;
        for k in i..=j {
            ranks[k] = avg;
        }
        i = j + 1;
    }
    let sum_x: f64 = combined
        .iter()
        .zip(&ranks)
        .filter(|(c, _)| c.1 == 0)
        .map(|(_, r)| r)
        .sum();
    let u_x = sum_x - n * (n + 1.0) / 2.0;
    let u_y = n * m - u_x;
    let u = u_x.min(u_y);
    // tie correction
    let mut tie_sum = 0.0;
    let mut k = 0;
    while k < combined.len() {
        let mut j = k;
        while j + 1 < combined.len() && (combined[j + 1].0 - combined[k].0).abs() < 1e-12 {
            j += 1;
        }
        let t = (j - k + 1) as f64;
        tie_sum += t * (t * t - 1.0);
        k = j + 1;
    }
    let n_all = (n + m) as f64;
    let sigma =
        ((n * m / 2.0) * ((n_all + 1.0) / 6.0 - tie_sum / (12.0 * n_all * (n_all - 1.0)))).sqrt();
    let z = (u - n * m / 2.0) / sigma;
    let p = 2.0
        * (1.0
            - NormalDist {
                mu: 0.0,
                sigma: 1.0,
            }
            .cdf(z.abs()));
    TestResult {
        name: "Mann-Whitney U",
        statistic: u,
        p_value: p,
        df: None,
        extra: vec![("U_x".into(), u_x), ("U_y".into(), u_y)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::NormalDist;

    #[test]
    fn ttest_knows_equal_means() {
        let a: Vec<f64> = (0..30)
            .map(|i| 5.0 + ((i % 3) as f64 - 1.0) * 0.1)
            .collect();
        let r = one_sample_ttest(&a, 5.0);
        assert!(r.p_value > 0.05, "p={}", r.p_value);
    }

    #[test]
    fn anova_detects_difference() {
        let g1: Vec<f64> = (0..20).map(|i| 0.01 * (i % 3) as f64).collect();
        let g2: Vec<f64> = (0..20).map(|i| 5.0 + 0.01 * (i % 3) as f64).collect();
        let r = anova_oneway(&[g1, g2]);
        assert!(r.p_value < 0.001);
    }

    #[test]
    fn mann_whitney_small() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let r = mann_whitney_u(&x, &y);
        assert!(r.p_value < 0.05);
    }

    #[test]
    fn ks_accepts_normal() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let d = NormalDist {
            mu: 0.0,
            sigma: 1.0,
        };
        let mut rng = StdRng::seed_from_u64(7);
        let xs: Vec<f64> = (0..1000)
            .map(|_| Normal::new(0.0, 1.0).unwrap().sample(&mut rng))
            .collect();
        eprintln!(
            "cdf(0)={:.3} cdf(2)={:.3} cdf(-2)={:.3}",
            d.cdf(0.0),
            d.cdf(2.0),
            d.cdf(-2.0)
        );
        let r = ks_test_one_sample(&xs, |x| d.cdf(x));
        eprintln!("KS d={:.4} p={:.4} n={}", r.statistic, r.p_value, xs.len());
        assert!(r.p_value > 0.01, "p={}", r.p_value);
    }
}
