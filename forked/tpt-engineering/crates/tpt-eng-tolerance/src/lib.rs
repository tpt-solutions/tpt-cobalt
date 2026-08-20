//! Tolerance analysis for dimension and parameter stack-up.
//!
//! Provides the standard stack-up methods used in mechanical tolerancing:
//! worst-case, root-sum-square (RSS), and Monte-Carlo evaluation, together
//! with sensitivity and contributor-ranking helpers.

use rand::Rng;
use rand_distr::{Distribution as Distr, Uniform};

/// A single dimension with a bilateral (±) tolerance about its nominal value.
///
/// The tolerance may be specified symmetrically (via [`DimTol::new`]) or
/// asymmetrically (via [`DimTol::asymmetric`], giving distinct `tol_plus` and
/// `tol_minus` one-sided deviations). When the asymmetric fields are `None`,
/// the symmetric `tol` is used for both sides.
#[derive(Debug, Clone)]
pub struct DimTol {
    /// Identifier for the dimension.
    pub name: String,
    /// Nominal (design) value.
    pub nominal: f64,
    /// Symmetric bilateral tolerance; used as the default for both sides when
    /// `tol_plus`/`tol_minus` are `None`.
    pub tol: f64,
    /// Optional asymmetric positive deviation. Falls back to `tol` when `None`.
    pub tol_plus: Option<f64>,
    /// Optional asymmetric negative deviation. Falls back to `tol` when `None`.
    pub tol_minus: Option<f64>,
}

impl DimTol {
    /// Build a dimension with a symmetric (±`tol`) tolerance.
    pub fn new(name: impl Into<String>, nominal: f64, tol: f64) -> Self {
        Self {
            name: name.into(),
            nominal,
            tol: tol.abs(),
            tol_plus: None,
            tol_minus: None,
        }
    }

    /// Build a dimension with explicit one-sided tolerances.
    pub fn asymmetric(
        name: impl Into<String>,
        nominal: f64,
        tol_plus: f64,
        tol_minus: f64,
    ) -> Self {
        Self {
            name: name.into(),
            nominal,
            tol: tol_plus.max(tol_minus).abs(),
            tol_plus: Some(tol_plus.abs()),
            tol_minus: Some(tol_minus.abs()),
        }
    }

    fn plus(&self) -> f64 {
        self.tol_plus.unwrap_or(self.tol).abs()
    }

    fn minus(&self) -> f64 {
        self.tol_minus.unwrap_or(self.tol).abs()
    }

    /// Lower bound of the tolerance interval.
    pub fn min(&self) -> f64 {
        self.nominal - self.minus()
    }

    /// Upper bound of the tolerance interval.
    pub fn max(&self) -> f64 {
        self.nominal + self.plus()
    }
}

/// Result of a Monte-Carlo stack-up evaluation.
#[derive(Debug, Clone)]
pub struct StackupResult {
    /// Number of samples.
    pub n: usize,
    /// Mean of the stack-up.
    pub mean: f64,
    /// Standard deviation of the stack-up.
    pub std: f64,
    /// Minimum observed stack-up.
    pub min: f64,
    /// Maximum observed stack-up.
    pub max: f64,
    /// Estimated yield (fraction of samples within `spec`), if a spec was given.
    pub yield_fraction: Option<f64>,
}

/// Worst-case stack-up interval: `[sum nominal - sum |tol|, sum nominal + sum |tol|]`.
pub fn worst_case(dims: &[DimTol]) -> (f64, f64) {
    let lo: f64 = dims.iter().map(|d| d.min()).sum();
    let hi: f64 = dims.iter().map(|d| d.max()).sum();
    (lo, hi)
}

/// Root-sum-square (RSS) stack-up interval assuming independent, normally
/// distributed dimensions: `[sum nominal - 3*sigma, sum nominal + 3*sigma]`
/// where `sigma = sqrt(sum tol_i^2 / 3)` (each ±tol spans ±3σ).
pub fn rss(dims: &[DimTol]) -> (f64, f64) {
    let nom: f64 = dims.iter().map(|d| d.nominal).sum();
    let var: f64 = dims
        .iter()
        .map(|d| {
            let t = ((d.plus().powi(2) + d.minus().powi(2)) / 2.0).sqrt();
            t * t / 9.0
        })
        .sum();
    let sigma = var.sqrt();
    (nom - 3.0 * sigma, nom + 3.0 * sigma)
}

/// Monte-Carlo stack-up evaluation. Each dimension is sampled uniformly in
/// its tolerance interval; the stack-up is the sum. If `spec` is provided as
/// `(low, high)`, the yield fraction is also reported.
///
/// # Panics
///
/// Panics if any dimension has non-finite tolerance bounds.
pub fn monte_carlo<R: Rng + ?Sized>(
    dims: &[DimTol],
    n: usize,
    spec: Option<(f64, f64)>,
    rng: &mut R,
) -> StackupResult {
    let mut samples = Vec::with_capacity(n);
    let mut in_spec = 0usize;
    for _ in 0..n {
        let mut total = 0.0;
        for d in dims {
            let u = Uniform::new(d.min(), d.max()).expect("finite tolerance bounds");
            total += u.sample(rng);
        }
        if let Some((lo, hi)) = spec {
            if total >= lo && total <= hi {
                in_spec += 1;
            }
        }
        samples.push(total);
    }
    let yield_fraction = spec.map(|_| in_spec as f64 / n as f64);
    StackupResult {
        n,
        mean: tpt_math_stats::mean(&samples),
        std: tpt_math_stats::std_dev(&samples),
        min: tpt_math_stats::min(&samples),
        max: tpt_math_stats::max(&samples),
        yield_fraction,
    }
}

/// RSS variance share of each dimension: `tol_i^2 / sum(tol^2)`.
///
/// Returns a vector aligned with `dims` (empty if no tolerance).
pub fn rss_contributions(dims: &[DimTol]) -> Vec<f64> {
    let shares: Vec<f64> = dims
        .iter()
        .map(|d| {
            let t = ((d.plus().powi(2) + d.minus().powi(2)) / 2.0).sqrt();
            t * t
        })
        .collect();
    let total: f64 = shares.iter().sum();
    if total == 0.0 {
        return vec![0.0; dims.len()];
    }
    shares.iter().map(|s| s / total).collect()
}

/// Rank dimensions by their RSS contribution (largest first).
///
/// Returns `(original_index, contribution)` pairs sorted descending. Uses a
/// total order (`f64::total_cmp`) so non-finite contributions (from
/// non-finite tolerance values in `dims`) sort deterministically instead of
/// panicking.
pub fn rank_contributors(dims: &[DimTol]) -> Vec<(usize, f64)> {
    let shares = rss_contributions(dims);
    let mut ranked: Vec<(usize, f64)> = shares.into_iter().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked
}

/// Pearson correlation of each dimension with the stack-up output, estimated
/// from a Monte-Carlo run. This measures linear sensitivity of the stack-up
/// to each input's variation.
///
/// # Panics
///
/// Panics if any dimension has non-finite tolerance bounds.
pub fn monte_carlo_sensitivities<R: Rng + ?Sized>(
    dims: &[DimTol],
    n: usize,
    rng: &mut R,
) -> Vec<f64> {
    let m = dims.len();
    let mut inputs: Vec<Vec<f64>> = (0..m).map(|_| Vec::with_capacity(n)).collect();
    let mut outputs = Vec::with_capacity(n);
    for _ in 0..n {
        let mut total = 0.0;
        for (j, d) in dims.iter().enumerate() {
            let u = Uniform::new(d.min(), d.max()).expect("finite tolerance bounds");
            let v = u.sample(rng);
            inputs[j].push(v);
            total += v;
        }
        outputs.push(total);
    }
    let out_mean = outputs.iter().sum::<f64>() / n as f64;
    let out_var = outputs.iter().map(|o| (o - out_mean).powi(2)).sum::<f64>();
    let out_std = out_var.sqrt();
    let mut corr = Vec::with_capacity(m);
    for col in &inputs {
        let in_mean = col.iter().sum::<f64>() / n as f64;
        let cov = col
            .iter()
            .zip(&outputs)
            .map(|(x, y)| (x - in_mean) * (y - out_mean))
            .sum::<f64>();
        let in_std = col
            .iter()
            .map(|x| (x - in_mean).powi(2))
            .sum::<f64>()
            .sqrt();
        let denom = in_std * out_std;
        corr.push(if denom > 0.0 { cov / denom } else { 0.0 });
    }
    corr
}

/// A single contributor to a 1-D tolerance stack-up.
///
/// `sign` is `+1.0` or `-1.0` and indicates the direction in which the member
/// contributes to the overall stack. `tol_plus` and `tol_minus` are the
/// one-sided deviations in the member's own positive and negative directions.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StackupMember {
    /// Nominal contribution of the member.
    pub nominal: f32,
    /// Allowable positive deviation of the member.
    pub tol_plus: f32,
    /// Allowable negative deviation of the member.
    pub tol_minus: f32,
    /// Direction of contribution, `+1.0` or `-1.0`.
    pub sign: f64,
}

/// A one-dimensional tolerance stack-up composed of several members.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Stackup {
    /// The contributing members.
    pub members: Vec<StackupMember>,
}

impl Stackup {
    /// Construct a stack-up from its members.
    #[must_use]
    pub fn new(members: Vec<StackupMember>) -> Self {
        Self { members }
    }

    /// Nominal stack-up length: the signed sum of member nominal contributions.
    #[must_use]
    pub fn nominal(&self) -> f32 {
        self.members.iter().map(|m| m.sign as f32 * m.nominal).sum()
    }

    /// Worst-case bounds `(lower, upper)`.
    ///
    /// Each member contributes a one-sided range
    /// `[sign*nominal - (sign>0 ? tol_minus : tol_plus),
    ///    sign*nominal + (sign>0 ? tol_plus   : tol_minus)]`,
    /// and the bounds are the sums of those per-member ranges.
    #[must_use]
    pub fn worst_case(&self) -> (f32, f32) {
        let mut lo = 0.0_f32;
        let mut hi = 0.0_f32;
        for m in &self.members {
            let signed_nominal = m.sign as f32 * m.nominal;
            let (neg, pos) = if m.sign >= 0.0 {
                (m.tol_minus, m.tol_plus)
            } else {
                (m.tol_plus, m.tol_minus)
            };
            lo += signed_nominal - neg;
            hi += signed_nominal + pos;
        }
        (lo, hi)
    }

    /// Root-sum-square bounds `(lower, upper)`.
    ///
    /// Each member is reduced to an equivalent symmetric tolerance
    /// `t_i = sqrt((tol_plus^2 + tol_minus^2) / 2)`, and the band is the nominal
    /// plus/minus `sqrt(Σ t_i^2)`.
    #[must_use]
    pub fn rss(&self) -> (f32, f32) {
        let mut sum_sq = 0.0_f32;
        for m in &self.members {
            let t_i = ((m.tol_plus * m.tol_plus + m.tol_minus * m.tol_minus) / 2.0).sqrt();
            sum_sq += t_i * t_i;
        }
        let dev = sum_sq.sqrt();
        let n = self.nominal();
        (n - dev, n + dev)
    }
}

impl StackupMember {
    /// Construct a symmetric member (equal `tol_plus` and `tol_minus`).
    #[must_use]
    pub fn symmetric(nominal: f32, tolerance: f32, sign: f64) -> Self {
        Self {
            nominal,
            tol_plus: tolerance,
            tol_minus: tolerance,
            sign,
        }
    }
}

/// Result of a Monte-Carlo tolerance stack-up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonteCarloResult {
    /// Mean of the resulting stack-up length over all samples.
    pub mean: f32,
    /// Standard deviation of the resulting stack-up length.
    pub std_dev: f32,
    /// Approximate lower bound at ±3σ.
    pub lower_3sigma: f32,
    /// Approximate upper bound at ±3σ.
    pub upper_3sigma: f32,
}

impl Stackup {
    /// Estimate the stack-up distribution by Monte-Carlo sampling.
    ///
    /// Each member's deviation is sampled uniformly from its `-tol_minus ..
    /// +tol_plus` one-sided range (the usual statistical-tolerance assumption),
    /// added to its signed nominal. Returns the sample mean, standard deviation,
    /// and a rough ±3σ band. `seed` initializes the internal generator.
    #[must_use]
    pub fn monte_carlo(&self, samples: u32, seed: u64) -> MonteCarloResult {
        let n = samples.max(1) as usize;
        let mut state = seed | 1;
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for _ in 0..n {
            let mut total = 0.0_f64;
            for m in &self.members {
                let (neg, pos) = if m.sign >= 0.0 {
                    (m.tol_minus, m.tol_plus)
                } else {
                    (m.tol_plus, m.tol_minus)
                };
                let (neg, pos) = (neg as f64, pos as f64);
                let u = lcg_uniform(&mut state);
                let dev = -neg + u * (neg + pos);
                total += m.sign * (m.nominal as f64 + dev);
            }
            sum += total;
            sum_sq += total * total;
        }
        let mean = (sum / n as f64) as f32;
        let variance = ((sum_sq / n as f64) - (sum / n as f64).powi(2)).max(0.0);
        let std = variance.sqrt() as f32;
        MonteCarloResult {
            mean,
            std_dev: std,
            lower_3sigma: mean - 3.0 * std,
            upper_3sigma: mean + 3.0 * std,
        }
    }
}

/// Deterministic xorshift64* generator state -> next state.
fn lcg_next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// Uniform `f64` in `[0, 1)` from the generator state.
fn lcg_uniform(state: &mut u64) -> f64 {
    (lcg_next(state) >> 11) as f64 / (1u64 << 53) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn dims() -> Vec<DimTol> {
        vec![
            DimTol::new("a", 10.0, 0.1),
            DimTol::new("b", 20.0, 0.2),
            DimTol::new("c", 5.0, 0.1),
        ]
    }

    #[test]
    fn worst_case_bounds() {
        let (lo, hi) = worst_case(&dims());
        assert_relative_eq!(lo, 35.0 - 0.4);
        assert_relative_eq!(hi, 35.0 + 0.4);
    }

    #[test]
    fn rss_is_tighter_than_worst_case() {
        let (wlo, whi) = worst_case(&dims());
        let (rlo, rhi) = rss(&dims());
        assert!(rlo > wlo);
        assert!(rhi < whi);
    }

    #[test]
    fn monte_carlo_mean_near_nominal() {
        let mut rng = rand::rng();
        let r = monte_carlo(&dims(), 50_000, None, &mut rng);
        assert_relative_eq!(r.mean, 35.0, epsilon = 0.05);
    }

    #[test]
    fn yield_within_wide_spec() {
        let mut rng = rand::rng();
        // Worst-case bounds are 34.6..35.4; a wider spec yields ~100%.
        let r = monte_carlo(&dims(), 20_000, Some((34.0, 36.0)), &mut rng);
        assert!((r.yield_fraction.unwrap() - 1.0).abs() < 0.02);
    }

    #[test]
    fn contributor_ranking() {
        let ranked = rank_contributors(&dims());
        // "b" has the largest tolerance (0.2) so should rank first.
        assert_eq!(ranked[0].0, 1);
        let shares: Vec<f64> = ranked.iter().map(|(_, s)| *s).collect();
        let sum: f64 = shares.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn contributor_ranking_handles_non_finite_tolerance() {
        let dims = vec![
            DimTol::new("a", 10.0, 0.1),
            DimTol::new("b", 20.0, f64::NAN),
            DimTol::new("c", 5.0, 0.1),
        ];
        // Must not panic; NaN contributions sort deterministically via total_cmp.
        let ranked = rank_contributors(&dims);
        assert_eq!(ranked.len(), 3);
    }

    #[test]
    fn sensitivity_uniform_correlations() {
        let mut rng = rand::rng();
        let corr = monte_carlo_sensitivities(&dims(), 50_000, &mut rng);
        // For a sum of independent uniforms, corr(X_i, sum) = sqrt(var_i / sum_var),
        // so it grows with each dimension's tolerance. "b" (tol 0.2) must rank
        // highest and "a"/"c" (tol 0.1) must be equal.
        assert!(corr[1] > corr[0], "b should correlate most");
        assert!(corr[1] > corr[2], "b should correlate most");
        assert_relative_eq!(corr[0], corr[2], epsilon = 0.05);
        assert!(corr[0] > 0.0);
    }
}
