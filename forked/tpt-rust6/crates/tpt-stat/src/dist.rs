//! Probability distributions with vectorized PDF/CDF and parallel sampling.

use rand::distributions::{Distribution as RandDist, Uniform as RandUniform};
use rand::rngs::ThreadRng;
use rand_distr::{Bernoulli, Beta, Binomial, Exp, Gamma, Normal, Poisson, StudentT};
use rayon::prelude::*;

use crate::special;
use crate::StatError;

/// A univariate continuous/discrete distribution.
pub trait Distribution {
    /// Probability density (continuous) or mass (discrete) at `x`.
    fn pdf(&self, x: f64) -> f64;
    /// Cumulative distribution function `P(X <= x)`.
    fn cdf(&self, x: f64) -> f64;
    fn mean(&self) -> f64;
    fn variance(&self) -> f64;
    /// Draw a single sample.
    fn sample(&self, rng: &mut ThreadRng) -> Result<f64, StatError>;
    /// Draw `n` samples in parallel.
    fn sample_n(&self, n: usize) -> Result<Vec<f64>, StatError>
    where
        Self: Sync,
    {
        (0..n)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                self.sample(&mut rng)
            })
            .collect()
    }
}

fn invalid(msg: impl std::fmt::Display) -> StatError {
    StatError::InvalidParams(msg.to_string())
}

pub struct NormalDist {
    pub mu: f64,
    pub sigma: f64,
}
impl NormalDist {
    /// Validate `sigma > 0` (and finite parameters) before constructing.
    pub fn try_new(mu: f64, sigma: f64) -> Result<Self, StatError> {
        if !mu.is_finite() {
            return Err(invalid(format!("Normal: mu must be finite, got {mu}")));
        }
        if !(sigma > 0.0) || !sigma.is_finite() {
            return Err(invalid(format!("Normal: sigma must be > 0, got {sigma}")));
        }
        Ok(Self { mu, sigma })
    }
}
impl Distribution for NormalDist {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        (-0.5 * z * z).exp() / (self.sigma * (2.0 * std::f64::consts::PI).sqrt())
    }
    fn cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + special::erf((x - self.mu) / (self.sigma * (2.0f64).sqrt())))
    }
    fn mean(&self) -> f64 {
        self.mu
    }
    fn variance(&self) -> f64 {
        self.sigma * self.sigma
    }
    fn sample(&self, rng: &mut ThreadRng) -> Result<f64, StatError> {
        let d = Normal::new(self.mu, self.sigma)
            .map_err(|e| invalid(format!("Normal(mu={}, sigma={}): {e}", self.mu, self.sigma)))?;
        Ok(d.sample(rng))
    }
}

pub struct UniformDist {
    pub a: f64,
    pub b: f64,
}
impl UniformDist {
    /// Validate `a < b` before constructing.
    pub fn try_new(a: f64, b: f64) -> Result<Self, StatError> {
        if !a.is_finite() || !b.is_finite() {
            return Err(invalid(format!(
                "Uniform: bounds must be finite, got a={a}, b={b}"
            )));
        }
        if !(a < b) {
            return Err(invalid(format!(
                "Uniform: requires a < b, got a={a}, b={b}"
            )));
        }
        Ok(Self { a, b })
    }
}
impl Distribution for UniformDist {
    fn pdf(&self, x: f64) -> f64 {
        if self.a <= x && x <= self.b {
            1.0 / (self.b - self.a)
        } else {
            0.0
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.a {
            0.0
        } else if x >= self.b {
            1.0
        } else {
            (x - self.a) / (self.b - self.a)
        }
    }
    fn mean(&self) -> f64 {
        0.5 * (self.a + self.b)
    }
    fn variance(&self) -> f64 {
        (self.b - self.a).powi(2) / 12.0
    }
    fn sample(&self, rng: &mut ThreadRng) -> Result<f64, StatError> {
        if !self.a.is_finite() || !self.b.is_finite() || !(self.a < self.b) {
            return Err(invalid(format!(
                "Uniform: requires finite a < b, got a={}, b={}",
                self.a, self.b
            )));
        }
        let d = RandUniform::new(self.a, self.b);
        Ok(d.sample(rng))
    }
}

pub struct ExponentialDist {
    pub lambda: f64,
}
impl ExponentialDist {
    /// Validate `lambda > 0` before constructing.
    pub fn try_new(lambda: f64) -> Result<Self, StatError> {
        if !(lambda > 0.0) || !lambda.is_finite() {
            return Err(invalid(format!(
                "Exponential: lambda must be > 0, got {lambda}"
            )));
        }
        Ok(Self { lambda })
    }
}
impl Distribution for ExponentialDist {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.lambda * (-self.lambda * x).exp()
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            1.0 - (-self.lambda * x).exp()
        }
    }
    fn mean(&self) -> f64 {
        1.0 / self.lambda
    }
    fn variance(&self) -> f64 {
        1.0 / (self.lambda * self.lambda)
    }
    fn sample(&self, rng: &mut ThreadRng) -> Result<f64, StatError> {
        let d = Exp::new(self.lambda)
            .map_err(|e| invalid(format!("Exponential(lambda={}): {e}", self.lambda)))?;
        Ok(d.sample(rng))
    }
}

pub struct GammaDist {
    pub shape: f64,
    pub scale: f64,
}
impl GammaDist {
    /// Validate `shape > 0` and `scale > 0` before constructing.
    pub fn try_new(shape: f64, scale: f64) -> Result<Self, StatError> {
        if !(shape > 0.0) || !shape.is_finite() {
            return Err(invalid(format!("Gamma: shape must be > 0, got {shape}")));
        }
        if !(scale > 0.0) || !scale.is_finite() {
            return Err(invalid(format!("Gamma: scale must be > 0, got {scale}")));
        }
        Ok(Self { shape, scale })
    }
}
impl Distribution for GammaDist {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let k = self.shape;
        x.powf(k - 1.0) * (-x / self.scale).exp() / (self.scale.powf(k) * special::lgamma(k).exp())
    }
    fn cdf(&self, x: f64) -> f64 {
        special::gammap(self.shape, x / self.scale)
    }
    fn mean(&self) -> f64 {
        self.shape * self.scale
    }
    fn variance(&self) -> f64 {
        self.shape * self.scale * self.scale
    }
    fn sample(&self, rng: &mut ThreadRng) -> Result<f64, StatError> {
        let d = Gamma::new(self.shape, self.scale).map_err(|e| {
            invalid(format!(
                "Gamma(shape={}, scale={}): {e}",
                self.shape, self.scale
            ))
        })?;
        Ok(d.sample(rng))
    }
}

pub struct ChiSquaredDist {
    pub k: f64,
}
impl ChiSquaredDist {
    /// Validate `k > 0` degrees of freedom before constructing.
    pub fn try_new(k: f64) -> Result<Self, StatError> {
        if !(k > 0.0) || !k.is_finite() {
            return Err(invalid(format!("ChiSquared: k must be > 0, got {k}")));
        }
        Ok(Self { k })
    }
}
impl Distribution for ChiSquaredDist {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        x.powf(self.k / 2.0 - 1.0) * (-x / 2.0).exp()
            / (2.0f64.powf(self.k / 2.0) * special::lgamma(self.k / 2.0).exp())
    }
    fn cdf(&self, x: f64) -> f64 {
        special::gammap(self.k / 2.0, x / 2.0)
    }
    fn mean(&self) -> f64 {
        self.k
    }
    fn variance(&self) -> f64 {
        2.0 * self.k
    }
    fn sample(&self, rng: &mut ThreadRng) -> Result<f64, StatError> {
        let d = Gamma::new(self.k / 2.0, 2.0)
            .map_err(|e| invalid(format!("ChiSquared(k={}): {e}", self.k)))?;
        Ok(d.sample(rng))
    }
}

pub struct StudentTDist {
    pub nu: f64,
}
impl StudentTDist {
    /// Validate `nu > 0` degrees of freedom before constructing.
    pub fn try_new(nu: f64) -> Result<Self, StatError> {
        if !(nu > 0.0) || !nu.is_finite() {
            return Err(invalid(format!("StudentT: nu must be > 0, got {nu}")));
        }
        Ok(Self { nu })
    }
}
impl Distribution for StudentTDist {
    fn pdf(&self, x: f64) -> f64 {
        let nu = self.nu;
        let c = special::lgamma((nu + 1.0) / 2.0).exp()
            / (special::lgamma(nu / 2.0).exp() * (nu * std::f64::consts::PI).sqrt());
        c * (1.0 + x * x / nu).powf(-(nu + 1.0) / 2.0)
    }
    fn cdf(&self, x: f64) -> f64 {
        let nu = self.nu;
        let t = if x > 0.0 { 1.0 } else { -1.0 };
        let xt = x / (1.0 + x * x / nu).sqrt();
        let ib = special::betai(nu / 2.0, 0.5, nu / (nu + xt * xt));
        0.5 * (1.0 + t * (1.0 - ib))
    }
    fn mean(&self) -> f64 {
        if self.nu > 1.0 {
            0.0
        } else {
            f64::NAN
        }
    }
    fn variance(&self) -> f64 {
        if self.nu > 2.0 {
            self.nu / (self.nu - 2.0)
        } else {
            f64::NAN
        }
    }
    fn sample(&self, rng: &mut ThreadRng) -> Result<f64, StatError> {
        let d = StudentT::new(self.nu)
            .map_err(|e| invalid(format!("StudentT(nu={}): {e}", self.nu)))?;
        Ok(d.sample(rng))
    }
}

pub struct BetaDist {
    pub a: f64,
    pub b: f64,
}
impl BetaDist {
    /// Validate both shape parameters are `> 0` before constructing.
    pub fn try_new(a: f64, b: f64) -> Result<Self, StatError> {
        if !(a > 0.0) || !a.is_finite() {
            return Err(invalid(format!("Beta: a must be > 0, got {a}")));
        }
        if !(b > 0.0) || !b.is_finite() {
            return Err(invalid(format!("Beta: b must be > 0, got {b}")));
        }
        Ok(Self { a, b })
    }
}
impl Distribution for BetaDist {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            return 0.0;
        }
        x.powf(self.a - 1.0) * (1.0 - x).powf(self.b - 1.0)
            / (special::lgamma(self.a).exp() * special::lgamma(self.b).exp()
                / special::lgamma(self.a + self.b).exp())
    }
    fn cdf(&self, x: f64) -> f64 {
        special::betai(self.a, self.b, x)
    }
    fn mean(&self) -> f64 {
        self.a / (self.a + self.b)
    }
    fn variance(&self) -> f64 {
        self.a * self.b / ((self.a + self.b).powi(2) * (self.a + self.b + 1.0))
    }
    fn sample(&self, rng: &mut ThreadRng) -> Result<f64, StatError> {
        let d = Beta::new(self.a, self.b)
            .map_err(|e| invalid(format!("Beta(a={}, b={}): {e}", self.a, self.b)))?;
        Ok(d.sample(rng))
    }
}

pub struct BernoulliDist {
    pub p: f64,
}
impl BernoulliDist {
    /// Validate `0 <= p <= 1` before constructing.
    pub fn try_new(p: f64) -> Result<Self, StatError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(invalid(format!("Bernoulli: p must be in [0, 1], got {p}")));
        }
        Ok(Self { p })
    }
}
impl Distribution for BernoulliDist {
    fn pdf(&self, x: f64) -> f64 {
        if (x - 0.0).abs() < 1e-9 {
            1.0 - self.p
        } else if (x - 1.0).abs() < 1e-9 {
            self.p
        } else {
            0.0
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else if x < 1.0 {
            1.0 - self.p
        } else {
            1.0
        }
    }
    fn mean(&self) -> f64 {
        self.p
    }
    fn variance(&self) -> f64 {
        self.p * (1.0 - self.p)
    }
    fn sample(&self, rng: &mut ThreadRng) -> Result<f64, StatError> {
        let d =
            Bernoulli::new(self.p).map_err(|e| invalid(format!("Bernoulli(p={}): {e}", self.p)))?;
        Ok(d.sample(rng) as u8 as f64)
    }
}

pub struct BinomialDist {
    pub n: u64,
    pub p: f64,
}
impl BinomialDist {
    /// Validate `0 <= p <= 1` for a valid trial count before constructing.
    pub fn try_new(n: u64, p: f64) -> Result<Self, StatError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(invalid(format!("Binomial: p must be in [0, 1], got {p}")));
        }
        if n > i64::MAX as u64 {
            return Err(invalid(format!("Binomial: n is too large, got {n}")));
        }
        Ok(Self { n, p })
    }
}
impl Distribution for BinomialDist {
    fn pdf(&self, x: f64) -> f64 {
        let k = x.round() as i64;
        if k < 0 || k > self.n as i64 {
            return 0.0;
        }
        let nf = self.n as f64;
        let kf = k as f64;
        let comb = (special::lgamma(nf + 1.0)
            - special::lgamma(kf + 1.0)
            - special::lgamma(nf - kf + 1.0))
        .exp();
        comb * self.p.powf(kf) * (1.0 - self.p).powf(nf - kf)
    }
    fn cdf(&self, x: f64) -> f64 {
        let k = x.floor() as i64;
        let mut s = 0.0;
        for j in 0..=k.max(0) {
            s += self.pdf(j as f64);
        }
        s
    }
    fn mean(&self) -> f64 {
        self.n as f64 * self.p
    }
    fn variance(&self) -> f64 {
        self.n as f64 * self.p * (1.0 - self.p)
    }
    fn sample(&self, rng: &mut ThreadRng) -> Result<f64, StatError> {
        let d = Binomial::new(self.n, self.p)
            .map_err(|e| invalid(format!("Binomial(n={}, p={}): {e}", self.n, self.p)))?;
        Ok(d.sample(rng) as f64)
    }
}

pub struct PoissonDist {
    pub lambda: f64,
}
impl PoissonDist {
    /// Validate `lambda > 0` before constructing.
    pub fn try_new(lambda: f64) -> Result<Self, StatError> {
        if !(lambda > 0.0) || !lambda.is_finite() {
            return Err(invalid(format!(
                "Poisson: lambda must be > 0, got {lambda}"
            )));
        }
        Ok(Self { lambda })
    }
}
impl Distribution for PoissonDist {
    fn pdf(&self, x: f64) -> f64 {
        let k = x.round() as i64;
        if k < 0 {
            return 0.0;
        }
        let kf = k as f64;
        (self.lambda.powf(kf) * (-self.lambda).exp()) / special::lgamma(kf + 1.0).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        let k = x.floor() as i64;
        let mut s = 0.0;
        for j in 0..=k.max(0) {
            s += self.pdf(j as f64);
        }
        s
    }
    fn mean(&self) -> f64 {
        self.lambda
    }
    fn variance(&self) -> f64 {
        self.lambda
    }
    fn sample(&self, rng: &mut ThreadRng) -> Result<f64, StatError> {
        let d = Poisson::new(self.lambda)
            .map_err(|e| invalid(format!("Poisson(lambda={}): {e}", self.lambda)))?;
        Ok(d.sample(rng) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn normal_pdf_cdf() {
        let n = NormalDist {
            mu: 0.0,
            sigma: 1.0,
        };
        assert!(approx(n.pdf(0.0), 0.3989, 1e-3));
        assert!(approx(n.cdf(0.0), 0.5, 1e-9));
        assert!(approx(n.cdf(1.96), 0.975, 1e-2));
    }

    #[test]
    fn uniform_sampling_in_range() {
        let u = UniformDist { a: 2.0, b: 5.0 };
        let mut rng = thread_rng();
        for _ in 0..100 {
            let s = u.sample(&mut rng).unwrap();
            assert!((2.0..=5.0).contains(&s));
        }
        assert!(approx(u.mean(), 3.5, 1e-9));
    }

    #[test]
    fn chi_squared_mean() {
        let c = ChiSquaredDist { k: 4.0 };
        assert!(approx(c.mean(), 4.0, 1e-9));
        assert!(approx(c.variance(), 8.0, 1e-9));
    }

    #[test]
    fn parallel_sampling_size() {
        let n = NormalDist {
            mu: 1.0,
            sigma: 2.0,
        };
        let s = n.sample_n(1000).unwrap();
        assert_eq!(s.len(), 1000);
    }
}
