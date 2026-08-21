//! Bayesian inference: Metropolis-Hastings and a basic HMC, parallelized across
//! chains via Rayon, with a typed [`Posterior`].

use rand::distributions::Distribution as RandDist;
use rand::thread_rng;
use rand::Rng;
use rand_distr::{Normal, Uniform};
use rayon::prelude::*;

use crate::StatError;

/// A single MCMC chain.
#[derive(Debug, Clone)]
pub struct Chain {
    pub draws: Vec<Vec<f64>>,
    pub log_posts: Vec<f64>,
}

impl Chain {
    pub fn dim(&self) -> usize {
        self.draws.first().map(|d| d.len()).unwrap_or(0)
    }
    /// Marginal samples for parameter index `d`.
    pub fn marginal(&self, d: usize) -> Vec<f64> {
        self.draws.iter().map(|row| row[d]).collect()
    }
    pub fn mean(&self, d: usize) -> f64 {
        let m = self.marginal(d);
        m.iter().sum::<f64>() / m.len() as f64
    }
}

/// Aggregate of one or more chains.
#[derive(Debug, Clone)]
pub struct Posterior {
    pub chains: Vec<Chain>,
    pub names: Vec<String>,
}

impl Posterior {
    pub fn new(chains: Vec<Chain>, names: Vec<String>) -> Self {
        Self { chains, names }
    }

    pub fn dim(&self) -> usize {
        self.chains.first().map(|c| c.dim()).unwrap_or(0)
    }

    /// Flattened marginal across all chains for parameter `d`.
    pub fn marginal(&self, d: usize) -> Vec<f64> {
        self.chains.iter().flat_map(|c| c.marginal(d)).collect()
    }

    pub fn mean(&self, name: &str) -> f64 {
        let d = self.index(name);
        let m = self.marginal(d);
        m.iter().sum::<f64>() / m.len() as f64
    }

    fn index(&self, name: &str) -> usize {
        self.names.iter().position(|n| n == name).unwrap_or(0)
    }

    /// Equal-tailed credible interval at the given level (e.g. 0.95).
    pub fn credible_interval(&self, name: &str, level: f64) -> (f64, f64) {
        let mut m = self.marginal(self.index(name));
        if m.is_empty() {
            return (f64::NAN, f64::NAN);
        }
        m.sort_by(|a, b| a.total_cmp(b));
        let last = m.len() - 1;
        let lo = ((m.len() as f64 * (1.0 - level) / 2.0) as usize).min(last);
        let hi = ((m.len() as f64 * (1.0 + level) / 2.0) as usize).min(last);
        (m[lo], m[hi])
    }

    /// Gelman-Rubin R-hat across chains (1.0 == converged).
    pub fn rhat(&self, name: &str) -> Result<f64, StatError> {
        let d = self.index(name);
        let chains: Vec<Vec<f64>> = self.chains.iter().map(|c| c.marginal(d)).collect();
        if chains.len() < 2 {
            return Err(StatError::Degenerate(
                "rhat requires >= 2 chains".to_string(),
            ));
        }
        if chains.iter().any(|c| c.is_empty()) {
            return Err(StatError::Degenerate(
                "rhat requires non-empty chains".to_string(),
            ));
        }
        let m = chains.len() as f64;
        let n = chains[0].len() as f64;
        let chain_means: Vec<f64> = chains.iter().map(|c| c.iter().sum::<f64>() / n).collect();
        let grand = chain_means.iter().sum::<f64>() / m;
        let b = n * chain_means
            .iter()
            .map(|cm| (cm - grand).powi(2))
            .sum::<f64>()
            / (m - 1.0);
        let w = chains
            .iter()
            .map(|c| c.iter().map(|v| (v - grand).powi(2)).sum::<f64>() / (n - 1.0))
            .sum::<f64>()
            / m;
        let v = (n - 1.0) / n * w + b / n;
        Ok((v / w).sqrt())
    }
}

/// Log-posterior density type: `params -> log density`.
pub type LogPost = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;

fn numerical_grad(f: &LogPost, x: &[f64], eps: f64) -> Vec<f64> {
    x.iter()
        .enumerate()
        .map(|(i, _)| {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[i] += eps;
            xm[i] -= eps;
            (f(&xp) - f(&xm)) / (2.0 * eps)
        })
        .collect()
}

/// Metropolis-Hastings (random-walk) sampler.
pub fn sample_mh(
    log_post: &LogPost,
    init: &[f64],
    proposal_sd: &[f64],
    n_samples: usize,
    rng: &mut impl Rng,
) -> Chain {
    let dim = init.len();
    let mut current = init.to_vec();
    let mut lp = log_post(&current);
    let mut draws = Vec::with_capacity(n_samples);
    let mut log_posts = Vec::with_capacity(n_samples);
    let normal = Normal::new(0.0, 1.0).unwrap();
    for _ in 0..n_samples {
        let proposal: Vec<f64> = (0..dim)
            .map(|i| current[i] + proposal_sd[i] * normal.sample(rng))
            .collect();
        let lp_prop = log_post(&proposal);
        let log_accept = lp_prop - lp;
        if log_accept >= 0.0 || rng.r#gen::<f64>().ln() < log_accept {
            current = proposal;
            lp = lp_prop;
        }
        draws.push(current.clone());
        log_posts.push(lp);
    }
    Chain { draws, log_posts }
}

/// Hamiltonian Monte Carlo with a simple leapfrog integrator and numerical
/// gradients.
pub fn sample_hmc(
    log_post: &LogPost,
    init: &[f64],
    n_samples: usize,
    step_size: f64,
    n_leapfrog: usize,
    rng: &mut impl Rng,
) -> Chain {
    let dim = init.len();
    let mut q = init.to_vec();
    let mut lp = log_post(&q);
    let mut draws = Vec::with_capacity(n_samples);
    let mut log_posts = Vec::with_capacity(n_samples);
    let normal = Normal::new(0.0, 1.0).unwrap();
    for _ in 0..n_samples {
        let mut p: Vec<f64> = (0..dim).map(|_| normal.sample(rng)).collect();
        let current_q = q.clone();
        let current_lp = lp;
        let current_p = p.clone();
        let grad = numerical_grad(log_post, &q, 1e-4);
        for i in 0..dim {
            p[i] += 0.5 * step_size * grad[i];
        }
        for _ in 0..n_leapfrog {
            for i in 0..dim {
                q[i] += step_size * p[i];
            }
            let g = numerical_grad(log_post, &q, 1e-4);
            for i in 0..dim {
                p[i] += step_size * g[i];
            }
            for i in 0..dim {
                p[i] -= 0.5 * step_size * g[i];
            }
        }
        let lp_new = log_post(&q);
        let kinetic = |p: &[f64]| p.iter().map(|v| v * v / 2.0).sum::<f64>();
        let current_k = kinetic(&current_p);
        let new_k = kinetic(&p);
        let log_accept = (lp_new - current_lp) + (current_k - new_k);
        if log_accept >= 0.0 || rng.r#gen::<f64>().ln() < log_accept {
            lp = lp_new;
        } else {
            q = current_q;
            lp = current_lp;
        }
        draws.push(q.clone());
        log_posts.push(lp);
    }
    Chain { draws, log_posts }
}

/// Run `n_chains` chains in parallel (each with its own RNG) and combine.
///
/// Non-deterministic: each chain's RNG is seeded from the system entropy
/// source. For reproducible runs use [`sample_parallel_seeded`].
pub fn sample_parallel(
    log_post: &LogPost,
    inits: &[Vec<f64>],
    method: &Sampler,
    n_samples: usize,
) -> Posterior {
    sample_parallel_seeded(log_post, inits, method, n_samples, rand::thread_rng().gen())
}

/// Deterministic variant of [`sample_parallel`]: chain `i` is seeded with
/// `seed + i`, so identical inputs always produce identical draws (stable
/// statistical gates / regression tests).
pub fn sample_parallel_seeded(
    log_post: &LogPost,
    inits: &[Vec<f64>],
    method: &Sampler,
    n_samples: usize,
    seed: u64,
) -> Posterior {
    use rand::SeedableRng;
    let names: Vec<String> = (0..inits[0].len()).map(|i| format!("p{}", i)).collect();
    let chains: Vec<Chain> = inits
        .par_iter()
        .enumerate()
        .map(|(ci, init)| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(ci as u64));
            match method {
                Sampler::MH { proposal_sd } => {
                    sample_mh(log_post, init, proposal_sd, n_samples, &mut rng)
                }
                Sampler::HMC {
                    step_size,
                    n_leapfrog,
                } => sample_hmc(log_post, init, n_samples, *step_size, *n_leapfrog, &mut rng),
                Sampler::NUTS {
                    step_size,
                    max_depth,
                } => sample_nuts(log_post, init, n_samples, *step_size, *max_depth, &mut rng),
            }
        })
        .collect();
    Posterior::new(chains, names)
}

/// Sampler configuration.
#[derive(Clone)]
pub enum Sampler {
    MH {
        proposal_sd: Vec<f64>,
    },
    HMC {
        step_size: f64,
        n_leapfrog: usize,
    },
    /// No-U-Turn Sampler: adaptively sized leapfrog trajectory.
    NUTS {
        step_size: f64,
        max_depth: usize,
    },
}

/// Build a jittered set of initial points around `center`.
pub fn jitter_inits(center: &[f64], n_chains: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    let u = Uniform::new(-0.5, 0.5);
    (0..n_chains)
        .map(|_| center.iter().map(|c| c + u.sample(&mut rng)).collect())
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn sub_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// One leapfrog step using numerical gradients of `log_post`.
fn leapfrog(q: &[f64], r: &[f64], eps: f64, log_post: &LogPost) -> (Vec<f64>, Vec<f64>, f64) {
    let g0 = numerical_grad(log_post, q, 1e-4);
    let r1: Vec<f64> = r
        .iter()
        .zip(&g0)
        .map(|(ri, gi)| ri + 0.5 * eps * gi)
        .collect();
    let q1: Vec<f64> = q.iter().zip(&r1).map(|(qi, ri)| qi + eps * ri).collect();
    let g1 = numerical_grad(log_post, &q1, 1e-4);
    let r2: Vec<f64> = r1
        .iter()
        .zip(&g1)
        .map(|(ri, gi)| ri + 0.5 * eps * gi)
        .collect();
    let lp = log_post(&q1);
    (q1, r2, lp)
}

/// Margin (in log-space) beyond which a subtree is abandoned as divergent.
const NUTS_DELTA_MAX: f64 = 1000.0;

struct Tree {
    theta_minus: Vec<f64>,
    r_minus: Vec<f64>,
    theta_plus: Vec<f64>,
    r_plus: Vec<f64>,
    theta: Vec<f64>,
    n: usize,
    s: bool,
    alpha: f64,
}

/// Recursive NUTS subtree builder (Hoffman & Gelman, 2014).
fn build_tree(
    theta: &[f64],
    r: &[f64],
    logu: f64,
    v: f64,
    depth: usize,
    eps: f64,
    log_post: &LogPost,
) -> Tree {
    if depth == 0 {
        let (q, p, lp) = leapfrog(theta, r, v * eps, log_post);
        let joint = lp - 0.5 * dot(&p, &p);
        let n = if joint > logu { 1 } else { 0 };
        let s = joint - logu < NUTS_DELTA_MAX;
        return Tree {
            theta_minus: q.clone(),
            r_minus: p.clone(),
            theta_plus: q.clone(),
            r_plus: p.clone(),
            theta: q,
            n,
            s,
            alpha: 1.0,
        };
    }
    let mut sub = build_tree(theta, r, logu, v, depth - 1, eps, log_post);
    if v == -1.0 {
        let left = build_tree(
            &sub.theta_minus,
            &sub.r_minus,
            logu,
            v,
            depth - 1,
            eps,
            log_post,
        );
        sub.theta_minus = left.theta_minus.clone();
        sub.r_minus = left.r_minus.clone();
        let accept = (sub.n + left.n) > 0
            && (left.n as f64 / (sub.n + left.n) as f64) > thread_rng().gen::<f64>();
        if accept {
            sub.theta = left.theta.clone();
        }
        sub.n += left.n;
        sub.s = sub.s && left.s;
        sub.alpha += left.alpha;
    } else {
        let right = build_tree(
            &sub.theta_plus,
            &sub.r_plus,
            logu,
            v,
            depth - 1,
            eps,
            log_post,
        );
        sub.theta_plus = right.theta_plus.clone();
        sub.r_plus = right.r_plus.clone();
        let accept = (sub.n + right.n) > 0
            && (right.n as f64 / (sub.n + right.n) as f64) > thread_rng().gen::<f64>();
        if accept {
            sub.theta = right.theta.clone();
        }
        sub.n += right.n;
        sub.s = sub.s && right.s;
        sub.alpha += right.alpha;
    }
    sub
}

/// Hamiltonian Monte Carlo with the No-U-Turn sampler: adaptively builds a
/// balanced binary tree of leapfrog steps so the trajectory stops before it
/// doubles back on itself, removing the need to tune the leapfrog count.
pub fn sample_nuts(
    log_post: &LogPost,
    init: &[f64],
    n_samples: usize,
    step_size: f64,
    max_depth: usize,
    rng: &mut impl Rng,
) -> Chain {
    let dim = init.len();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut q = init.to_vec();
    let mut lp = log_post(&q);
    let mut draws = Vec::with_capacity(n_samples);
    let mut log_posts = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let p: Vec<f64> = (0..dim).map(|_| normal.sample(rng)).collect();
        let joint0 = lp - 0.5 * dot(&p, &p);
        let logu = joint0 + rng.gen::<f64>().ln();
        let mut theta_minus = q.clone();
        let mut r_minus = p.clone();
        let mut theta_plus = q.clone();
        let mut r_plus = p.clone();
        let mut theta = q.clone();
        let mut n: usize = 1;
        let mut s = true;
        let mut depth = 0;
        while s && depth < max_depth {
            let v = if rng.gen::<f64>() < 0.5 { -1.0 } else { 1.0 };
            let tree = if v == -1.0 {
                build_tree(&theta_minus, &r_minus, logu, v, depth, step_size, log_post)
            } else {
                build_tree(&theta_plus, &r_plus, logu, v, depth, step_size, log_post)
            };
            let accept =
                (n + tree.n) > 0 && (tree.n as f64 / (n + tree.n) as f64) > rng.gen::<f64>();
            if accept {
                theta = tree.theta.clone();
            }
            theta_minus = tree.theta_minus.clone();
            r_minus = tree.r_minus.clone();
            theta_plus = tree.theta_plus.clone();
            r_plus = tree.r_plus.clone();
            n += tree.n;
            s = tree.s
                && dot(&sub_vec(&theta_plus, &theta_minus), &r_minus) > 0.0
                && dot(&sub_vec(&theta_plus, &theta_minus), &r_plus) > 0.0;
            depth += 1;
        }
        q = theta;
        lp = log_post(&q);
        draws.push(q.clone());
        log_posts.push(lp);
    }
    Chain { draws, log_posts }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mh_recovers_normal_mean() {
        // Posterior: N(mu | 3, 1) on a single parameter.
        let target: LogPost = Box::new(|x: &[f64]| -0.5 * (x[0] - 3.0).powi(2));
        let inits = jitter_inits(&[0.0], 4);
        let post = sample_parallel(
            &target,
            &inits,
            &Sampler::MH {
                proposal_sd: vec![0.5],
            },
            2000,
        );
        let m = post.mean("p0");
        assert!((m - 3.0).abs() < 0.2, "mean={}", m);
        let r = post.rhat("p0").expect("rhat");
        assert!(r < 1.1, "rhat={}", r);
    }

    #[test]
    fn hmc_two_dim_gaussian() {
        // N([0,0], I)
        let target: LogPost = Box::new(|x: &[f64]| -0.5 * x.iter().map(|v| v * v).sum::<f64>());
        let inits = jitter_inits(&[0.0, 0.0], 2);
        let post = sample_parallel(
            &target,
            &inits,
            &Sampler::HMC {
                step_size: 0.25,
                n_leapfrog: 20,
            },
            1500,
        );
        assert!((post.mean("p0")).abs() < 0.2);
        assert!(post.rhat("p1").expect("rhat") < 1.1);
    }

    #[test]
    fn nuts_two_dim_gaussian() {
        // N([0,0], I) — NUTS should match HMC's recovery.
        // Seeded: this is a reproducible regression gate, not a lottery.
        let target: LogPost = Box::new(|x: &[f64]| -0.5 * x.iter().map(|v| v * v).sum::<f64>());
        let inits = jitter_inits(&[0.0, 0.0], 2);
        let post = sample_parallel_seeded(
            &target,
            &inits,
            &Sampler::NUTS {
                step_size: 0.25,
                max_depth: 6,
            },
            3000,
            42,
        );
        assert!((post.mean("p0")).abs() < 0.2, "mean={}", post.mean("p0"));
        assert!((post.mean("p1")).abs() < 0.2, "mean={}", post.mean("p1"));
        let r0 = post.rhat("p0").expect("rhat");
        let r1 = post.rhat("p1").expect("rhat");
        assert!(r0 < 1.1, "rhat={}", r0);
        assert!(r1 < 1.1, "rhat={}", r1);
    }
}
