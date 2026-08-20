//! Optimizers operating on a flat parameter vector.
//!
//! Every optimizer implements [`Optimizer::step`], which updates `params`
//! in-place from `grads` (same ordering as [`crate::model::Model::params`]).
//! State vectors are lazily sized on the first step.

/// A gradient-descent style parameter update rule.
pub trait Optimizer {
    /// Apply one update of `params` using `grads`.
    fn step(&mut self, params: &mut Vec<f64>, grads: &[f64]);
    /// Current learning rate.
    fn lr(&self) -> f64;
    /// Override the learning rate (used by [`crate::trainer::LrSchedule`]).
    fn set_lr(&mut self, lr: f64);
    /// Drop accumulated moments/momentum.
    fn reset(&mut self) {}
}

fn fit_state(v: &mut Vec<f64>, n: usize) {
    if v.len() != n {
        v.clear();
        v.resize(n, 0.0);
    }
}

fn l2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Stochastic gradient descent with optional (classical) momentum and L2 decay.
#[derive(Clone, Debug)]
pub struct Sgd {
    /// Learning rate.
    pub lr: f64,
    /// Momentum coefficient (0 disables momentum).
    pub momentum: f64,
    /// Coupled L2 weight decay added to the gradient.
    pub weight_decay: f64,
    velocity: Vec<f64>,
}

impl Sgd {
    /// SGD with the given learning rate, no momentum and no decay.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            velocity: Vec::new(),
        }
    }
    /// Set the momentum coefficient.
    pub fn momentum(mut self, m: f64) -> Self {
        self.momentum = m;
        self
    }
    /// Set the coupled L2 weight decay.
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, params: &mut Vec<f64>, grads: &[f64]) {
        fit_state(&mut self.velocity, params.len());
        for i in 0..params.len().min(grads.len()) {
            let g = grads[i] + self.weight_decay * params[i];
            self.velocity[i] = self.momentum * self.velocity[i] + g;
            params[i] -= self.lr * self.velocity[i];
        }
    }
    fn lr(&self) -> f64 {
        self.lr
    }
    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
    fn reset(&mut self) {
        self.velocity.clear();
    }
}

/// Shared first/second-moment state for Adam-family optimizers.
#[derive(Clone, Debug)]
struct Moments {
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
}

impl Moments {
    fn new() -> Self {
        Self {
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
    /// Advance the step counter and return bias-correction denominators.
    fn advance(&mut self, n: usize, b1: f64, b2: f64) -> (f64, f64) {
        fit_state(&mut self.m, n);
        fit_state(&mut self.v, n);
        self.t += 1;
        let t = self.t as i32;
        (1.0 - b1.powi(t), 1.0 - b2.powi(t))
    }
    fn update(&mut self, i: usize, g: f64, b1: f64, b2: f64) {
        self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
        self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
    }
    fn clear(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }
}

/// Adam (Kingma & Ba, 2015) with bias correction and coupled L2 decay.
#[derive(Clone, Debug)]
pub struct Adam {
    /// Learning rate.
    pub lr: f64,
    /// First-moment decay.
    pub beta1: f64,
    /// Second-moment decay.
    pub beta2: f64,
    /// Numerical stability term.
    pub eps: f64,
    /// Coupled L2 weight decay (added to the gradient).
    pub weight_decay: f64,
    state: Moments,
}

impl Adam {
    /// Adam with defaults `beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8`.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            state: Moments::new(),
        }
    }
    /// Override the moment decay rates.
    pub fn betas(mut self, b1: f64, b2: f64) -> Self {
        self.beta1 = b1;
        self.beta2 = b2;
        self
    }
    /// Set the coupled L2 weight decay.
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut Vec<f64>, grads: &[f64]) {
        let (c1, c2) = self.state.advance(params.len(), self.beta1, self.beta2);
        for i in 0..params.len().min(grads.len()) {
            let g = grads[i] + self.weight_decay * params[i];
            self.state.update(i, g, self.beta1, self.beta2);
            let mh = self.state.m[i] / c1;
            let vh = self.state.v[i] / c2;
            params[i] -= self.lr * mh / (vh.sqrt() + self.eps);
        }
    }
    fn lr(&self) -> f64 {
        self.lr
    }
    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
    fn reset(&mut self) {
        self.state.clear();
    }
}

/// AdamW (Loshchilov & Hutter, 2019): Adam with *decoupled* weight decay.
#[derive(Clone, Debug)]
pub struct AdamW {
    /// Learning rate.
    pub lr: f64,
    /// First-moment decay.
    pub beta1: f64,
    /// Second-moment decay.
    pub beta2: f64,
    /// Numerical stability term.
    pub eps: f64,
    /// Decoupled weight decay applied directly to the parameters.
    pub weight_decay: f64,
    state: Moments,
}

impl AdamW {
    /// AdamW with the standard `weight_decay = 0.01`.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            state: Moments::new(),
        }
    }
    /// Set the decoupled weight decay.
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &mut Vec<f64>, grads: &[f64]) {
        let (c1, c2) = self.state.advance(params.len(), self.beta1, self.beta2);
        for i in 0..params.len().min(grads.len()) {
            self.state.update(i, grads[i], self.beta1, self.beta2);
            let mh = self.state.m[i] / c1;
            let vh = self.state.v[i] / c2;
            params[i] -= self.lr * (mh / (vh.sqrt() + self.eps) + self.weight_decay * params[i]);
        }
    }
    fn lr(&self) -> f64 {
        self.lr
    }
    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
    fn reset(&mut self) {
        self.state.clear();
    }
}

/// LAMB (You et al., 2020): Adam plus a layer-wise trust ratio.
///
/// **Scoped down:** the trust ratio is computed over the whole flat parameter
/// vector (one "layer") instead of per tensor, because [`Optimizer`] receives an
/// untyped `Vec<f64>`. The update rule itself is the published one.
#[derive(Clone, Debug)]
pub struct Lamb {
    /// Learning rate.
    pub lr: f64,
    /// First-moment decay.
    pub beta1: f64,
    /// Second-moment decay.
    pub beta2: f64,
    /// Numerical stability term.
    pub eps: f64,
    /// Decoupled weight decay folded into the trust-ratio numerator.
    pub weight_decay: f64,
    state: Moments,
}

impl Lamb {
    /// LAMB with defaults `beta1 = 0.9`, `beta2 = 0.999`, `weight_decay = 0.0`.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-6,
            weight_decay: 0.0,
            state: Moments::new(),
        }
    }
    /// Set the weight decay.
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for Lamb {
    fn step(&mut self, params: &mut Vec<f64>, grads: &[f64]) {
        let n = params.len().min(grads.len());
        let (c1, c2) = self.state.advance(params.len(), self.beta1, self.beta2);
        let mut update = vec![0.0; n];
        for (i, u) in update.iter_mut().enumerate() {
            self.state.update(i, grads[i], self.beta1, self.beta2);
            let mh = self.state.m[i] / c1;
            let vh = self.state.v[i] / c2;
            *u = mh / (vh.sqrt() + self.eps) + self.weight_decay * params[i];
        }
        let (pn, un) = (l2(&params[..n]), l2(&update));
        let trust = if pn > 0.0 && un > 0.0 { pn / un } else { 1.0 };
        for i in 0..n {
            params[i] -= self.lr * trust * update[i];
        }
    }
    fn lr(&self) -> f64 {
        self.lr
    }
    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
    fn reset(&mut self) {
        self.state.clear();
    }
}

/// Lion (Chen et al., 2023): sign of an interpolated momentum.
///
/// `update = sign(b1 * m + (1 - b1) * g)`, then `m = b2 * m + (1 - b2) * g`.
/// Weight decay is decoupled, as in the reference implementation.
#[derive(Clone, Debug)]
pub struct Lion {
    /// Learning rate (typically ~10x smaller than Adam's).
    pub lr: f64,
    /// Interpolation factor used for the update direction.
    pub beta1: f64,
    /// Momentum decay.
    pub beta2: f64,
    /// Decoupled weight decay.
    pub weight_decay: f64,
    m: Vec<f64>,
}

impl Lion {
    /// Lion with defaults `beta1 = 0.9`, `beta2 = 0.99`, no weight decay.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
            m: Vec::new(),
        }
    }
    /// Set the decoupled weight decay.
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for Lion {
    fn step(&mut self, params: &mut Vec<f64>, grads: &[f64]) {
        fit_state(&mut self.m, params.len());
        for i in 0..params.len().min(grads.len()) {
            let g = grads[i];
            let c = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            let dir = if c > 0.0 {
                1.0
            } else if c < 0.0 {
                -1.0
            } else {
                0.0
            };
            params[i] -= self.lr * (dir + self.weight_decay * params[i]);
            self.m[i] = self.beta2 * self.m[i] + (1.0 - self.beta2) * g;
        }
    }
    fn lr(&self) -> f64 {
        self.lr
    }
    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
    fn reset(&mut self) {
        self.m.clear();
    }
}

/// Uppercase alias for [`Sgd`].
pub type SGD = Sgd;
/// Uppercase alias for [`Lamb`].
pub type LAMB = Lamb;
