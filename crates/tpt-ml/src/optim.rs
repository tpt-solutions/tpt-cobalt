use tpt_tensor::Tensor;

/// Parameter update rule. Operates on a slice of parameters (the same ordering
/// returned by [`crate::Module::parameters`]); `step` writes updated values
/// back into those tensors in place (the caller then writes them into the model
/// via `set_parameters`).
pub trait Optimizer {
    fn step(&mut self, params: &mut [Tensor]);

    /// Current learning rate. Used by [`LrScheduler`]s to read/adjust the rate.
    fn lr(&self) -> f64;

    /// Overwrite the learning rate (called by [`LrScheduler`]s each epoch).
    fn set_lr(&mut self, lr: f64);
}

/// Stochastic gradient descent: `param -= lr * grad`.
pub struct Sgd {
    lr: f64,
}

impl Sgd {
    pub fn new(lr: f64) -> Self {
        Sgd { lr }
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, params: &mut [Tensor]) {
        for p in params.iter_mut() {
            if let Some(grad) = p.grad() {
                let update = p.add(&grad.scale(-self.lr));
                p.set_values(update.to_vec::<f64>().unwrap());
            }
        }
    }

    fn lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

/// AdamW with decoupled weight decay (Loshchilov & Hutter):
/// `param -= lr * (mhat / (sqrt(vhat) + eps) + wd * param)`.
pub struct AdamW {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    t: usize,
    m: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
}

impl AdamW {
    pub fn new(lr: f64) -> Self {
        Self::with_config(lr, 0.9, 0.999, 1e-8, 0.01)
    }

    pub fn with_config(lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        AdamW {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &mut [Tensor]) {
        // Lazily (re)size moment buffers to match the parameter set.
        if self.m.len() != params.len() {
            self.m = params.iter().map(|p| vec![0.0; p.numel()]).collect();
            self.v = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        }
        self.t += 1;
        let t = self.t as f64;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);
        for (i, p) in params.iter_mut().enumerate() {
            if let Some(grad) = p.grad() {
                let g = grad.to_vec::<f64>().unwrap();
                let val = p.to_vec::<f64>().unwrap();
                let (m, v) = (&mut self.m[i], &mut self.v[i]);
                let mut new_val = vec![0.0f64; val.len()];
                for j in 0..val.len() {
                    m[j] = self.beta1 * m[j] + (1.0 - self.beta1) * g[j];
                    v[j] = self.beta2 * v[j] + (1.0 - self.beta2) * g[j] * g[j];
                    let mhat = m[j] / bc1;
                    let vhat = v[j] / bc2;
                    new_val[j] =
                        val[j] - self.lr * (mhat / (vhat.sqrt() + self.eps) + self.weight_decay * val[j]);
                }
                p.set_values(new_val);
            }
        }
    }

    fn lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

/// A learning-rate schedule: adjusts an optimizer's `lr` each epoch.
pub trait LrScheduler {
    /// Advance one epoch and recompute the learning rate on the held optimizer.
    fn step(&mut self);
    /// The learning rate the scheduler would apply on the next step.
    fn get_lr(&self) -> f64;
}

/// Decay the LR by `gamma` every `step_size` epochs: `lr = base * gamma^floor(epoch/step_size)`.
pub struct StepLR<'a, O: Optimizer> {
    opt: &'a mut O,
    step_size: usize,
    gamma: f64,
    epoch: usize,
    base_lr: f64,
}

impl<'a, O: Optimizer> StepLR<'a, O> {
    pub fn new(opt: &'a mut O, step_size: usize, gamma: f64) -> Self {
        let base_lr = opt.lr();
        StepLR {
            opt,
            step_size,
            gamma,
            epoch: 0,
            base_lr,
        }
    }
}

impl<'a, O: Optimizer> LrScheduler for StepLR<'a, O> {
    fn step(&mut self) {
        self.epoch += 1;
        let factor = (self.epoch / self.step_size) as f64;
        self.opt.set_lr(self.base_lr * self.gamma.powf(factor));
    }

    fn get_lr(&self) -> f64 {
        self.opt.lr()
    }
}

/// Multiply the LR by `gamma` every epoch: `lr = base * gamma^epoch`.
pub struct ExponentialLR<'a, O: Optimizer> {
    opt: &'a mut O,
    gamma: f64,
    epoch: usize,
    base_lr: f64,
}

impl<'a, O: Optimizer> ExponentialLR<'a, O> {
    pub fn new(opt: &'a mut O, gamma: f64) -> Self {
        let base_lr = opt.lr();
        ExponentialLR {
            opt,
            gamma,
            epoch: 0,
            base_lr,
        }
    }
}

impl<'a, O: Optimizer> LrScheduler for ExponentialLR<'a, O> {
    fn step(&mut self) {
        self.epoch += 1;
        self.opt.set_lr(self.base_lr * self.gamma.powf(self.epoch as f64));
    }

    fn get_lr(&self) -> f64 {
        self.opt.lr()
    }
}

/// Cosine annealing from `base_lr` down to `eta_min` over `t_max` epochs, restarting.
pub struct CosineAnnealingLR<'a, O: Optimizer> {
    opt: &'a mut O,
    t_max: usize,
    eta_min: f64,
    epoch: usize,
    base_lr: f64,
}

impl<'a, O: Optimizer> CosineAnnealingLR<'a, O> {
    pub fn new(opt: &'a mut O, t_max: usize, eta_min: f64) -> Self {
        let base_lr = opt.lr();
        CosineAnnealingLR {
            opt,
            t_max,
            eta_min,
            epoch: 0,
            base_lr,
        }
    }
}

impl<'a, O: Optimizer> LrScheduler for CosineAnnealingLR<'a, O> {
    fn step(&mut self) {
        self.epoch += 1;
        let t = ((self.epoch - 1) % self.t_max + 1) as f64;
        let t_max = self.t_max as f64;
        let cos = (std::f64::consts::PI * t / t_max).cos();
        let lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + cos);
        self.opt.set_lr(lr);
    }

    fn get_lr(&self) -> f64 {
        self.opt.lr()
    }
}

/// Linear warmup: ramp `0 -> base_lr` over `warmup_epochs`, then hold at `base_lr`.
pub struct LinearLR<'a, O: Optimizer> {
    opt: &'a mut O,
    warmup_epochs: usize,
    epoch: usize,
    base_lr: f64,
}

impl<'a, O: Optimizer> LinearLR<'a, O> {
    pub fn new(opt: &'a mut O, warmup_epochs: usize) -> Self {
        let base_lr = opt.lr();
        LinearLR {
            opt,
            warmup_epochs,
            epoch: 0,
            base_lr,
        }
    }
}

impl<'a, O: Optimizer> LrScheduler for LinearLR<'a, O> {
    fn step(&mut self) {
        self.epoch += 1;
        let lr = if self.warmup_epochs == 0 {
            self.base_lr
        } else {
            self.base_lr * (self.epoch as f64 / self.warmup_epochs as f64).min(1.0)
        };
        self.opt.set_lr(lr);
    }

    fn get_lr(&self) -> f64 {
        self.opt.lr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Linear;
    use crate::module::Module;
    use tpt_autograd::backward;
    use tpt_tensor::Tensor;

    #[test]
    fn linear_backward_and_sgd() {
        // y = x @ W^T + b, x = [1,2,3] (batch 1). d(sum y)/dW[o][in] = x[in].
        let mut model = Linear::new(3, 2, true);
        let x = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0])
            .reshape(&[1, 3])
            .unwrap();
        let y = model.forward(&x);
        assert_eq!(y.shape(), &[1, 2]);
        // use sum of outputs as scalar loss (seed = ones on backward)
        backward(&y);

        let w_grad = model.weight.grad().expect("weight should have grad");
        assert_eq!(w_grad.shape(), &[3, 2]);
        // d(sum y)/dW[in][out] = x[in]  (x = [1,2,3])
        assert_eq!(w_grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);

        let before = model.weight.to_vec::<f64>().unwrap();
        let mut params = model.parameters();
        let mut opt = Sgd::new(0.1);
        opt.step(&mut params);
        model.set_parameters(params);
        let after = model.weight.to_vec::<f64>().unwrap();
        // each row r should have moved by -0.1 * (r+1)
        for r in 0..3 {
            for c in 0..2 {
                let expected = before[r * 2 + c] - 0.1 * (r as f64 + 1.0);
                assert!((after[r * 2 + c] - expected).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn adamw_changes_params() {
        let model = Linear::new(4, 3, true);
        let x = Tensor::from_typed(vec![0.5_f64, -1.0, 1.5, 2.0])
            .reshape(&[1, 4])
            .unwrap();
        let y = model.forward(&x);
        backward(&y);

        let before = model.parameters();
        let mut params = before.clone();
        let mut opt = AdamW::new(0.01);
        opt.step(&mut params);
        // at least one parameter must have changed
        let changed = params
            .iter()
            .zip(&before)
            .any(|(a, b)| a.to_vec::<f64>().unwrap() != b.to_vec::<f64>().unwrap());
        assert!(changed);
    }

    #[test]
    fn step_lr_decays_every_step_size() {
        let mut opt = Sgd::new(1.0);
        let mut sched = StepLR::new(&mut opt, 2, 0.1);
        assert!((sched.get_lr() - 1.0).abs() < 1e-12);
        sched.step(); // epoch 1
        assert!((sched.get_lr() - 1.0).abs() < 1e-12);
        sched.step(); // epoch 2 -> decay
        assert!((sched.get_lr() - 0.1).abs() < 1e-12);
        sched.step();
        sched.step(); // epoch 4 -> decay again
        assert!((sched.get_lr() - 0.01).abs() < 1e-12);
    }

    #[test]
    fn exponential_lr_shrinks() {
        let mut opt = Sgd::new(1.0);
        let mut sched = ExponentialLR::new(&mut opt, 0.5);
        sched.step();
        assert!((sched.get_lr() - 0.5).abs() < 1e-12);
        sched.step();
        assert!((sched.get_lr() - 0.25).abs() < 1e-12);
    }

    #[test]
    fn cosine_annealing_reaches_eta_min() {
        let mut opt = Sgd::new(1.0);
        let mut sched = CosineAnnealingLR::new(&mut opt, 4, 0.0);
        // after 4 steps (epoch 4) -> eta_min
        for _ in 0..4 {
            sched.step();
        }
        assert!((sched.get_lr() - 0.0).abs() < 1e-9);
        // epoch 5 wraps back to epoch 1 of next cycle -> not at min
        sched.step();
        assert!(sched.get_lr() > 0.0);
    }

    #[test]
    fn linear_lr_warms_up_then_holds() {
        let mut opt = Sgd::new(2.0);
        let mut sched = LinearLR::new(&mut opt, 4);
        sched.step(); // epoch1 -> 0.25*base = 0.5
        assert!((sched.get_lr() - 0.5).abs() < 1e-12);
        sched.step();
        sched.step();
        sched.step(); // epoch4 -> base
        assert!((sched.get_lr() - 2.0).abs() < 1e-12);
        sched.step(); // capped at base
        assert!((sched.get_lr() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn end_to_end_mlp_reduces_loss() {
        use crate::activations::relu;
        use crate::loss::mse;
        // Fit a tiny MLP to the constant target 1.0, proving the whole stack
        // (Linear -> ReLU -> Linear -> MSE -> SGD) backprops and trains.
        let mut w1 = Linear::new(1, 8, true);
        let mut w2 = Linear::new(8, 1, true);
        let mut opt = Sgd::new(0.05);
        let sample = || {
            let x = Tensor::from_typed(vec![0.5_f64]).reshape(&[1, 1]).unwrap();
            let y = w2.forward(&relu(&w1.forward(&x)));
            let t = Tensor::from_typed(vec![1.0_f64]).reshape(&[1, 1]).unwrap();
            mse(&y, &t).to_vec::<f64>().unwrap()[0]
        };
        let init_loss = sample();
        for _ in 0..300 {
            let x = Tensor::from_typed(vec![0.5_f64])
                .reshape(&[1, 1])
                .unwrap()
                .with_autograd();
            let y = w2.forward(&relu(&w1.forward(&x)));
            let t = Tensor::from_typed(vec![1.0_f64]).reshape(&[1, 1]).unwrap();
            let loss = mse(&y, &t);
            backward(&loss);
            let mut params: Vec<Tensor> = Vec::new();
            params.extend(w1.parameters());
            params.extend(w2.parameters());
            opt.step(&mut params);
            let mut it = params.into_iter();
            let p1: Vec<Tensor> = it.by_ref().take(2).collect();
            let p2: Vec<Tensor> = it.collect();
            w1.set_parameters(p1);
            w2.set_parameters(p2);
        }
        let final_loss = sample();
        assert!(
            final_loss < init_loss,
            "loss should decrease: {init_loss} -> {final_loss}"
        );
        assert!(final_loss < 0.1, "mlp should converge: {final_loss}");
    }
}
