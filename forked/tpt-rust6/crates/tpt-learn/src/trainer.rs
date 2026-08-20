//! Declarative training loop: loss, LR schedule, early stopping, checkpointing.

use ndarray::Array2;
use tpt_omni::Tensor;

use crate::data::{as_matrix, from_matrix, rows, DataLoader};
use crate::error::LearnError;
use crate::model::Model;
use crate::optim::{Optimizer, Sgd};

/// Supported objectives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loss {
    /// Mean squared error, averaged over every element of the batch.
    MeanSquaredError,
    /// Softmax cross-entropy over logits. Targets may be one-hot `[B, OUT]`
    /// or a single column of class indices `[B, 1]`.
    CrossEntropy,
}

fn softmax_rows(logits: &Array2<f64>) -> Array2<f64> {
    let mut out = logits.clone();
    for mut row in out.rows_mut() {
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        row.mapv_inplace(|v| (v - max).exp());
        let sum: f64 = row.iter().sum();
        if sum > 0.0 {
            row.mapv_inplace(|v| v / sum);
        }
    }
    out
}

/// Expand class-index targets to one-hot when needed.
fn one_hot_like(pred: &Array2<f64>, target: &Array2<f64>) -> Array2<f64> {
    if target.ncols() == pred.ncols() {
        return target.to_owned();
    }
    let mut out = Array2::zeros(pred.raw_dim());
    for (i, row) in target.rows().into_iter().enumerate() {
        let k = row[0].round();
        if k >= 0.0 && (k as usize) < pred.ncols() {
            out[[i, k as usize]] = 1.0;
        }
    }
    out
}

impl Loss {
    /// Scalar loss for a batch of predictions.
    ///
    /// # Panics
    /// If `pred` and `target` have different row counts.
    pub fn value(&self, pred: &Array2<f64>, target: &Array2<f64>) -> f64 {
        assert_eq!(pred.nrows(), target.nrows(), "loss: batch size mismatch");
        match self {
            Loss::MeanSquaredError => {
                let n = pred.len().max(1) as f64;
                pred.iter()
                    .zip(target.iter())
                    .map(|(p, t)| (p - t) * (p - t))
                    .sum::<f64>()
                    / n
            }
            Loss::CrossEntropy => {
                let p = softmax_rows(pred);
                let t = one_hot_like(pred, target);
                let b = pred.nrows().max(1) as f64;
                -p.iter()
                    .zip(t.iter())
                    .map(|(p, t)| t * (p + 1e-12).ln())
                    .sum::<f64>()
                    / b
            }
        }
    }

    /// Gradient of [`Loss::value`] with respect to `pred`.
    ///
    /// # Panics
    /// If `pred` and `target` have different row counts.
    pub fn grad(&self, pred: &Array2<f64>, target: &Array2<f64>) -> Array2<f64> {
        assert_eq!(pred.nrows(), target.nrows(), "loss: batch size mismatch");
        match self {
            Loss::MeanSquaredError => {
                let n = pred.len().max(1) as f64;
                let mut g = pred.clone();
                g.zip_mut_with(target, |p, t| *p = 2.0 * (*p - t) / n);
                g
            }
            Loss::CrossEntropy => {
                let b = pred.nrows().max(1) as f64;
                let mut g = softmax_rows(pred);
                let t = one_hot_like(pred, target);
                g.zip_mut_with(&t, |p, t| *p = (*p - t) / b);
                g
            }
        }
    }
}

/// Learning-rate schedules evaluated once per epoch.
#[derive(Clone, Copy, Debug)]
pub enum LrSchedule {
    /// Keep the base learning rate.
    Constant,
    /// Multiply by `gamma` every `step_size` epochs.
    Step {
        /// Epochs between decays.
        step_size: usize,
        /// Multiplicative decay factor.
        gamma: f64,
    },
    /// Cosine anneal from the base rate down to `min_lr` over `t_max` epochs.
    Cosine {
        /// Length of the annealing cycle in epochs.
        t_max: usize,
        /// Floor of the schedule.
        min_lr: f64,
    },
}

impl LrSchedule {
    /// Learning rate for `epoch` (0-based) given the base rate.
    pub fn lr(&self, base: f64, epoch: usize) -> f64 {
        match *self {
            LrSchedule::Constant => base,
            LrSchedule::Step { step_size, gamma } => {
                let k = epoch.checked_div(step_size).unwrap_or(0);
                base * gamma.powi(k as i32)
            }
            LrSchedule::Cosine { t_max, min_lr } => {
                let t = if t_max == 0 {
                    1.0
                } else {
                    (epoch.min(t_max) as f64) / t_max as f64
                };
                min_lr + 0.5 * (base - min_lr) * (1.0 + (std::f64::consts::PI * t).cos())
            }
        }
    }
}

/// Metric watched by early stopping / best-checkpoint selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Validation loss (default).
    ValLoss,
    /// Training loss.
    TrainLoss,
}

/// One row of the training history returned by [`Trainer::fit_detailed`].
#[derive(Clone, Copy, Debug)]
pub struct EpochLog {
    /// 0-based epoch index.
    pub epoch: usize,
    /// Mean training loss over the epoch.
    pub train_loss: f64,
    /// Loss on the validation split.
    pub val_loss: f64,
    /// Learning rate used for the epoch.
    pub lr: f64,
}

/// Full-batch evaluation of a model under a loss.
pub fn evaluate<M: Model>(
    model: &M,
    x: &Tensor<f64>,
    y: &Tensor<f64>,
    loss: Loss,
) -> Result<f64, LearnError> {
    let target = as_matrix(y)?;
    let pred = as_matrix(&model.forward(x)?)?;
    if pred.nrows() != target.nrows() {
        return Err(LearnError::RowMismatch(pred.nrows(), target.nrows()));
    }
    Ok(loss.value(&pred, &target))
}

/// Per-epoch logging hook: `(epoch, train_loss, val_loss)`.
pub type EpochCallback = Box<dyn Fn(usize, f64, f64)>;

/// Builder-style training loop.
///
/// ```
/// use tpt_learn::prelude::*;
/// use tpt_omni::{ndarray::ArrayD, Tensor};
///
/// let x = Tensor::new(ArrayD::from_shape_vec(vec![4, 1], vec![0., 1., 2., 3.]).unwrap());
/// let y = Tensor::new(ArrayD::from_shape_vec(vec![4, 1], vec![1., 3., 5., 7.]).unwrap());
/// let model = Trainer::new(Linear::<1, 1>::new())
///     .optimizer(Adam::new(0.1))
///     .loss(Loss::MeanSquaredError)
///     .epochs(100)
///     .batch_size(4)
///     .fit(&x, &y, &x, &y)
///     .unwrap();
/// assert!(evaluate(&model, &x, &y, Loss::MeanSquaredError).unwrap() < 0.1);
/// ```
pub struct Trainer<M: Model> {
    model: M,
    optimizer: Box<dyn Optimizer>,
    loss: Loss,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    schedule: LrSchedule,
    early_stopping: Option<(usize, Metric)>,
    checkpoint_dir: Option<std::path::PathBuf>,
    on_epoch: Option<EpochCallback>,
    verbose: bool,
}

impl<M: Model> Trainer<M> {
    /// Start from an initialized model with SGD(0.01) / MSE defaults.
    pub fn new(model: M) -> Self {
        Self {
            model,
            optimizer: Box::new(Sgd::new(0.01)),
            loss: Loss::MeanSquaredError,
            epochs: 10,
            batch_size: 32,
            lr: 0.01,
            schedule: LrSchedule::Constant,
            early_stopping: None,
            checkpoint_dir: None,
            on_epoch: None,
            verbose: false,
        }
    }

    /// Set the optimizer; its learning rate becomes the schedule's base rate.
    pub fn optimizer<O: Optimizer + 'static>(mut self, opt: O) -> Self {
        self.lr = opt.lr();
        self.optimizer = Box::new(opt);
        self
    }
    /// Choose the objective.
    pub fn loss(mut self, loss: Loss) -> Self {
        self.loss = loss;
        self
    }
    /// Maximum number of epochs.
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }
    /// Mini-batch size (the last batch may be shorter).
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
    /// Override the base learning rate.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }
    /// Stop after `patience` epochs without improvement of `metric`.
    pub fn early_stopping(mut self, patience: usize, metric: Metric) -> Self {
        self.early_stopping = Some((patience, metric));
        self
    }
    /// Write the best model so far to `dir/best_model.bin` after each epoch.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn checkpoint(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.checkpoint_dir = Some(dir.into());
        self
    }
    /// Apply a per-epoch learning-rate schedule.
    pub fn lr_schedule(mut self, schedule: LrSchedule) -> Self {
        self.schedule = schedule;
        self
    }
    /// Register a logging hook called as `(epoch, train_loss, val_loss)`.
    pub fn on_epoch<F: Fn(usize, f64, f64) + 'static>(mut self, f: F) -> Self {
        self.on_epoch = Some(Box::new(f));
        self
    }
    /// Print one line per epoch to stdout.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Train and return the best model (by the monitored metric).
    ///
    /// Errors are stringified; use [`Trainer::fit_detailed`] for typed errors
    /// and the epoch history.
    pub fn fit(
        self,
        train_x: &Tensor<f64>,
        train_y: &Tensor<f64>,
        val_x: &Tensor<f64>,
        val_y: &Tensor<f64>,
    ) -> Result<M, String> {
        self.fit_detailed(train_x, train_y, val_x, val_y)
            .map(|(m, _)| m)
            .map_err(|e| e.to_string())
    }

    /// Like [`Trainer::fit`], but also returns the per-epoch history.
    pub fn fit_detailed(
        mut self,
        train_x: &Tensor<f64>,
        train_y: &Tensor<f64>,
        val_x: &Tensor<f64>,
        val_y: &Tensor<f64>,
    ) -> Result<(M, Vec<EpochLog>), LearnError> {
        self.validate(train_x, train_y)?;
        self.validate(val_x, val_y)?;

        let monitor = self
            .early_stopping
            .map(|(_, m)| m)
            .unwrap_or(Metric::ValLoss);
        let n_train = rows(train_x)?.max(1) as f64;

        let mut params = self.model.params();
        let mut best_params = params.clone();
        let mut best = f64::INFINITY;
        let mut waited = 0usize;
        let mut history = Vec::with_capacity(self.epochs);

        for epoch in 0..self.epochs {
            let lr = self.schedule.lr(self.lr, epoch);
            self.optimizer.set_lr(lr);

            let mut running = 0.0;
            for batch in DataLoader::new(train_x, Some(train_y), self.batch_size)? {
                let target = as_matrix(batch.y.as_ref().expect("loader built with targets"))?;
                let pred = as_matrix(&self.model.forward(&batch.x)?)?;
                running += self.loss.value(&pred, &target) * batch.len as f64;
                let grad_out = from_matrix(self.loss.grad(&pred, &target));
                let grads = self.model.backward(&batch.x, &grad_out)?;
                self.optimizer.step(&mut params, &grads);
                self.model.set_params(&params);
            }
            let train_loss = running / n_train;
            let val_loss = evaluate(&self.model, val_x, val_y, self.loss)?;
            history.push(EpochLog {
                epoch,
                train_loss,
                val_loss,
                lr,
            });

            if let Some(cb) = &self.on_epoch {
                cb(epoch, train_loss, val_loss);
            }
            #[cfg(not(target_arch = "wasm32"))]
            if self.verbose {
                println!(
                    "epoch {epoch:>4}  lr {lr:.5}  train_loss {train_loss:.6}  val_loss {val_loss:.6}"
                );
            }

            let watched = match monitor {
                Metric::ValLoss => val_loss,
                Metric::TrainLoss => train_loss,
            };
            if watched.is_finite() && watched < best - 1e-12 {
                best = watched;
                best_params.clone_from(&params);
                waited = 0;
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(dir) = &self.checkpoint_dir {
                    self.model.save(dir.join("best_model.bin"))?;
                }
            } else {
                waited += 1;
                if let Some((patience, _)) = self.early_stopping {
                    if waited >= patience {
                        break;
                    }
                }
            }
        }

        // Restore the best-scoring parameters seen during training.
        self.model.set_params(&best_params);
        Ok((self.model, history))
    }

    fn validate(&self, x: &Tensor<f64>, y: &Tensor<f64>) -> Result<(), LearnError> {
        if self.batch_size == 0 {
            return Err(LearnError::Config("batch_size must be > 0".into()));
        }
        let xm = as_matrix(x)?;
        let ym = as_matrix(y)?;
        if xm.ncols() != M::IN_DIM {
            return Err(LearnError::Shape {
                expected: M::IN_DIM,
                found: xm.ncols(),
            });
        }
        let class_indices = self.loss == Loss::CrossEntropy && ym.ncols() == 1 && M::OUT_DIM > 1;
        if ym.ncols() != M::OUT_DIM && !class_indices {
            return Err(LearnError::Shape {
                expected: M::OUT_DIM,
                found: ym.ncols(),
            });
        }
        if xm.nrows() != ym.nrows() {
            return Err(LearnError::RowMismatch(xm.nrows(), ym.nrows()));
        }
        Ok(())
    }
}
