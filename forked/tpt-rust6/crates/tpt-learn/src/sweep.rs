//! Parallel hyperparameter sweeps for `tpt-learn` models.
//!
//! A grid sweep builds one model per hyperparameter combination and trains each
//! trial on Rayon's thread pool, then ranks them by a validation metric. This is
//! the same parallelism engine `tpt-dag`'s executor uses, applied to model
//! selection (the crate intentionally ships no distributed scheduler).

use std::collections::HashMap;

use rayon::prelude::*;

use crate::model::Model;

/// One evaluated trial: its hyperparameters, the trained model, and the metric.
pub struct SweepTrial<M> {
    /// Hyperparameter values used for this trial.
    pub params: HashMap<String, f64>,
    /// The trained model.
    pub model: M,
    /// Validation metric (higher is better).
    pub metric: f64,
}

impl<M> SweepTrial<M> {
    /// Read a hyperparameter value (or `NaN` if absent).
    pub fn get(&self, name: &str) -> f64 {
        self.params.get(name).copied().unwrap_or(f64::NAN)
    }
}

/// The outcome of a sweep: every trial plus the index of the best one.
pub struct SweepResult<M> {
    /// All evaluated trials.
    pub trials: Vec<SweepTrial<M>>,
    /// Index of the trial with the highest metric.
    pub best: Option<usize>,
}

impl<M> SweepResult<M> {
    /// Index of the best trial.
    pub fn best_index(&self) -> Option<usize> {
        self.best
    }
    /// The best-trained model.
    pub fn best_model(&self) -> Option<&M> {
        self.best.map(|i| &self.trials[i].model)
    }
    /// Hyperparameters of the best trial.
    pub fn best_params(&self) -> Option<&HashMap<String, f64>> {
        self.best.map(|i| &self.trials[i].params)
    }
    /// Best achieved metric.
    pub fn best_metric(&self) -> Option<f64> {
        self.best.map(|i| self.trials[i].metric)
    }
}

/// Builder for a grid hyperparameter sweep.
pub struct GridSweep {
    dims: Vec<(String, Vec<f64>)>,
}

impl Default for GridSweep {
    fn default() -> Self {
        Self::new()
    }
}

impl GridSweep {
    /// Start an empty sweep.
    pub fn new() -> Self {
        GridSweep { dims: Vec::new() }
    }

    /// Add a named hyperparameter dimension with its candidate values.
    pub fn dim(mut self, name: &str, values: &[f64]) -> Self {
        self.dims.push((name.to_string(), values.to_vec()));
        self
    }

    /// Enumerate every combination of hyperparameter values.
    fn combinations(&self) -> Vec<HashMap<String, f64>> {
        let mut out: Vec<HashMap<String, f64>> = vec![HashMap::new()];
        for (name, vals) in &self.dims {
            let mut next = Vec::with_capacity(out.len() * vals.len());
            for base in &out {
                for v in vals {
                    let mut m = base.clone();
                    m.insert(name.clone(), *v);
                    next.push(m);
                }
            }
            out = next;
        }
        out
    }

    /// Run the sweep. `build` turns a parameter map into a fresh (untrained)
    /// model; `train` fits the model and returns `(model, metric)` where a
    /// higher metric is better. Trials run concurrently on the Rayon pool.
    pub fn run<M, B, T>(self, build: B, train: T) -> SweepResult<M>
    where
        M: Model + Send + 'static,
        B: Fn(&HashMap<String, f64>) -> M + Sync,
        T: Fn(M, &HashMap<String, f64>) -> (M, f64) + Sync,
    {
        let combos = self.combinations();
        if combos.is_empty() {
            return SweepResult {
                trials: Vec::new(),
                best: None,
            };
        }
        let trials: Vec<SweepTrial<M>> = combos
            .par_iter()
            .map(|params| {
                let model = build(params);
                let (model, metric) = train(model, params);
                SweepTrial {
                    params: params.clone(),
                    model,
                    metric,
                }
            })
            .collect();
        let best = trials
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.metric
                    .partial_cmp(&b.1.metric)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);
        SweepResult { trials, best }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use tpt_omni::{ndarray::ArrayD, Tensor};

    #[test]
    fn grid_sweep_finds_best_lr_and_epochs() {
        let xs: Vec<f64> = (0..32).map(|i| i as f64 / 4.0).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x + 1.0).collect();
        let x = Tensor::new(ArrayD::from_shape_vec(vec![32, 1], xs.clone()).unwrap());
        let y = Tensor::new(ArrayD::from_shape_vec(vec![32, 1], ys.clone()).unwrap());

        let res = GridSweep::new()
            .dim("lr", &[0.001, 0.05, 0.5])
            .dim("epochs", &[50.0, 300.0])
            .run(
                |_params| Linear::<1, 1>::new(),
                |model, params| {
                    let lr = params.get("lr").unwrap();
                    let epochs = params.get("epochs").unwrap().round() as usize;
                    let m = Trainer::new(model)
                        .optimizer(Adam::new(*lr))
                        .loss(Loss::MeanSquaredError)
                        .epochs(epochs)
                        .batch_size(8)
                        .fit(&x, &y, &x, &y)
                        .unwrap();
                    let loss = evaluate(&m, &x, &y, Loss::MeanSquaredError).unwrap();
                    (m, -loss)
                },
            );

        assert_eq!(res.trials.len(), 6);
        assert!(res.best_index().is_some());
        let best_loss = -res.best_metric().unwrap();
        assert!(best_loss < 0.1, "best_loss={best_loss}");
        assert_eq!(
            res.best_params().unwrap().get("epochs").unwrap().round() as usize,
            300
        );
    }
}
