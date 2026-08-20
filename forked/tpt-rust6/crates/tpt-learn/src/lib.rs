//! # tpt-learn — High-Level ML Training Abstractions
//!
//! Declare *what* you want to train instead of hand-writing a training loop:
//!
//! ```
//! use tpt_learn::prelude::*;
//! use tpt_omni::{ndarray::ArrayD, Tensor};
//!
//! let xs: Vec<f64> = (0..16).map(|i| i as f64 / 8.0).collect();
//! let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x + 1.0).collect();
//! let x = Tensor::new(ArrayD::from_shape_vec(vec![16, 1], xs).unwrap());
//! let y = Tensor::new(ArrayD::from_shape_vec(vec![16, 1], ys).unwrap());
//!
//! let model = Trainer::new(Linear::<1, 1>::new())
//!     .optimizer(Adam::new(0.05))
//!     .loss(Loss::MeanSquaredError)
//!     .epochs(300)
//!     .batch_size(8)
//!     .lr_schedule(LrSchedule::Cosine { t_max: 300, min_lr: 1e-4 })
//!     .early_stopping(50, Metric::ValLoss)
//!     .on_epoch(|epoch, train, val| {
//!         let _ = (epoch, train, val); // wire up your logger here
//!     })
//!     .fit(&x, &y, &x, &y)?;
//!
//! assert!(evaluate(&model, &x, &y, Loss::MeanSquaredError)? < 0.1);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## What is implemented
//!
//! * [`Model`] — compile-time input/output widths ([`Model::IN_DIM`] /
//!   [`Model::OUT_DIM`]) plus a flat parameter view for optimizers, with
//!   [`Linear<IN, OUT>`](Linear) as the built-in const-generic layer.
//! * [`Optimizer`] — [`Sgd`], [`Adam`], [`AdamW`], [`Lamb`], [`Lion`].
//! * [`Trainer`] — batching, LR schedules ([`LrSchedule`]), early stopping,
//!   best-checkpoint saving and a logging callback.
//! * [`DataLoader`] — sequential mini-batches sliced along dim 0.
//! * Serialization — single-file bincode via [`Model::save`] / [`Model::load`]
//!   (and wasm-safe [`Model::to_bytes`] / [`Model::from_bytes`]).
//!
//! ## Scoped-down areas
//!
//! * Gradients are analytic per layer (`Model::backward`) rather than taped
//!   autodiff; `tpt-grad` is currently an empty scaffold, so no dependency on
//!   it is taken and the crate builds standalone.
//! * [`Lamb`] applies one trust ratio to the whole flat parameter vector
//!   instead of per tensor; [`Lion`] omits the optional gradient clipping.
//! * [`Model::export_wasm`] / [`Model::to_onnx`] are implemented for [`Linear`]
//!   behind the `wasm-export` and `onnx` features respectively; other model
//!   kinds still return [`LearnError::Unsupported`]. Binary export
//!   ([`Model::to_bytes`] / [`Model::save`]) is fully implemented regardless.
//! * Core paths avoid threads and filesystem access; file IO is compiled out on
//!   `wasm32`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod data;
pub mod error;
pub mod model;
pub mod optim;
pub mod sweep;
pub mod trainer;

#[cfg(feature = "onnx")]
mod onnx;
#[cfg(feature = "wasm-export")]
mod wasm_export;

pub use data::{as_matrix, from_matrix, Batch, DataLoader};
pub use error::LearnError;
pub use model::{Linear, Model};
pub use optim::{Adam, AdamW, Lamb, Lion, Optimizer, Sgd, LAMB, SGD};
pub use sweep::{GridSweep, SweepResult, SweepTrial};
pub use trainer::{evaluate, EpochLog, Loss, LrSchedule, Metric, Trainer};

/// Everything needed for a typical training script.
pub mod prelude {
    pub use crate::data::{as_matrix, from_matrix, Batch, DataLoader};
    pub use crate::error::LearnError;
    pub use crate::model::{Linear, Model};
    pub use crate::optim::{Adam, AdamW, Lamb, Lion, Optimizer, Sgd, LAMB, SGD};
    pub use crate::sweep::{GridSweep, SweepResult, SweepTrial};
    pub use crate::trainer::{evaluate, EpochLog, Loss, LrSchedule, Metric, Trainer};
}
