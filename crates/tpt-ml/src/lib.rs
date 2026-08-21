//! # tpt-ml — The Standard ML API (Phase 2, spec §5.3)
//!
//! Starting point: `tpt-rust6::tpt-learn` (nn / optim / loss / data). This
//! scaffold builds on [`tpt_tensor`] + [`tpt_autograd`]: a minimal but real
//! `Module` abstraction, a `Linear` layer, an `SGD` optimizer, and `AdamW`
//! (decoupled weight decay). Layers are differentiable through `tpt-autograd`;
//! `backward` then populates parameter gradients for the optimizer to consume.

pub mod activations;
pub mod attention;
pub mod conv;
pub mod data;
pub mod embedding;
pub mod layers;
pub mod loss;
pub mod module;
pub mod norm;
pub mod optim;

pub use activations::{gelu, relu, sigmoid_act, tanh};
pub use attention::{MultiHeadAttention, TransformerBlock};
pub use conv::{Conv1d, Conv2d, Conv3d};
pub use data::{Dataset, DataLoader, TensorDataset};
pub use embedding::Embedding;
pub use layers::{Linear, Sequential};
pub use loss::{
    binary_cross_entropy, binary_cross_entropy_with_logits, cross_entropy, huber, mae, mse,
    nll_loss,
};
pub use module::Module;
pub use norm::{BatchNorm2d, LayerNorm};
pub use optim::{
    AdamW, CosineAnnealingLR, ExponentialLR, LrScheduler, LinearLR, Optimizer, Sgd, StepLR,
};
