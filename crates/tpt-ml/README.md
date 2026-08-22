# tpt-ml

The standard ML API — modules, layers, losses, optimizers, and data loading
built on [`tpt-tensor`](https://crates.io/crates/tpt-tensor) and
[`tpt-autograd`](https://crates.io/crates/tpt-autograd).

`tpt-ml` gives you a minimal but real deep-learning stack: a [`Module`] trait
with stable parameter ordering, dense/conv/embedding/norm/attention layers, a
full loss zoo, SGD + AdamW optimizers with LR schedulers, and an in-memory
`DataLoader` — all differentiable end-to-end through `tpt-autograd::backward`.

## Features

- **`Module` trait** — `forward` / `parameters` / `set_parameters` with stable
  parameter ordering so optimizers can key moment state by index.
- **Layers** — [`Linear`], [`Sequential`], [`Embedding`], [`LayerNorm`],
  [`BatchNorm2d`], [`Conv1d`]/[`Conv2d`]/[`Conv3d`].
- **Attention** — [`MultiHeadAttention`] and a pre-norm [`TransformerBlock`]
  (`x + MHA(x)` → LN → FFN → LN) over `[B, T, D]` sequences.
- **Activations** — `relu`, `gelu`, `tanh`, `sigmoid` (each with a correct
  backward, via custom nodes or composed autograd primitives).
- **Losses** — `mse`, `mae`, `huber`, `cross_entropy`, `nll_loss`,
  `binary_cross_entropy`, `binary_cross_entropy_with_logits`.
- **Optimizers** — [`Sgd`], [`AdamW`] (decoupled weight decay), plus the
  `step_attached` helper that re-attaches fresh leaf nodes after an update
  (so epoch 2+ backward passes keep working).
- **LR schedulers** — `StepLR`, `ExponentialLR`, `CosineAnnealingLR`,
  `LinearLR` (warmup), all behind the [`LrScheduler`] trait.
- **Data** — the [`Dataset`] trait, in-memory [`TensorDataset`], and a
  [`DataLoader`] that stacks shuffled `[B, ...]` batches.
- **Deterministic init** — Xavier-ish LCG initialization; no external RNG dep.

## Installation

```toml
[dependencies]
tpt-ml = "0.1"
```

## Quick start

```rust
use tpt_autograd::backward;
use tpt_ml::{mse, Linear, Module, Optimizer, Sequential, Sgd};
use tpt_tensor::Tensor;

fn main() {
    let mut net = Sequential::new();
    net.push(Linear::new(1, 16, true));
    net.push(Linear::new(16, 1, true));

    let mut opt = Sgd::new(0.05);
    let x = Tensor::from_typed(vec![1.0_f64]).reshape(&[1, 1]).unwrap();
    let y = Tensor::from_typed(vec![2.0_f64]).reshape(&[1, 1]).unwrap();

    for _ in 0..100 {
        let pred = net.forward(&x);
        let loss = mse(&pred, &y);
        backward(&loss);

        let mut params = net.parameters();
        opt.step(&mut params);
        net.set_parameters(params.into_iter().map(|p| p.with_autograd()).collect());
    }
}
```

## Examples

```sh
cargo run -p tpt-ml --example train_mlp
cargo run -p tpt-ml --example data_loader
```

## Status

The core training contract is complete. Deferred (see the workspace todo):
multi-threaded / Arrow-backed data prefetching, causal attention masks, and
GPU-backed parameter updates.

## License

Dual-licensed under the workspace license (see repository root).
