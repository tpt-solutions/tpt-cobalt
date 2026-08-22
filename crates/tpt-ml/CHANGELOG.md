# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-22

### Added
- Module trait with stable parameter ordering (orward / parameters / set_parameters).
- Layers: Linear, Sequential, Embedding, LayerNorm, BatchNorm2d.
- Convolutions: Conv1d, Conv2d, Conv3d with custom-node backward.
- Attention: MultiHeadAttention and pre-norm TransformerBlock over [B, T, D].
- Activations: elu, gelu, 	anh, sigmoid with correct gradients.
- Losses: mse, mae, huber, cross_entropy, 
ll_loss, inary_cross_entropy, inary_cross_entropy_with_logits.
- Optimizers: Sgd, AdamW (decoupled weight decay) and step_attached re-attachment helper.
- LR schedulers: StepLR, ExponentialLR, CosineAnnealingLR, LinearLR behind LrScheduler.
- Data: Dataset trait, TensorDataset, DataLoader with deterministic shuffling and batch stacking.
- Two runnable examples (	rain_mlp, data_loader) and a comprehensive README.

[0.1.0]: https://github.com/tpt-solutions/tpt-crucible/releases/tag/tpt-ml-v0.1.0
