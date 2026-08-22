# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-22

### Added
- Eager reverse-mode autodiff tape over 	pt-tensor::Tensor.
- Differentiable ops: dd, sub, mul, div, 
eg, xp, log, bs, sigmoid, matmul, mm, sum, mean, sum_lastdim, log_softmax.
- ackward(output) with ones-seeded gradients and reverse topological traversal.
- custom_vjp for registering hand-written vector-Jacobian products (used by 	pt-sci).
- Broadcast-aware gradient reduction via sum_to.
- GradAccumulator for multi-device gradient collection and reduce.
- Two runnable examples (irst_gradients, custom_vjp_square) and a comprehensive README.

### Notes
- Planned for later Phase 1 work: traced/compiled graphs, gradient checkpointing, cross-device accumulation over real backends.

[0.1.0]: https://github.com/tpt-solutions/tpt-crucible/releases/tag/tpt-autograd-v0.1.0
