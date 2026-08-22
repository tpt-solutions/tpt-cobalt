# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-22

### Added
- Differentiable IVP integration: uler_step, k4_step, solve_ivp — gradients flow through the solver into the initial state and vector-field parameters.
- Differentiable dense linear solve K u = f (solve_linear) with adjoint VJP (K^T lambda = g) for gradients w.r.t. stiffness matrix and force vector.
- PINN training: pinn_mlp, pinn_residual_loss, pinn_ic_loss, and 	rain_pinn_ode fitting a 	pt-ml MLP to an ODE residual.
- Two runnable examples (integrate_ode, 	rain_pinn) and a comprehensive README.

[0.1.0]: https://github.com/tpt-solutions/tpt-crucible/releases/tag/tpt-sci-v0.1.0
