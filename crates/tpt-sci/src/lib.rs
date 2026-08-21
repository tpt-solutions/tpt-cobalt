//! # tpt-sci — Differentiable Scientific Computing (Phase 3, spec §5.4)
//!
//! Physics internalization onto `tpt-tensor`/`tpt-autograd`. This glue crate
//! makes classic scientific solvers differentiable so gradients flow back into
//! model parameters:
//!
//! - `ode`: an IVP integrator (RK4 / Euler) whose trajectory is itself a
//!   differentiable tensor — backprop runs *through the solver*.
//! - `fea`: a differentiable dense linear solve `K u = f` (the inner kernel of
//!   any static FEA/PDE discretization), with an adjoint VJP.
//! - `pinn`: a physics-informed neural-network training demo that fits a
//!   `tpt-ml` MLP to an ODE by minimizing a residual.

pub mod fea;
pub mod ode;
pub mod pinn;

pub use fea::solve_linear;
pub use ode::{euler_step, rk4_step, solve_ivp};
pub use pinn::train_pinn_ode;
