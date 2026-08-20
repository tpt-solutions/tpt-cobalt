//! AI-native multiphysics coupling layer for `tpt-physics`.
//!
//! `tpt-sci-sim-core` (from the sibling `tpt-science` repo) already provides
//! the generic multi-scale engine this crate needs: heterogeneous
//! sub-model time-stepping ([`tpt_sci_sim_core::Simulation`]), cross-scale
//! coupling ([`tpt_sci_sim_core::Coupling`]), and checkpointing. Rather than
//! duplicate that, `tpt-phys-orchestrator` re-exports it as its co-simulation
//! foundation and adds the physics-domain-specific pieces spec2 calls for on
//! top: RL/differentiable hooks (see [`rl`]) for adaptive solver control, and
//! (as `tpt-phys-fsi`/`tpt-phys-thermal-struct`/`tpt-phys-electro-thermal`
//! land) the [`SubModel`] adapters that let those coupling crates plug into a
//! [`Simulation`] directly.

pub mod adapters;
pub mod rl;

pub use adapters::{
    build_demo_simulation, build_demo_simulation_for, ElectroThermalSubModel, FsiSubModel,
    ThermalStructSubModel,
};
pub use tpt_sci_sim_core::{Checkpoint, Coupling, CouplingFn, Simulation, SubModel};
