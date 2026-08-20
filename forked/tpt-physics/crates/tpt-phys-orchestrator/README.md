# tpt-phys-orchestrator

AI-native coupling layer: adaptive time-stepping, solver coordination,
co-simulation API, and reinforcement-learning hooks.

Built on top of `tpt-sci-sim-core` (the sibling `tpt-science` repo's
multi-scale simulation engine — heterogeneous sub-model time-stepping,
cross-scale coupling, checkpointing) rather than reimplementing it. This
crate adds the physics-domain layer on top: RL/differentiable hooks (the
`rl` module, ported from the removed `tpt-physics-ai` crate) and, as
`tpt-phys-fsi`/`tpt-phys-thermal-struct`/`tpt-phys-electro-thermal` mature,
`SubModel` adapters wiring them into a `Simulation`.

## Quick start

```rust
use tpt_phys_orchestrator::build_demo_simulation;

let mut sim = build_demo_simulation();
for _ in 0..200 {
    sim.step(1e-4);
}
```

`build_demo_simulation` registers an electro-thermal rod, a thermal-structural
tetrahedral model, and an FSI channel, then couples the electro-thermal
temperature field into the thermal-structural model.

## Modules

| Module | Description |
| --- | --- |
| `rl` | `DifferentiablePlant` (AD-via-forward-mode transition) and `GymWrapper` (Gymnasium-like `reset`/`step`), with `HarmonicOscillator` and `Pendulum` reference plants. |
| `adapters` | `SubModel` adapters `FsiSubModel`, `ElectroThermalSubModel`, `ThermalStructSubModel` wiring the domain crates into a `tpt_sci_sim_core::Simulation`; `build_demo_simulation` / `build_demo_simulation_for`. |
| (root) | Re-exports of `Simulation`, `SubModel`, `Coupling`, `CouplingFn`, `Checkpoint` from `tpt-sci-sim-core`. |

## Examples

Runnable with `cargo run --release --example <name> -p tpt-phys-orchestrator`.

| Example | Demonstrates |
| --- | --- |
| `rl_pendulum` | A differentiable pendulum as a Gym environment; forward-mode AD Jacobians match the analytic derivatives. |
| `uq_coupled` | Monte-Carlo UQ (`tpt-phys-core`) sweeping structural-material scatter through the coupled co-simulation, propagating into the electro-thermal hotspot. |
| `coupled_simulation` | Builds the three-crate demo `Simulation` and runs it, reporting each sub-model's state (electro-thermal heating drives the loop). |

## Validations

- Demo `Simulation` steps and stays finite across all sub-models.
- RL plants: physical stepping; forward-mode AD Jacobians match analytic
  derivatives; the Gym wrapper runs and terminates.

## License

Dual-licensed under [MIT](../../LICENSE-MIT) and [Apache-2.0](../../LICENSE-APACHE-2.0).
Copyright TPT Solutions.
