# tpt-phys-cfd

Computational Fluid Dynamics for `tpt-physics`. Two native, pure-Rust solvers:

- `Lbm2D` — a D2Q9 (9-velocity, 2-D) Lattice Boltzmann Method for
  incompressible flow.
- `Sph2D` — a weakly-compressible Smoothed-Particle Hydrodynamics (WCSPH,
  Müller et al. 2003) solver for free-surface / dam-break flow.

No CFD code exists in the sibling `tpt-math` / `tpt-fem` workspaces, so this
is net-new.

## Quick start (LBM)

```rust
use tpt_phys_cfd::{Lbm2D, XBoundary};

let mut lbm = Lbm2D::new(200, 60, 0.53);
lbm.set_x_boundary(XBoundary::Inlet(0.08));
lbm.set_horizontal_walls();
lbm.add_circle(60.0, 30.0, 8.0); // bluff body
lbm.initialise(1.0, [0.08, 0.0]);
for _ in 0..5_000 { lbm.step([0.0, 0.0]); }
```

## Quick start (SPH)

```rust
use tpt_phys_cfd::sph::Sph2D;

let h = 0.04;
let block = Sph2D::block(15, 30, h / 1.3, [0.02, 0.02]);
let mut sim = Sph2D::new(block, h, 1000.0, 20.0, 1.0, 0.5, [0.0, -9.81], [1.0, 1.0], 1e-4);
for _ in 0..2_000 { sim.step(); }
```

## Modules

| Module | Description |
| --- | --- |
| `lattice` | `D2Q9` definition: velocity set, weights, equilibrium distribution. |
| `lib` | `Lbm2D` solver — BGK collision, streaming with half-way bounce-back, macroscopic field recovery, `step_par` (`rayon`) variant; `XBoundary` (periodic / inlet–outlet / open). |
| `sph` | `Sph2D` weakly-compressible SPH: Poly6/Spiky/viscous kernels, Tait EOS, uniform-grid neighbour search. |

### `Lbm2D` features

- half-way bounce-back solid boundaries (stationary or moving lids);
- periodic, or velocity-inlet / zero-gradient-outlet `x` boundaries, plus a
  non-reflecting `Open` (periodic-upstream / zero-gradient-downstream) mode;
- a body-force (Guo-style velocity-shift) term for pressure-gradient /
  gravity-driven flow;
- circular (`add_circle`) and rectangular (`add_rect`) obstacle primitives;
- `rayon`-parallel collision sweep via `step_par`.

## Examples

Runnable with `cargo run --release --example <name> -p tpt-phys-cfd`.

| Example | Demonstrates |
| --- | --- |
| `cavity` | Lid-driven cavity: top lid drags fluid `+x`, establishing a primary recirculating vortex. |
| `poiseuille` | Pressure-driven channel flow; verifies the analytic parabolic (`u(y) ∝ y(H−y)`) Poiseuille profile (R² > 0.97). |
| `flow_past_cylinder` | Uniform stream past a cylinder: steady wake, recirculation bubble, and periodic von Kármán shedding (Strouhal ≈ 0.2 at Re ≈ 130). |
| `dam_break_sph` | SPH dam-break collapse; tracks the leading edge and confirms weakly-compressible density conservation. |

## Validations

- `tests/lid_driven_cavity.rs` — lid-driven cavity: primary recirculating vortex.
- `tests/flow_past_cylinder.rs` — steady symmetric wake + recirculation at
  Re ≈ 22, von Kármán vortex shedding at Re ≈ 72.
- SPH: `dam_break_stays_finite_and_bounded`, `settles_under_gravity`.

## License

Dual-licensed under [MIT](../../LICENSE-MIT) and [Apache-2.0](../../LICENSE-APACHE-2.0).
Copyright TPT Solutions.
