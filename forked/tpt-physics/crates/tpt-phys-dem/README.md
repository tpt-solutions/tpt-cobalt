# tpt-phys-dem

Discrete Element Method: Hertz–Mindlin contact, uniform-grid spatial hashing,
SIMD narrow-phase resolution, and a minimal time-stepping driver.

No DEM / particle code exists in the sibling `tpt-math` / `tpt-fem` workspaces,
so everything here is net-new.

## Modules

| Module | Description |
| --- | --- |
| `particle` | `Particle` — spherical state (position, velocity, radius, mass). |
| `contact` | Hertz–Mindlin law: `hertz_normal_force`, `contact_force` (normal + damping + Coulomb-capped Mindlin tangential friction), reduced modulus/radius/mass. |
| `broadphase` | `SpatialHash` — uniform-grid broad phase with candidate-pair and neighbour-list queries. |
| `simd` | SIMD-accelerated narrow-phase contact resolution. |
| `obstacle` | Fixed boundaries: `Obstacle::Cylinder` and `Obstacle::Plane` (half-space walls / hopper funnels / rectangular containments). |
| `world` | `World` driver — semi-implicit Euler with gravity, planar floor (inelastic), obstacle boundaries, optional fluidized driving term, and a speed clamp. `World::step_par` is the `rayon` parallel contact sweep. `World::relax` is a position-based overlap-heal warmup (no energy injected) for stable settling after random placement. |

## Quick start

```rust
use tpt_phys_dem::{
    Particle, World,
    scenarios::{granular_pile, PileParams, run},
};

// Drop and settle a cubic pile.
let params = PileParams { count: 400, edge: 1.5, height: 3.0, grain_r: 0.05, density: 2500.0 };
let mut world = granular_pile(params);
let report = run(&mut world, 2000);
println!("settled bed height: {:.3} m", report.bed_height());
```

Or build a scene by hand with fixed obstacles:

```rust
use tpt_phys_dem::{Particle, World, obstacle::Obstacle};

let particles: Vec<Particle> = (0..200)
    .map(|i| Particle::new([(i % 10) as f64 * 0.11, 2.0 + i as f64 * 0.01, 0.0], [0.0; 3], 0.05, 2500.0))
    .collect();
let obstacles = vec![Obstacle::Plane { point: [0.0, 0.0, 0.0], normal: [0.0, 1.0, 0.0], y_range: None }];
let mut world = World::with_obstacles(particles, 1e-4, obstacles);
world.gravity = -9.81;
for _ in 0..2000 { world.step(); }
```

Cohesive / wet grains: set `world.bond_stiffness` and `world.bond_strength`,
then call `world.create_bonds(fraction)` to freeze contacts formed at the
current state (used by the `cohesive_bonds` example).

## Examples

Runnable with `cargo run --release --example <name> -p tpt-phys-dem`.

| Example | Demonstrates |
| --- | --- |
| `hopper_discharge` | `scenarios::hopper_discharge` / `HopperParams`; orifice-size sweep reproduces the Beverloo mass-flow trend. |
| `cohesive_bonds` | Cohesive bonds (`create_bonds`, `bond_stiffness`, `bond_strength`, `active_bonds`); comparison of a snap-to-pieces pile vs a bonded cohesive pile. |
| `obstacles_ssi` | Soil–structure interaction: a granular bed settling around an embedded cylindrical spacer (`Obstacle::Cylinder`), with no penetration and a settling KE-decay curve. |
| `heat_conduction` | Thermal contacts (`heat_transfer_coeff`, `specific_heat`); a 9-grain chain conserves energy and converges to analytic Fourier conduction (λ ≈ 0.170 W/m·K). |
| `checkpoint` | Determinism + restart: `to_checkpoint` / `from_checkpoint` / `save_checkpoint` / `load_checkpoint` give a bit-exact resume. |
| `parallel_step` | `step` vs `step_par` parity and a scaling measurement; documents where the parallel stepper does / does not win. |

## Validations

- `tests/granular_settling.rs` — generic settling without blow-up.
- `tests/hopper_discharge.rs` — arching below ~1 particle-diameter orifice,
  increasing discharge with orifice size (Beverloo trend).
- `tests/pile_cage_flow.rs` — wet-concrete aggregate flow through a pile cage.
- `tests/ssi_spacer.rs` — soil–structure interaction around an embedded cylinder.
- `tests/random_close_packing.rs` — mono-disperse random close packing (~0.64).
- `tests/large_scale.rs` — >100k-particle bed advanced stably via `step_par`.

## License

Dual-licensed under [MIT](../../LICENSE-MIT) and [Apache-2.0](../../LICENSE-APACHE-2.0).
Copyright TPT Solutions.
