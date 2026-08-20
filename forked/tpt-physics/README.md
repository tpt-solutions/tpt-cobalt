# tpt-physics

**A pure-Rust, AI-native engineering & physics simulation framework.**

TPT Solutions · Dual-licensed [MIT](LICENSE-MIT) / [Apache-2.0](LICENSE-APACHE-2.0)

`tpt-physics` is a ground-up, 100% pure-Rust rewrite of core physics
simulation paradigms — Finite Element Analysis (FEA), Discrete Element Method
(DEM), and Computational Fluid Dynamics (CFD) — with no C/C++ FFI, no
`bindgen`, and no system-level dependency hell. It is data-oriented,
unit-safe, and designed from day one to expose differentiable simulation state
to reinforcement-learning agents.

## Why

The engineering simulation industry is dominated by monolithic, decades-old
C++/Fortran codebases (OpenFOAM, CalculiX, ANSYS) that suffer from
build-system hell, opaque memory management, restrictive (GPL) licenses, and
poor integration with modern AI/ML workflows. `tpt-physics` replaces that with
a modular, composable, permissively-licensed Rust workspace.

## Design principles

- **100% Pure Rust.** Zero C/C++ FFI. Cross-compiles trivially to WASM,
  Linux, Windows, and macOS.
- **Strict MIT / Apache-2.0 licensing.** All dependencies are audited via
  [`cargo-deny`](deny.toml); any copyleft (GPL/LGPL/AGPL/SSPL) crate is
  rejected.
- **Compile-time unit safety.** SI units are typed through
  [`tpt-math-units`](https://github.com/tpt-solutions/tpt-math) (a wrap over
  `uom`), preventing catastrophic unit mismatches at compile time.
- **Data-oriented design.** Structure-of-arrays layouts and `rayon` parallelism
  for fearless concurrency on large problems.
- **AI-native & differentiable.** Simulation state is exposed as Gymnasium-like
  environments backed by [`tpt-math-autodiff`].

## Architecture

Per the 2026-08-19 re-scope (`spec2.txt`), `tpt-physics` owns only what
`tpt-fem` doesn't already implement: **DEM**, **meshless CFD** (SPH/LBM), and
**multiphysics coupling orchestration**. FEM is delegated to `tpt-fem` and
rigid-body dynamics to `tpt-science`. Crate names carry the `tpt-phys-*` prefix.
(`spec.txt` is the original Phase-0 design doc this re-scope superseded; kept
for history, not current scope.)

```
tpt-phys/
├── tpt-phys-core          # Material database + CAD→tpt-fem-mesh ingestion adapter
├── tpt-phys-dem           # Granular physics: Hertz–Mindlin, spatial hashing, SIMD, obstacles
├── tpt-phys-cfd           # Meshless CFD: Lattice Boltzmann (D2Q9) + native SPH (free-surface)
├── tpt-phys-fsi           # Partitioned fluid–structure interaction (mapping + coupling driver)
├── tpt-phys-thermal-struct# Thermal-to-structural coupling (ported from the old FEA crate)
├── tpt-phys-electro-thermal # Electro-thermal: Joule heating, resistive losses, T(σ)
├── tpt-phys-orchestrator  # Multiphysics co-simulation (tpt-sci-sim-core) + RL wrappers
├── tpt-phys-gallery       # Example-gallery runner (core/dem/cfd/fsi/thermal/electro/orchestrator)
└── tpt-physics-wasm       # WebAssembly playground bindings (DEM + CFD + SPH)
```

(Validated: DEM, LBM cylinder wake (cavity is ⚠️ experimental), SPH dam-break,
electro-thermal Joule heating, thermal-struct coupling, orchestrator
co-simulation. FSI coupling is ⚠️ experimental — the current driver uses a
lumped `LumpedStructure`, not a real `tpt-fem`-backed FEM solve. The 3-D beam /
shell / J2-plasticity / linear-FEA code moved to `tpt-fem`; the iterative
solvers / GPU dispatch moved to `tpt-fem-solve` / `tpt-phys-*`.)

## Crate-reuse map

Per the 2026-08-15 re-scope, `tpt-physics` reuses the sibling
[`tpt-math`](https://github.com/tpt-solutions/tpt-math) and
[`tpt-fem`](https://github.com/tpt-solutions/tpt-fem) workspaces **directly**
(no wrapper crates, no re-export shims). Each `tpt-phys-*` crate depends on
exactly the sibling crates it needs:

| Capability | Directly-used sibling crate |
| --- | --- |
| SI units | `tpt-math-units`, `tpt-math-units-dyn` |
| Dense linear algebra | `tpt-math-linalg-dense`, `tpt-math-linalg-fixed` |
| Geometry / autodiff | `tpt-math-geometry`, `tpt-math-autodiff` |
| Mesh + generation | `tpt-fem-mesh`, `tpt-fem-mesh-gen` |
| Reference elements / quadrature | `tpt-fem-element`, `tpt-fem-quadrature` |
| Sparse assembly + solve | `tpt-fem-sparse` |
| BCs / elasticity / thermal / eigen / solve | `tpt-fem-assembly`, `tpt-fem-elasticity`, `tpt-fem-thermal`, `tpt-fem-eigen`, `tpt-fem-solve` |

Only the genuinely net-new code lives in `tpt-phys-*`: the material database,
the CAD→`tpt-fem-mesh` ingestion adapter, the DEM contact model, the LBM and SPH
CFD solvers, the FSI coupling driver, electro-thermal Joule heating,
thermal-to-structural coupling, and the multiphysics orchestrator (re-exporting
`tpt-science`'s `tpt-sci-sim-core` co-simulation engine). FEM, linear solvers
and GPU dispatch are consumed from `tpt-fem` / `tpt-science` as needed.

## Building

```sh
cargo build --workspace
cargo test  --workspace
```

Requires a Rust toolchain ≥ 1.84. The sibling `tpt-math`, `tpt-fem`, and
`tpt-science` repos must be checked out as siblings of this directory (the
`[workspace.dependencies]` in `Cargo.toml` reference them by relative path).
`tpt-phys-orchestrator` re-exports `tpt-sci-sim-core` from `tpt-science`.

## Quickstart

```sh
# 1. Verify the sibling dependency workspaces are present.
./scripts/bootstrap.sh        # or: just setup

# 2. Build, test, and run the example gallery.
just test                     # cargo test --workspace
just examples                 # build every example
cargo run --release --example granular_pile -p tpt-phys-dem
```

### Sibling-checkout directory layout

`tpt-physics` pulls `tpt-math`, `tpt-fem`, and `tpt-science` in as **relative
path** dependencies (see `[workspace.dependencies]` in `Cargo.toml`). They must
therefore be checked out as siblings of this directory:

```
your-workspace/
├── tpt-math/          # https://github.com/tpt-solutions/tpt-math
├── tpt-fem/           # https://github.com/tpt-solutions/tpt-fem
├── tpt-science/       # https://github.com/tpt-solutions/tpt-science
└── tpt-physics/       # this repo
```

`scripts/bootstrap.sh` / `bootstrap.ps1` (or `just setup`) clones the siblings
for you. Without them, `cargo build` fails with a `failed to load source for
dependency …` error.

### Quickstart via template

Scaffold a standalone application that depends on the `tpt-phys-*` crates:

```sh
cargo generate --path ./template
cd <project-name>
cargo run
```

The template wires the sibling path dependencies for you (see
[`template/`](template/) and [`CONTRIBUTING.md`](CONTRIBUTING.md)).

A minimal DEM drop using the material database and the granular `World`:

```rust
use tpt_phys_core::MaterialRegistry;
use tpt_phys_dem::particle::Particle;
use tpt_phys_dem::world::World;

let steel = MaterialRegistry::with_defaults().get("Structural Steel").unwrap().clone();
let ps = vec![Particle::new([0.0, 1.0, 0.0], [0.0; 3], 0.5, steel.density)];
let mut world = World::new(ps, 2e-4);
for _ in 0..4000 {
    world.step();
}
println!("kinetic energy = {:.4e} J", world.kinetic_energy());
```

Materials round-trip through JSON (`MaterialRegistry::from_json` / `to_json`),
and the LBM/SPH CFD solvers live in `tpt-phys-cfd`.

## Validated vs. experimental

Capabilities are split by verification maturity. Anything *experimental* is
useful and tested for the cases noted, but has **not** been validated against a
full benchmark suite and may need more work before production use.

| Capability | Status | Notes |
| --- | --- | --- |
| DEM (Hertz–Mindlin, SIMD, >100k) | ✅ Validated | multiple physics tests |
| LBM (Poiseuille, flow-past-cylinder) | ✅ Validated | analytic + shedding benchmarks |
| SPH (free-surface dam break) | ✅ Validated | WCSPH; stays bounded, settles |
| FSI coupling driver | ⚠️ Experimental | explicit + relaxed sub-iterations; **lumped `LumpedStructure`` only — a real `tpt-fem`-backed FEM structure is future work** |
| Thermal-to-structural coupling | ✅ Validated | `thermal_load_vector` integration test |
| Electro-thermal Joule heating | ✅ Validated | heats under load, self-limits |
| Multiphysics co-simulation (orchestrator) | ✅ Validated | `SubModel` adapters + `Simulation` step |
| Differentiable plants (oscillator, pendulum, DEM bulk) | ✅ Validated | analytic/finite-diff Jacobian match (in `tpt-phys-orchestrator`) |
| Cohesive bonds + inter-particle heat | ✅ Validated | new in this pass |
| Lid-driven cavity | ⚠️ Experimental | `#[ignore]`d — primary-vortex convergence issue |
| FSI on a full FEM structure | ⚠️ Experimental | scaffold uses a lumped `LumpedStructure`; real `tpt-fem` elasticity solve is future work |

## Roadmap

- **Phase 1 — Foundation & FEA MVP:** workspace, solvers, linear/nonlinear
  FEA. Milestone: simulate the 3D-printed pile cage spacer (now delegated to
  `tpt-fem`).
- **Phase 2 — Granular & performance:** DEM (concrete aggregate flow),
  GPU acceleration for >100k particles, nonlinear FEA extensions.
- **Phase 3 — Fluids & AI:** LBM CFD, differentiable gym environments,
  documentation, benchmarks, and the "Spacer Benchmark" case study.
- **Phase 5 — DEM / meshless CFD / multiphysics (current):** `tpt-phys-*`
  rename, FSI + thermal-struct + electro-thermal + orchestrator crates,
  native SPH solver, and co-simulation wiring via `tpt-sci-sim-core`.

### External-adoption note (path-dependency model)

`tpt-physics` consumes `tpt-math` / `tpt-fem` / `tpt-science` as **relative
path** dependencies. This is ideal for in-repo development, but it means
`cargo add tpt-phys-dem` does **not** work for an external user who hasn't
checked out the three sibling repos alongside this one. Broader external
adoption (publishing `tpt-physics` to crates.io, or as a standalone git dep)
requires either publishing the sibling crates and switching to version/git deps,
or vendoring them — tracked as a roadmap item, not yet implemented.

See [`todo.md`](todo.md) for the full checklist.

## Validation status

Every non-trivial capability is backed by an integration test (or unit test)
that asserts the *qualitative physics*, not just "it compiles". Highlights:

| Area | Validation | What it proves |
| --- | --- | --- |
| DEM | `tests/granular_settling.rs` | a poured pile settles without blow-up or interpenetration |
| DEM | `tests/ssi_spacer.rs` | soil settles around an embedded cylindrical spacer, no penetration |
| DEM | `tests/pile_cage_flow.rs` | fluidized aggregate flows past a pile cage and comes to rest |
| DEM | `tests/hopper_discharge.rs` | discharge rate follows the Beverloo trend (arching when `D < d`) |
| DEM | `tests/random_close_packing.rs` | mono-disperse bed packs to the RCP fraction (~0.64) |
| DEM | `tests/large_scale.rs` | the `rayon` stepper advances **>100k** particles stably |
| CFD | `src/sph.rs` | SPH dam break stays bounded and settles (weakly compressible) |
| CFD | `tests/lid_driven_cavity.rs` | lid-driven cavity: **known-failing / `#[ignore]`d** (primary-vortex convergence issue; run `cargo test --release -- --ignored`) |
| CFD | `tests/flow_past_cylinder.rs` | steady symmetric wake (low Re) and von Kármán shedding (moderate Re) |
| FSI | `src/coupling.rs` | structure displaces under steady flow and relaxes when flow stops |
| Electro-thermal | `src/lib.rs` | rod heats under voltage, stays finite, self-limits |
| Orchestrator | `src/adapters.rs` | `Simulation` step drives FSI + electro-thermal + thermal-struct |

## Benchmarks & examples

Long-term performance tracking uses a [`criterion`](https://github.com/japaric/criterion.rs)
harness (replacing the old `eprintln!`-timing examples):

```sh
cargo bench --workspace            # all crates
cargo bench -p tpt-phys-dem        # DEM parallel stepper (>100k particles)
cargo bench -p tpt-phys-cfd        # LBM lattice step + SPH step
cargo bench -p tpt-phys-fsi        # FSI coupling-iteration driver
```

Runnable examples (see [`GALLERY.md`](GALLERY.md) for the full
index, or run `./scripts/run_gallery.sh`):

```sh
cargo run --release --example cavity        -p tpt-phys-cfd   # lid-driven cavity
cargo run --release --example granular_pile -p tpt-phys-dem   # settling pile
cargo run --release --example rl_pendulum  -p tpt-phys-orchestrator # differentiable env + Jacobians
cargo run --release --example uq_coupled   -p tpt-phys-orchestrator # UQ x co-simulation
cargo run -p tpt-phys-gallery                              # all-domain "hello world" runner
```

The DEM `rayon` stepper ([`World::step_par`]) is the CPU-acceleration path for
large counts; multiphysics coupling is orchestrated by `tpt-phys-orchestrator`
over `tpt-sci-sim-core` from the sibling `tpt-science` repo.

## WebAssembly playground

`crates/tpt-physics-wasm` ships a browser playground that runs the DEM, CFD
(LBM), and SPH solvers directly in WebGL (via `wasm-bindgen`) — no server
required. Scenes are loaded as JSON and state is pulled back as flat
`Float32Array`s for rendering. The `dem_cfd` scene runs a combined DEM+CFD loop.
See [`crates/tpt-physics-wasm/README.md`](crates/tpt-physics-wasm/README.md)
for the build/run flow (`just wasm` / `just serve-wasm`) and the exact
constructor JSON schema.

> Note: the playground currently exposes **DEM**, **CFD**, and
> **electro-thermal**; FSI / orchestrator bindings are planned but not yet wired
> into the frontend.

## Troubleshooting

- **`failed to load source for dependency tpt-math-...` / `tpt-sci-sim-core`** —
  a sibling repo is missing. Run `./scripts/bootstrap.sh` (or `scripts/bootstrap.ps1`)
  and clone `tpt-math` / `tpt-fem` / `tpt-science` as siblings of this directory
  (`tpt-phys-orchestrator` re-exports `tpt-sci-sim-core` from `tpt-science`).
- **`lid_driven_cavity` test "fails"** — it is intentionally `#[ignore]`d
  (experimental). Run it explicitly with
  `cargo test --release -- --ignored` if you want to inspect it.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE-2.0) at
your option. Copyright TPT Solutions.
