# tpt-physics-wasm

**WebAssembly bindings for the `tpt-physics` solvers** — a browser playground
that runs the pure-Rust DEM, CFD, and electro-thermal engines directly via
WebGL, with no server.

This crate is the repo's WebAssembly front door. It exposes four solvers to
JavaScript via [`wasm-bindgen`](https://crates.io/crates/wasm-bindgen):

* **DEM** ([`tpt_phys_dem`]) — a granular `World` you drive step-by-step.
* **CFD** ([`tpt_phys_cfd`]) — the D2Q9 LBM (`Lbm2D`) solver.
* **SPH** ([`tpt_phys_cfd`]) — the 2-D weakly-compressible `Sph2D` solver, for
  free-surface flow the LBM mesh can't represent.
* **Electro-thermal** ([`tpt_phys_electro_thermal`]) — the 1-D Joule-heating
  `ElectroThermalRod`.

Load a scene (particles/obstacles or a lattice setup) as JSON, advance it
step-by-step, and pull state back out as flat `Float32Array`s ready to upload
to a WebGL buffer.

> **Note (2026-08-20):** the playground exposes **DEM**, **CFD**, **SPH**, and
> **electro-thermal**. FSI / orchestrator / thermal-struct bindings are planned
> but not yet wired into the WebGL frontend — see `src/lib.rs` for the exact
> surface.

## Build

Requires `wasm-pack` (or `cargo` + `wasm-bindgen-cli`) on `PATH`.

```sh
# from the workspace root
just wasm          # runs scripts/build_wasm.ps1 (or .sh)
just serve-wasm    # serves crates/tpt-physics-wasm/www on :8080
```

`just wasm` produces `crates/tpt-physics-wasm/www/pkg`; open
`crates/tpt-physics-wasm/www/index.html` (served by `just serve-wasm`) to use
the playground.

## Public surface

* `DemSimulation` — `new(json)`, `step()`, `count()`, `positions()`,
  `velocities()`, `temperatures()`, `kinetic_energy()`.
* `CfdSimulation` — `new(json)`, `step()`, `nx()`, `ny()`, `velocity()`,
  `solid()`.
* `SphSimulation` — `new(json)`, `step()`, `count()`, `smoothing_length()`,
  `positions()`, `speeds()`, `densities()`, `kinetic_energy()`.
* `ElectroThermalSimulation` — `new(json)`, `step(dt)`, `temperatures()`,
  `max_temperature()`.

See the doc comments on each constructor for the exact JSON scene schema
(particles + obstacles for DEM; lattice size, walls, obstacles, moving lid for
CFD; node count / voltage / convection for electro-thermal).

## Examples

The binding surface is exercised from Rust in `tests/bindings.rs` (build a
scene from JSON, step it, read scalar state), and live in the browser via
`www/playground.js`.

`examples/coupled_dem_cfd.rs` is a resolved, one-way **CFD-DEM** coupling demo:
the LBM fluid (`Lbm2D`) is driven by a body force and its mean x-velocity is
sampled each step and injected into the DEM bed (`World`) as a uniform drag
body acceleration via `World::external_accel` — the hook a partitioned
CFD-DEM solver uses. Two-way back-coupling (granular drag on the fluid) is a
follow-up; the `dem_cfd` playground scene runs both engines in one loop.

## License

Dual-licensed under [MIT](../../LICENSE-MIT) and [Apache-2.0](../../LICENSE-APACHE-2.0).
Copyright TPT Solutions.
