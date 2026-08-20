# tpt-engineering

[![CI](https://github.com/tpt-solutions/tpt-engineering/actions/workflows/ci.yml/badge.svg)](https://github.com/tpt-solutions/tpt-engineering/actions/workflows/ci.yml)
[![license: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

Applied engineering primitives for the [TPT Solutions](https://github.com/tpt-solutions)
physical-systems verticals.

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).

> **29 of 43 crates are published on [crates.io](https://crates.io)** at
> `0.1.0` (Batches 1-5, released 2026-08-15 — see `RELEASE.md`). The 14
> newest domain crates (`props-mixture`, `electrical`, `schedule`, `biomech`,
> `crystallography`, `geotech`, `heat-transfer`, `vehicle-dynamics`,
> `power-components`, `pcb`, `renewables`, `building-sys`, `thermal-mgmt`,
> `unit-ops`) are not yet published — see `PUBLISH_TRACKING.md`. There is no
> `v0.1.0` git tag yet. APIs are unstable until the first tagged release.

## Scope

A set of small, single-purpose `tpt-eng-*` crates providing reusable
engineering maths — fluid/gas properties, time-series conditioning,
infrastructure topology, controls, structural/mechanical analysis,
geometry/CAD primitives, and domain-specific component models — drawn out of
the product verticals as foundation-pillar building blocks rather than
product-shaped packages. The workspace has grown from the original 14-crate
Phase-0 scope (see `spec.txt`) to 43 crates: structural/mechanical and
geometry/CAD domains were added post-Phase-4, and a further 14
electronics/energy/transport/medical/process/earth/construction
component-model crates were added per `spec2.txt` (Phase 9).

All crates are new implementations built from scratch (the spec's claimed
consolidation source `tpt-rust2` does not exist on this machine, and the
external vertical repos were not available for audit from this build
environment).

## Release & publish status

29 of the 43 `tpt-eng-*` crates are published on
[crates.io](https://crates.io) at `0.1.0` (Batches 1–5, 2026-08-15); the
remaining 14 newest domain crates are tracked for release in
[`PUBLISH_TRACKING.md`](PUBLISH_TRACKING.md) (Batches 6–8). The full batch plan
lives in [`RELEASE.md`](RELEASE.md).

Per-crate publish state is kept consistent with those tracking docs by
`cargo xtask publish-status` (also run as part of `cargo xtask check` in CI),
which asserts the README/CHANGELOG counts and the pending-crate set against
`Cargo.toml` / `RELEASE.md` / `PUBLISH_TRACKING.md`.

| Status | Crates | Source |
|--------|--------|--------|
| Published (`0.1.0`) | 29 | `RELEASE.md` Batches 1–5 |
| Pending (Batches 6–8) | 14 | `PUBLISH_TRACKING.md` |

> There is no `v0.1.0` git tag yet. APIs are unstable until the first tagged
> release.

## Crate inventory

| Crate | Domain | no_std | Description |
|-------|--------|--------|-------------|
| `tpt-eng-props-water` | fluid-properties | yes | IAPWS-IF97 water/steam property tables |
| `tpt-eng-props-air` | fluid-properties | yes | ASHRAE psychrometrics / moist-air |
| `tpt-eng-props-fuels` | fluid-properties | yes | Fuel heating values, density, combustion |
| `tpt-eng-props` | fluid-properties | yes | Umbrella: re-exports water + air + fuels |
| `tpt-eng-timeseries-core` | timeseries | | Core time-series types |
| `tpt-eng-timeseries-align` | timeseries | | Irregular multi-rate stream alignment |
| `tpt-eng-timeseries-gap` | timeseries | | Staleness/gap detection + interpolation |
| `tpt-eng-timeseries` | timeseries | | Umbrella: re-exports core + align + gap |
| `tpt-eng-geo-asset` | geo/topology | | Geographic asset registry |
| `tpt-eng-geo-topology` | geo/topology | | Directional infrastructure graphs |
| `tpt-eng-network-matrix` | geo/topology | | Incidence/admittance matrix generation |
| `tpt-eng-controls` | controls | | PID / state-space / transfer-function |
| `tpt-eng-structural` | structural/mechanical | | Loads, beam/frame analysis, code checks |
| `tpt-eng-materials` | structural/mechanical | | Material property library + data provenance |
| `tpt-eng-sections` | structural/mechanical | | Cross-section properties (rect/circle/I-section) |
| `tpt-eng-standards` | structural/mechanical | | Load cases, design factors, code-check results |
| `tpt-eng-tolerance` | structural/mechanical | | 1-D dimensional tolerance stack-up (worst-case/RSS/Monte-Carlo) |
| `tpt-eng-gdt` | structural/mechanical | | GD&T data model (zones, datum frames, material modifiers) |
| `tpt-eng-reliability` | structural/mechanical | | Fatigue, Weibull, FMEA |
| `tpt-eng-safety` | structural/mechanical | | Limit-state margins / utilization ratios |
| `tpt-eng-geometry` | geometry/CAD | | Points, vectors, frames |
| `tpt-eng-nurbs` | geometry/CAD | | B-spline / NURBS curves and surfaces |
| `tpt-eng-mesh` | geometry/CAD | | Triangle mesh data structure, quality/repair |
| `tpt-eng-cad` | geometry/CAD | | SDF solid modeling / CSG |
| `tpt-eng-io` | geometry/CAD | | STL/OBJ/JSON/CSV file I/O |
| `tpt-eng-plot` | output | | Charts and section drawings (via `plotters`) |
| `tpt-eng-report` | output | | Markdown/HTML/JSON calculation reports |
| `tpt-eng-cli` | integration | | Command-line tool over the above |
| `tpt-eng-examples` | integration | | Cross-crate scenarios composing multiple domains |
| `tpt-eng-electrical` | electrical | no | Impedance/reactance, per-unit systems, three-phase power |
| `tpt-eng-heat-transfer` | thermal | no | Convection/radiation correlations, conduction, thermal-resistance networks |
| `tpt-eng-props-mixture` | fluid-properties | yes | General real-gas / VLE property lookups for arbitrary mixtures |
| `tpt-eng-pcb` | electronics | no | PCB layer stackup, trace routing, via/footprint definitions |
| `tpt-eng-thermal-mgmt` | cross-domain | no | Heat sink sizing, fan curves, thermal-resistance networks |
| `tpt-eng-power-components` | energy | no | Transformer/generator models, transmission-line parameters |
| `tpt-eng-renewables` | energy | no | Solar PV, wind turbine, battery degradation models |
| `tpt-eng-vehicle-dynamics` | transport | no | Tire models, aero drag, suspension kinematics |
| `tpt-eng-biomech` | medical | no | Hyperelastic tissue models, implant geometry |
| `tpt-eng-unit-ops` | process | no | Distillation, heat exchangers, pump/compressor curves |
| `tpt-eng-crystallography` | materials | no | Miller indices, slip systems, symmetry operations |
| `tpt-eng-geotech` | earth | no | Soil constitutive models, borehole stratigraphy |
| `tpt-eng-building-sys` | construction | no | HVAC loads, plumbing fixture units, panel scheduling |
| `tpt-eng-schedule` | construction | no | CPM/PERT, resource leveling, Earned Value Management |

## Building

This workspace depends on the [`tpt-math`](https://github.com/tpt-solutions/tpt-math)
substrate (`tpt-math-units`, `tpt-math-numeric`, `tpt-math-linalg`, `tpt-math-stats`),
resolved as regular versioned dependencies from crates.io — no sibling checkout needed:

```sh
git clone https://github.com/tpt-solutions/tpt-engineering.git
cd tpt-engineering
cargo build --workspace
cargo test --workspace
```

- **Edition:** `2024`.
- **MSRV:** `1.85` (`rust-version` in `[workspace.package]`). The toolchain
  version is additionally fixed by `rust-toolchain.toml` via `rustup`.

The `tpt-eng-props-*` family is `no_std`; it builds for bare-metal targets
with the default features disabled.

## Which crate do I need?

The inventory above is a flat list. For choosing a starting point by task:

| If you need to… | Start with |
|-------|--------|
| Compute water/steam thermodynamic properties (IAPWS-IF97) | `tpt-eng-props-water` |
| Compute moist-air / psychrometric properties (ASHRAE) | `tpt-eng-props-air` |
| Compute fuel heating values, density, combustion properties | `tpt-eng-props-fuels` |
| Work with any/all fluid properties at once | `tpt-eng-props` (umbrella) |
| Condition/timestamp irregular sensor streams | `tpt-eng-timeseries-core` |
| Align multi-rate streams (e.g. 1 Hz CAN vs 10 s Modbus) onto one grid | `tpt-eng-timeseries-align` |
| Detect staleness/dropouts and interpolate gaps | `tpt-eng-timeseries-gap` |
| Use all time-series primitives together | `tpt-eng-timeseries` (umbrella) |
| Register geographic assets / devices | `tpt-eng-geo-asset` |
| Model pipes/wires/ducts as directed graphs | `tpt-eng-geo-topology` |
| Generate incidence/admittance matrices from topology | `tpt-eng-network-matrix` |
| Implement PID / state-space / transfer-function control | `tpt-eng-controls` |
| Compute loads, beam/frame analysis, code checks | `tpt-eng-structural` |
| Look up material properties (with data provenance) | `tpt-eng-materials` |
| Compute cross-section properties (rect/circle/I-section) | `tpt-eng-sections` |
| Apply code load cases / design factors / pass-fail checks | `tpt-eng-standards` |
| Roll up 1-D dimensional tolerance stack-ups (worst-case/RSS/Monte-Carlo) | `tpt-eng-tolerance` |
| Model GD&T zones, datum frames, material modifiers | `tpt-eng-gdt` |
| Compute fatigue life, Weibull reliability, run an FMEA | `tpt-eng-reliability` |
| Compute limit-state margins / utilization ratios | `tpt-eng-safety` |
| Work with points/vectors/frames in 2D/3D | `tpt-eng-geometry` |
| Build B-spline / NURBS curves and surfaces | `tpt-eng-nurbs` |
| Represent/repair/query a triangle mesh | `tpt-eng-mesh` |
| Build solids via SDF/CSG modeling | `tpt-eng-cad` |
| Read/write STL, OBJ, JSON, or CSV files | `tpt-eng-io` |
| Draw a chart or section diagram | `tpt-eng-plot` |
| Generate a calculation report (Markdown/HTML/JSON) | `tpt-eng-report` |
| Run any of the above from a command line, no Rust required | `tpt-eng-cli` |
| A full worked cross-crate scenario | `tpt-eng-examples` |
| Compute electrical impedance / per-unit / three-phase power | `tpt-eng-electrical` |
| Compute convection/radiation heat transfer and conduction | `tpt-eng-heat-transfer` |
| Look up general real-gas / VLE mixture properties | `tpt-eng-props-mixture` |
| Design a PCB stackup / trace / via | `tpt-eng-pcb` |
| Size a heat sink or solve a fan-curve operating point | `tpt-eng-thermal-mgmt` |
| Model transformers, generators, or transmission lines | `tpt-eng-power-components` |
| Model solar PV, wind turbines, or battery degradation | `tpt-eng-renewables` |
| Model tire forces, aero drag, or suspension kinematics | `tpt-eng-vehicle-dynamics` |
| Model hyperelastic tissue or implant geometry | `tpt-eng-biomech` |
| Size distillation columns / heat exchangers, pump curves | `tpt-eng-unit-ops` |
| Work with Miller indices, slip systems, crystal symmetry | `tpt-eng-crystallography` |
| Model soil behavior or borehole stratigraphy | `tpt-eng-geotech` |
| Compute HVAC loads, plumbing, or electrical panel schedules | `tpt-eng-building-sys` |
| Plan/schedule projects (CPM/PERT, Earned Value Management) | `tpt-eng-schedule` |

## Quickstart

Add the crates you need (they re-export `tpt-math`'s `uom`-typed units):

```sh
cargo add tpt-eng-props tpt-eng-controls tpt-eng-structural
```

```rust
use tpt_eng_controls::{Pid, PidGains};

let mut pid = Pid::new(PidGains::new(2.0, 1.0, 0.0)).with_output_limit(1.0);
pid.set_setpoint(10.0);
let mut y = 0.0;
for _ in 0..1000 {
    let u = pid.update(y, 0.01);
    y += 0.01 * u; // first-order plant, τ = 1 s
}
assert!((y - 10.0).abs() < 0.5);
```

For a full, cross-crate worked example (topology → network matrix → controls →
time-series conditioning → structural check), see
[`tpt-eng-examples`](crates/tpt-eng-examples).

## Developer tooling

`cargo xtask` provides one-stop hygiene and scaffolding (`check`, `test`,
`doctest`, `doc`, `no-std-matrix`, `new-crate`). A root [`justfile`](justfile)
mirrors these for non-Cargo users.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the issues-only contribution
policy and where to report problems.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE)
at your option.

Copyright (c) 2026 TPT Solutions.
