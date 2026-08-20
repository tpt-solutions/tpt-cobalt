# tpt-phys-core

Foundational, **net-new** data structures for the [`tpt-physics`](../../README.md)
workspace: a type-safe engineering **material database**, a **CAD → mesh
ingestion adapter**, and an optional **Monte-Carlo uncertainty-quantification**
sweep over material-property scatter.

Per the 2026-08-15 re-scope, `tpt-phys-core` contains *only* code that is
genuinely new to the physics workspace. SI units, mesh containers, reference
elements, sparse assembly and the FEM solvers are consumed **directly** from the
sibling [`tpt-math`](https://github.com/tpt-solutions/tpt-math) /
[`tpt-fem`](https://github.com/tpt-solutions/tpt-fem) crates — there is no
wrapper or re-export shim.

## What lives here

| Module | Status | Description |
| --- | --- | --- |
| [`material`] | **new** | Type-safe [`Material`] / [`MaterialRegistry`] with `serde` JSON serialization. Stores base-SI `f64` fields (`E` in Pa, `ν`, `ρ` in kg/m³, `α` in 1/K) and exposes compile-time-typed accessors via [`tpt_math_units`], plus derived isotropic moduli (Lamé `λ`, shear `G`, bulk `K`). |
| [`cad`] | **new** | [`CadIngestor`] lowers a tessellated boundary-representation solid ([`CadSolid`]) into a [`tpt_fem_mesh::Mesh`] by fan-triangulating each polygon face and preserving per-face physical-group `region` tags. |
| [`uq`] | **new**, feature `uq` | Monte-Carlo sweeps ([`monte_carlo`], [`monte_carlo_seeded`]) that sample materials from a relative tolerance band ([`tol_band`]) and summarise a scalar response ([`Statistics`]: mean/std/min/max/percentiles/CoV). Includes worked forward models: [`cantilever_tip_deflection`] and [`cantilever_natural_frequency`]. |

The sibling units crate is re-exported as `tpt_phys_core::units` so consumers can
get compile-time-typed quantities from a single import.

## Why a material database at all?

`tpt-fem` / `tpt-math` pass material constants as bare function arguments. Real
engineering models need a *named, serializable, unit-checked* library so that a
project can ship its own vetted data alongside a model and round-trip it through
JSON. Materials are stored in base SI and validated by the isotropic-elasticity
identity `E = 3K(1 − 2ν) = 2G(1 + ν)`.

## Feature flags

| Feature | Default | Effect |
| --- | --- | --- |
| `uq` | off | Enables the [`uq`] module (adds a `proptest` dependency used as the reproducible sampling engine). |

## Quick start

```rust
use tpt_phys_core::MaterialRegistry;

let reg = MaterialRegistry::with_defaults();
let steel = reg.get("Structural Steel").expect("built-in");
assert!(steel.youngs_modulus > 100e9);

// Derived isotropic moduli, all in Pa.
let (g, k) = (steel.shear_modulus(), steel.bulk_modulus());
assert!(g > 0.0 && k > 0.0);

// Persist / restore a whole library.
let json = reg.to_json().unwrap();
let back = MaterialRegistry::from_json(&json).unwrap();
assert_eq!(reg, back);
```

## Examples

Run any with `cargo run --example <name> -p tpt-phys-core` (add `--release` for
the sweeps):

| Example | Feature | What it shows |
| --- | --- | --- |
| [`material_registry`](examples/material_registry.rs) | — | Load the built-in library, derive `λ/G/K`, add a custom alloy, use unit-safe accessors, and JSON round-trip. |
| [`cad_ingest`](examples/cad_ingest.rs) | — | Build a unit cube as a B-rep solid, ingest it into a triangulated `tpt-fem-mesh`, and exercise both ingestion error paths. |
| [`uq_cantilever`](examples/uq_cantilever.rs) | `uq` | Monte-Carlo tip-deflection scatter for a ±10% `E` / ±5% `ν` band. |
| [`uq_natural_frequency`](examples/uq_natural_frequency.rs) | `uq` | Correlated `E` **and** `ρ` scatter propagated into the fundamental modal frequency (`f₁ ∝ sqrt(E/ρ)`). |

```sh
cargo run --example material_registry -p tpt-phys-core
cargo run --example cad_ingest        -p tpt-phys-core
cargo run --release --example uq_natural_frequency -p tpt-phys-core --features uq
```

## Testing

```sh
cargo test -p tpt-phys-core                 # material + CAD unit tests
cargo test -p tpt-phys-core --features uq   # also the Monte-Carlo UQ tests
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache-2.0](../../LICENSE-APACHE-2.0)
at your option. Copyright TPT Solutions.

[`material`]: src/material.rs
[`cad`]: src/cad.rs
[`uq`]: src/uq.rs
[`Material`]: src/material.rs
[`MaterialRegistry`]: src/material.rs
[`CadIngestor`]: src/cad.rs
[`CadSolid`]: src/cad.rs
[`monte_carlo`]: src/uq.rs
[`monte_carlo_seeded`]: src/uq.rs
[`tol_band`]: src/uq.rs
[`Statistics`]: src/uq.rs
[`cantilever_tip_deflection`]: src/uq.rs
[`cantilever_natural_frequency`]: src/uq.rs
[`tpt_math_units`]: https://github.com/tpt-solutions/tpt-math
[`tpt_fem_mesh::Mesh`]: https://github.com/tpt-solutions/tpt-fem
