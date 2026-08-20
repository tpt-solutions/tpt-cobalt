# tpt-eng-materials

Material property modeling for the TPT engineering ecosystem.

This crate models materials as a named, categorized collection of `Material`
values, each carrying typed `Property` data and provenance. Three property forms
are supported:

- **scalar** — a single value with a unit string,
- **temperature-dependent** — `(temperature, value)` samples evaluated by linear
  interpolation (`Property::value_at`), and
- **anisotropic** — per-direction scalars (a hook for directional or
  tensor-valued properties).

## Features

- **Material model** (`material`): named, categorized `Material` values carrying
  typed properties and provenance.
- **Property forms** (`property`): scalar, temperature-dependent, and
  anisotropic properties.
- **Categories** (`category`): a `MaterialCategory` taxonomy.
- **Library** (`library`): in-memory `MaterialLibrary` with JSON and CSV
  load/save.
- **Data policy**: only original, user-provided, or openly licensed data is
  accepted; every material must record a source, and `MaterialLibrary::validate`
  enforces that rule. No proprietary material tables are bundled or scraped.

## Data policy

**Only original, user-provided, or openly licensed data is accepted.** Every
material must record a source (`provenance::DataSource`), and
`MaterialLibrary::validate` enforces that rule: a material with no recorded
source, or a disallowed `license` attribute, fails validation. No proprietary
material tables are bundled or scraped.

## Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
tpt-eng-materials = "0.1"
```

## Quick start

```rust
use tpt_eng_materials::{Material, MaterialCategory, MaterialLibrary, Property};

let mut lib = MaterialLibrary::new();
lib.add(
    Material::new("steel-s355", "S355", MaterialCategory::Metal)
        .with_property(
            "youngs-modulus",
            Property::Scalar {
                value: 210.0,
                unit: "GPa".into(),
            },
        ),
);
let e = lib.get_by_id("steel-s355").unwrap().value("youngs-modulus", 0.0).unwrap();
assert!((e - 210.0).abs() < 1e-12);
```

## Persistence

Libraries load/save as JSON (`MaterialLibrary::from_json` /
`MaterialLibrary::to_json`) and as CSV (`MaterialLibrary::from_csv` /
`MaterialLibrary::to_csv`).

An optional embedded-database backend was evaluated and deliberately **not**
adopted: the in-memory + JSON/CSV model keeps the dependency tree small and
fully license-clean (all of `serde`, `serde_json`, and `csv` are
MIT/Apache-2.0 compatible).

## Crate modules

| Module | Purpose |
| --- | --- |
| `material` | `Material` value with typed properties and provenance. |
| `property` | `Property` (scalar / temperature-dependent / anisotropic) and `TempPoint`. |
| `category` | `MaterialCategory` taxonomy. |
| `library` | `MaterialLibrary` with JSON/CSV persistence and `validate`. |
| `error` | `MaterialError` and `Result` type. |

The `prelude` module re-exports the most commonly used items.

## Related crates

- [`tpt-eng-safety`](../tpt-eng-safety) — consolidated utilization/pass-fail evaluation.
- [`tpt-eng-structural`](../tpt-eng-structural) — structural analysis and code checks.
- [`tpt-eng-sections`](../tpt-eng-sections) — cross-section properties.
- [`tpt-eng-standards`](../tpt-eng-standards) — standards modeling as data.

## Status

Initial `0.1.0` release. Persistence is JSON/CSV; no database dependency is
introduced.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR
[Apache-2.0](../../LICENSE-APACHE).
