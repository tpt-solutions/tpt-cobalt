# tpt-eng-pcb

PCB (printed-circuit-board) engineering primitives: dielectric layer
stackups, transmission-line/trace models (microstrip characteristic
impedance, IPC-2221 current capacity), vias, and surface-mount pad
footprints.

All scalar quantities are plain `f64` in SI units unless a parameter name or
doc comment says otherwise. Where a formula is defined in a non-SI unit
(notably the IPC-2221 current-capacity relation, which works in thousandths
of an inch, "mils"), the unit is called out explicitly and the crate
provides SI-to-mil helpers (`MIL_TO_M`, `trace_area_mil2`).

## Features

- **[`stackup::Layer`] / [`stackup::Stackup`]** — ordered dielectric/conductor
  layer stackup: `total_thickness`, thickness-weighted-harmonic-mean
  `effective_dielectric_constant`, per-layer `sheet_resistance`.
- **[`trace::microstrip_impedance`]** — surface-microstrip characteristic
  impedance (narrow-strip Wheeler/IPC and wide-strip Edwards closed forms).
- **[`trace::ipc_2221_current_capacity`]** — IPC-2221 temperature-rise
  nomograph current capacity, `I = k·ΔT^0.44·A^0.725`.
- **[`trace::trace_area_mil2`]** — SI trace geometry to mils² cross-section.
- **[`trace::trace_dc_resistance`]** — trace DC resistance via
  `tpt-eng-electrical::dc_resistance`.
- **[`via::Via`]** — plated-through-hole via: `aspect_ratio`,
  `annular_ring_m`.
- **[`footprint::Pad`]** — rectangular SMT/through-hole pad:
  `is_surface_mount`, `pitch_to`.

## Installation

```toml
[dependencies]
tpt-eng-pcb = "0.1"
```

## Quick start

```rust
use tpt_eng_pcb::microstrip_impedance;

// A 50 Ohm microstrip: ~1.8 mm trace on 1.6 mm FR-4 (er = 4.4).
let z0 = microstrip_impedance(1.8e-3, 1.6e-3, 4.4, 35e-6);
assert!((z0 - 50.0).abs() < 6.0);
```

## Crate items

| Item | Purpose |
| --- | --- |
| `stackup::Layer` / `stackup::Stackup` | PCB dielectric/conductor layer stackup. |
| `trace::microstrip_impedance` | Microstrip characteristic impedance. |
| `trace::ipc_2221_current_capacity` | IPC-2221 trace current capacity. |
| `trace::trace_area_mil2` / `MIL_TO_M` | SI-to-mil trace geometry helpers. |
| `trace::trace_dc_resistance` | Trace DC resistance. |
| `via::Via` | Plated through-hole via geometry checks. |
| `footprint::Pad` | Rectangular pad footprint. |

## Related crates

- [tpt-eng-electrical](../tpt-eng-electrical) — supplies `dc_resistance`,
  used by `trace::trace_dc_resistance`.
- [tpt-eng-materials](../tpt-eng-materials) — material property modeling
  (workspace dependency).

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
