# tpt-eng-geotech

Soil mechanics primitives for the TPT engineering ecosystem.

## What it provides

- **Mohr-Coulomb** (`mohr_coulomb`) — shear strength `τ_f = c + σ_n·tanφ` and
  factor of safety against shear failure.
- **Cam-Clay** (`cam_clay`) — reduced modified critical-state yield surface and
  void-ratio update under normal consolidation or swelling.
- **Bearing capacity** (`bearing_capacity`) — Terzaghi and Meyerhof ultimate
  bearing capacity with shape/depth factors, net and allowable capacities.
- **Consolidation** (`consolidation`) — coefficient of consolidation, 1-D primary
  consolidation settlement, and time-rate (time factor ↔ degree of consolidation).
- **Lateral earth pressure** (`lateral_earth_pressure`) — Rankine active/passive
  coefficients and resultants, plus the Coulomb active coefficient.
- **Atterberg limits / USCS** (`atterberg`) — plasticity index, liquidity index,
  consistency state, USCS fine-grained classification, and soil activity.
- **Borehole stratigraphy** — `SoilLayer` / `Borehole` with
  `tpt_eng_materials::DataSource` provenance tracking.

## Units

Stresses, pressures, and strengths are in **pascals (Pa)**, depths in **metres
(m)**, angles in **degrees**, and void ratios dimensionless, unless a function
states otherwise. SI units are the caller's responsibility; no `uom` is used.

## Example

```rust
use tpt_eng_geotech::bearing_capacity::{FoundationShape, terzaghi_ultimate_bearing_capacity};
use tpt_eng_geotech::mohr_coulomb::shear_strength;

let tau_f = shear_strength(0.0, 35.0, 100_000.0);
let q_ult = terzaghi_ultimate_bearing_capacity(5_000.0, 28.0, 19_000.0, 2.0, 1.0, FoundationShape::Square, 2.0);
assert!(q_ult > tau_f);
```

Run the bundled example with `cargo run -p tpt-eng-geotech --example basic`.

Dual-licensed under MIT OR Apache-2.0.
