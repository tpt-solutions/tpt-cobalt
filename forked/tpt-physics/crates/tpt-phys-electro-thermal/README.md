# tpt-phys-electro-thermal

Electro-thermal coupling for `tpt-physics`: Joule heating, resistive losses, and
temperature-dependent conductivity.

No prior art for this exists in `tpt-physics`, `tpt-fem`, or `tpt-science`, so the
crate is built from scratch. It closes the multiphysics loop:

```
electric field  →  current J = σ(T)·E  →  Joule heating q = σ(T)·|E|²
                                                      ↑
                            temperature T  →  σ(T)  (positive feedback)
                                                      ↑
                          heat conduction + surface convection
```

The reference solver is a 1-D resistive rod (`ElectroThermalRod`, explicit finite
differences) — the textbook Joule-heating geometry. The same `step` contract
generalises to an arbitrary mesh.

## Quick start

```rust
use tpt_phys_electro_thermal::ElectroThermalRod;

let mut rod = ElectroThermalRod::new(21, 300.0);
rod.dx = 0.01;
rod.set_voltage(10.0);
rod.convection = 50.0; // sink so a steady state exists
for _ in 0..2000 {
    rod.step(1e-4);
}
assert!(rod.temperatures().iter().all(|&t| t.is_finite()));
```

Conductivity is temperature-dependent (metallic, negative TCR):
`σ(T) = σ₀ / (1 + α (T − T_ref))`, so a hotter conductor carries *less* current
and self-limits its heating.

## API

| Item | Description |
| --- | --- |
| `ElectroThermalRod` | 1-D electro-thermal rod: electrical + thermal state, explicit `step`. |
| `ElectroThermalRod::conductivity` | `σ(T)` — temperature-dependent conductivity. |
| `ElectroThermalRod::joule_heating` | Volumetric source `σ(T) |E|²` at a node. |
| `ElectroThermalRod::total_joule_power` | Total deposited Joule power (W, per-unit cross-section). |
| `ElectroThermalRod::set_ends` | `EndCondition::Insulated` (adiabatic) or `Fixed(T)` (Dirichlet). |
| `ElectroThermalRod::set_voltage` | Applied voltage; uniform field `E = V / L`. |

## Examples

Runnable with `cargo run --release --example <name> -p tpt-phys-electro-thermal`.

| Example | Demonstrates |
| --- | --- |
| `heated_rod` | A 1-D rod driven by a voltage reaches a finite, stabilising steady temperature under surface convection. |
| `self_limiting` | The metallic negative-TCR `σ(T)` self-limits the heating: the rod runs cooler than an equivalent constant-`σ` conductor. |

## Validations

- Heats up under voltage and stays finite; no voltage ⇒ no heating.
- `σ(T)` is monotonically decreasing in temperature (metallic negative TCR).

## Status

Scaffold. The 1-D operator is verified; a 3-D mesh-based variant (reusing
`tpt-fem-mesh` / `tpt-fem-thermal`) is future work — see the crate-source
roadmap comment for the intended `∇·(σ(T)∇φ)` → `q = σ|∇φ|²` → conduction path.

## License

Dual-licensed under [MIT](../../LICENSE-MIT) and [Apache-2.0](../../LICENSE-APACHE-2.0).
Copyright TPT Solutions.
