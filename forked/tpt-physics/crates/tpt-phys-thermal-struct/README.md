# tpt-phys-thermal-struct

Thermal expansion and thermo-mechanical stress coupling (FEM → FEM).

Steady-state heat conduction is reused directly from `tpt-fem-thermal`, and
the structural solve from `tpt-fem-elasticity` — this crate's only job is the
coupling between them: converting a solved temperature field into a
thermal-strain load on the structural degrees of freedom.

Ported from `tpt-physics-fea` (removed) when this repo re-scoped to
multiphysics coupling per `spec2.txt`, with FEM itself delegated to `tpt-fem`.

## Quick start

```rust
use tpt_fem_mesh::{CellType, MeshBuilder};
use tpt_phys_core::Material;
use tpt_phys_thermal_struct::thermal_load_vector;

let mat = Material::new("Steel", 200e9, 0.3, 7850.0, 12e-6);
let mut b = MeshBuilder::new();
let n0 = b.add_node(vec![0.0, 0.0, 0.0]);
let n1 = b.add_node(vec![1.0, 0.0, 0.0]);
let n2 = b.add_node(vec![0.0, 1.0, 0.0]);
let n3 = b.add_node(vec![0.0, 0.0, 1.0]);
b.add_element(CellType::Tet, vec![n0, n1, n2, n3]);
let mesh = b.build();
let temps = vec![150.0, 50.0, 50.0, 50.0];
let load = thermal_load_vector(&mesh, 3, &mat, &temps, 20.0);
```

## Modules

| Module | Description |
| --- | --- |
| (root) | `tet4_thermal_load` — element thermal-strain load for a linear tetrahedron; `thermal_load_vector` — global assembly over a `Tet` mesh. Both return `None`/`skip` on a degenerate (inverted) element instead of producing `NaN`. |

For a linear tetrahedral element the stress-free thermal strain
`ε_th = α (T − T_ref) · [1,1,1,0,0,0]` produces the element load
`f_th = ∫ Bᵀ D ε_th dV`.

## Examples

Runnable with `cargo run --release --example <name> -p tpt-phys-thermal-struct`.

| Example | Demonstrates |
| --- | --- |
| `uniform_expansion` | A uniform temperature rise is stress-free (zero load); a gradient yields a non-zero, self-equilibrated load; degenerate elements return `None`. |
| `layered_strip` | A two-layer strip with a hot top / cold bottom layer: the thermal load is concentrated on the hot layer and the global assembly is self-equilibrated. |

## Validations

- zero temperature delta ⇒ zero load;
- single-tet load is self-equilibrated (net force ≈ 0);
- uniform mesh temperature ⇒ zero total load.

## License

Dual-licensed under [MIT](../../LICENSE-MIT) and [Apache-2.0](../../LICENSE-APACHE-2.0).
Copyright TPT Solutions.
