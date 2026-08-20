# tpt-phys-fsi

Partitioned fluid–structure interaction (FSI) coupling for `tpt-physics`.

This crate wires a fluid solver (`tpt-phys-cfd`) to a structural model
(`tpt-fem-mesh` from `tpt-fem`) through a partitioned, under-relaxed
Gauss–Seidel exchange. No FSI code exists in the sibling workspaces, so this
is net-new.

> Status: **scaffold**. The fluid→structure traction is a placeholder drag law
> (`traction_from_fluid`, a `coeff · ρ|u|²` term), not a real fluid-stress
> tensor. The coupling loop, mesh mapping, and relaxation machinery are real
> and exercised; the physics coefficients are stand-ins to be replaced by a
> proper boundary-integral stress transfer.

## Quick start

```rust
use tpt_fem_mesh::MeshBuilder;
use tpt_phys_cfd::{Lbm2D, XBoundary};
use tpt_phys_fsi::{FsiDriver, LumpedStructure, StructuralModel};

let mut sim = Lbm2D::new(32, 16, 0.6);
sim.set_x_boundary(XBoundary::Inlet(0.1));
sim.set_horizontal_walls();
sim.add_rect(30, 1, 30, 14); // compliant wall
sim.initialise(1.0, [0.1, 0.0]);

let mut b = MeshBuilder::new();
b.add_node(vec![30.0, 8.0, 0.0]);
let mesh = b.build();

let mut structure = LumpedStructure::new(1, 1.0, 10.0, 2.0);
let mut driver = FsiDriver::new(&sim, &structure, &mesh);
for _ in 0..200 {
    driver.step(&mut sim, &mut structure, 1.0);
}
```

## Modules

| Module | Description |
| --- | --- |
| (root) | `nearest_node_map` — nearest-neighbour mesh mapping between non-matching fluid/structural interface discretizations. |
| `coupling` | `FsiDriver` partitioned loop, `couple_explicit` sub-step, `LumpedStructure` (anchored mass–spring–damper) `StructuralModel`, `fluid_interface_points` / `FluidInterfacePoint`. |

### Coupling loop

`FsiDriver::step` advances the fluid by one LBM step and performs `substeps`
Gauss–Seidel coupling iterations:

1. advance the fluid (`Lbm2D::step`);
2. sample surface traction at the fluid interface and map it onto the nearest
   structural node (`nearest_node_map`);
3. advance the structural model;
4. map the structural velocity back onto the fluid boundary as a moving wall
   (Ladd bounce-back correction), under `relaxation` under-relaxation.

## Examples

Runnable with `cargo run --release --example <name> -p tpt-phys-fsi`.

| Example | Demonstrates |
| --- | --- |
| `compliant_wall` | A vertical wall driven by channel flow; the partitioned driver maps traction onto a lumped structural node, which deflects downstream and relaxes when the flow stops. |
| `mesh_mapping` | The `nearest_node_map` primitive: builds a fluid interface and verifies every fluid point maps to its closest structural node. |

## Validations

- `nearest_node_map` maps to the closest node; empty structural mesh yields an
  empty map.
- `structure_displaces_under_steady_flow_and_relaxes` — flow pushes the
  structure downstream; it relaxes when the flow is removed.
- `no_interface_no_coupling` — a domain with no walls has an empty interface.

## License

Dual-licensed under [MIT](../../LICENSE-MIT) and [Apache-2.0](../../LICENSE-APACHE-2.0).
Copyright TPT Solutions.
