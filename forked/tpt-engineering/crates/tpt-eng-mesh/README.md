# tpt-eng-mesh

A license-clean triangle-mesh crate for engineering/CAD workloads, built on [`tpt-eng-geometry`].

It provides a simple indexed [`Mesh`] data model together with:

- **Normals** — per-face and area-weighted smooth vertex normals.
- **Quality metrics** — triangle angles, aspect ratios, edge lengths, and degenerate-face counts.
- **Refinement & repair** — midpoint subdivision, degenerate-face removal, and vertex welding.
- **Format conversion (in-house)** — binary/ASCII STL and Wavefront OBJ, implemented from scratch with no third-party format crates.

`Point3` and `Vector3` are re-exported aliases over [`glam::Vec3`] (positions vs. directions).

## Features

- Indexed `Mesh` with optional per-vertex normals, texture coordinates, and per-corner normal indices.
- Per-face and area-weighted smooth vertex normals.
- Quality metrics: min/max triangle angle, aspect ratio, edge lengths, degenerate-face counts.
- Refinement & repair: `subdivide_midpoint`, `remove_degenerate_faces`, `weld_vertices`.
- In-house binary/ASCII STL and Wavefront OBJ codecs carrying vertices, faces, texture coordinates, and per-corner normals.

## Installation

```toml
[dependencies]
tpt-eng-mesh = "0.1"
```

## Quick start

```rust
use tpt_eng_mesh::Mesh;
use tpt_eng_geometry::Point3;

let tri = [
    Point3::new(0.0, 0.0, 0.0),
    Point3::new(1.0, 0.0, 0.0),
    Point3::new(0.0, 1.0, 0.0),
];
let mesh = Mesh::from_triangles(&[tri]).with_smooth_normals();
assert_eq!(mesh.face_count(), 1);
assert_eq!(mesh.vertex_count(), 3);

// In-house STL/OBJ codecs.
let stl = mesh.to_stl_binary();
let parsed = Mesh::from_stl_binary(&stl).unwrap();
let obj = parsed.to_obj();
let from_obj = Mesh::from_obj(&obj).unwrap();
```

## Crate modules

| Module | Purpose |
| --- | --- |
| `Mesh` | Indexed triangle mesh with normals, quality metrics, refinement/repair, and STL/OBJ codecs |

## Related crates

- [tpt-eng-geometry](../tpt-eng-geometry/) — geometric primitives (`Point3`, `Vector3`, queries) that `tpt-eng-mesh` builds on.
- [tpt-eng-cad](../tpt-eng-cad/) — extracts `tpt_eng_mesh::Mesh` surfaces from signed-distance fields.
- [tpt-eng-nurbs](../tpt-eng-nurbs/) — tessellates NURBS surfaces into `tpt_eng_mesh::Mesh`.
- [tpt-eng-io](../tpt-eng-io/) — engineering file I/O (STL/OBJ) on top of `tpt-eng-mesh`.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
