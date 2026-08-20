# tpt-eng-io

Engineering file I/O (STL and OBJ meshes, plus JSON and CSV) for the TPT ecosystem, built on [`tpt-eng-mesh`].

This crate provides traits and implementations for reading/writing common engineering file formats including JSON, CSV, STL, and OBJ. STL/OBJ files are exchanged as [`tpt_eng_mesh::Mesh`] values (this crate adds no duplicate geometry model of its own). `tpt-eng-mesh` is the consolidation point for mesh codecs in the workspace.

## Features

- `read_stl` / `write_stl` / `write_stl_ascii` operating on `tpt_eng_mesh::Mesh`.
- `read_obj` / `write_obj` operating on `tpt_eng_mesh::Mesh`.
- `read_json` / `write_json` and `read_csv` / `write_csv` for tabular/serialized data.
- `ReadFromFile`, `WriteToFile`, and `EngineeringData` traits for file-backed (de)serialization with `serde`.

## Installation

```toml
[dependencies]
tpt-eng-io = "0.1"
```

## Quick start

```rust
use tpt_eng_io::{read_stl, write_stl};
use tpt_eng_io::Mesh;

# fn main() -> tpt_eng_io::Result<()> {
let mesh: Mesh = read_stl("part.stl")?;
let _ = mesh.triangle_count();
write_stl(&mesh, "part_out.stl")?;
# Ok(())
# }
```

## Crate modules

| Module | Purpose |
| --- | --- |
| `stl` | STL read/write (binary and ASCII) over `tpt_eng_mesh::Mesh` |
| `obj` | Wavefront OBJ read/write over `tpt_eng_mesh::Mesh` |
| `json` | JSON read/write via `serde` |
| `csv` | CSV read/write of records |
| `error` | `Error` / `Result` types |
| `Mesh` | Re-export of the canonical mesh type used for STL/OBJ exchange |

## Related crates

- [tpt-eng-mesh](../tpt-eng-mesh/) — the consolidation point for mesh codecs; STL/OBJ are exchanged as `tpt_eng_mesh::Mesh`.
- [tpt-eng-geometry](../tpt-eng-geometry/) — geometric primitives used by the underlying mesh.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
