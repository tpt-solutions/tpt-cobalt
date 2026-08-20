# tpt-eng-network-matrix

Automated generation of network matrices from an infrastructure
[`Topology`](tpt_eng_geo_topology::Topology) graph: the reduced incidence matrix
`A` and the nodal admittance (Laplacian) matrix `Y`, returned as in-house
[`DMatrix`](tpt_math_linalg::tpt_math_linalg_dense::DMatrix) from
[`tpt-math-linalg`](tpt_math_linalg).

## Features

- **[`incidence_matrix`]** — the reduced node–edge incidence matrix `A`
  (node rows, edge columns; `−1` at a branch tail, `+1` at its head).
- **[`admittance_matrix`]** — the nodal admittance (Laplacian) matrix
  `Y = A · diag(y) · Aᵀ`, built directly from per-branch admittances (taken here
  from each edge's `capacity`).
- Deterministic node/edge ordering (sorted node ids; topology edge order). Edges
  with missing endpoint nodes are skipped, so a malformed topology cannot trigger
  an out-of-bounds index.

## Installation

```toml
[dependencies]
tpt-eng-network-matrix = "0.1"
```

## Quick start

```rust
use tpt_eng_geo_topology::{Edge, EdgeKind, Topology};
use tpt_eng_network_matrix::{admittance_matrix, incidence_matrix};

let mut t = Topology::new();
for n in ["A", "B", "C"] { t.add_node(n); }
t.add_edge(Edge::new("e1", "A", "B", EdgeKind::Wire, 1.0));
t.add_edge(Edge::new("e2", "B", "C", EdgeKind::Wire, 1.0));
t.add_edge(Edge::new("e3", "C", "A", EdgeKind::Wire, 1.0));

let y = admittance_matrix(&t);
assert_eq!((y.nrows(), y.ncols()), (3, 3));
// Each node touches two unit branches → diagonal 2, off-diagonal −1.
assert_eq!(y[(0, 0)], 2.0);
assert_eq!(y[(0, 1)], -1.0);

let a = incidence_matrix(&t);
assert_eq!((a.nrows(), a.ncols()), (3, 3));
```

## Crate items

| Item | Purpose |
| --- | --- |
| `incidence_matrix` | Reduced node–edge incidence matrix `A`. |
| `admittance_matrix` | Nodal admittance (Laplacian) matrix `Y`. |

## Related crates

- [tpt-eng-geo-topology](../tpt-eng-geo-topology/) — the `Topology` graph these
  matrices are generated from.
- [tpt-eng-geo-asset](../tpt-eng-geo-asset/) — node identities behind the topology.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
