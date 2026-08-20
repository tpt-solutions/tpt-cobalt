# tpt-eng-geo-topology

Directional infrastructure graphs for pipes / wires / ducts, with
upstream/downstream traversal and flow-direction logic.

Each [`Edge`] has a direction (`from → to`) representing the nominal flow
direction, enabling upstream/downstream traversal and reachability logic ("what
is fed by node X?", "what feeds node Y?"). Node ids correspond to
[`tpt_eng_geo_asset::Asset`] ids in the wider model.

## Features

- **[`Edge`] / [`EdgeKind`]** — a directed connection carrying a medium (`Pipe`,
  `Wire`, `Duct`) and a nominal capacity/rating.
- **[`Topology`]** — add nodes/edges; query `outgoing` / `incoming` edges,
  `downstream` / `upstream` node sets, and `reaches` reachability.
- Built on [`tpt_eng_geo_asset`] node identities.

## Installation

```toml
[dependencies]
tpt-eng-geo-topology = "0.1"
```

## Quick start

```rust
use tpt_eng_geo_topology::{Edge, EdgeKind, Topology};

let mut topo = Topology::new();
topo.add_node("tank");
topo.add_node("pump");
topo.add_node("valve");
topo.add_edge(Edge::new("e1", "tank", "pump", EdgeKind::Pipe, 1.0));
topo.add_edge(Edge::new("e2", "pump", "valve", EdgeKind::Pipe, 1.0));

// Downstream of the tank is the pump and then the valve.
let mut down = topo.downstream("tank");
down.sort();
assert_eq!(down, vec!["pump", "valve"]);

// Upstream of the valve is the pump and the tank.
let mut up = topo.upstream("valve");
up.sort();
assert_eq!(up, vec!["pump", "tank"]);

// A directed path tank → valve exists.
assert!(topo.reaches("tank", "valve"));
```

## Crate items

| Item | Purpose |
| --- | --- |
| `Topology` | A directional infrastructure graph. |
| `Edge` / `EdgeKind` | A directed, medium-tagged connection. |
| `downstream` / `upstream` | BFS reachability along/against flow. |
| `reaches` / `outgoing` / `incoming` | Path and adjacency queries. |

## Related crates

- [tpt-eng-geo-asset](../tpt-eng-geo-asset/) — node identities behind `Topology`.
- [tpt-eng-network-matrix](../tpt-eng-network-matrix/) — turns a `Topology` into
  incidence / admittance matrices.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
