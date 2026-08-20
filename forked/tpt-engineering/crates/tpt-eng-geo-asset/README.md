# tpt-eng-geo-asset

Geographic asset registry: maps geographic coordinates to logical device /
network nodes. Supports id lookup and spatial queries (nearest asset, assets
within a radius) using the haversine great-circle distance.

## Features

- **[`GeoCoord`]** — latitude/longitude in decimal degrees, with latitude
  clamping and longitude wrapping into `[−180, 180]`.
- **[`Asset`]** — an id, kind ([`AssetKind`]), a coordinate, and the
  `logical_node` it maps onto in the infrastructure model.
- **[`AssetRegistry`]** — register/lookup assets; `nearest` and `within_radius`
  spatial queries by great-circle distance.
- **[`haversine`]** — great-circle distance (metres) between two coordinates.
- Non-finite coordinates are ignored by spatial queries rather than causing a
  panic, so malformed registry entries cannot crash a caller built on untrusted
  data.

## Installation

```toml
[dependencies]
tpt-eng-geo-asset = "0.1"
```

## Quick start

```rust
use tpt_eng_geo_asset::{Asset, AssetKind, AssetRegistry, GeoCoord, haversine};

let mut r = AssetRegistry::new();
r.register(Asset::new("A", AssetKind::Sensor, GeoCoord::new(0.0, 0.0), "n1"));
r.register(Asset::new("B", AssetKind::Meter, GeoCoord::new(1.0, 1.0), "n2"));
assert_eq!(r.get("A").unwrap().logical_node, "n1");
assert_eq!(r.nearest(GeoCoord::new(0.1, 0.1)).unwrap().id, "A");

// Everything within ~200 km of the origin (≈ 2° at the equator).
assert_eq!(r.within_radius(GeoCoord::new(0.0, 0.0), 200_000.0).len(), 1);

// ~111 km per degree of latitude near the equator.
let d = haversine(GeoCoord::new(0.0, 0.0), GeoCoord::new(1.0, 0.0));
assert!((d - 111_195.0).abs() < 1_000.0);
```

## Crate items

| Item | Purpose |
| --- | --- |
| `GeoCoord` | Latitude/longitude in decimal degrees. |
| `Asset` / `AssetKind` | A located device and its kind. |
| `AssetRegistry` | Register/lookup plus `nearest` / `within_radius` queries. |
| `haversine` | Great-circle distance (m) between two coordinates. |

## Related crates

- [tpt-eng-geo-topology](../tpt-eng-geo-topology/) — uses `Asset` ids as graph
  node identities.
- [tpt-eng-network-matrix](../tpt-eng-network-matrix/) — builds solver matrices
  from a topology of these assets.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
