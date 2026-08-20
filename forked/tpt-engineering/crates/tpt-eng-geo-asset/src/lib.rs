//! # tpt-eng-geo-asset
//!
//! Geographic asset registry: maps geographic coordinates to logical
//! device / network nodes for the physical-systems verticals.
//!
//! An [`Asset`] carries an id, a kind, a [`GeoCoord`] (WGS-84-style decimal
//! degrees), and the `logical_node` it maps onto in the infrastructure model.
//! [`AssetRegistry`] supports lookup by id and spatial queries (nearest
//! asset, assets within a radius) using the haversine great-circle distance.

use std::collections::HashMap;

/// A geographic coordinate in decimal degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoCoord {
    /// Latitude in degrees, clamped to [−90, 90].
    pub lat: f64,
    /// Longitude in degrees, wrapped to [−180, 180].
    pub lon: f64,
}

impl GeoCoord {
    /// Construct, clamping latitude and wrapping longitude into range.
    pub fn new(lat: f64, lon: f64) -> Self {
        let lat = lat.clamp(-90.0, 90.0);
        let mut lon = lon % 360.0;
        if lon > 180.0 {
            lon -= 360.0;
        } else if lon < -180.0 {
            lon += 360.0;
        }
        GeoCoord { lat, lon }
    }
}

/// The kind of physical asset at a location.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetKind {
    /// A measurement sensor.
    Sensor,
    /// A control actuator.
    Actuator,
    /// A passive junction / tee.
    Junction,
    /// A metering point.
    Meter,
    /// A source or sink.
    Source,
}

/// A registered asset: a real device at a geographic point that maps onto a
/// logical node in the infrastructure topology.
#[derive(Debug, Clone, PartialEq)]
pub struct Asset {
    /// Unique asset identifier.
    pub id: String,
    /// What the asset is.
    pub kind: AssetKind,
    /// Where it is.
    pub coord: GeoCoord,
    /// The logical node id in the infrastructure model it belongs to.
    pub logical_node: String,
}

impl Asset {
    /// Construct an asset.
    pub fn new(id: &str, kind: AssetKind, coord: GeoCoord, logical_node: &str) -> Self {
        Asset {
            id: id.to_string(),
            kind,
            coord,
            logical_node: logical_node.to_string(),
        }
    }
}

/// A registry of geographic assets.
#[derive(Debug, Clone, Default)]
pub struct AssetRegistry {
    by_id: HashMap<String, Asset>,
    list: Vec<Asset>,
}

impl AssetRegistry {
    /// A new, empty registry.
    pub fn new() -> Self {
        AssetRegistry::default()
    }

    /// Register an asset (replaces any existing id).
    pub fn register(&mut self, asset: Asset) {
        self.by_id.insert(asset.id.clone(), asset.clone());
        self.list.push(asset);
    }

    /// Look up an asset by id.
    pub fn get(&self, id: &str) -> Option<&Asset> {
        self.by_id.get(id)
    }

    /// All registered assets.
    pub fn all(&self) -> &[Asset] {
        &self.list
    }

    /// The asset nearest `target` by great-circle distance, if any.
    ///
    /// Assets whose coordinates are non-finite (NaN/inf) are ignored rather
    /// than causing a panic, so a single malformed registry entry cannot crash
    /// a call built from untrusted asset data.
    pub fn nearest(&self, target: GeoCoord) -> Option<&Asset> {
        self.list
            .iter()
            .filter(|a| a.coord.lat.is_finite() && a.coord.lon.is_finite())
            .min_by(|a, b| {
                haversine(a.coord, target)
                    .partial_cmp(&haversine(b.coord, target))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Assets within `radius_m` metres of `target` (inclusive).
    ///
    /// Assets whose coordinates are non-finite (NaN/inf) are ignored, matching
    /// the behaviour of [`AssetRegistry::nearest`].
    pub fn within_radius(&self, target: GeoCoord, radius_m: f64) -> Vec<&Asset> {
        self.list
            .iter()
            .filter(|a| a.coord.lat.is_finite() && a.coord.lon.is_finite())
            .filter(|a| haversine(a.coord, target) <= radius_m)
            .collect()
    }
}

const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Great-circle distance (metres) between two coordinates (haversine).
pub fn haversine(a: GeoCoord, b: GeoCoord) -> f64 {
    let lat1 = a.lat.to_radians();
    let lat2 = b.lat.to_radians();
    let dlat = (b.lat - a.lat).to_radians();
    let dlon = (b.lon - a.lon).to_radians();
    let h = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_M * h.sqrt().asin()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coord_clamps_and_wraps() {
        let c = GeoCoord::new(120.0, 200.0);
        assert!((c.lat - 90.0).abs() < 1e-9);
        assert!((c.lon - (-160.0)).abs() < 1e-9);
    }

    #[test]
    fn registry_lookup_and_nearest() {
        let mut r = AssetRegistry::new();
        r.register(Asset::new(
            "A",
            AssetKind::Sensor,
            GeoCoord::new(0.0, 0.0),
            "n1",
        ));
        r.register(Asset::new(
            "B",
            AssetKind::Meter,
            GeoCoord::new(1.0, 1.0),
            "n2",
        ));
        assert_eq!(r.get("A").unwrap().logical_node, "n1");
        let near = r.nearest(GeoCoord::new(0.1, 0.1)).unwrap();
        assert_eq!(near.id, "A");
    }

    #[test]
    fn haversine_known() {
        // ~111 km per degree of latitude near the equator.
        let d = haversine(GeoCoord::new(0.0, 0.0), GeoCoord::new(1.0, 0.0));
        assert!((d - 111_195.0).abs() < 1_000.0);
    }

    #[test]
    fn within_radius_filters() {
        let mut r = AssetRegistry::new();
        r.register(Asset::new(
            "A",
            AssetKind::Sensor,
            GeoCoord::new(0.0, 0.0),
            "n1",
        ));
        r.register(Asset::new(
            "B",
            AssetKind::Sensor,
            GeoCoord::new(10.0, 10.0),
            "n2",
        ));
        assert_eq!(r.within_radius(GeoCoord::new(0.0, 0.0), 200_000.0).len(), 1);
    }

    #[test]
    fn non_finite_coords_ignored_in_radius() {
        let mut r = AssetRegistry::new();
        r.register(Asset::new(
            "A",
            AssetKind::Sensor,
            GeoCoord::new(f64::NAN, 0.0),
            "n1",
        ));
        // The malformed entry must not match as a zero-distance (NaN) hit.
        assert_eq!(r.within_radius(GeoCoord::new(0.0, 0.0), 1.0).len(), 0);
    }
}
