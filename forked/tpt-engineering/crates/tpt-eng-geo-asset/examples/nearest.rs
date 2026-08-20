//! Runnable example: nearest-asset selection with `tpt-eng-geo-asset`.
//!
//! Builds a larger registry of mixed-kind devices and uses `nearest` /
//! `within_radius` to answer the classic "which field device serves this point?"
//! and "how many devices are in range?" questions, including a non-finite
//! coordinate that spatial queries must ignore.

use tpt_eng_geo_asset::{Asset, AssetKind, AssetRegistry, GeoCoord, haversine};

fn main() {
    let mut reg = AssetRegistry::new();

    // Scattered water network devices (lat, lon in decimal degrees).
    let devices = [
        ("w-1", AssetKind::Meter, -36.852, 174.764),
        ("w-2", AssetKind::Sensor, -36.858, 174.770),
        ("w-3", AssetKind::Actuator, -36.844, 174.758),
        ("w-4", AssetKind::Junction, -36.850, 174.780),
        ("w-5", AssetKind::Source, -36.840, 174.772),
        ("w-6", AssetKind::Sensor, -36.856, 174.762),
    ];
    for (id, kind, lat, lon) in devices {
        reg.register(Asset::new(id, kind, GeoCoord::new(lat, lon), "net"));
    }

    // A malformed entry: a NaN coordinate must NOT be selected as the nearest
    // (NaN compares false) nor matched as a zero-distance hit.
    reg.register(Asset::new(
        "broken",
        AssetKind::Sensor,
        GeoCoord::new(f64::NAN, 0.0),
        "net",
    ));

    let query = GeoCoord::new(-36.851, 174.766);

    // Nearest device to the query point.
    let near = reg.nearest(query).expect("at least one finite asset");
    println!(
        "nearest device to query: {} ({:?}) at {:.3} m",
        near.id,
        near.kind,
        haversine(near.coord, query)
    );

    // Ranked nearest N by explicit sort over great-circle distance.
    let mut ranked: Vec<_> = reg
        .all()
        .iter()
        .filter(|a| a.coord.lat.is_finite() && a.coord.lon.is_finite())
        .map(|a| (a, haversine(a.coord, query)))
        .collect();
    ranked.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
    println!("closest 3 devices:");
    for (a, d) in ranked.iter().take(3) {
        println!("  {} @ {:.3} m", a.id, d);
    }

    // Count devices inside a coverage radius at two scales.
    for radius in [500.0, 2_000.0] {
        let n = reg.within_radius(query, radius).len();
        println!("devices within {:.3} m: {}", radius, n);
    }
}
