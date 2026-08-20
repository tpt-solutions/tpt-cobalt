//! Runnable example: core `tpt-eng-geo-asset` API.
//!
//! Registers a small geographic asset registry, looks assets up by id, and runs
//! the haversine-based spatial queries (`nearest`, `within_radius`).

use tpt_eng_geo_asset::{Asset, AssetKind, AssetRegistry, GeoCoord, haversine};

fn main() {
    let mut reg = AssetRegistry::new();

    // A handful of field devices scattered around a city centre.
    reg.register(Asset::new(
        "meter-01",
        AssetKind::Meter,
        GeoCoord::new(-36.852, 174.764),
        "node-a",
    ));
    reg.register(Asset::new(
        "sensor-02",
        AssetKind::Sensor,
        GeoCoord::new(-36.858, 174.770),
        "node-b",
    ));
    reg.register(Asset::new(
        "pump-03",
        AssetKind::Actuator,
        GeoCoord::new(-36.844, 174.758),
        "node-c",
    ));
    reg.register(Asset::new(
        "tee-04",
        AssetKind::Junction,
        GeoCoord::new(-36.850, 174.780),
        "node-c",
    ));

    println!("registered {} assets", reg.all().len());
    for a in reg.all() {
        println!(
            "  {} [{:?}] @ ({:.4}, {:.4}) -> {}",
            a.id, a.kind, a.coord.lat, a.coord.lon, a.logical_node
        );
    }

    // Latitude clamps and longitude wraps into [-180, 180].
    let wrapped = GeoCoord::new(120.0, 200.0);
    println!(
        "GeoCoord::new(120, 200) -> ({:.3}, {:.3})",
        wrapped.lat, wrapped.lon
    );

    // Id lookup.
    let a = reg.get("pump-03").expect("pump-03 present");
    println!(
        "lookup 'pump-03': kind={:?}, node={}",
        a.kind, a.logical_node
    );

    // Nearest asset to a query point near the centre.
    let query = GeoCoord::new(-36.851, 174.766);
    let near = reg.nearest(query).expect("registry non-empty");
    let d = haversine(near.coord, query);
    println!("nearest to query: {} at {:.3} m", near.id, d);

    // Everything within 1.5 km of the query point (inclusive).
    let radius = 1_500.0;
    let inside = reg.within_radius(query, radius);
    println!("assets within {:.3} m of query: {}", radius, inside.len());
    for a in inside {
        println!("  {} @ {:.3} m", a.id, haversine(a.coord, query));
    }

    // Great-circle distance between two registreed assets.
    let a1 = reg.get("meter-01").unwrap();
    let a2 = reg.get("sensor-02").unwrap();
    println!(
        "haversine(meter-01, sensor-02) = {:.3} m",
        haversine(a1.coord, a2.coord)
    );
}
