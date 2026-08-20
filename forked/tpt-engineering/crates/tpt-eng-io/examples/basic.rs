//! tpt-eng-io basic usage: JSON, CSV, and STL round-trips to a temp directory.
//!
//! Run with: `cargo run --example basic -p tpt-eng-io`

use serde::{Deserialize, Serialize};

use tpt_eng_geometry::Point3;
use tpt_eng_io::{
    CsvRecord, Mesh,
    csv::{read_csv_with_headers, write_csv_with_headers},
    read_json, write_json, write_stl,
};

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Part {
    name: String,
    mass_kg: f64,
    coordinates: Vec<f64>,
}

fn main() {
    let dir = std::env::temp_dir();

    // --- JSON ---
    let part = Part {
        name: "bracket".into(),
        mass_kg: 2.5,
        coordinates: vec![0.0, 1.0, 2.0],
    };
    let json_path = dir.join("tpt_io_basic.json");
    write_json(&part, &json_path).unwrap();
    let loaded: Part = read_json(&json_path).unwrap();
    println!(
        "JSON round-trip: {} (mass {:.1} kg)",
        loaded.name, loaded.mass_kg
    );
    assert_eq!(part, loaded);

    // --- CSV with headers ---
    let headers = vec!["node".into(), "x".into(), "y".into()];
    let rows = vec![
        CsvRecord::new(vec!["1".into(), "0.0".into(), "0.0".into()]),
        CsvRecord::new(vec!["2".into(), "1.0".into(), "0.0".into()]),
        CsvRecord::new(vec!["3".into(), "0.0".into(), "1.0".into()]),
    ];
    let csv_path = dir.join("tpt_io_basic.csv");
    write_csv_with_headers(&headers, &rows, &csv_path).unwrap();
    let (h, recs) = read_csv_with_headers(&csv_path).unwrap();
    println!(
        "CSV round-trip: {} columns, {} data rows",
        h.len(),
        recs.len()
    );
    assert_eq!(h, headers);
    assert_eq!(recs, rows);

    // --- STL mesh round-trip ---
    let tri = [[
        Point3::ZERO,
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ]];
    let mesh: Mesh = Mesh::from_triangles(&tri);
    let stl_path = dir.join("tpt_io_basic.stl");
    write_stl(&mesh, &stl_path).unwrap();
    let loaded_mesh: Mesh = tpt_eng_io::read_stl(&stl_path).unwrap();
    println!(
        "STL round-trip: {} faces (expected {})",
        loaded_mesh.face_count(),
        mesh.face_count()
    );
}
