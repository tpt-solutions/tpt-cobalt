//! Richer scenario: maintain a project material catalogue — persist it as JSON,
//! exchange it as CSV, and enforce the crate's data-provenance policy.
//!
//! Run with `cargo run -p tpt-eng-materials --example library_io`.

use tpt_eng_materials::{
    ALLOWED_LICENSES, DataSource, Material, MaterialCategory, MaterialLibrary, Property, TempPoint,
};

/// Build the project catalogue: four materials, each with a recorded source.
fn catalogue() -> MaterialLibrary {
    let mut lib = MaterialLibrary::new();

    lib.add(
        Material::new(
            "steel-s355",
            "S355 structural steel",
            MaterialCategory::Metal,
        )
        .with_source(DataSource::standard("EN 10025-2 (user-entered)"))
        .with_property(
            "youngs-modulus",
            Property::Scalar {
                value: 210.0,
                unit: "GPa".into(),
            },
        )
        .with_property(
            "density",
            Property::Scalar {
                value: 7850.0,
                unit: "kg/m^3".into(),
            },
        )
        .with_property(
            "yield-strength",
            Property::TemperatureDependent {
                unit: "MPa".into(),
                points: vec![
                    TempPoint {
                        temp: 20.0,
                        value: 355.0,
                    },
                    TempPoint {
                        temp: 300.0,
                        value: 280.0,
                    },
                    TempPoint {
                        temp: 600.0,
                        value: 110.0,
                    },
                ],
            },
        ),
    );

    lib.add(
        Material::new("alu-6061-t6", "AA6061-T6", MaterialCategory::Metal)
            .with_source(DataSource::file("mill-certificate-6061.pdf"))
            .with_property(
                "youngs-modulus",
                Property::Scalar {
                    value: 68.9,
                    unit: "GPa".into(),
                },
            )
            .with_property(
                "density",
                Property::Scalar {
                    value: 2700.0,
                    unit: "kg/m^3".into(),
                },
            )
            .with_property(
                "yield-strength",
                Property::TemperatureDependent {
                    unit: "MPa".into(),
                    points: vec![
                        TempPoint {
                            temp: 20.0,
                            value: 276.0,
                        },
                        TempPoint {
                            temp: 150.0,
                            value: 220.0,
                        },
                        TempPoint {
                            temp: 300.0,
                            value: 60.0,
                        },
                    ],
                },
            ),
    );

    lib.add(
        Material::new(
            "concrete-c30",
            "Concrete C30/37",
            MaterialCategory::Concrete,
        )
        .with_source(DataSource::standard("project mix design (user-entered)"))
        .with_property(
            "youngs-modulus",
            Property::Scalar {
                value: 33.0,
                unit: "GPa".into(),
            },
        )
        .with_property(
            "density",
            Property::Scalar {
                value: 2400.0,
                unit: "kg/m^3".into(),
            },
        ),
    );

    let mut timber = Material::new("glulam-gl28h", "Glulam GL28h", MaterialCategory::Wood)
        .with_source(DataSource::standard("supplier datasheet (user-entered)"))
        .with_property(
            "youngs-modulus",
            Property::Scalar {
                value: 12.6,
                unit: "GPa".into(),
            },
        )
        .with_property(
            "density",
            Property::Scalar {
                value: 460.0,
                unit: "kg/m^3".into(),
            },
        );
    // Metadata attributes are free-form; `license` is the one the policy reads.
    timber
        .metadata
        .attributes
        .insert("license".into(), "user-provided".into());
    lib.add(timber);

    lib
}

fn main() {
    let lib = catalogue();
    println!("catalogue holds {} materials", lib.len());

    // --- 1. Specific stiffness ranking (E / density), a common screening step ---
    println!();
    println!(
        "{:<24} {:>10} {:>12} {:>14}",
        "material", "E [GPa]", "rho [kg/m3]", "E/rho [MJ/kg]"
    );
    let mut ranked: Vec<(&str, f64, f64)> = lib
        .materials
        .iter()
        .filter_map(|m| {
            let e = m.value("youngs-modulus", 20.0)?;
            let rho = m.value("density", 20.0)?;
            Some((m.name.as_str(), e, rho))
        })
        .collect();
    ranked.sort_by(|a, b| (b.1 / b.2).total_cmp(&(a.1 / a.2)));
    for (name, e, rho) in &ranked {
        // E in GPa over density in kg/m^3 gives MJ/kg (1 GPa/(kg/m^3) = 1 MJ/kg).
        println!(
            "{name:<24} {e:>10.3} {rho:>12.3} {:>14.3}",
            e / rho * 1000.0
        );
    }

    // --- 2. Elevated-temperature strength retention ---
    println!();
    println!("yield strength retention (fraction of the 20 degC value)");
    for m in &lib.materials {
        if !m.has_temperature_dependence() {
            continue;
        }
        let base = m.value("yield-strength", 20.0).expect("20 degC sample");
        let mut line = format!("{:<24}", m.name);
        for temp in [100.0, 200.0, 300.0, 400.0] {
            let fy = m.value("yield-strength", temp).expect("interpolated");
            line.push_str(&format!(" {:>7.3}", fy / base));
        }
        println!("{line}   (100/200/300/400 degC)");
    }

    // --- 3. JSON round-trip (the crate's persistence format) ---
    let json = lib.to_json().expect("serialize library");
    let restored = MaterialLibrary::from_json(&json).expect("deserialize library");
    println!();
    println!("JSON size          = {} bytes", json.len());
    println!("JSON round-trip ok = {}", restored == lib);

    // --- 4. CSV exchange, flattened at a reference temperature ---
    let mut csv_bytes: Vec<u8> = Vec::new();
    lib.to_csv(&mut csv_bytes, 200.0).expect("write csv");
    let csv_text = String::from_utf8(csv_bytes.clone()).expect("utf-8 csv");
    println!();
    println!("CSV export at 200 degC reference temperature:");
    for line in csv_text.lines() {
        println!("  {line}");
    }

    let from_csv = MaterialLibrary::from_csv(csv_bytes.as_slice()).expect("read csv");
    let steel_fy = from_csv
        .get_by_id("steel-s355")
        .and_then(|m| m.value("yield-strength", 0.0))
        .expect("flattened fy");
    println!(
        "re-imported {} materials; S355 fy frozen at 200 degC = {steel_fy:.3} MPa",
        from_csv.len()
    );

    // --- 5. Data policy: every material needs a source, licenses must be open ---
    println!();
    println!("allowed licenses   = {}", ALLOWED_LICENSES.join(", "));
    match lib.validate() {
        Ok(()) => println!("catalogue validates: every material is traceable"),
        Err(e) => println!("unexpected validation failure: {e}"),
    }

    let mut tainted = lib;
    let mut vendor = Material::new("vendor-alloy", "Vendor alloy X", MaterialCategory::Metal);
    vendor
        .metadata
        .attributes
        .insert("license".into(), "proprietary".into());
    tainted.add(vendor); // no source *and* a disallowed license
    match tainted.validate() {
        Ok(()) => println!("unexpected: proprietary data accepted"),
        Err(e) => println!("policy rejects the vendor entry -> {e}"),
    }
    println!("tainted library compliant = {}", tainted.is_compliant());
}
