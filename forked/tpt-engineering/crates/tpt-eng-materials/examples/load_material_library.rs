//! End-to-end example: build a small material library, look up a material, and
//! evaluate a property at a temperature.
//! Run with `cargo run --example load_material_library`.

use tpt_eng_materials::{
    DataSource, Material, MaterialCategory, MaterialLibrary, Property, TempPoint,
};

fn main() {
    let mut lib = MaterialLibrary::new();
    lib.add(
        Material::new("steel-s355", "S355", MaterialCategory::Metal)
            .with_source(DataSource::standard("EN 10025"))
            .with_property(
                "youngs-modulus",
                Property::Scalar {
                    value: 210.0,
                    unit: "GPa".into(),
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
                            temp: 100.0,
                            value: 345.0,
                        },
                    ],
                },
            ),
    );

    let steel = lib.get_by_id("steel-s355").unwrap();
    println!("E @ 20C  = {:?} GPa", steel.value("youngs-modulus", 20.0));
    println!("fy @ 60C = {:?} MPa", steel.value("yield-strength", 60.0));
    println!("library compliant = {}", lib.is_compliant());
}
