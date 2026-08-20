//! Basic `tpt-eng-materials` usage: the three property forms, material queries,
//! and a small library with provenance tracking.
//!
//! Run with `cargo run -p tpt-eng-materials --example basic`.

use std::collections::BTreeMap;

use tpt_eng_materials::{
    DataSource, Material, MaterialCategory, MaterialLibrary, Property, TempPoint,
};

fn main() {
    // --- 1. A metal: scalar stiffness + temperature-dependent yield strength ---
    let steel = Material::new(
        "steel-s355",
        "Structural steel S355",
        MaterialCategory::Metal,
    )
    .with_description("Hot-rolled structural steel, user-entered values")
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
                    temp: 200.0,
                    value: 320.0,
                },
                TempPoint {
                    temp: 400.0,
                    value: 230.0,
                },
                TempPoint {
                    temp: 600.0,
                    value: 110.0,
                },
            ],
        },
    );

    println!("== {} ({}) ==", steel.name, steel.category);
    println!("source            : {}", steel.source.label);
    let e = steel
        .value("youngs-modulus", 20.0)
        .expect("modulus present");
    let rho = steel.value("density", 20.0).expect("density present");
    println!("E                 = {e:.3} GPa");
    println!("density           = {rho:.3} kg/m^3");

    // `value` interpolates linearly between samples and clamps outside them.
    for temp in [20.0, 100.0, 300.0, 500.0, 800.0] {
        let fy = steel.value("yield-strength", temp).expect("fy present");
        println!("fy @ {temp:>5.0} degC   = {fy:.3} MPa");
    }
    println!(
        "fy unit           = {}",
        steel.property("yield-strength").expect("fy present").unit()
    );
    println!(
        "temp-dependent    = {}, anisotropic = {}",
        steel.has_temperature_dependence(),
        steel.has_anisotropy()
    );

    // --- 2. A composite: direction-dependent (anisotropic) stiffness ---
    let mut moduli = BTreeMap::new();
    moduli.insert("11".to_string(), 135.0); // fibre direction
    moduli.insert("22".to_string(), 10.0); // transverse
    moduli.insert("33".to_string(), 10.0); // through-thickness

    let cfrp = Material::new(
        "cfrp-ud",
        "Unidirectional CFRP",
        MaterialCategory::Composite,
    )
    .with_source(DataSource::file("coupon-tests-2024.csv"))
    .with_property(
        "youngs-modulus",
        Property::Anisotropic {
            unit: "GPa".into(),
            values: moduli,
        },
    );

    println!();
    println!("== {} ({}) ==", cfrp.name, cfrp.category);
    for dir in cfrp
        .property("youngs-modulus")
        .expect("modulus present")
        .directions()
    {
        let ed = cfrp
            .anisotropic_value("youngs-modulus", &dir)
            .expect("direction present");
        println!("E_{dir}              = {ed:.3} GPa");
    }
    // `value` on an anisotropic property returns the isotropic-equivalent mean.
    let e_mean = cfrp.value("youngs-modulus", 20.0).expect("mean available");
    println!("E (mean estimate) = {e_mean:.3} GPa");
    println!("anisotropic       = {}", cfrp.has_anisotropy());

    // --- 3. Collect into a library and check the data policy ---
    let mut lib = MaterialLibrary::new();
    lib.add(steel);
    lib.add(cfrp);

    println!();
    println!("library size      = {}", lib.len());
    println!(
        "metals            = {}",
        lib.filter_by_category(MaterialCategory::Metal).len()
    );
    println!(
        "composites        = {}",
        lib.filter_by_category(MaterialCategory::Composite).len()
    );
    println!(
        "lookup by name    = {:?}",
        lib.get_by_name("Unidirectional CFRP").map(|m| &m.id)
    );
    println!("policy compliant  = {}", lib.is_compliant());
}
