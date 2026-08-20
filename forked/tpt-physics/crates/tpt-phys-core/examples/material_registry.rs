//! Hello-world for `tpt-phys-core`: the typed material database.
//!
//! Loads the built-in engineering materials, derives the elastic moduli that
//! every downstream solver needs (Lamé `λ`, shear `G`, bulk `K`), adds a custom
//! alloy, and round-trips the whole registry through JSON — the persistence
//! format `tpt-physics` uses for material libraries.
//!
//! Run with:
//!
//! ```text
//! cargo run --example material_registry -p tpt-phys-core
//! ```

use tpt_phys_core::{Material, MaterialRegistry};

fn main() {
    // 1. The built-in library of representative engineering materials.
    let mut reg = MaterialRegistry::with_defaults();

    // 2. Register a custom material. `insert` matches on `name`, so calling it
    //    twice with the same name updates in place rather than duplicating.
    reg.insert(Material::new(
        "Phosphor Bronze (C51000)",
        110e9,  // E  [Pa]
        0.34,   // ν  [-]
        8800.0, // ρ  [kg/m³]
        18e-6,  // α  [1/K]
    ));

    println!("Material registry ({} entries)", reg.materials.len());
    println!(
        "{:<28} {:>9} {:>9} {:>9} {:>9} {:>10}",
        "name", "E [GPa]", "G [GPa]", "K [GPa]", "λ [GPa]", "ρ [kg/m³]"
    );
    println!("{}", "-".repeat(78));
    for m in &reg.materials {
        println!(
            "{:<28} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>10.0}",
            m.name,
            m.youngs_modulus / 1e9,
            m.shear_modulus() / 1e9,
            m.bulk_modulus() / 1e9,
            m.lame_lambda() / 1e9,
            m.density,
        );
    }

    // 3. Look a material up by name and use the compile-time-typed accessors.
    //    These return `tpt-math-units` quantities, so a `Pressure` can never be
    //    silently confused with a `MassDensity` at a call site.
    let steel = reg.get("Structural Steel").expect("built-in material");
    println!();
    println!("Typed (unit-safe) accessors for {:?}:", steel.name);
    println!("  E = {:?}", steel.youngs_modulus_q());
    println!("  ρ = {:?}", steel.density_q());
    println!("  α = {:?}", steel.thermal_expansion_q());

    // 4. A consistency identity every isotropic material must satisfy:
    //    E = 3K(1 - 2ν) = 2G(1 + ν).
    let from_bulk = 3.0 * steel.bulk_modulus() * (1.0 - 2.0 * steel.poissons_ratio);
    let from_shear = 2.0 * steel.shear_modulus() * (1.0 + steel.poissons_ratio);
    println!();
    println!("Isotropic elasticity identities for {:?}:", steel.name);
    println!("  E              = {:.4e} Pa", steel.youngs_modulus);
    println!("  3K(1 - 2ν)     = {from_bulk:.4e} Pa");
    println!("  2G(1 + ν)      = {from_shear:.4e} Pa");
    assert!((from_bulk - steel.youngs_modulus).abs() < 1.0);
    assert!((from_shear - steel.youngs_modulus).abs() < 1.0);

    // 5. Serialize the library to JSON and read it back — this is how a project
    //    ships its own vetted material data alongside a model.
    let json = reg.to_json().expect("serialize registry");
    let restored = MaterialRegistry::from_json(&json).expect("deserialize registry");
    assert_eq!(reg, restored);
    println!();
    println!(
        "JSON round-trip: {} bytes, {} materials, exact match ✓",
        json.len(),
        restored.materials.len()
    );
    println!("First 3 lines of the serialized form:");
    for line in json.lines().take(3) {
        println!("  {line}");
    }
}
