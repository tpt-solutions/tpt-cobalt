//! Built-in material property library for the CLI `materials inspect` command.
//!
//! Backed by [`tpt_eng_materials::MaterialLibrary`]; the embedded library is
//! seeded with representative reference values, each carrying a
//! [`tpt_eng_materials::DataSource`] so the library passes
//! [`tpt_eng_materials::MaterialLibrary::validate`].

use std::sync::OnceLock;

use tpt_eng_materials::{DataSource, Material, MaterialCategory, MaterialLibrary, Property};

fn build_library() -> MaterialLibrary {
    let mut lib = MaterialLibrary::new();
    let entries = [
        (
            "steel",
            "Steel",
            MaterialCategory::Metal,
            7850.0,
            200e9,
            250e6,
            0.30,
        ),
        (
            "aluminium",
            "Aluminium",
            MaterialCategory::Metal,
            2700.0,
            69e9,
            95e6,
            0.33,
        ),
        (
            "concrete",
            "Concrete",
            MaterialCategory::Other,
            2400.0,
            30e9,
            40e6,
            0.20,
        ),
        (
            "timber",
            "Timber",
            MaterialCategory::Other,
            500.0,
            11e9,
            30e6,
            0.40,
        ),
        (
            "titanium",
            "Titanium",
            MaterialCategory::Metal,
            4500.0,
            116e9,
            880e6,
            0.32,
        ),
        (
            "copper",
            "Copper",
            MaterialCategory::Metal,
            8960.0,
            117e9,
            70e6,
            0.34,
        ),
    ];
    for (id, name, cat, density, e, yield_s, poisson) in &entries {
        let m = Material::new(*id, *name, *cat)
            .with_source(DataSource::standard("representative reference values"))
            .with_property(
                "density",
                Property::Scalar {
                    value: *density,
                    unit: "kg/m^3".into(),
                },
            )
            .with_property(
                "youngs-modulus",
                Property::Scalar {
                    value: *e,
                    unit: "Pa".into(),
                },
            )
            .with_property(
                "yield-strength",
                Property::Scalar {
                    value: *yield_s,
                    unit: "Pa".into(),
                },
            )
            .with_property(
                "poisson-ratio",
                Property::Scalar {
                    value: *poisson,
                    unit: String::new(),
                },
            );
        lib.add(m);
    }
    // The embedded library must satisfy the data-policy rules.
    debug_assert!(
        lib.validate().is_ok(),
        "embedded material library failed validation"
    );
    lib
}

fn library() -> &'static MaterialLibrary {
    static LIB: OnceLock<MaterialLibrary> = OnceLock::new();
    LIB.get_or_init(build_library)
}

/// Look up a material by id or name (case-insensitive). Returns `None` if not found.
pub fn find(name: &str) -> Option<&'static Material> {
    let needle = name.trim();
    library()
        .materials
        .iter()
        .find(|m| m.id.eq_ignore_ascii_case(needle) || m.name.eq_ignore_ascii_case(needle))
}

/// All built-in materials (for listing).
pub fn list() -> &'static [Material] {
    &library().materials
}

/// Format a material as a multi-line property listing.
pub fn describe(material: &Material) -> String {
    let density = material.value("density", 0.0).unwrap_or(0.0);
    let e = material.value("youngs-modulus", 0.0).unwrap_or(0.0);
    let yield_s = material.value("yield-strength", 0.0).unwrap_or(0.0);
    let poisson = material.value("poisson-ratio", 0.0).unwrap_or(0.0);
    format!(
        "Material: {}\n  density:          {:.0} kg/m^3\n  Young's modulus:  {:.2e} Pa\n  yield strength:   {:.2e} Pa\n  Poisson's ratio:  {:.2}",
        material.name, density, e, yield_s, poisson
    )
}
