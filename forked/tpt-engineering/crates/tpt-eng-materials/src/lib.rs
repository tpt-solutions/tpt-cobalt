//! # tpt-eng-materials
//!
//! Material property modeling for the TPT engineering ecosystem.
//!
//! This crate models materials as a named, categorized collection of
//! [`material::Material`] values, each carrying typed [`property::Property`]
//! data and provenance. Three property forms are supported:
//!
//! * **scalar** — a single value with a unit string,
//! * **temperature-dependent** — `(temperature, value)` samples evaluated by
//!   linear interpolation ([`property::Property::value_at`]), and
//! * **anisotropic** — per-direction scalars (a hook for directional or
//!   tensor-valued properties).
//!
//! ## Data policy
//!
//! **Only original, user-provided, or openly licensed data is accepted.** Every
//! material must record a source ([`provenance::DataSource`]), and
//! [`library::MaterialLibrary::validate`] enforces that rule: a material with no
//! recorded source, or a disallowed `license` attribute, fails validation. No
//! proprietary material tables are bundled or scraped.
//!
//! ## Persistence
//!
//! Libraries load/save as JSON ([`library::MaterialLibrary::from_json`] /
//! [`library::MaterialLibrary::to_json`]) and as CSV
//! ([`library::MaterialLibrary::from_csv`] / [`library::MaterialLibrary::to_csv`]).
//!
//! An optional embedded-database backend was evaluated and deliberately **not**
//! adopted: the in-memory + JSON/CSV model keeps the dependency tree small and
//! fully license-clean (all of `serde`, `serde_json`, and `csv` are
//! MIT/Apache-2.0 compatible).
//!
//! ## Example
//!
//! ```rust
//! use tpt_eng_materials::{Material, MaterialCategory, MaterialLibrary, Property};
//!
//! let mut lib = MaterialLibrary::new();
//! lib.add(
//!     Material::new("steel-s355", "S355", MaterialCategory::Metal)
//!         .with_property(
//!             "youngs-modulus",
//!             Property::Scalar {
//!                 value: 210.0,
//!                 unit: "GPa".into(),
//!             },
//!         ),
//! );
//! let e = lib.get_by_id("steel-s355").unwrap().value("youngs-modulus", 0.0).unwrap();
//! assert!((e - 210.0).abs() < 1e-12);
//! ```
//!
//! #![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod category;
pub mod error;
pub mod library;
pub mod material;
pub mod property;
pub mod provenance;

pub use category::MaterialCategory;
pub use error::{MaterialError, Result};
pub use library::{ALLOWED_LICENSES, MaterialLibrary};
pub use material::Material;
pub use property::{Property, TempPoint};
pub use provenance::{DataSource, Metadata};

/// The most commonly used items, in one `use`.
pub mod prelude {
    pub use crate::{
        category::MaterialCategory,
        error::{MaterialError, Result},
        library::{ALLOWED_LICENSES, MaterialLibrary},
        material::Material,
        property::{Property, TempPoint},
        provenance::{DataSource, Metadata},
    };
}
