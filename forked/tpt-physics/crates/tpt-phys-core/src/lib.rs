//! Foundational physics data structures for `tpt-physics`.
//!
//! `tpt-phys-core` contains only the code that is genuinely net-new to the
//! physics workspace: a typed material database and the CAD→[`tpt_fem_mesh`]
//! ingestion adapter. Everything else (SI units, mesh containers, reference
//! elements, sparse assembly, linear/nonlinear elasticity, thermal, eigen,
//! …) is consumed **directly** from the sibling `tpt-math` / `tpt-fem`
//! crates — no wrapper layer.
//!
//! ```
//! use tpt_phys_core::MaterialRegistry;
//! let reg = MaterialRegistry::with_defaults();
//! let steel = reg.get("Structural Steel").unwrap();
//! assert!(steel.youngs_modulus > 100e9);
//! ```

pub mod cad;
pub mod material;

// Re-export the primary public types at the crate root for ergonomic use.
pub use cad::{CadError, CadFace, CadIngestor, CadSolid, CadVertex};
pub use material::{Material, MaterialRegistry};

// Monte-Carlo uncertainty quantification over material property scatter.
// Gated behind the `uq` feature so consumers that don't need it pay no
// `proptest` dependency cost.
#[cfg(feature = "uq")]
pub mod uq;
#[cfg(feature = "uq")]
pub use uq::{
    cantilever_natural_frequency, cantilever_tip_deflection, monte_carlo, monte_carlo_seeded,
    tol_band, Statistics,
};

// Re-export the sibling unit crate so consumers get compile-time-typed
// quantities from a single `tpt-phys-core` import if they want them.
pub use tpt_math_units as units;
