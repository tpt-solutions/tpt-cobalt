//! # tpt-eng-standards
//!
//! Standards modeling as **structured, parameterized data** for the TPT
//! engineering ecosystem.
//!
//! > **Legal rule.** This crate contains **no** copyrighted standard text, no
//! > proprietary code clauses, and no scraped proprietary tables. It models the
//! > *shape* of standards-based calculations — load cases, load combinations,
//! > load/safety factors, and limit-state logic — as generic, user-filled data
//! > structures. The actual factors and combinations are supplied by the user
//! > (who is responsible for the license of the values they enter).
//!
//! ## Modules
//!
//! * [`load`] — [`load::LoadCase`] and [`load::LoadType`].
//! * [`combinations`] — [`combinations::LoadCombination`] and the arithmetic to
//!   evaluate one against a demand map. Combinations and their factors are
//!   user-provided data.
//! * [`factors`] — [`factors::FactorSet`], a user-supplied bag of named partial
//!   factors.
//! * [`limit_states`] — [`limit_states::LimitState`] and the parameterized
//!   [`limit_states::DemandCapacity`] utilization check.
//! * [`design`] — the [`design::DesignBasis`] aggregate and
//!   [`design::evaluate_check`]/[`design::CheckResult`] workflow that ties a
//!   combination to a limit state and a capacity.
//!
//! ## User-provided data
//!
//! Every factor, combination, and factor set is plain data the caller provides.
//! See the module examples for building a [`design::DesignBasis`] from your own
//! (correctly licensed) values.
//!
//! ## Example
//!
//! ```rust
//! use std::collections::HashMap;
//! use tpt_eng_standards::{
//!     DesignBasis, LimitState, LoadCase, LoadCombination, LoadType,
//! };
//!
//! let basis = DesignBasis::new()
//!     .with_case(LoadCase::new("G", "dead", LoadType::Dead))
//!     .with_case(LoadCase::new("Q", "live", LoadType::Live))
//!     .with_combination(
//!         LoadCombination::new("ULS", "ULS")
//!             .with_factor("G", 1.35)
//!             .with_factor("Q", 1.5),
//!     );
//! let mut demands = HashMap::new();
//! demands.insert("G".to_string(), 10.0);
//! demands.insert("Q".to_string(), 4.0);
//! let results = basis.run_checks(&demands, 30.0, 1.0, LimitState::Ultimate);
//! // 1.35*10 + 1.5*4 = 19.5
//! assert!((results[0].combined_demand - 19.5).abs() < 1e-9);
//! ```
//!
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod combinations;
pub mod design;
pub mod factors;
pub mod limit_states;
pub mod load;
pub mod metadata;

pub use combinations::{CombinationFactor, LoadCombination};
pub use design::{CheckResult, DesignBasis, evaluate_check};
pub use factors::{FactorSet, LoadFactor};
pub use limit_states::{DemandCapacity, LimitState};
pub use load::{LoadCase, LoadType};

/// The most commonly used items, in one `use`.
pub mod prelude {
    pub use crate::{
        combinations::{CombinationFactor, LoadCombination},
        design::{CheckResult, DesignBasis, evaluate_check},
        factors::{FactorSet, LoadFactor},
        limit_states::{DemandCapacity, LimitState},
        load::{LoadCase, LoadType},
    };
}
