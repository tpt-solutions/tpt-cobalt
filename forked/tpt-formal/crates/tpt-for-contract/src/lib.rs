#![no_std]
//! Design-by-contract primitives.
//!
//! Consolidates the four tightly-coupled facets of one contract concern into a
//! single coherent surface:
//!
//! - **Preconditions** ([`requires!`]) — checked at function entry.
//! - **Postconditions** ([`ensures!`]) — checked at function exit.
//! - **Invariants** ([`Invariant`] trait + [`invariant!`]) — hold for the
//!   whole lifetime of a value.
//! - **Loop invariants** ([`loop_invariant!`]) — hold on entry and after every
//!   iteration.
//!
//! All checks are `no_std` and panic with the offending condition's source
//! text, file, and line. A debug-only variant ([`debug_requires!`] and friends)
//! is provided for checks that should be skipped in release builds.
//!
//! # Numeric predicates
//!
//! The crate is generic over any value and predicate, so it needs no external
//! numeric types. Where a contract reasons about numbers, supply the predicate
//! directly (e.g. `requires!(x >= 0)`). Richer numeric types from
//! `tpt-math-numeric` compose transparently because contracts only require a
//! `bool` condition — wire that dependency in once `tpt-math` is published.

pub mod error;
pub mod invariant;

pub use error::{report, ContractError};
pub use invariant::Invariant;

/// Assert a precondition: `cond` must hold when the caller enters a function.
///
/// # Panics
///
/// Panics if `cond` is false, labelling the failure as a `precondition`.
///
/// ```
/// let s = [1, 2, 3];
/// tpt_for_contract::requires!(!s.is_empty());
/// ```
#[macro_export]
macro_rules! requires {
    ($cond:expr $(,)?) => {
        if !($cond) {
            $crate::report($crate::ContractError {
                kind: "precondition",
                expr: ::core::stringify!($cond),
                file: ::core::file!(),
                line: ::core::line!(),
            });
        }
    };
    ($cond:expr, $msg:expr $(,)?) => {
        if !($cond) {
            $crate::report($crate::ContractError {
                kind: "precondition",
                expr: ::core::concat!(::core::stringify!($cond), " — ", $msg),
                file: ::core::file!(),
                line: ::core::line!(),
            });
        }
    };
}

/// Assert a postcondition: `cond` must hold when a function returns.
///
/// # Panics
///
/// Panics if `cond` is false, labelling the failure as a `postcondition`.
#[macro_export]
macro_rules! ensures {
    ($cond:expr $(,)?) => {
        if !($cond) {
            $crate::report($crate::ContractError {
                kind: "postcondition",
                expr: ::core::stringify!($cond),
                file: ::core::file!(),
                line: ::core::line!(),
            });
        }
    };
    ($cond:expr, $msg:expr $(,)?) => {
        if !($cond) {
            $crate::report($crate::ContractError {
                kind: "postcondition",
                expr: ::core::concat!(::core::stringify!($cond), " — ", $msg),
                file: ::core::file!(),
                line: ::core::line!(),
            });
        }
    };
}

/// Assert a value's invariant holds.
///
/// # Panics
///
/// Panics if `cond` is false, labelling the failure as an `invariant`.
#[macro_export]
macro_rules! invariant {
    ($cond:expr $(,)?) => {
        if !($cond) {
            $crate::report($crate::ContractError {
                kind: "invariant",
                expr: ::core::stringify!($cond),
                file: ::core::file!(),
                line: ::core::line!(),
            });
        }
    };
}

/// Assert a loop invariant. Place at the top of a loop body; it verifies the
/// condition holds on every iteration (and thus, with the entry check, across
/// the whole loop).
///
/// # Panics
///
/// Panics if `cond` is false, labelling the failure as a `loop-invariant`.
#[macro_export]
macro_rules! loop_invariant {
    ($cond:expr $(,)?) => {
        if !($cond) {
            $crate::report($crate::ContractError {
                kind: "loop-invariant",
                expr: ::core::stringify!($cond),
                file: ::core::file!(),
                line: ::core::line!(),
            });
        }
    };
}

/// Debug-only variants of the contract macros. These compile to nothing in
/// release builds (`cfg(not(debug_assertions))`), so they can guard expensive
/// checks without affecting production performance.
#[macro_export]
macro_rules! debug_requires {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        $crate::requires!($($arg)*);
    };
}

#[macro_export]
macro_rules! debug_ensures {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        $crate::ensures!($($arg)*);
    };
}

#[macro_export]
macro_rules! debug_invariant {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        $crate::invariant!($($arg)*);
    };
}

#[macro_export]
macro_rules! debug_loop_invariant {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        $crate::loop_invariant!($($arg)*);
    };
}
