#![no_std]
//! Type-level safety encoding toolkit.
//!
//! Consolidates the techniques that let Rust encode invariants at the type
//! level rather than at runtime:
//!
//! - [`phantom`] — variance-controlled phantom markers and the sealed-trait
//!   pattern.
//! - [`ghost`] — ghost/typestate tokens that carry a logical state in the
//!   type system.
//! - [`newtype`] — newtype wrappers with `From`/`Into` glue over an inner type.
//! - [`bounded`] — range-checked numeric values, both runtime-checked and
//!   type-level named bounds.
//! - [`safe_cast`] — infallible widening casts and fallible narrowing casts
//!   that fail instead of truncating.

#[cfg(test)]
extern crate std;

pub mod bounded;
pub mod ghost;
pub mod newtype;
pub mod phantom;
pub mod safe_cast;
