//! Proof AST: representation of terms, formulas, and proof trees.
//!
//! A small, serializable-agnostic AST for first-order proofs. It models
//! *terms* (the objects proofs talk about), *formulas* (the statements),
//! and *proofs* (trees built from inference rules). It intentionally includes
//! only the structure — soundness checking is left to downstream tools.

pub mod formula;
pub mod proof;
pub mod term;

pub use formula::Formula;
pub use proof::Proof;
pub use term::Term;
