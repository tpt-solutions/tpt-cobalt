//! # tpt-lang — The TPT Script Language Runtime (Phase 6, spec §6)
//!
//! Scaffold of the execution engine: the dynamic [`Value`] model with
//! **first-class tensors** (`Value::Tensor` is a native variant with zero
//! indirection), numeric promotion rules, truthiness, and scoped
//! environments.
//!
//! Deliberate divergences from the spec sketch (documented in todo.md): List
//! and Dict use `Arc<Mutex<…>>` instead of a tracing `Gc` — shared mutable
//! semantics without adding a GC dependency yet; cycles will leak until a GC
//! lands.

mod env;
mod ops;
mod value;

pub use env::Environment;
pub use ops::{value_add, value_div, value_eq, value_mul, value_sub, LangError};
pub use value::{Truthiness, Value};