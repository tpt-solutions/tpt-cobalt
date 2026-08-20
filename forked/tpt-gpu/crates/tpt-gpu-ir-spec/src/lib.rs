//! TPTIR Dialect Specification
//!
//! This crate is the canonical, versioned definition of the TPTIR intermediate
//! representation shared across the TPT compute suite (tpt-gpu, tpt-crucible,
//! tpt-spark). Any tool that reads or writes TPTIR text should depend on this
//! crate rather than re-defining the dialect inline.
//!
//! # Stability
//! Types in this crate are stable from v0.1.0 onward. Additive changes increment
//! the minor version; breaking changes increment the major version.

pub mod attr;
pub mod dialect;
pub mod ops;
pub mod text;
pub mod types;

pub use attr::{Attr, AttrValue};
pub use dialect::{Dialect, DialectRegistry, TPTIR_DIALECT_VERSION};
pub use ops::{Op, OpCategory, OpDef};
pub use text::{emit, parse, parse_op, parse_region, parse_type, EmitOptions, ParseError};
pub use types::{AddressSpace, ElemType, Type, TypeKind};
