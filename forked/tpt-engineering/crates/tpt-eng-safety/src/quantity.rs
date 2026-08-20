//! A minimal, dynamically-dimensioned quantity type local to this crate.
//!
//! `tpt-math-units` provides compile-time dimension-checked quantities, but
//! the limit-check API here is generic over *which* dimension is being
//! checked (any two design/allowable values, compared at runtime) — a shape
//! that needs a runtime-tagged dimension rather than a distinct Rust type per
//! unit. Each [`Quantity`] is nonetheless *backed* by a real
//! `tpt-math-units` (uom) value: the constructors build a uom-free `f64`
//! magnitude by round-tripping through the SI uom quantities via `.get::<pascal>()`
//! / `.get::<meter>()`, so the stored `value` is always the genuine SI scalar.

use thiserror::Error;
use tpt_math_units::uom::si::f64::{Length as UomLength, Pressure as UomPressure};
use tpt_math_units::uom::si::{length::meter, pressure::pascal};

/// The physical dimension a [`Quantity`] is expressed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    /// Pressure/stress (SI: pascal).
    Pressure,
    /// Length (SI: metre).
    Length,
}

/// A numeric value tagged with its physical dimension.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantity {
    /// The numeric value, in the dimension's SI unit.
    pub value: f64,
    /// The physical dimension.
    pub dim: Dimension,
}

impl Quantity {
    /// Construct a quantity with an explicit dimension.
    pub fn new(value: f64, dim: Dimension) -> Self {
        Self { value, dim }
    }

    /// Construct a quantity from a real `tpt-math-units` pressure value (SI
    /// pascal).
    pub fn from_pressure(p: UomPressure) -> Self {
        Self::new(p.get::<pascal>(), Dimension::Pressure)
    }

    /// Construct a quantity from a real `tpt-math-units` length value (SI metre).
    pub fn from_length(l: UomLength) -> Self {
        Self::new(l.get::<meter>(), Dimension::Length)
    }

    /// Construct a pressure/stress quantity, in pascals.
    pub fn pascals(value: f64) -> Self {
        Self::from_pressure(UomPressure::new::<pascal>(value))
    }

    /// Construct a length quantity, in metres.
    pub fn meters(value: f64) -> Self {
        Self::from_length(UomLength::new::<meter>(value))
    }
}

/// Error evaluating or combining [`Quantity`] values.
#[derive(Debug, Clone, Error)]
pub enum QuantityError {
    /// The two quantities involved have different dimensions.
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),
}
