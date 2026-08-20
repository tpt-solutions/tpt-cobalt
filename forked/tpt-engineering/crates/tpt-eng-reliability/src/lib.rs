//! Reliability and life analysis.
//!
//! Covers fatigue (S–N and Miner's rule), life distributions (Weibull and
//! exponential), failure-rate helpers, FMEA structures, and probabilistic
//! design (reliability index / normal-integration helpers).

pub mod fatigue;
pub mod fmea;
pub mod life;
pub mod probabilistic;

pub use fatigue::*;
pub use fmea::*;
pub use life::*;
pub use probabilistic::*;

use thiserror::Error;

/// Errors produced by reliability routines.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ReliabilityError {
    /// An input parameter was outside its valid domain.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}
