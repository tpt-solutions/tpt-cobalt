//! # tpt-eng-pcb
//!
//! PCB (printed-circuit-board) engineering primitives: dielectric layer
//! stackups, transmission-line / trace models (microstrip characteristic
//! impedance, IPC-2221 current capacity), vias and surface-mount pad
//! footprints.
//!
//! All scalar quantities are plain `f64` in SI units unless a parameter name
//! or doc comment says otherwise. Where a formula is defined in a non-SI unit
//! (notably the IPC-2221 current-capacity relation, which works in **thousandths
//! of an inch, "mils"**), the unit is called out explicitly and the crate
//! provides SI↔mil helpers.
//!
//! ## Example
//!
//! ```
//! use tpt_eng_pcb::microstrip_impedance;
//!
//! // A 50 Ω microstrip: ~1.8 mm trace on 1.6 mm FR-4 (er = 4.4).
//! let z0 = microstrip_impedance(1.8e-3, 1.6e-3, 4.4, 35e-6);
//! assert!((z0 - 50.0).abs() < 6.0);
//! ```

pub mod footprint;
pub mod stackup;
pub mod trace;
pub mod via;

pub use footprint::Pad;
pub use stackup::{Layer, Stackup};
pub use trace::{
    ipc_2221_current_capacity, microstrip_impedance, trace_area_mil2, trace_dc_resistance,
};
pub use via::Via;

/// One mil = 1/1000 inch = 25.4 µm, expressed in metres.
pub const MIL_TO_M: f64 = 2.54e-5;
