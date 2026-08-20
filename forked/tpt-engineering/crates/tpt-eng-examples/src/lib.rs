//! # tpt-eng-examples
//!
//! End-to-end scenarios that compose the `tpt-eng-*` primitives the way a
//! real physical-systems vertical would:
//!
//! 1. **Thermal loop** — build an infrastructure topology, derive its network
//!    matrix, regulate a plant with a PID controller sized from a fuel's heating
//!    value, condition the gappy telemetry that feeds the controller, and run a
//!    structural check on the supporting structure ([`thermal_loop`]).
//! 2. **Mechanical design** — define a cross-section and material, roll up a
//!    dimensional tolerance stack-up, check measured points against a GD&T zone,
//!    and render the result as a calculation report ([`mechanical_design`]).
//! 3. **Solar PV sizing** — sweep a single-diode PV cell to its maximum-power
//!    point, scale to an array, and size the three-phase grid-tie connection
//!    ([`solar_pv`]).
//!
//! Each scenario lives in its own module as a self-contained function plus a
//! unit test, so the crate doubles as executable documentation for "how do I
//! use these crates together?".
//!
//! ## Example
//!
//! ```
//! use tpt_eng_examples::solar_pv::run_solar_pv_sizing;
//!
//! let report = run_solar_pv_sizing();
//! // The array DC power must be recovered by the three-phase tie.
//! let p_ac = 3.0f64.sqrt() * report.grid_v_ll * report.line_current_a * report.power_factor;
//! assert!((p_ac - report.array_dc_w).abs() < 1e-6);
//! ```

pub mod mechanical_design;
pub mod solar_pv;
pub mod thermal_loop;

pub use mechanical_design::{MechanicalDesignReport, design_report, run_mechanical_design};
pub use solar_pv::{SolarPvReport, run_solar_pv_sizing};
pub use thermal_loop::ThermalLoopReport;
