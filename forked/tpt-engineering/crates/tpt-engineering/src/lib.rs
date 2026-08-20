//! # tpt-engineering
//!
//! Umbrella meta-crate re-exporting every `tpt-eng-*` crate behind its own
//! feature flag, so a consumer can depend on the whole toolkit with one crate
//! and enable only the domains they need.
//!
//! Enable domains via features (named after the crate without the `tpt-eng-`
//! prefix; hyphens become underscores in the module name):
//!
//! ```toml
//! [dependencies]
//! tpt-engineering = { version = "0.1", features = ["structural", "props"] }
//! ```
//!
//! ```rust,ignore
//! use tpt_engineering::structural;
//! use tpt_engineering::props;
//! ```
#[cfg(feature = "biomech")]
pub use tpt_eng_biomech as biomech;
#[cfg(feature = "building-sys")]
pub use tpt_eng_building_sys as building_sys;
#[cfg(feature = "cad")]
pub use tpt_eng_cad as cad;
#[cfg(feature = "controls")]
pub use tpt_eng_controls as controls;
#[cfg(feature = "crystallography")]
pub use tpt_eng_crystallography as crystallography;
#[cfg(feature = "electrical")]
pub use tpt_eng_electrical as electrical;
#[cfg(feature = "examples")]
pub use tpt_eng_examples as examples;
#[cfg(feature = "gdt")]
pub use tpt_eng_gdt as gdt;
#[cfg(feature = "geo-asset")]
pub use tpt_eng_geo_asset as geo_asset;
#[cfg(feature = "geo-topology")]
pub use tpt_eng_geo_topology as geo_topology;
#[cfg(feature = "geometry")]
pub use tpt_eng_geometry as geometry;
#[cfg(feature = "geotech")]
pub use tpt_eng_geotech as geotech;
#[cfg(feature = "heat-transfer")]
pub use tpt_eng_heat_transfer as heat_transfer;
#[cfg(feature = "io")]
pub use tpt_eng_io as io;
#[cfg(feature = "materials")]
pub use tpt_eng_materials as materials;
#[cfg(feature = "mesh")]
pub use tpt_eng_mesh as mesh;
#[cfg(feature = "network-matrix")]
pub use tpt_eng_network_matrix as network_matrix;
#[cfg(feature = "nurbs")]
pub use tpt_eng_nurbs as nurbs;
#[cfg(feature = "pcb")]
pub use tpt_eng_pcb as pcb;
#[cfg(feature = "plot")]
pub use tpt_eng_plot as plot;
#[cfg(feature = "power-components")]
pub use tpt_eng_power_components as power_components;
#[cfg(feature = "props")]
pub use tpt_eng_props as props;
#[cfg(feature = "props-air")]
pub use tpt_eng_props_air as props_air;
#[cfg(feature = "props-fuels")]
pub use tpt_eng_props_fuels as props_fuels;
#[cfg(feature = "props-mixture")]
pub use tpt_eng_props_mixture as props_mixture;
#[cfg(feature = "props-water")]
pub use tpt_eng_props_water as props_water;
#[cfg(feature = "reliability")]
pub use tpt_eng_reliability as reliability;
#[cfg(feature = "renewables")]
pub use tpt_eng_renewables as renewables;
#[cfg(feature = "report")]
pub use tpt_eng_report as report;
#[cfg(feature = "safety")]
pub use tpt_eng_safety as safety;
#[cfg(feature = "schedule")]
pub use tpt_eng_schedule as schedule;
#[cfg(feature = "sections")]
pub use tpt_eng_sections as sections;
#[cfg(feature = "standards")]
pub use tpt_eng_standards as standards;
#[cfg(feature = "structural")]
pub use tpt_eng_structural as structural;
#[cfg(feature = "thermal-mgmt")]
pub use tpt_eng_thermal_mgmt as thermal_mgmt;
#[cfg(feature = "timeseries")]
pub use tpt_eng_timeseries as timeseries;
#[cfg(feature = "timeseries-align")]
pub use tpt_eng_timeseries_align as timeseries_align;
#[cfg(feature = "timeseries-core")]
pub use tpt_eng_timeseries_core as timeseries_core;
#[cfg(feature = "timeseries-gap")]
pub use tpt_eng_timeseries_gap as timeseries_gap;
#[cfg(feature = "tolerance")]
pub use tpt_eng_tolerance as tolerance;
#[cfg(feature = "unit-ops")]
pub use tpt_eng_unit_ops as unit_ops;
#[cfg(feature = "vehicle-dynamics")]
pub use tpt_eng_vehicle_dynamics as vehicle_dynamics;
