//! # tpt-eng-sections
//!
//! Cross-section properties for the TPT engineering ecosystem.
//!
//! Every section implements the [`section::Section`] trait, which exposes area,
//! centroid, centroidal second moments, elastic/plastic section moduli, and the
//! torsional constant. Properties are assembled into a single
//! [`properties::SectionProperties`] bundle via [`section::Section::properties`].
//!
//! Supported section types ([`shapes`]):
//!
//! * [`shapes::Rectangle`]
//! * [`shapes::Circle`]
//! * [`shapes::Tube`] (circular hollow)
//! * [`shapes::ISection`]
//! * [`shapes::Channel`]
//! * [`shapes::Angle`]
//! * [`polygon::CustomPolygon`] (arbitrary simply-connected polygon)
//!
//! Composite sections (I-section, channel, angle) are evaluated by rectangle
//! decomposition ([`compose`]); arbitrary polygons use exact Green's-theorem
//! formulas for area/centroid/second moments, with plastic moduli and torsion
//! computed on a grid confined to the polygon.
//!
//! All quantities are reported in the section's own consistent length units;
//! the caller is responsible for unit consistency (integration with
//! `tpt-math-units` is deferred).
//!
//! ## Relationship to `tpt-eng-geometry`
//!
//! This crate models **2-D cross-section properties** (area, centroid, second
//! moments, section/plastic moduli) as pure geometry. `tpt-eng-geometry`
//! models **3-D solid geometry** (frames, surfaces, intersections). The split is
//! a deliberate, permanent domain separation — 2-D section math and 3-D solid
//! geometry have distinct concerns and are intentionally not merged — not a
//! pending integration awaiting a crate that does not exist.
//!
//! ## Example
//!
//! ```rust
//! use tpt_eng_sections::{ISection, Section};
//!
//! // W-shape: depth 10, flange width 6, flange thickness 1, web thickness 0.5.
//! let s = ISection::new(10.0, 6.0, 1.0, 0.5);
//! // Area = 2*6*1 + 0.5*(10 - 2) = 16.
//! assert!((s.area() - 16.0).abs() < 1e-9);
//! let props = s.properties();
//! assert!(props.area > 0.0);
//! ```
//!
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod compose;
pub mod polygon;
pub mod properties;
pub mod section;
pub mod shapes;

pub use polygon::CustomPolygon;
pub use properties::SectionProperties;
pub use section::Section;
pub use shapes::{Angle, Channel, Circle, ISection, Rectangle, Tube};

/// The most commonly used items, in one `use`.
pub mod prelude {
    pub use crate::{
        compose::{Rect, centroid, plastic_x, plastic_y, second_moments, torsion},
        polygon::CustomPolygon,
        properties::SectionProperties,
        section::Section,
        shapes::{Angle, Channel, Circle, ISection, Rectangle, Tube},
    };
}
