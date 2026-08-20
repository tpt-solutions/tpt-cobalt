//! # Mechanical-design integration scenario
//!
//! A self-contained example of how the `tpt-eng-*` crates fit together for a
//! mechanical part-design workflow:
//!
//! 1. **Sections** — define a rectangular cross-section and read its geometric
//!    properties (area, second moments of area).
//! 2. **Materials** — build a steel material with provenance-tracked properties,
//!    register it in a `MaterialLibrary`, and look it back up.
//! 3. **Tolerance** — roll up a 1-D stack of manufacturing dimensions with
//!    worst-case and RSS methods.
//! 4. **GD&T** — define a datum reference frame and a cylindrical position
//!    tolerance zone, then check measured points for conformance.
//! 5. **Report** — assemble a calculation report from the above and render it to
//!    Markdown/HTML/JSON.

use tpt_eng_gdt::{Datum, DatumReferenceFrame, ToleranceZone};
use tpt_eng_geometry::frame::Frame3;
use tpt_eng_geometry::{Point3, Vector3};
use tpt_eng_materials::{DataSource, Material, MaterialCategory, MaterialLibrary, Property};
use tpt_eng_report::{NamedValue, Report, ResultEntry};
use tpt_eng_sections::{Rectangle, Section};
use tpt_eng_tolerance::{DimTol, rss, worst_case};

/// Output of [`run_mechanical_design`].
#[derive(Debug, Clone, PartialEq)]
pub struct MechanicalDesignReport {
    /// Cross-section area (m²).
    pub area: f64,
    /// Second moment of area about the x-axis (m⁴).
    pub i_x: f64,
    /// Second moment of area about the y-axis (m⁴).
    pub i_y: f64,
    /// Steel Young's modulus (Pa).
    pub youngs_modulus: f64,
    /// Steel yield strength (Pa).
    pub yield_strength: f64,
    /// Nominal stack-up sum (m).
    pub stackup_nominal: f64,
    /// Worst-case stack-up lower bound (m).
    pub stackup_worst_lo: f64,
    /// Worst-case stack-up upper bound (m).
    pub stackup_worst_hi: f64,
    /// RSS (3σ) stack-up lower bound (m).
    pub stackup_rss_lo: f64,
    /// RSS (3σ) stack-up upper bound (m).
    pub stackup_rss_hi: f64,
    /// Whether the measured points passed the GD&T zone.
    pub gdt_passed: bool,
    /// Maximum GD&T deviation (mm).
    pub gdt_max_deviation: f32,
}

/// Run the full mechanical-design scenario and return a numeric summary.
///
/// # Panics
///
/// Panics if the steel material is missing from the local `MaterialLibrary`
/// (it is registered unconditionally at the start of this function, so this
/// only fails if that registration is changed).
pub fn run_mechanical_design() -> MechanicalDesignReport {
    // 1. Section geometry: a 100 mm x 200 mm rectangle.
    let rect = Rectangle::new(0.1, 0.2);
    let area = rect.area();
    let (i_x, i_y, _) = rect.second_moments();

    // 2. Material: structural steel with provenance-tracked properties.
    let steel = Material::new("steel", "Structural Steel", MaterialCategory::Metal)
        .with_source(DataSource::standard("EN 10025"))
        .with_property(
            "youngs-modulus",
            Property::Scalar {
                value: 210e9,
                unit: "Pa".into(),
            },
        )
        .with_property(
            "yield-strength",
            Property::Scalar {
                value: 355e6,
                unit: "Pa".into(),
            },
        );
    let mut lib = MaterialLibrary::new();
    lib.add(steel);
    let looked_up = lib.get_by_id("steel").expect("steel registered");
    let youngs_modulus = looked_up.value("youngs-modulus", 0.0).expect("has E");
    let yield_strength = looked_up.value("yield-strength", 0.0).expect("has Sy");

    // 3. Tolerance stack-up: three coaxial manufacturing dimensions (metres).
    let dims = vec![
        DimTol::new("plate", 0.010, 0.001),
        DimTol::new("shim", 0.002, 0.0002),
        DimTol::new("bolt", 0.012, 0.0005),
    ];
    let nominal_sum: f64 = dims.iter().map(|d| d.nominal).sum();
    let (worst_lo, worst_hi) = worst_case(&dims);
    let (rss_lo, rss_hi) = rss(&dims);

    // 4. GD&T: a cylindrical position zone of Ø0.1 mm on datum A, checked
    //    against three measured points (all well inside the zone).
    let drf = DatumReferenceFrame::new(Datum::new('A', Frame3::IDENTITY));
    let zone = ToleranceZone::Cylindrical { diameter: 0.1 };
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.03, 0.0, 0.0),
        Point3::new(0.0, 0.04, 0.0),
    ];
    let conformance = drf.check_conformance(&points, &zone, Vector3::Z, Point3::ZERO);

    MechanicalDesignReport {
        area,
        i_x,
        i_y,
        youngs_modulus,
        yield_strength,
        stackup_nominal: nominal_sum,
        stackup_worst_lo: worst_lo,
        stackup_worst_hi: worst_hi,
        stackup_rss_lo: rss_lo,
        stackup_rss_hi: rss_hi,
        gdt_passed: conformance.passed,
        gdt_max_deviation: conformance.max_deviation,
    }
}

/// Build a calculation [`Report`] from a [`MechanicalDesignReport`].
pub fn design_report(r: &MechanicalDesignReport) -> Report {
    Report::new("Mechanical Part Design")
        .with_author("tpt-eng-examples")
        .with_summary(
            "Cross-section geometry, material properties, dimensional stack-up, and GD&T check.",
        )
        .assumptions(vec![
            NamedValue::new("Section", 0.0).with_description("rectangle 100 mm x 200 mm"),
            NamedValue::new("Area", r.area).with_unit("m^2"),
            NamedValue::new("I_x", r.i_x).with_unit("m^4"),
            NamedValue::new("I_y", r.i_y).with_unit("m^4"),
            NamedValue::new("Young's modulus", r.youngs_modulus).with_unit("Pa"),
            NamedValue::new("Yield strength", r.yield_strength).with_unit("Pa"),
        ])
        .results(vec![
            ResultEntry::with_limits(
                "Stack-up nominal",
                r.stackup_nominal,
                Some("m".into()),
                None,
                None,
            ),
            ResultEntry::with_limits(
                "Stack-up worst-case (lo)",
                r.stackup_worst_lo,
                Some("m".into()),
                None,
                None,
            ),
            ResultEntry::with_limits(
                "Stack-up worst-case (hi)",
                r.stackup_worst_hi,
                Some("m".into()),
                None,
                None,
            ),
            ResultEntry::with_limits(
                "Stack-up RSS (lo)",
                r.stackup_rss_lo,
                Some("m".into()),
                None,
                None,
            ),
            ResultEntry::with_limits(
                "Stack-up RSS (hi)",
                r.stackup_rss_hi,
                Some("m".into()),
                None,
                None,
            ),
            ResultEntry::with_limits(
                "GD&T max deviation",
                r.gdt_max_deviation as f64,
                Some("mm".into()),
                None,
                None,
            ),
        ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenario_assembles_and_reports() {
        let r = run_mechanical_design();

        // Rectangle 100 mm x 200 mm: A = 0.02 m², I_x = b h³/12 = 0.1*0.2³/12.
        assert!((r.area - 0.02).abs() < 1e-12);
        assert!((r.i_x - 0.1 * 0.2f64.powi(3) / 12.0).abs() < 1e-15);
        // Material lookup from the library must recover the authored values.
        assert!((r.youngs_modulus - 210e9).abs() < 1e-3);
        assert!((r.yield_strength - 355e6).abs() < 1e-3);

        // Stack-up nominal = 0.010 + 0.002 + 0.012 = 0.024 m.
        assert!((r.stackup_nominal - 0.024).abs() < 1e-15);
        // Worst-case width = 2 * (0.001 + 0.0002 + 0.0005) = 0.0034 m.
        assert!((r.stackup_worst_hi - r.stackup_worst_lo - 0.0034).abs() < 1e-15);
        // RSS interval must be narrower than the worst-case interval.
        assert!(r.stackup_rss_hi - r.stackup_rss_lo < r.stackup_worst_hi - r.stackup_worst_lo);

        // All measured points lie inside the Ø0.1 mm cylindrical zone.
        assert!(r.gdt_passed);
        assert!(r.gdt_max_deviation <= 0.0);

        // The report renders and captures the key results.
        let report = design_report(&r);
        let md = tpt_eng_report::to_markdown(&report);
        assert!(md.contains("Mechanical Part Design"));
        assert!(md.contains("Young's modulus"));
        assert!(md.contains("Stack-up nominal"));
    }
}
