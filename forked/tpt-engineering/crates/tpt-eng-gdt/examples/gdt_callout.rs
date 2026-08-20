//! Richer scenario: the complete GD&T callout for a bearing housing bore, with
//! inspection sign-off and an ISO fit selection.
//!
//! The drawing carries four annotations (flatness of the mounting face,
//! perpendicularity of the bore, position of the bore, and total runout of the
//! seat). Simulated CMM data for the derived bore axis is then checked against
//! the position zone in the A|B|C datum reference frame, an ISO H7/g6 running
//! fit is computed, and the axial location stack-up is evaluated.
//!
//! Linear values are millimetres.
//!
//! Run with `cargo run -p tpt-eng-gdt --example gdt_callout`.

use tpt_eng_gdt::{
    Datum, DatumReference, DatumReferenceFrame, FitType, GdtAnnotation, GeometricCharacteristic,
    MaterialCondition, Stackup, StackupMember, ToleranceFrame, ToleranceZone, allowance, clearance,
    fit_type, iso_hole_limits, iso_shaft_limits, iso_tolerance, iso_tolerance_unit_um,
};
use tpt_eng_geometry::frame::Frame3;
use tpt_eng_geometry::{Point3, Quat, Vector3};

/// Nominal bore diameter, mm.
const BORE_DIA: f32 = 50.0;
/// Bore centre relative to datum A, mm.
const BORE_X: f32 = 60.0;
const BORE_Y: f32 = 40.0;

/// Build the drawing's feature control frames.
fn drawing_callouts() -> Vec<GdtAnnotation> {
    vec![
        GdtAnnotation {
            id: "FCF-1".to_string(),
            feature: "mounting face (datum A)".to_string(),
            frame: ToleranceFrame::new(
                GeometricCharacteristic::Flatness,
                ToleranceZone::ParallelPlanes { tolerance: 0.02 },
            ),
        },
        GdtAnnotation {
            id: "FCF-2".to_string(),
            feature: "bore dia 50 H7 axis".to_string(),
            frame: ToleranceFrame::new(
                GeometricCharacteristic::Perpendicularity,
                ToleranceZone::Cylindrical { diameter: 0.03 },
            )
            .with_datum(DatumReference::new('A', MaterialCondition::Rfs)),
        },
        GdtAnnotation {
            id: "FCF-3".to_string(),
            feature: "bore dia 50 H7 location".to_string(),
            frame: ToleranceFrame::new(
                GeometricCharacteristic::Position,
                ToleranceZone::Cylindrical { diameter: 0.05 },
            )
            .with_datum(DatumReference::new('A', MaterialCondition::Rfs))
            .with_datum(DatumReference::new('B', MaterialCondition::Mmc))
            .with_datum(DatumReference::new('C', MaterialCondition::Mmc)),
        },
        GdtAnnotation {
            id: "FCF-4".to_string(),
            feature: "seal seat face".to_string(),
            frame: ToleranceFrame::new(
                GeometricCharacteristic::TotalRunout,
                ToleranceZone::TotalRunoutBand { tolerance: 0.04 },
            )
            .with_datum(DatumReference::new('B', MaterialCondition::Rfs)),
        },
    ]
}

/// Simulated derived-median-line points of the as-machined bore: four axial
/// stations, each displaced from nominal by `(dx, dy)` millimetres.
fn measured_axis(dx: f32, dy: f32) -> Vec<Point3> {
    [4.0_f32, 12.0, 20.0, 28.0]
        .iter()
        .map(|&z| Point3::new(BORE_X + dx, BORE_Y + dy, z))
        .collect()
}

fn main() {
    // --- 1. The drawing ----------------------------------------------------
    println!("Bearing housing, dia {BORE_DIA:.1} H7 bore at ({BORE_X:.1}, {BORE_Y:.1}) from A\n");
    println!(
        "{:<7}{:<30}{:<18}{:<12}{:>9}  datums",
        "id", "feature", "characteristic", "category", "zone"
    );
    for a in &drawing_callouts() {
        let datums: Vec<String> = a
            .frame
            .datum_refs
            .iter()
            .map(|d| format!("{}({:?})", d.label, d.condition))
            .collect();
        println!(
            "{:<7}{:<30}{:<18}{:<12}{:>9.3}  {}",
            a.id,
            a.feature,
            format!("{:?}", a.frame.characteristic),
            format!("{:?}", a.frame.characteristic.category()),
            a.frame.zone.magnitude(),
            if datums.is_empty() {
                "-".to_string()
            } else {
                datums.join(" | ")
            }
        );
        // (columns above are padded as plain strings since derived `Debug`
        // ignores width specifiers)
    }

    // --- 2. The datum reference frame A|B|C --------------------------------
    // A: mounting face at the part origin.
    // B: bore axis, offset from A (secondary datums are relative to the primary).
    // C: a machined side face rotated 90 deg about Z, clocking the last degree
    //    of freedom.
    let drf = DatumReferenceFrame::new(Datum::new('A', Frame3::IDENTITY))
        .with_secondary(Datum::new(
            'B',
            Frame3::from_origin(Point3::new(BORE_X, BORE_Y, 0.0)),
        ))
        .with_tertiary(Datum::new(
            'C',
            Frame3::new(
                Point3::ZERO,
                Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
            ),
        ));
    let world = drf.world_frame();
    println!();
    println!(
        "DRF origin (world)   = ({:.3}, {:.3}, {:.3})",
        world.origin.x, world.origin.y, world.origin.z
    );
    println!(
        "DRF local X in world = ({:.3}, {:.3}, {:.3}) after datum C clocking",
        world.x_axis().x,
        world.x_axis().y,
        world.x_axis().z
    );

    // --- 3. Inspection of the bore position -------------------------------
    let position_zone = ToleranceZone::Cylindrical { diameter: 0.05 };
    println!();
    println!(
        "Position check against a dia {:.3} mm cylindrical zone about datum B:",
        position_zone.magnitude()
    );
    for (label, dx, dy) in [
        ("part 1 (0.018 mm off)", 0.018_f32, 0.000_f32),
        ("part 2 (0.021 mm off)", 0.015, 0.015),
        ("part 3 (0.040 mm off)", 0.040, 0.000),
    ] {
        let points = measured_axis(dx, dy);
        let report = drf.check_conformance(&points, &position_zone, Vector3::Z, Point3::ZERO);
        // Radial offset of the derived axis, from the raw measurement.
        let radial = (dx * dx + dy * dy).sqrt();
        println!(
            "  {label:<22} radial {radial:.4} mm, {} points, max deviation {:+.4} mm -> {}",
            report.sample_count,
            report.max_deviation,
            if report.passed { "ACCEPT" } else { "REJECT" }
        );
    }
    // Perpendicularity uses the tighter zone from FCF-2 on the same data.
    let perp_zone = ToleranceZone::Cylindrical { diameter: 0.03 };
    let perp = drf.check_conformance(
        &measured_axis(0.018, 0.0),
        &perp_zone,
        Vector3::Z,
        Point3::ZERO,
    );
    println!(
        "  part 1 re-checked against the dia {:.3} mm perpendicularity zone -> {} ({:+.4} mm)",
        perp_zone.magnitude(),
        if perp.passed { "ACCEPT" } else { "REJECT" },
        perp.max_deviation
    );

    // --- 4. ISO fit selection for the bearing seat ------------------------
    println!();
    println!("ISO system of limits and fits at dia {BORE_DIA:.1} mm:");
    println!(
        "  tolerance unit i   = {:.4} um",
        iso_tolerance_unit_um(BORE_DIA)
    );
    for grade in [6_u8, 7, 8] {
        let t = iso_tolerance(grade, BORE_DIA).expect("supported grade");
        println!("  IT{grade} tolerance    = {t:.4} mm");
    }

    let housing = iso_hole_limits(BORE_DIA, 7, "H").expect("H7 hole");
    let shaft = iso_shaft_limits(BORE_DIA, 6, "g").expect("g6 shaft");
    println!(
        "  H7 hole            = {:.4} .. {:.4} mm (tol {:.4})",
        housing.lower,
        housing.upper,
        housing.tolerance()
    );
    println!(
        "  g6 shaft           = {:.4} .. {:.4} mm (tol {:.4})",
        shaft.lower,
        shaft.upper,
        shaft.tolerance()
    );
    let fit = fit_type(&housing, &shaft);
    println!(
        "  min clearance      = {:.4} mm, allowance = {:.4} mm, fit = {fit:?}",
        clearance(&housing, &shaft),
        allowance(&housing, &shaft)
    );
    println!(
        "  suitability        = {}",
        match fit {
            FitType::Clearance => "running fit: the shaft always turns freely",
            FitType::Transition => "locational fit: assembly may be tight",
            FitType::Interference => "press fit: requires force or heating",
        }
    );
    // Unsupported deviation letters are reported rather than silently guessed.
    match iso_shaft_limits(BORE_DIA, 6, "k") {
        Ok(l) => println!("  unexpected k6 result {:.4}..{:.4}", l.lower, l.upper),
        Err(e) => println!("  k6 request rejected: {e}"),
    }

    // --- 5. Axial location stack-up of the bore ---------------------------
    // Bore centre height = boss height - flange thickness - shim.
    let axial = Stackup::new(vec![
        StackupMember::symmetric(BORE_Y, 0.05, 1.0),
        StackupMember::symmetric(6.0, 0.02, -1.0),
        StackupMember::symmetric(1.5, 0.01, -1.0),
    ]);
    let (lo, hi) = axial.worst_case();
    let (rlo, rhi) = axial.rss();
    let mc = axial.monte_carlo(50_000, 2026);
    println!();
    println!("Axial location stack-up of the bore centre:");
    println!("  nominal      = {:.4} mm", axial.nominal());
    println!("  worst case   = [{lo:.4}, {hi:.4}] mm");
    println!("  RSS          = [{rlo:.4}, {rhi:.4}] mm");
    println!(
        "  Monte Carlo  = mean {:.4} mm, std {:.4} mm, 3 sigma [{:.4}, {:.4}] mm",
        mc.mean, mc.std_dev, mc.lower_3sigma, mc.upper_3sigma
    );
    println!(
        "  positional budget used = {:.3} % of the dia {:.3} mm zone radius",
        100.0 * (hi - axial.nominal()) / (position_zone.magnitude() / 2.0),
        position_zone.magnitude()
    );
}
