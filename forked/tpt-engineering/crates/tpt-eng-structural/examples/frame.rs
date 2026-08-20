//! Richer scenario: a single-bay portal frame, analysed member by member.
//!
//! The frame is decomposed the way a hand calculation would be: the rafter is a
//! simply-supported member carrying the roof loads, and its support reactions
//! become the column axial loads. Three load cases are run to build a moment
//! envelope, the rafter is checked in bending with [`SectionCheck`], and the
//! column is checked in compression through [`tpt_eng_safety`] — the same
//! utilisation logic the structural crate itself delegates to.
//!
//! Run with `cargo run -p tpt-eng-structural --example frame`.

use tpt_eng_safety::{ApplicationClass, CheckStatus, Quantity, evaluate_limit, max_limit};
use tpt_eng_structural::{Beam, Load, SectionCheck};
use tpt_math_units::uom::si::f64::{Area, Force, Length, Pressure, Volume};
use tpt_math_units::uom::si::{
    area::square_millimeter, force::kilonewton, length::meter, pressure::megapascal,
    torque::kilonewton_meter, volume::cubic_meter,
};

/// Rafter span (eaves to eaves), m.
const SPAN_M: f64 = 12.0;
/// Column height (base to eaves), m.
const COLUMN_HEIGHT_M: f64 = 6.0;

/// One named load case for the rafter, as a set of transverse loads.
struct RafterCase {
    name: &'static str,
    /// Total uniformly distributed force over the whole span, kN (downward +).
    udl_total_kn: f64,
    /// A concentrated service load, kN, and its position from the left, m.
    point_kn: f64,
    point_at_m: f64,
}

/// Build the rafter model for one load case.
fn rafter(case: &RafterCase) -> Beam {
    let span = Length::new::<meter>(SPAN_M);
    let mut beam = Beam::new(span);
    beam.add(Load::uniform(
        Length::new::<meter>(0.0),
        span,
        Force::new::<kilonewton>(case.udl_total_kn),
    ));
    if case.point_kn != 0.0 {
        beam.add(Load::point(
            Length::new::<meter>(case.point_at_m),
            Force::new::<kilonewton>(case.point_kn),
        ));
    }
    beam
}

fn main() {
    println!("Portal frame: {SPAN_M:.3} m span, {COLUMN_HEIGHT_M:.3} m columns, pinned bases\n");

    // Rafter: 610x229x101 UB, Z = 2.87e-3 m^3.
    let rafter_check = SectionCheck::new(
        Volume::new::<cubic_meter>(2.87e-3),
        Pressure::new::<megapascal>(165.0),
    );
    // Column: 305x305x137 UC, A = 17400 mm^2, Z = 2.30e-3 m^3.
    let column_area = Area::new::<square_millimeter>(17400.0);
    let column_check = SectionCheck::new(
        Volume::new::<cubic_meter>(2.30e-3),
        Pressure::new::<megapascal>(165.0),
    );
    // Reduced allowable for the column in compression: 275 MPa yield knocked
    // down for buckling over the 6 m unbraced height (slenderness factor 0.55).
    let column_allowable = Pressure::new::<megapascal>(275.0 * 0.55);

    let cases = [
        RafterCase {
            name: "G  dead only",
            udl_total_kn: 96.0,
            point_kn: 0.0,
            point_at_m: 0.0,
        },
        RafterCase {
            name: "G+Q gravity",
            udl_total_kn: 210.0,
            point_kn: 35.0,
            point_at_m: 4.0,
        },
        RafterCase {
            name: "G+S snow",
            udl_total_kn: 186.0,
            point_kn: 0.0,
            point_at_m: 0.0,
        },
    ];

    // --- Load-case envelope -------------------------------------------------
    println!(
        "{:<14}{:>10}{:>10}{:>13}{:>9}",
        "case", "R_A [kN]", "R_B [kN]", "M_max[kN*m]", "util"
    );
    let mut governing: Option<(&str, f64, f64)> = None; // (case, moment, axial)
    for case in &cases {
        let beam = rafter(case);
        let ra = beam.reaction_a().get::<kilonewton>();
        let rb = beam.reaction_b().get::<kilonewton>();
        let m_max = beam.max_bending_moment();
        let util = rafter_check.utilization(m_max);
        println!(
            "{:<14}{:>10.3}{:>10.3}{:>13.3}{:>9.3}",
            case.name,
            ra,
            rb,
            m_max.get::<kilonewton_meter>(),
            util
        );
        let axial = ra.max(rb);
        if governing.is_none_or(|(_, m, _)| util > m) {
            governing = Some((case.name, util, axial));
        }
    }
    let (gov_name, gov_util, gov_axial) = governing.expect("at least one load case");
    println!();
    println!("governing rafter case = {gov_name} (utilisation {gov_util:.3})");

    // --- Rafter bending detail for the governing case ----------------------
    let gov_case = cases
        .iter()
        .find(|c| c.name == gov_name)
        .expect("governing case in list");
    let beam = rafter(gov_case);
    println!();
    println!("Rafter shear / moment diagram (governing case):");
    println!("{:>8}{:>14}{:>16}", "x [m]", "shear [kN]", "moment [kN*m]");
    for i in 0..=6 {
        let x = Length::new::<meter>(SPAN_M * i as f64 / 6.0);
        println!(
            "{:>8.3}{:>14.3}{:>16.3}",
            x.get::<meter>(),
            beam.shear_at(x).get::<kilonewton>(),
            beam.moment_at(x).get::<kilonewton_meter>()
        );
    }
    let m_max = beam.max_bending_moment();
    println!(
        "peak moment = {:.3} kN*m, rafter utilisation = {:.3}",
        m_max.get::<kilonewton_meter>(),
        rafter_check.utilization(m_max)
    );

    // --- Column: the rafter reaction becomes an axial compression ----------
    // Force / Area is dimensionally a Pressure, so `uom` returns the stress
    // directly with no manual unit bookkeeping.
    let axial = Force::new::<kilonewton>(gov_axial);
    let stress: Pressure = axial / column_area;
    println!();
    println!("Column check (rafter reaction carried down the leg):");
    println!("  axial force N     = {:.3} kN", axial.get::<kilonewton>());
    println!(
        "  gross area  A     = {:.3} mm^2",
        column_area.get::<square_millimeter>()
    );
    println!(
        "  axial stress      = {:.3} MPa",
        stress.get::<megapascal>()
    );
    println!(
        "  allowable (buckling-reduced) = {:.3} MPa",
        column_allowable.get::<megapascal>()
    );

    // The safety crate consumes the real `uom` values and reports the margin.
    let report = evaluate_limit(
        "column compression",
        Quantity::from_pressure(stress),
        &max_limit(Quantity::from_pressure(column_allowable)),
        Some(ApplicationClass::StaticGeneral.recommended_safety_factor()),
    )
    .expect("pressure compared with pressure");
    println!(
        "  utilisation = {:.3}, margin = {:.3} MPa, achieved SF = {:.3}",
        report.utilization,
        report.margin / 1.0e6,
        report.safety_factor
    );
    println!(
        "  required SF = {:.3} -> {:?}",
        ApplicationClass::StaticGeneral.recommended_safety_factor(),
        report.status
    );

    // Frame continuity also drives a moment into the column at the eaves. For a
    // pinned-base portal a common hand idealisation takes it as half the
    // simply-supported rafter moment.
    let m_eaves = m_max * 0.5;
    let column_bending_util = column_check.utilization(m_eaves);
    let combined_util = report.utilization + column_bending_util;
    println!(
        "  eaves moment      = {:.3} kN*m -> bending utilisation {column_bending_util:.3}",
        m_eaves.get::<kilonewton_meter>()
    );
    println!(
        "  linear interaction N/N_c + M/M_c = {combined_util:.3} -> {}",
        if combined_util <= 1.0 { "PASS" } else { "FAIL" }
    );

    // --- Frame summary -----------------------------------------------------
    let eaves_thrust_kn = m_eaves.get::<kilonewton_meter>() / COLUMN_HEIGHT_M;
    println!();
    println!("Frame summary:");
    println!("  rafter bending utilisation = {gov_util:.3}");
    println!("  column axial utilisation   = {:.3}", report.utilization);
    println!("  column bending utilisation = {column_bending_util:.3}");
    println!(
        "  horizontal eaves thrust    = {eaves_thrust_kn:.3} kN (eaves moment / column height)"
    );
    println!(
        "  overall verdict            = {}",
        if gov_util <= 1.0 && combined_util <= 1.0 && report.status == CheckStatus::Pass {
            "frame adequate"
        } else {
            "revise member sizes"
        }
    );
}
