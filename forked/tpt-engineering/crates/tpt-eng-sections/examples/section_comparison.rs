//! Richer scenario: pick a steel beam section for a fixed bending demand.
//!
//! Seven candidate sections (rolled, hollow, built-up, and a custom cold-formed
//! outline) are evaluated through the same [`Section`] interface, then ranked by
//! structural efficiency (elastic modulus per unit area). Dimensions are
//! millimetres, so `Sx` is mm^3 and the demand is entered in kN*m.
//!
//! Note: for [`CustomPolygon`], area/centroid/second moments are exact
//! (Green's theorem) while the plastic modulus and torsion constant are grid
//! estimates, so only the exact quantities are compared below.
//!
//! Run with `cargo run -p tpt-eng-sections --example section_comparison`.

use tpt_eng_sections::compose::{self, Rect};
use tpt_eng_sections::polygon::PolygonError;
use tpt_eng_sections::{Angle, Channel, CustomPolygon, ISection, Section, SectionProperties, Tube};

/// Steel density used to turn area (mm^2) into mass per metre (kg/m).
const STEEL_DENSITY: f64 = 7850.0e-9; // kg/mm^3

/// Design bending moment, kN*m.
const M_DESIGN: f64 = 90.0;
/// Allowable bending stress, MPa (= N/mm^2).
const SIGMA_ALLOW: f64 = 165.0;

/// Utilisation `M / (Sx * sigma_allow)` with consistent mm/N units.
fn utilisation(sx_mm3: f64) -> f64 {
    let m_nmm = M_DESIGN * 1.0e6; // kN*m -> N*mm
    m_nmm / (sx_mm3 * SIGMA_ALLOW)
}

fn main() {
    let mut candidates: Vec<(String, SectionProperties)> = Vec::new();

    // 1. Rolled I-section, roughly a 356 x 171 x 51 UB.
    let ub = ISection::new(355.0, 171.0, 11.5, 7.4);
    candidates.push(("UB 356x171 (I-section)".to_string(), ub.properties()));

    // 2. Deeper, lighter rolled I-section.
    let ub_deep = ISection::new(457.0, 152.0, 10.9, 7.6);
    candidates.push(("UB 457x152 (I-section)".to_string(), ub_deep.properties()));

    // 3. Channel, bent about its strong axis.
    let pfc = Channel::new(300.0, 90.0, 13.0, 9.0);
    candidates.push(("PFC 300x90 (channel)".to_string(), pfc.properties()));

    // 4. Equal angle, an intentionally poor bending member.
    let angle = Angle::new(150.0, 150.0, 12.0);
    candidates.push(("L 150x150x12 (angle)".to_string(), angle.properties()));

    // 5. Circular hollow section.
    let chs = Tube::new(323.9, 311.9);
    candidates.push(("CHS 323.9x6.0 (tube)".to_string(), chs.properties()));

    // 6. Built-up welded plate girder, assembled from rectangles with the
    //    `compose` helpers (the same decomposition the rolled shapes use).
    let girder = [
        Rect::new(-100.0, 375.0, 200.0, 20.0),  // top flange 200 x 20
        Rect::new(-100.0, -395.0, 200.0, 20.0), // bottom flange 200 x 20
        Rect::new(-5.0, -375.0, 10.0, 750.0),   // web 10 x 750
    ];
    let (gcx, gcy) = compose::centroid(&girder);
    let (gix, giy, gixy) = compose::second_moments(&girder);
    let girder_props = SectionProperties::new(
        compose::area(&girder),
        gcx,
        gcy,
        gix,
        giy,
        gixy,
        gix / compose::y_extreme(&girder, gcy),
        giy / compose::x_extreme(&girder, gcx),
        compose::plastic_x(&girder, gcy),
        compose::plastic_y(&girder, gcx),
        compose::torsion(&girder),
    );
    candidates.push(("Plate girder 790 deep (built-up)".to_string(), girder_props));

    // 7. A custom cold-formed outline: a 200 x 80 x 4 lipless channel traced as
    //    a single closed (non-convex) boundary.
    let cold_formed = CustomPolygon::new(vec![
        (0.0, 0.0),
        (80.0, 0.0),
        (80.0, 4.0),
        (4.0, 4.0),
        (4.0, 196.0),
        (80.0, 196.0),
        (80.0, 200.0),
        (0.0, 200.0),
    ]);
    match cold_formed.validate() {
        Ok(()) => candidates.push((
            "CF channel 200x80x4 (polygon)".to_string(),
            cold_formed.properties(),
        )),
        Err(e) => println!("cold-formed outline rejected: {e}"),
    }

    // --- Report ------------------------------------------------------------
    println!("Design check: M = {M_DESIGN:.3} kN*m, allowable stress = {SIGMA_ALLOW:.3} MPa\n");
    println!(
        "{:<34}{:>10}{:>9}{:>13}{:>11}{:>8}{:>7}",
        "section", "A [mm2]", "kg/m", "Ix [mm4]", "Sx [mm3]", "rx", "util"
    );
    for (name, p) in &candidates {
        let mass = p.area * STEEL_DENSITY * 1000.0; // kg per metre of member
        let rx = (p.second_moment_x / p.area).sqrt();
        println!(
            "{:<34}{:>10.3}{:>9.3}{:>13.3e}{:>11.3e}{:>8.3}{:>7.3}",
            name,
            p.area,
            mass,
            p.second_moment_x,
            p.section_modulus_x,
            rx,
            utilisation(p.section_modulus_x)
        );
    }

    // Efficiency ranking: elastic modulus delivered per unit of material.
    println!();
    println!("ranked by bending efficiency Sx/A (higher is better):");
    let mut ranked: Vec<&(String, SectionProperties)> = candidates.iter().collect();
    ranked.sort_by(|a, b| {
        (b.1.section_modulus_x / b.1.area).total_cmp(&(a.1.section_modulus_x / a.1.area))
    });
    for (rank, (name, p)) in ranked.iter().enumerate() {
        let util = utilisation(p.section_modulus_x);
        let verdict = if util <= 1.0 { "PASS" } else { "FAIL" };
        println!(
            "  {}. {:<34} Sx/A = {:>8.3} mm   util = {:.3}  {verdict}",
            rank + 1,
            name,
            p.section_modulus_x / p.area,
            util
        );
    }

    // Torsion and asymmetry notes drawn straight from the property bundle.
    println!();
    let (_, angle_props) = candidates
        .iter()
        .find(|(n, _)| n.starts_with("L 150"))
        .expect("angle in candidate list");
    println!(
        "angle centroid offset = ({:.3}, {:.3}) mm, Ixy = {:.3e} mm^4 (unsymmetric bending)",
        angle_props.centroid_x, angle_props.centroid_y, angle_props.product_moment
    );
    let (_, chs_props) = candidates
        .iter()
        .find(|(n, _)| n.starts_with("CHS"))
        .expect("tube in candidate list");
    println!(
        "closed CHS torsion J  = {:.3e} mm^4 vs open plate girder J = {:.3e} mm^4",
        chs_props.torsional_constant, girder_props.torsional_constant
    );

    // A degenerate outline is rejected instead of silently reporting zeros.
    let collapsed = CustomPolygon::new(vec![(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]);
    match collapsed.validate() {
        Ok(()) => println!("unexpected: collinear polygon accepted"),
        Err(PolygonError::ZeroArea { area }) => {
            println!("collinear polygon rejected: signed area = {area:.3}");
        }
        Err(e) => println!("collinear polygon rejected: {e}"),
    }
}
