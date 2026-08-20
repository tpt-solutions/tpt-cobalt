// Runnable example for `tpt-eng-geotech` demonstrating the core soil-mechanics
// primitives plus the Phase-9e extensions (bearing capacity, consolidation,
// lateral earth pressure, Atterberg limits).

use tpt_eng_geotech::atterberg::uscs_fine_grained;
use tpt_eng_geotech::bearing_capacity::{FoundationShape, terzaghi_ultimate_bearing_capacity};
use tpt_eng_geotech::consolidation::{consolidation_settlement, time_factor_from_degree};
use tpt_eng_geotech::lateral_earth_pressure::rankine_ka;
use tpt_eng_geotech::mohr_coulomb::shear_strength;

fn main() {
    // Mohr-Coulomb shear strength of a sandy soil at 100 kPa normal stress.
    let tau_f = shear_strength(0.0, 35.0, 100_000.0);
    println!("Mohr-Coulomb shear strength (35°, 100 kPa): {tau_f:.1} Pa");

    // Terzaghi ultimate bearing capacity of a 2 m square footing at 1 m depth
    // in soil with c = 5 kPa, φ = 28°, γ = 19 kN/m³.
    let q_ult = terzaghi_ultimate_bearing_capacity(
        5_000.0,
        28.0,
        19_000.0,
        2.0,
        1.0,
        FoundationShape::Square,
        2.0,
    );
    println!("Terzaghi q_ult (square footing): {q_ult:.1} Pa");

    // Rankine active earth-pressure coefficient for the same friction angle.
    println!("Rankine K_a (28°): {:.3}", rankine_ka(28.0));

    // 1-D consolidation settlement under 30 kPa added stress.
    let s = consolidation_settlement(0.3, 0.9, 5.0, 50_000.0, 30_000.0);
    println!("Primary consolidation settlement: {s:.3} m");

    // Time factor to reach 90% consolidation.
    println!(
        "Time factor at U = 90%: {:.3}",
        time_factor_from_degree(90.0)
    );

    // USCS classification of a clay (LL = 45, PL = 20).
    println!(
        "USCS group (LL=45, PL=20): {}",
        uscs_fine_grained(45.0, 20.0)
    );
}
