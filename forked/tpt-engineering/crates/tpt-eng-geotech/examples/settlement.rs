// Settlement and consolidation example for `tpt-eng-geotech`.
//
// Models 1-D primary consolidation of a clay layer and its time-rate, plus
// Terzaghi/Meyerhof bearing capacity and Rankine/Coulomb lateral earth pressure
// for the same subsoil profile. Soil data is provenance-tagged.

use tpt_eng_geotech::bearing_capacity::{
    FoundationShape, meyerhof_ultimate_bearing_capacity, terzaghi_ultimate_bearing_capacity,
};
use tpt_eng_geotech::consolidation::{
    coeff_consolidation, consolidation_settlement, consolidation_time, time_factor_from_degree,
};
use tpt_eng_geotech::lateral_earth_pressure::{
    coulomb_ka, rankine_ka, rankine_kp, rankine_passive_force,
};
use tpt_eng_materials::DataSource;

fn main() {
    // --- 1-D consolidation settlement of a 4 m clay layer ---
    // Cc = 0.35, e0 = 0.9, in-situ stress 60 kPa, added stress 40 kPa.
    let s = consolidation_settlement(0.35, 0.9, 4.0, 60_000.0, 40_000.0);
    println!("Primary settlement (4 m clay): {:.3} m", s);

    // --- Time-rate: coefficient of consolidation and time to 90% ---
    let cv = coeff_consolidation(1e-9, 0.9, 2.0e-7, 9_810.0);
    let tv90 = time_factor_from_degree(90.0);
    let t90 = consolidation_time(cv, 2.0, 90.0); // double drainage, H_dr = 2 m
    println!("c_v                       : {:.3e} m^2/s", cv);
    println!("Time factor U=90%         : {:.3}", tv90);
    println!("Time to 90% consolidation : {:.1} days", t90 / 86400.0);

    // --- Bearing capacity of a 2 m square footing at 1.5 m depth ---
    let q_terz = terzaghi_ultimate_bearing_capacity(
        10_000.0,
        30.0,
        19_000.0,
        2.0,
        1.5,
        FoundationShape::Square,
        2.0,
    );
    let q_mey = meyerhof_ultimate_bearing_capacity(10_000.0, 30.0, 19_000.0, 2.0, 1.5, 2.0);
    println!("Terzaghi q_ult           : {:.1} kPa", q_terz / 1e3);
    println!("Meyerhof q_ult           : {:.1} kPa", q_mey / 1e3);

    // --- Lateral earth pressure on a 6 m retaining wall ---
    let ka = rankine_ka(30.0);
    let kp = rankine_kp(30.0);
    let kp_coul = coulomb_ka(30.0, 15.0, 0.0, 0.0);
    let p_passive = rankine_passive_force(19_000.0, 6.0, 0.0, kp);
    println!("Rankine K_a (30°)        : {:.3}", ka);
    println!("Rankine K_p (30°)        : {:.3}", kp);
    println!("Coulomb K_a (δ=15°)      : {:.3}", kp_coul);
    println!("Passive resultant        : {:.1} kN/m", p_passive / 1e3);

    // --- Provenance-tagged soil datum ---
    let src = DataSource::standard("ASTM D2487");
    println!("Soil layer source        : {:?} / {}", src.kind, src.label);
}
