// Richer example: binary distillation column design with McCabe–Thiele.
//
// Given a feed composition and relative volatility, we size the column using
// the four classic short-cut methods and report the full stage count with the
// feed-stage location. Relative volatility is pulled from Peng–Robinson
// saturation pressures via `tpt_eng_unit_ops::relative_volatility`.

use tpt_eng_props::mixture::{Component, pr_saturation_pressure};
use tpt_eng_unit_ops::{
    fenske_min_stages, gilliland_stages, mccabe_thiele_stages, relative_volatility,
    separation_factor, underwood_rmin, underwood_theta,
};

fn main() {
    println!("=== Distillation column short-cut design ===");

    // Ethane / propane split at 260 K.
    let light = Component::from_name("ethane").unwrap();
    let heavy = Component::from_name("propane").unwrap();

    // Feed: 50 mol% light key; we want 95% overhead, 5% bottoms.
    let xf = 0.50;
    let xd = 0.95;
    let xb = 0.05;

    let alpha = relative_volatility(260.0, light, heavy).unwrap();
    let pe = pr_saturation_pressure(260.0, light);
    let ph = pr_saturation_pressure(260.0, heavy);
    println!("\nRelative volatility α = P_sat,light / P_sat,heavy");
    println!("  P_sat,ethane  = {pe:.3} Pa");
    println!("  P_sat,propane = {ph:.3} Pa");
    println!("  α             = {alpha:.3} (at 260 K)");

    // Separation sharpness used for quick screening.
    let s = separation_factor(xd, xb).unwrap();
    println!("\nSeparation sharpness S = {s:.3}");

    // 1) Total-reflux minimum (Fenske).
    let n_min = fenske_min_stages(xd, xb, alpha).unwrap();
    println!("\n1) Fenske minimum stages (total reflux) : {n_min:.3}");

    // 2) Minimum reflux (Underwood). q = 0 → saturated-vapour feed.
    let q = 0.0;
    let theta = underwood_theta(xf, q, alpha).unwrap();
    let rmin = underwood_rmin(xd, xb, xf, q, alpha).unwrap();
    println!("2) Underwood θ = {theta:.3}, R_min = {rmin:.3}");

    // 3) Pick an actual reflux ratio and bridge with Gilliland.
    let r = 3.0 * rmin;
    let n_gill = gilliland_stages(n_min, rmin, r).unwrap();
    println!("3) Gilliland stages @ R = {r:.3}        : {n_gill:.3}");

    // 4) Exact McCabe–Thiele stepping with this design reflux.
    let mt = mccabe_thiele_stages(xd, xb, xf, q, alpha, r).unwrap();
    println!(
        "4) McCabe–Thiele stages                : {:.3}  (feed enters at stage {:?})",
        mt.stages, mt.feed_stage
    );

    println!(
        "\nDesign summary: {:.1} theoretical stages including reboiler,",
        mt.stages
    );
    println!(
        "feed stage {}, operating at R = {:.3} (~{:.0}% above R_min).",
        mt.feed_stage.unwrap(),
        r,
        (r / rmin - 1.0) * 100.0
    );
}
