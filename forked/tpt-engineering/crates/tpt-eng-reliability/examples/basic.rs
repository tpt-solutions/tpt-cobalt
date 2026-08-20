//! Runnable demo for the core public API of `tpt-eng-reliability`.
//!
//! Exercises every module: fatigue (Basquin / Miner), life distributions
//! (Weibull / exponential), FMEA ranking, and probabilistic-design helpers.

use tpt_eng_reliability::{
    FmeaItem, basquin_cycles, basquin_stress, exponential_mean, exponential_reliability,
    miners_rule, prob_failure_below, rank_by_rpn, reliability_strength_vs_stress, weibull_b_life,
    weibull_mean, weibull_reliability,
};

fn main() {
    // --- Fatigue: Basquin S-N and Miner's rule -------------------------------
    // S-N slope m = 3.0, intercept log10(Nf) at s = 1 -> 12.0.
    let n = basquin_cycles(120.0, 3.0, 12.0).unwrap();
    let s_back = basquin_stress(n, 3.0, 12.0).unwrap();
    println!("Basquin: N(120 MPa) = {n:.3} cycles, recovered s = {s_back:.3} MPa");

    let damage = miners_rule(&[(4000.0, 10_000.0), (3000.0, 12_000.0)]);
    println!(
        "Miner's damage = {damage:.3} ({})",
        if damage >= 1.0 { "failed" } else { "safe" }
    );

    // --- Life distributions: Weibull and exponential -------------------------
    let r = weibull_reliability(1000.0, 5000.0, 1.5).unwrap();
    let b10 = weibull_b_life(10.0, 5000.0, 1.5).unwrap();
    let mean = weibull_mean(5000.0, 1.5).unwrap();
    println!("Weibull(eta=5000, beta=1.5): R(1000) = {r:.3}, B10 = {b10:.1} h, mean = {mean:.1} h");

    let er = exponential_reliability(1000.0, 1e-4).unwrap();
    let em = exponential_mean(1e-4).unwrap();
    println!("Exponential(lambda=1e-4): R(1000) = {er:.3}, mean = {em:.1} h");

    // --- FMEA ranking ---------------------------------------------------------
    let items = vec![
        FmeaItem::new("A", "pump", "leak", "seal wear", "fluid loss", 7, 3, 4),
        FmeaItem::new("B", "pump", "stall", "bearing seize", "no flow", 9, 2, 5),
        FmeaItem::new("C", "valve", "stick", "corrosion", "reduced flow", 4, 2, 2),
    ];
    for it in rank_by_rpn(&items) {
        println!(
            "FMEA {}: RPN={} criticality={} (S{} O{} D{})",
            it.id,
            it.rpn(),
            it.criticality(),
            it.severity,
            it.occurrence,
            it.detection
        );
    }

    // --- Probabilistic design: strength vs. stress ----------------------------
    // Strength S ~ N(100, 8), stress L ~ N(70, 10).
    let r_ss = reliability_strength_vs_stress(100.0, 8.0, 70.0, 10.0);
    let p_below = prob_failure_below(100.0, 8.0, 85.0);
    println!("Strength>stress reliability = {r_ss:.3}, P(S < 85) = {p_below:.3}");
}
