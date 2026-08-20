//! Basic `tpt-eng-standards` usage: load cases, a user-supplied load
//! combination, partial factors, and a limit-state demand/capacity check.
//!
//! The crate ships no code tables — every factor below is data the engineer
//! enters. Demands here are bending moments in kN*m.
//!
//! Run with `cargo run -p tpt-eng-standards --example basic`.

use std::collections::HashMap;

use tpt_eng_standards::combinations::demand_map;
use tpt_eng_standards::{
    DemandCapacity, DesignBasis, FactorSet, LimitState, LoadCase, LoadCombination, LoadType,
    evaluate_check,
};

fn main() {
    // --- 1. Load cases: identity and kind only, never magnitudes ------------
    let cases = vec![
        LoadCase::new("G", "Self weight and finishes", LoadType::Dead),
        LoadCase::new("Q", "Imposed floor load", LoadType::Live),
        LoadCase::new("W", "Wind pressure", LoadType::Wind),
    ];
    for c in &cases {
        println!("case {:>2}: {:<28} type = {:?}", c.id, c.name, c.load_type);
    }

    // --- 2. Per-analysis demands, keyed by case id -------------------------
    let demands: HashMap<String, f64> = demand_map(&cases, &[42.0, 28.0, 16.0]);
    println!();
    for c in &cases {
        println!("unfactored demand {:>2} = {:.3} kN*m", c.id, demands[&c.id]);
    }

    // --- 3. A user-entered combination: 1.35 G + 1.5 Q + 0.9 W -------------
    let uls = LoadCombination::new("ULS-1", "1.35G + 1.5Q + 0.9W")
        .with_factor("G", 1.35)
        .with_factor("Q", 1.5)
        .with_factor("W", 0.9);
    println!();
    println!("combination {} references {:?}", uls.id, uls.case_ids());
    println!("combined demand = {:.3} kN*m", uls.evaluate(&demands));

    // `evaluate_checked` refuses to silently drop a missing case.
    let mut partial = HashMap::new();
    partial.insert("G".to_string(), 42.0);
    match uls.evaluate_checked(&partial) {
        Ok(v) => println!("unexpected success: {v:.3}"),
        Err(e) => println!("checked evaluation caught the gap: {e}"),
    }

    // --- 4. Limit-state check: demand * factor vs capacity ------------------
    let capacity = 130.0; // kN*m of section moment capacity
    let dc = DemandCapacity::new(uls.evaluate(&demands), capacity, 1.0);
    println!();
    println!(
        "utilization = {:.3}, passes = {} (capacity {:.3} kN*m)",
        dc.utilization(),
        dc.passes(),
        capacity
    );

    // The same check through the design-check workflow, which records context.
    let result = evaluate_check(&uls, &demands, capacity, 1.0, LimitState::Ultimate);
    println!(
        "{}: demand = {:.3}, capacity = {:.3}, util = {:.3}, passed = {} ({:?})",
        result.combination_id,
        result.combined_demand,
        result.capacity,
        result.utilization,
        result.passed,
        result.limit_state
    );

    // --- 5. Partial factors recorded as data, and a basis that owns it all --
    let mut factors = FactorSet::new();
    factors.insert("gamma_G", 1.35).insert("gamma_Q", 1.5);
    factors.insert("psi_0_wind", 0.6);
    println!();
    println!("factor set holds {} entries", factors.len());
    println!(
        "gamma_G = {:?}, missing = {:?}",
        factors.get("gamma_G"),
        factors.get("gamma_M")
    );

    let basis = DesignBasis::new()
        .with_case(cases[0].clone())
        .with_case(cases[1].clone())
        .with_case(cases[2].clone())
        .with_combination(uls)
        .with_factors(factors);
    for r in basis.run_checks(
        &basis.demands(&[42.0, 28.0, 16.0]),
        capacity,
        1.0,
        LimitState::Ultimate,
    ) {
        println!(
            "basis check {}: util = {:.3} -> {}",
            r.combination_id,
            r.utilization,
            if r.passed { "PASS" } else { "FAIL" }
        );
    }

    println!();
    println!("available limit states: {:?}", LimitState::ALL);
    println!("available load types  : {:?}", LoadType::ALL);
}
