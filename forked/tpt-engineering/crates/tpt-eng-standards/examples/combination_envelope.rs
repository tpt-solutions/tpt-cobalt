//! Richer scenario: a full combination envelope for one roof beam.
//!
//! A design basis carries four load cases and six user-entered combinations
//! (five ultimate, one serviceability). Every combination is evaluated, the
//! governing case is identified, the wind-uplift reversal is checked against the
//! reduced reverse capacity, and the basis is round-tripped through JSON so it
//! can be stored with the calculation record.
//!
//! Demands are bending moments in kN*m; the serviceability check uses
//! deflections in mm.
//!
//! Run with `cargo run -p tpt-eng-standards --example combination_envelope`.

use tpt_eng_standards::{
    CheckResult, DemandCapacity, DesignBasis, FactorSet, LimitState, LoadCase, LoadCombination,
    LoadType, evaluate_check,
};

/// Section moment capacity with the bottom flange restrained, kN*m.
const CAPACITY_ULS: f64 = 155.0;
/// Reduced capacity when the load reverses and the top flange is unrestrained.
const CAPACITY_REVERSED: f64 = 95.0;
/// Deflection limit (span/360 for a 7.2 m span), mm.
const DEFLECTION_LIMIT: f64 = 20.0;

/// Build the design basis: cases plus the engineer's own combination table.
fn roof_beam_basis() -> DesignBasis {
    let mut factors = FactorSet::new();
    factors.insert("gamma_G_unfav", 1.35);
    factors.insert("gamma_G_fav", 1.0);
    factors.insert("gamma_Q", 1.5);
    factors.insert("psi_0_wind", 0.6);
    factors.insert("psi_0_snow", 0.5);

    DesignBasis::new()
        .with_case(LoadCase::new("G", "Self weight + roofing", LoadType::Dead))
        .with_case(LoadCase::new("Q", "Imposed roof load", LoadType::Live))
        .with_case(LoadCase::new("S", "Snow", LoadType::Snow))
        .with_case(LoadCase::new("W", "Wind uplift", LoadType::Wind))
        // Ultimate combinations: each leading variable action in turn.
        .with_combination(LoadCombination::new("ULS-1", "1.35G").with_factor("G", 1.35))
        .with_combination(
            LoadCombination::new("ULS-2", "1.35G + 1.5Q + 0.5S")
                .with_factor("G", 1.35)
                .with_factor("Q", 1.5)
                .with_factor("S", 0.5 * 1.5),
        )
        .with_combination(
            LoadCombination::new("ULS-3", "1.35G + 1.5S + 0.6*1.5Q")
                .with_factor("G", 1.35)
                .with_factor("S", 1.5)
                .with_factor("Q", 0.6 * 1.5),
        )
        .with_combination(
            LoadCombination::new("ULS-4", "1.35G + 1.5W + 0.6*1.5Q")
                .with_factor("G", 1.35)
                .with_factor("W", 1.5)
                .with_factor("Q", 0.6 * 1.5),
        )
        // Uplift case: favourable dead load only, wind reversing the moment.
        .with_combination(
            LoadCombination::new("ULS-5", "1.0G + 1.5W (uplift)")
                .with_factor("G", 1.0)
                .with_factor("W", 1.5),
        )
        // Serviceability: characteristic combination, no partial factors.
        .with_combination(
            LoadCombination::new("SLS-1", "G + Q + 0.5S")
                .with_factor("G", 1.0)
                .with_factor("Q", 1.0)
                .with_factor("S", 0.5),
        )
        .with_factors(factors)
}

/// Print one check row.
fn row(r: &CheckResult, capacity_unit: &str) {
    let verdict = if r.passed { "PASS" } else { "FAIL" };
    println!(
        "  {:<7} demand = {:>9.3} {capacity_unit:<5} capacity = {:>8.3}  util = {:>6.3}  {verdict}",
        r.combination_id, r.combined_demand, r.capacity, r.utilization
    );
}

fn main() {
    let basis = roof_beam_basis();
    println!(
        "design basis: {} load cases, {} combinations, {} recorded factors",
        basis.cases.len(),
        basis.combinations.len(),
        basis.factors.len()
    );

    // Unfactored action effects, in the same order as the basis's cases.
    // Wind is negative: uplift reverses the sagging moment.
    let moments = basis.demands(&[48.0, 35.0, 12.0, -60.0]);
    println!();
    for c in &basis.cases {
        println!(
            "  {:<2} {:<24} {:?}: M = {:>7.3} kN*m",
            c.id, c.name, c.load_type, moments[&c.id]
        );
    }

    // --- Ultimate limit state envelope ------------------------------------
    println!();
    println!("Ultimate limit state (capacity {CAPACITY_ULS:.3} kN*m):");
    let uls_results: Vec<CheckResult> = basis
        .combinations
        .iter()
        .filter(|c| c.id.starts_with("ULS"))
        .map(|c| evaluate_check(c, &moments, CAPACITY_ULS, 1.0, LimitState::Ultimate))
        .collect();
    for r in &uls_results {
        row(r, "kN*m");
    }

    // The governing combination is the one with the largest utilization.
    let governing = uls_results
        .iter()
        .max_by(|a, b| a.utilization.total_cmp(&b.utilization))
        .expect("at least one ULS combination");
    println!(
        "  -> governing: {} at util = {:.3} ({:.3} kN*m of {:.3} kN*m)",
        governing.combination_id,
        governing.utilization,
        governing.combined_demand,
        governing.capacity
    );
    println!(
        "  -> spare capacity = {:.3} kN*m ({:.3} % of the section)",
        governing.capacity - governing.combined_demand,
        100.0 * (1.0 - governing.utilization)
    );
    println!(
        "  -> {} of {} ultimate combinations pass",
        uls_results.iter().filter(|r| r.passed).count(),
        uls_results.len()
    );

    // Reversal check: uplift flips the moment sign, so the magnitude must be
    // re-checked against the (lower) reverse capacity.
    let uplift = uls_results
        .iter()
        .find(|r| r.combination_id == "ULS-5")
        .expect("uplift combination");
    println!(
        "  -> ULS-5 net moment = {:.3} kN*m ({})",
        uplift.combined_demand,
        if uplift.combined_demand < 0.0 {
            "reversed: top flange now in compression"
        } else {
            "no reversal"
        }
    );
    let reversal = DemandCapacity::new(uplift.combined_demand.abs(), CAPACITY_REVERSED, 1.0);
    println!(
        "  -> reversal check: {:.3} kN*m vs reduced capacity {CAPACITY_REVERSED:.3} kN*m, util = {:.3} -> {}",
        uplift.combined_demand.abs(),
        reversal.utilization(),
        if reversal.passes() { "PASS" } else { "FAIL" }
    );

    // Deflections likewise reverse, so report the uplift case separately.
    let sls_deflections = [7.5, 9.0, 3.0, -14.0];

    // --- Serviceability limit state, in deflection terms -------------------
    // Reuse the SLS combination with per-case deflection contributions.
    let deflections = basis.demands(&sls_deflections);
    let sls = basis
        .combinations
        .iter()
        .find(|c| c.id == "SLS-1")
        .expect("SLS combination");
    let sls_result = evaluate_check(
        sls,
        &deflections,
        DEFLECTION_LIMIT,
        1.0,
        LimitState::Serviceability,
    );
    println!();
    println!("Serviceability limit state (limit span/360 = {DEFLECTION_LIMIT:.3} mm):");
    row(&sls_result, "mm");

    // --- Persist the basis with the calculation record ---------------------
    let json = serde_json::to_string_pretty(&basis).expect("serialize basis");
    let restored: DesignBasis = serde_json::from_str(&json).expect("deserialize basis");
    println!();
    println!(
        "basis JSON = {} bytes, round-trip identical = {}",
        json.len(),
        restored == basis
    );

    // Re-running the restored basis reproduces the same numbers.
    let replay = restored.run_checks(&moments, CAPACITY_ULS, 1.0, LimitState::Ultimate);
    let replay_max = replay
        .iter()
        .map(|r| r.utilization)
        .fold(f64::NEG_INFINITY, f64::max);
    println!(
        "replayed governing utilization = {replay_max:.3} (matches = {})",
        (replay_max - governing.utilization).abs() < 1e-12
    );
}
