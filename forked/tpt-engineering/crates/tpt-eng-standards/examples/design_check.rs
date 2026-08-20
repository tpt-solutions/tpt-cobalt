//! End-to-end example: run a standards-based limit-state design check over a
//! user-supplied load combination. Run with `cargo run --example design_check`.

use std::collections::HashMap;

use tpt_eng_standards::{DesignBasis, LimitState, LoadCase, LoadCombination, LoadType};

fn main() {
    let basis = DesignBasis::new()
        .with_case(LoadCase::new("G", "dead", LoadType::Dead))
        .with_case(LoadCase::new("Q", "live", LoadType::Live))
        .with_combination(
            LoadCombination::new("ULS", "ULS")
                .with_factor("G", 1.35)
                .with_factor("Q", 1.5),
        );

    let mut demands = HashMap::new();
    demands.insert("G".to_string(), 10.0);
    demands.insert("Q".to_string(), 4.0);

    for r in basis.run_checks(&demands, 30.0, 1.0, LimitState::Ultimate) {
        println!(
            "{}: demand = {:.3}, utilization = {:.3}, passed = {}",
            r.combination_id, r.combined_demand, r.utilization, r.passed
        );
    }
}
