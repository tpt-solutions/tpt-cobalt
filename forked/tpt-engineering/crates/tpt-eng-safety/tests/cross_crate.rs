//! Cross-crate integration test: a tolerance stack-up (tpt-eng-tolerance)
//! drives a safety limit check (tpt-eng-safety) and a reliability estimate
//! (tpt-eng-reliability), with quantities expressed via tpt-eng-safety's
//! own `Quantity` type.

use rand::SeedableRng;
use rand::rngs::StdRng;
use tpt_eng_reliability::prob_failure_below;
use tpt_eng_safety::{CheckStatus, Dimension, Limit, LimitSense, Quantity, evaluate_limit};
use tpt_eng_tolerance::{DimTol, monte_carlo, worst_case};

#[test]
fn stackup_drives_safety_and_reliability() {
    // Three plates in a stack; total thickness must clear a 29.7 mm minimum.
    let dims = vec![
        DimTol::new("a", 10.0, 0.1),
        DimTol::new("b", 10.0, 0.1),
        DimTol::new("c", 10.0, 0.1),
    ];
    let (wlo, _whi) = worst_case(&dims); // 30.0 ± 0.3 -> low bound 29.7 mm
    assert!((wlo - 29.7).abs() < 1e-9);

    let mut rng = StdRng::seed_from_u64(7);
    let mc = monte_carlo(&dims, 20_000, None, &mut rng);

    // Reliability (tpt-eng-reliability): probability total thickness < 29.7 mm.
    let p_fail = prob_failure_below(mc.mean, mc.std, 29.7);
    assert!(
        p_fail < 0.05,
        "unexpectedly high failure probability: {p_fail}"
    );

    // Safety (tpt-eng-safety): design thickness must be at least 29.7 mm.
    let limit = Limit {
        value: Quantity::new(29.7, Dimension::Length),
        sense: LimitSense::Above,
    };
    let report = evaluate_limit(
        "stack clearance",
        Quantity::new(mc.mean, Dimension::Length),
        &limit,
        Some(1.0),
    )
    .unwrap();
    assert_eq!(report.status, CheckStatus::Pass);
    assert!(report.safety_factor > 1.0);
}
