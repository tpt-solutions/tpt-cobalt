//! Integration tests for `tpt-eng-schedule`.
//!
//! These exercise the public API end-to-end (network build, CPM/PERT,
//! resource leveling, and EVM) against the scenarios described in the crate
//! brief. Unit-level checks of internal arithmetic live in `src/lib.rs`.

use tpt_eng_schedule::{
    Activity, Schedule, cpi, eac, leveling_peak, pert_expected, pert_variance, spi,
};

/// Two-activity chain A (dur 2) → B (dur 3).
fn chain() -> Schedule {
    Schedule::new(vec![
        Activity {
            id: "A".into(),
            duration: 2.0,
            predecessors: vec![],
        },
        Activity {
            id: "B".into(),
            duration: 3.0,
            predecessors: vec!["A".into()],
        },
    ])
    .expect("valid chain")
}

/// Parallel A (dur 2) & B (dur 1) from start, then C (dur 1) after both.
fn parallel() -> Schedule {
    Schedule::new(vec![
        Activity {
            id: "A".into(),
            duration: 2.0,
            predecessors: vec![],
        },
        Activity {
            id: "B".into(),
            duration: 1.0,
            predecessors: vec![],
        },
        Activity {
            id: "C".into(),
            duration: 1.0,
            predecessors: vec!["A".into(), "B".into()],
        },
    ])
    .expect("valid parallel network")
}

#[test]
fn chain_forward_backward_critical() {
    let s = chain();
    assert_eq!(s.early_start("B").unwrap(), 2.0);
    assert_eq!(s.early_finish("B").unwrap(), 5.0);
    assert_eq!(s.total_float("A").unwrap(), 0.0);
    assert_eq!(s.total_float("B").unwrap(), 0.0);
    assert_eq!(s.critical_path(), vec!["A".to_string(), "B".to_string()]);
    assert_eq!(s.project_duration(), 5.0);
}

#[test]
fn parallel_critical_path_and_float() {
    let s = parallel();
    assert!(s.is_critical("C").unwrap());
    assert!(s.is_critical("A").unwrap());
    assert!(!s.is_critical("B").unwrap());
    assert_eq!(s.total_float("B").unwrap(), 1.0);
    assert_eq!(s.critical_path(), vec!["A".to_string(), "C".to_string()]);
    assert_eq!(s.project_duration(), 3.0);
}

#[test]
fn pert_integration() {
    // Classic balanced three-point estimate.
    assert_eq!(pert_expected(2.0, 4.0, 6.0), 4.0);
    assert!(pert_variance(2.0, 6.0) > 0.0);
    // Expected lowers toward the optimistic side when m is near o.
    assert!(pert_expected(1.0, 2.0, 9.0) < 4.0);
}

#[test]
fn resource_leveling_peak() {
    let demands = [4.0, 8.0, 3.0, 8.0, 2.0];
    assert_eq!(leveling_peak(&demands), 8.0);
    assert_eq!(leveling_peak(&[]), 0.0);
}

#[test]
fn evm_integration() {
    // Under budget, behind schedule.
    assert!((cpi(50.0, 40.0) - 1.25).abs() < 1e-12);
    assert!((spi(50.0, 100.0) - 0.5).abs() < 1e-12);
    assert!((eac(100.0, 50.0, 40.0) - 80.0).abs() < 1e-12);
}
