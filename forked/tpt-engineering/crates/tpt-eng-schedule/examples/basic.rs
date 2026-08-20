//! Runnable demo for `tpt-eng-schedule`.
//!
//! Builds a small project network, prints the CPM solution and EVM indices,
//! and demonstrates PERT estimation and resource leveling.

use tpt_eng_schedule::{
    Activity, Schedule, cpi, eac, leveling_peak, leveling_smooth_variance_reduction, pert_expected,
    pert_variance,
};

fn main() {
    let sched = Schedule::new(vec![
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
    .expect("valid network");

    println!("Project duration: {}", sched.project_duration());
    for id in ["A", "B", "C"] {
        println!(
            "{id}: ES={:.1} EF={:.1} LS={:.1} LF={:.1} float={:.1} critical={}",
            sched.early_start(id).unwrap(),
            sched.early_finish(id).unwrap(),
            sched.late_start(id).unwrap(),
            sched.late_finish(id).unwrap(),
            sched.total_float(id).unwrap(),
            sched.is_critical(id).unwrap(),
        );
    }
    println!("Critical path: {:?}", sched.critical_path());

    // PERT.
    let te = pert_expected(1.0, 2.0, 4.0);
    let var = pert_variance(1.0, 4.0);
    println!("PERT expected={te} variance={var}");

    // Resource leveling.
    let demands = [1.0, 9.0, 1.0, 9.0, 1.0];
    println!("Peak demand: {}", leveling_peak(&demands));
    println!(
        "Variance reduction after smoothing: {:.3}",
        leveling_smooth_variance_reduction(&demands, 3)
    );

    // EVM.
    println!(
        "CPI={:.2} EAC={:.1}",
        cpi(50.0, 40.0),
        eac(100.0, 50.0, 40.0)
    );
}
