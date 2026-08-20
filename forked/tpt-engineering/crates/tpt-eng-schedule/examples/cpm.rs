//! Runnable example: CPM forward/backward pass and critical path.

use tpt_eng_schedule::{Activity, Schedule};

fn main() {
    let activities = vec![
        Activity {
            id: "A".into(),
            duration: 3.0,
            predecessors: vec![],
        },
        Activity {
            id: "B".into(),
            duration: 4.0,
            predecessors: vec!["A".into()],
        },
        Activity {
            id: "C".into(),
            duration: 2.0,
            predecessors: vec!["A".into()],
        },
        Activity {
            id: "D".into(),
            duration: 5.0,
            predecessors: vec!["B".into(), "C".into()],
        },
    ];
    let sched = Schedule::new(activities).expect("acyclic network");

    let duration = sched.project_duration();
    let critical = sched.critical_path();
    let float_c = sched.total_float("C").expect("known activity");

    println!("project duration     = {:.1}", duration);
    println!("critical path        = {:?}", critical);
    println!("total float of C     = {:.1}", float_c);

    assert!((duration - 12.0).abs() < 1e-9);
    assert!(critical.iter().any(|c| c == "A"));
    assert!(critical.iter().any(|c| c == "B"));
    assert!(critical.iter().any(|c| c == "D"));
    println!("schedule example passed");
}
