//! Richer PERT example for `tpt-eng-schedule`.
//!
//! Builds a realistic small project from three-point (optimistic / most-likely /
//! pessimistic) estimates, converts each to a PERT expected duration and
//! variance, runs the CPM forward/backward pass on the expected durations, and
//! reports the expected project duration plus the variance accumulated along
//! the critical path. It also demonstrates resource-leveling on a daily demand
//! profile derived from the activity schedule.

use tpt_eng_schedule::{
    Activity, Schedule, leveling_peak, leveling_smooth_variance_reduction, pert_expected,
    pert_variance,
};

struct Task {
    id: &'static str,
    o: f64,
    m: f64,
    p: f64,
    predecessors: &'static [&'static str],
}

fn main() {
    // Three-point estimates (in days) for a software-release project.
    let tasks = [
        Task {
            id: "Design",
            o: 4.0,
            m: 6.0,
            p: 10.0,
            predecessors: &[],
        },
        Task {
            id: "Backend",
            o: 6.0,
            m: 10.0,
            p: 18.0,
            predecessors: &["Design"],
        },
        Task {
            id: "Frontend",
            o: 5.0,
            m: 8.0,
            p: 15.0,
            predecessors: &["Design"],
        },
        Task {
            id: "Integrate",
            o: 3.0,
            m: 5.0,
            p: 9.0,
            predecessors: &["Backend", "Frontend"],
        },
        Task {
            id: "Test",
            o: 4.0,
            m: 6.0,
            p: 12.0,
            predecessors: &["Integrate"],
        },
        Task {
            id: "Launch",
            o: 1.0,
            m: 2.0,
            p: 4.0,
            predecessors: &["Test"],
        },
    ];

    // Convert each task to a PERT expected duration and remember its variance.
    let mut variance = std::collections::HashMap::new();
    let activities: Vec<Activity> = tasks
        .iter()
        .map(|t| {
            let te = pert_expected(t.o, t.m, t.p);
            variance.insert(t.id, pert_variance(t.o, t.p));
            Activity {
                id: t.id.into(),
                duration: te,
                predecessors: t.predecessors.iter().map(|s| s.to_string()).collect(),
            }
        })
        .collect();

    let sched = Schedule::new(activities).expect("acyclic PERT network");

    println!("PERT schedule (expected durations):");
    for t in &tasks {
        println!(
            "  {:<9} te={:6.3}  var={:6.3}  ES={:5.2} EF={:5.2} float={:5.2} critical={}",
            t.id,
            pert_expected(t.o, t.m, t.p),
            variance[t.id],
            sched.early_start(t.id).unwrap(),
            sched.early_finish(t.id).unwrap(),
            sched.total_float(t.id).unwrap(),
            sched.is_critical(t.id).unwrap(),
        );
    }

    let critical = sched.critical_path();
    println!("\nCritical path: {:?}", critical);
    println!(
        "Expected project duration: {:.3} days",
        sched.project_duration()
    );

    // Project variance = sum of variances along the critical path (independent
    // activity estimates assumed). Standard deviation is its square root.
    let proj_var: f64 = critical.iter().map(|id| variance[id.as_str()]).sum();
    let proj_sd = proj_var.sqrt();
    println!(
        "Critical-path variance: {:.3}  (std dev {:.3} days)",
        proj_var, proj_sd
    );

    // Resource leveling: assign each activity a constant daily crew demand over
    // its expected duration; build the day-by-day total-demand profile.
    let crew = [3.0_f64, 4.0, 4.0, 5.0, 2.0, 2.0];
    let horizon = sched.project_duration().ceil() as usize;
    let mut demands = vec![0.0_f64; horizon];
    for (i, t) in tasks.iter().enumerate() {
        let start = sched.early_start(t.id).unwrap().ceil() as usize;
        let end = sched.early_finish(t.id).unwrap().ceil() as usize;
        for d in demands.iter_mut().take(end).skip(start) {
            *d += crew[i];
        }
    }
    println!(
        "\nPeak daily crew demand: {:.1}  (variance reduction after smoothing: {:.3})",
        leveling_peak(&demands),
        leveling_smooth_variance_reduction(&demands, 3)
    );
}
