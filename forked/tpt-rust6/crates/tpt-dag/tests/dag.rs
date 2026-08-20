//! Behavioural tests for the local executor: fan-out, retries, caching and
//! genuine parallelism.

use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tpt_dag::prelude::*;

// ---------------------------------------------------------------------------
// 1. a -> b -> [c, d]
// ---------------------------------------------------------------------------

task! {
    #[task]
    fn seed(n: i64) -> i64 { n * 2 }
}
task! {
    #[task]
    fn double(x: i64) -> i64 { x * 2 }
}
task! {
    #[task]
    fn label(x: i64) -> String { format!("v={x}") }
}
task! {
    #[task]
    fn negate(x: i64) -> i64 { -x }
}

#[test]
fn fan_out_pipeline_produces_all_outputs() {
    let p = pipeline![ seed(5) -> double -> [label, negate] ];
    assert_eq!(p.node_names(), vec!["seed", "double", "label", "negate"]);
    assert_eq!(
        p.edges(),
        vec![
            ("seed", "double"),
            ("double", "label"),
            ("double", "negate")
        ]
    );

    let out = run(p).unwrap();
    assert_eq!(out.len(), 4);
    assert_eq!(out.get::<i64>("seed"), Some(10));
    assert_eq!(out.get::<i64>("double"), Some(20));
    assert_eq!(out.get::<String>("label").as_deref(), Some("v=20"));
    assert_eq!(out.get::<i64>("negate"), Some(-20));
}

#[test]
fn progress_callback_reports_every_node() {
    let seen = Arc::new(Mutex::new(Vec::<(String, f64)>::new()));
    let sink = Arc::clone(&seen);
    let ex = Executor::new().on_progress(move |name, frac| {
        sink.lock().unwrap().push((name.to_string(), frac));
    });
    ex.run(&pipeline![ seed(1) -> double -> [label, negate] ])
        .unwrap();
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 4);
    assert_eq!(seen[0].0, "seed");
    assert!((seen[3].1 - 1.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// 2. retries
// ---------------------------------------------------------------------------

static FLAKY_ATTEMPTS: AtomicUsize = AtomicUsize::new(0);

task! {
    #[task]
    fn one() -> i64 { 1 }
}
task! {
    #[task(retries = 3)]
    fn flaky(x: i64) -> Result<i64, String> {
        let n = FLAKY_ATTEMPTS.fetch_add(1, SeqCst);
        if n < 2 { Err(format!("transient failure #{n}")) } else { Ok(x + 41) }
    }
}

#[test]
fn failing_task_is_retried_until_it_succeeds() {
    let p = pipeline![ one() -> flaky ];
    // Metadata is published when the task is first called or wired into a DAG.
    assert_eq!(task_meta("flaky").unwrap().retries, 3);
    let out = Executor::new().run(&p).unwrap();
    assert_eq!(out.get::<i64>("flaky"), Some(42));
    // 2 failures + 1 success == 3 invocations of the body.
    assert_eq!(FLAKY_ATTEMPTS.load(SeqCst), 3);
}

static DOOMED_ATTEMPTS: AtomicUsize = AtomicUsize::new(0);

task! {
    #[task(retries = 2)]
    fn doomed(_x: i64) -> Result<i64, String> {
        DOOMED_ATTEMPTS.fetch_add(1, SeqCst);
        Err("always fails".to_string())
    }
}

#[test]
fn exhausted_retries_surface_as_an_error() {
    let err = Executor::new()
        .run(&pipeline![ one() -> doomed ])
        .unwrap_err();
    match err {
        DagError::TaskFailed {
            node,
            attempts,
            message,
        } => {
            assert_eq!(node, "doomed");
            assert_eq!(attempts, 3); // 1 initial + 2 retries
            assert!(message.contains("always fails"));
        }
        other => panic!("unexpected error: {other}"),
    }
    assert_eq!(DOOMED_ATTEMPTS.load(SeqCst), 3);
}

// ---------------------------------------------------------------------------
// 3. caching
// ---------------------------------------------------------------------------

static EXPENSIVE_CALLS: AtomicUsize = AtomicUsize::new(0);

task! {
    #[task]
    fn cache_seed() -> i64 { 6 }
}
task! {
    #[task(cache = true)]
    fn expensive(x: i64) -> i64 {
        EXPENSIVE_CALLS.fetch_add(1, SeqCst);
        x * 7
    }
}

#[test]
fn cached_task_body_runs_only_once() {
    let cache = Arc::new(Cache::new());
    let ex = Executor::with_cache(Arc::clone(&cache));
    let p = pipeline![ cache_seed() -> expensive ];

    let first = ex.run(&p).unwrap();
    let second = ex.run(&p).unwrap();

    assert_eq!(first.get::<i64>("expensive"), Some(42));
    assert_eq!(second.get::<i64>("expensive"), Some(42));
    assert_eq!(EXPENSIVE_CALLS.load(SeqCst), 1, "second run must hit cache");
    assert_eq!(cache.len(), 1);

    // Clearing the cache forces re-execution.
    cache.clear();
    ex.run(&p).unwrap();
    assert_eq!(EXPENSIVE_CALLS.load(SeqCst), 2);
}

// ---------------------------------------------------------------------------
// 4. parallelism of independent nodes
// ---------------------------------------------------------------------------

static PAR_COUNTER: Mutex<usize> = Mutex::new(0);
static OVERLAPS: AtomicUsize = AtomicUsize::new(0);

/// Increment the shared counter, then wait (bounded) for the sibling node to
/// do the same. Both siblings observe `2` only if they truly ran concurrently.
fn bump_and_watch() {
    {
        let mut g = PAR_COUNTER.lock().unwrap();
        *g += 1;
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        if *PAR_COUNTER.lock().unwrap() == 2 {
            OVERLAPS.fetch_add(1, SeqCst);
            return;
        }
        if Instant::now() >= deadline {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

task! {
    #[task]
    fn par_seed() -> i64 { 3 }
}
task! {
    #[task]
    fn par_left(x: i64) -> i64 { bump_and_watch(); x + 1 }
}
task! {
    #[task]
    fn par_right(x: i64) -> i64 { bump_and_watch(); x + 2 }
}

#[test]
fn independent_nodes_run_concurrently() {
    let out = Executor::new()
        .run(&pipeline![ par_seed() -> [par_left, par_right] ])
        .unwrap();
    assert_eq!(out.get::<i64>("par_left"), Some(4));
    assert_eq!(out.get::<i64>("par_right"), Some(5));
    assert_eq!(*PAR_COUNTER.lock().unwrap(), 2);
    if num_workers() > 1 {
        assert_eq!(
            OVERLAPS.load(SeqCst),
            2,
            "both nodes should have observed each other in flight"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. graph shapes / errors
// ---------------------------------------------------------------------------

task! {
    #[task]
    fn scale(x: i64, factor: i64) -> i64 { x * factor }
}

#[test]
fn nested_fan_out_and_extra_literal_arguments() {
    // `scale` receives the upstream value plus a literal argument; the second
    // branch nests another chain inside the bracket group.
    let out = run(pipeline![ seed(2) -> [ scale(10) -> negate, label ] ]).unwrap();
    assert_eq!(out.get::<i64>("scale"), Some(40));
    assert_eq!(out.get::<i64>("negate"), Some(-40));
    assert_eq!(out.get::<String>("label").as_deref(), Some("v=4"));
    assert_eq!(out.final_node.as_deref(), Some("negate"));
}

#[test]
fn reused_task_gets_a_unique_node_name() {
    let out = run(pipeline![ seed(1) -> double -> double ]).unwrap();
    assert_eq!(out.get::<i64>("double"), Some(4));
    assert_eq!(out.get::<i64>("double#1"), Some(8));
    assert_eq!(out.final_value::<i64>(), Some(8));
}
