//! tpt-dag — in-process concurrent pipeline executor: `task!`/`#[task]`,
//! `pipeline![]` DAGs, rayon-parallel execution, retries, content-addressed
//! caching and progress reporting.
//!
//! # Quick start
//!
//! ```
//! use tpt_dag::prelude::*;
//!
//! task! {
//!     #[task(cache = true, retries = 2)]
//!     fn ingest(n: i64) -> i64 { n * 2 }
//! }
//! task! {
//!     #[task]
//!     fn clean(x: i64) -> i64 { x + 1 }
//! }
//! task! {
//!     #[task]
//!     fn train(x: i64) -> String { format!("model({x})") }
//! }
//! task! {
//!     #[task]
//!     fn evaluate(x: i64) -> f64 { x as f64 / 2.0 }
//! }
//!
//! // `train` and `evaluate` run concurrently on the rayon pool.
//! let out = run(pipeline![ ingest(20) -> clean -> [train, evaluate] ]).unwrap();
//! assert_eq!(out.get::<i64>("clean"), Some(41));
//! assert_eq!(out.get::<String>("train").unwrap(), "model(41)");
//! assert_eq!(out.get::<f64>("evaluate"), Some(20.5));
//! assert_eq!(task_meta("ingest").unwrap().retries, 2);
//! ```
//!
//! # Model
//!
//! * [`task!`] emits the *original, normally-callable* function plus a hidden
//!   same-named module carrying its [`TaskMeta`] (`retries`, `cache`), which is
//!   published into a global `OnceLock<Mutex<HashMap<String, TaskMeta>>>`
//!   registry and readable with [`task_meta`] (publication happens the first
//!   time the function is called or wired into a `pipeline!`, since
//!   `macro_rules!` cannot run life-before-main code).
//! * [`pipeline!`] turns a `a("x") -> b -> [c, d]` chain into a [`Pipeline`]:
//!   node names, edges, `stringify!`d argument expressions and, for each node,
//!   a boxed `dyn Fn() -> Result<Arc<dyn Any + Send + Sync>, String>` closure.
//!   Those closures read their upstream values by name from a shared store
//!   captured at build time, so nodes stay uniformly typed.
//! * [`Executor`] topologically sorts the DAG into levels and runs each level
//!   with `rayon`'s `par_iter`, applying per-task retries (linear backoff,
//!   panics are converted into retryable errors) and the [`Cache`].
//! * [`Executor::run`] yields [`Outputs`] — a `HashMap<String, Arc<dyn Any +
//!   Send + Sync>>` (via `Deref`) with `get::<T>(name)` downcast helpers.
//!
//! # Caching
//!
//! Cache keys are content-addressed over *the DAG definition*: `hash(task name,
//! source text of the literal arguments, keys of the upstream nodes)`. For pure
//! tasks this is equivalent to hashing the serialized inputs while requiring no
//! `serde` bound on user types. Reuse one [`Cache`] (or the process-wide
//! [`default_cache`] used by [`run`]) across runs to get hits.
//!
//! # Single-binary deployment
//!
//! There is no scheduler, broker or worker daemon: the DAG, the task registry,
//! the cache and the executor all live inside your `main`, so
//! `cargo build --release && ./my_pipeline --workers 16` *is* the deployment.
//! Map `--workers` onto rayon:
//!
//! ```no_run
//! rayon::ThreadPoolBuilder::new().num_threads(16).build_global().unwrap();
//! ```
//!
//! [`DistributedCoordinator`] provides a working, infrastructure-free
//! multi-worker execution path today (it runs the whole pipeline on a Rayon
//! thread pool sized to its `workers` count); the `endpoint` field reserves the
//! future cross-machine transport that will ship the same binary to peer nodes.
//!
//! # Scope reductions (vs. the full spec)
//!
//! * `#[task]` is spelled inside `task! { ... }`: `macro_rules!` cannot define
//!   attribute macros and this crate intentionally ships no proc-macro crate.
//! * Each node consumes exactly one upstream value (plus optional literals), so
//!   fan-*out* is supported but fan-*in* / diamonds are not.
//! * Task outputs must be `Clone + Send + Sync + 'static`.
//! * Execution uses CPU-level parallelism via Rayon's `par_iter` over the
//!   topologically-sorted levels. Async I/O via Tokio is **not** wired up
//!   (this crate has no `tokio` dependency); the `DistributedCoordinator` runs
//!   pipelines across `workers` Rayon threads today (infrastructure-free), with
//!   the cross-machine transport left for the future. No file-backed cache is
//!   provided.

mod macros;
mod runtime;

pub use runtime::*;

/// Everything you normally need: `use tpt_dag::prelude::*;`
pub mod prelude {
    pub use crate::runtime::{
        default_cache, num_workers, register_task, registered_tasks, run, task_meta, Cache,
        DagError, DistributedCoordinator, Executor, Node, NodeFn, NodeResult, Outputs, Pipeline,
        PipelineBuilder, Store, TaskMeta, Value,
    };
    pub use crate::{pipeline, task};
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    task! {
        #[task]
        fn unit_src() -> i64 { 7 }
    }
    task! {
        #[task(retries = 5, cache = true)]
        fn unit_inc(x: i64) -> i64 { x + 1 }
    }

    #[test]
    fn meta_is_registered_and_fn_is_callable() {
        assert_eq!(unit_src(), 7);
        assert_eq!(unit_inc(1), 2);
        let m = task_meta("unit_inc").expect("registered");
        assert_eq!((m.retries, m.cache), (5, true));
        assert!(registered_tasks().contains(&"unit_src".to_string()));
    }

    #[test]
    fn pipeline_describes_nodes_and_edges() {
        let p = pipeline![ unit_src() -> unit_inc ];
        assert_eq!(p.node_names(), vec!["unit_src", "unit_inc"]);
        assert_eq!(p.edges(), vec![("unit_src", "unit_inc")]);
        assert!(p.to_dot().contains("\"unit_src\" -> \"unit_inc\""));
    }

    #[test]
    fn cycles_are_detected() {
        // The `pipeline!` grammar is acyclic by construction, so build one by
        // hand through the builder API.
        let mut b = PipelineBuilder::new();
        b.add(
            TaskMeta::new("a", 0, false),
            vec!["b".into()],
            "a()",
            || Ok(std::sync::Arc::new(1i32)),
        );
        b.add(
            TaskMeta::new("b", 0, false),
            vec!["a".into()],
            "b()",
            || Ok(std::sync::Arc::new(2i32)),
        );
        let err = Executor::new().run(&b.build()).unwrap_err();
        assert!(matches!(err, DagError::Cycle(_)), "got {err}");
    }

    #[test]
    fn unknown_dependencies_are_rejected() {
        let mut b = PipelineBuilder::new();
        b.add(
            TaskMeta::new("a", 0, false),
            vec!["ghost".into()],
            "a()",
            || Ok(std::sync::Arc::new(1i32)),
        );
        let err = Executor::new().run(&b.build()).unwrap_err();
        assert!(matches!(err, DagError::UnknownDep { .. }), "got {err}");
    }

    #[test]
    fn distributed_runs_pipeline() {
        let p = pipeline![unit_src()];
        let out = DistributedCoordinator::new("tcp://localhost:9000", 4)
            .submit(&p)
            .expect("distributed submit should run the pipeline");
        assert!(!out.map.is_empty(), "distributed run produced no outputs");
    }
}
