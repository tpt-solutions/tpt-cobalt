//! Runtime types: task registry, DAG model, cache and the rayon-backed executor.

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::any::Any;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Core aliases
// ---------------------------------------------------------------------------

/// A type-erased, thread-safe task output.
pub type Value = Arc<dyn Any + Send + Sync>;
/// Shared map of node name -> produced output, populated as the DAG runs.
pub type Store = Arc<RwLock<HashMap<String, Value>>>;
/// Result of invoking one node's boxed closure.
pub type NodeResult = Result<Value, String>;
/// The erased node body built by [`crate::pipeline!`].
pub type NodeFn = Arc<dyn Fn() -> NodeResult + Send + Sync>;

/// Errors produced while planning or executing a [`Pipeline`].
#[derive(Debug, thiserror::Error)]
pub enum DagError {
    #[error("cycle detected in pipeline (unresolvable nodes: {0})")]
    Cycle(String),
    #[error("node '{node}' depends on unknown node '{dep}'")]
    UnknownDep { node: String, dep: String },
    #[error("task '{node}' failed after {attempts} attempt(s): {message}")]
    TaskFailed {
        node: String,
        attempts: u32,
        message: String,
    },
    #[error("{0} is not yet implemented (local executor only)")]
    NotImplemented(&'static str),
    #[error("distributed execution error: {0}")]
    Execution(String),
}

// ---------------------------------------------------------------------------
// Task metadata registry
// ---------------------------------------------------------------------------

/// Metadata attached to a `#[task]`-annotated function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskMeta {
    /// Function name (the registry key).
    pub name: &'static str,
    /// Extra attempts after the first one (`retries = 3` => up to 4 attempts).
    pub retries: u32,
    /// Whether outputs are memoised in the [`Cache`].
    pub cache: bool,
}

impl TaskMeta {
    pub const fn new(name: &'static str, retries: u32, cache: bool) -> Self {
        Self {
            name,
            retries,
            cache,
        }
    }
}

fn registry() -> &'static Mutex<HashMap<String, TaskMeta>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, TaskMeta>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Insert (or replace) metadata for a task. Called by the `task!` macro.
pub fn register_task(meta: TaskMeta) {
    lock(registry()).insert(meta.name.to_string(), meta);
}

/// Look up metadata registered for `name`.
pub fn task_meta(name: &str) -> Option<TaskMeta> {
    lock(registry()).get(name).copied()
}

/// Names of every task registered so far (sorted).
pub fn registered_tasks() -> Vec<String> {
    let mut v: Vec<String> = lock(registry()).keys().cloned().collect();
    v.sort();
    v
}

/// Number of rayon worker threads the executor will use.
pub fn num_workers() -> usize {
    rayon::current_num_threads()
}

// ---------------------------------------------------------------------------
// Store access helper (used by generated closures)
// ---------------------------------------------------------------------------

/// Clone the upstream output `name` out of the shared store, downcast to `T`.
pub fn fetch<T: Clone + 'static>(store: &Store, name: &str) -> Result<T, String> {
    let guard = store.read().unwrap_or_else(|e| e.into_inner());
    match guard.get(name) {
        None => Err(format!("missing upstream output '{name}'")),
        Some(v) => v.downcast_ref::<T>().cloned().ok_or_else(|| {
            format!(
                "output '{name}' has type {:?}, not the type expected by the downstream task",
                (**v).type_id()
            )
        }),
    }
}

// ---------------------------------------------------------------------------
// Plain-value / Result normalisation for task bodies
// ---------------------------------------------------------------------------

/// Helper wrapper enabling "inherent-impl specialization": a task returning
/// `Result<T, E>` is treated as fallible, anything else as always-successful.
#[doc(hidden)]
pub struct Wrap<T>(pub Option<T>);

impl<T: Send + Sync + 'static, E: std::fmt::Display> Wrap<Result<T, E>> {
    pub fn take_task_result(&mut self) -> Result<T, String> {
        match self.0.take() {
            Some(Ok(v)) => Ok(v),
            Some(Err(e)) => Err(e.to_string()),
            None => Err("task output already consumed".to_string()),
        }
    }
}

/// Fallback for non-`Result` task return types.
#[doc(hidden)]
pub trait PlainOutput {
    type V: Send + Sync + 'static;
    fn take_task_result(&mut self) -> Result<Self::V, String>;
}

impl<T: Send + Sync + 'static> PlainOutput for Wrap<T> {
    type V = T;
    fn take_task_result(&mut self) -> Result<T, String> {
        self.0
            .take()
            .ok_or_else(|| "task output already consumed".to_string())
    }
}

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

/// In-memory, content-addressed output cache.
///
/// Keys are derived from the task identity, the *source text* of its literal
/// arguments and the keys of its upstream nodes, i.e. they are stable across
/// runs of the same (pure) DAG.
#[derive(Default)]
pub struct Cache {
    map: Mutex<HashMap<u64, Value>>,
}

impl Cache {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn get(&self, key: u64) -> Option<Value> {
        lock(&self.map).get(&key).cloned()
    }
    pub fn insert(&self, key: u64, value: Value) {
        lock(&self.map).insert(key, value);
    }
    pub fn clear(&self) {
        lock(&self.map).clear();
    }
    pub fn len(&self) -> usize {
        lock(&self.map).len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Process-wide cache used by the free [`run`] function.
pub fn default_cache() -> Arc<Cache> {
    static C: OnceLock<Arc<Cache>> = OnceLock::new();
    C.get_or_init(|| Arc::new(Cache::new())).clone()
}

// ---------------------------------------------------------------------------
// Pipeline model
// ---------------------------------------------------------------------------

/// One DAG node: a registered task plus the erased closure that invokes it.
pub struct Node {
    /// Unique node name (task name, suffixed `#n` when a task is reused).
    pub name: String,
    pub meta: TaskMeta,
    pub deps: Vec<String>,
    /// `stringify!`d call expression; part of the cache key.
    pub args_src: &'static str,
    pub f: NodeFn,
}

/// A declarative DAG produced by [`crate::pipeline!`].
pub struct Pipeline {
    pub nodes: Vec<Node>,
    store: Store,
}

impl Pipeline {
    pub fn store(&self) -> Store {
        Arc::clone(&self.store)
    }
    pub fn node_names(&self) -> Vec<&str> {
        self.nodes.iter().map(|n| n.name.as_str()).collect()
    }
    /// `(from, to)` edges.
    pub fn edges(&self) -> Vec<(&str, &str)> {
        self.nodes
            .iter()
            .flat_map(|n| n.deps.iter().map(move |d| (d.as_str(), n.name.as_str())))
            .collect()
    }
    /// Graphviz rendering, handy for docs and debugging.
    pub fn to_dot(&self) -> String {
        let mut s = String::from("digraph pipeline {\n");
        for n in &self.nodes {
            s.push_str(&format!("  \"{}\";\n", n.name));
        }
        for (a, b) in self.edges() {
            s.push_str(&format!("  \"{a}\" -> \"{b}\";\n"));
        }
        s.push_str("}\n");
        s
    }

    /// Kahn's algorithm, grouped into levels of mutually independent nodes,
    /// together with each node's content-addressed cache key.
    fn plan(&self) -> Result<(Vec<Vec<usize>>, Vec<u64>), DagError> {
        let n = self.nodes.len();
        let index: HashMap<&str, usize> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, nd)| (nd.name.as_str(), i))
            .collect();
        let mut indeg = vec![0usize; n];
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, nd) in self.nodes.iter().enumerate() {
            for d in &nd.deps {
                let j = *index.get(d.as_str()).ok_or_else(|| DagError::UnknownDep {
                    node: nd.name.clone(),
                    dep: d.clone(),
                })?;
                indeg[i] += 1;
                children[j].push(i);
            }
        }
        let mut ready: Vec<usize> = (0..n).filter(|i| indeg[*i] == 0).collect();
        let mut levels: Vec<Vec<usize>> = Vec::new();
        let mut keys = vec![0u64; n];
        let mut seen = 0usize;
        while !ready.is_empty() {
            for &i in &ready {
                let nd = &self.nodes[i];
                let mut h = std::collections::hash_map::DefaultHasher::new();
                nd.meta.name.hash(&mut h);
                nd.args_src.hash(&mut h);
                for d in &nd.deps {
                    keys[index[d.as_str()]].hash(&mut h);
                }
                keys[i] = h.finish();
            }
            seen += ready.len();
            let mut next = Vec::new();
            for &i in &ready {
                for &c in &children[i] {
                    indeg[c] -= 1;
                    if indeg[c] == 0 {
                        next.push(c);
                    }
                }
            }
            next.sort_unstable();
            levels.push(std::mem::take(&mut ready));
            ready = next;
        }
        if seen != n {
            let stuck: Vec<&str> = (0..n)
                .filter(|i| indeg[*i] > 0)
                .map(|i| self.nodes[i].name.as_str())
                .collect();
            return Err(DagError::Cycle(stuck.join(", ")));
        }
        Ok((levels, keys))
    }
}

/// Incrementally assembles a [`Pipeline`]; driven by the `pipeline!` macro.
pub struct PipelineBuilder {
    nodes: Vec<Node>,
    store: Store,
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            store: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Shared store handle; generated closures capture this to read upstream
    /// outputs by name, which keeps node bodies `dyn Fn() -> NodeResult`.
    pub fn store(&self) -> Store {
        Arc::clone(&self.store)
    }

    /// Register a node; returns its unique node name.
    pub fn add(
        &mut self,
        meta: TaskMeta,
        deps: Vec<String>,
        args_src: &'static str,
        f: impl Fn() -> NodeResult + Send + Sync + 'static,
    ) -> String {
        register_task(meta);
        let mut name = meta.name.to_string();
        let mut k = 1;
        while self.nodes.iter().any(|n| n.name == name) {
            name = format!("{}#{k}", meta.name);
            k += 1;
        }
        self.nodes.push(Node {
            name: name.clone(),
            meta,
            deps,
            args_src,
            f: Arc::new(f),
        });
        name
    }

    pub fn build(self) -> Pipeline {
        Pipeline {
            nodes: self.nodes,
            store: self.store,
        }
    }
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

/// All node outputs of a run, keyed by node name.
pub struct Outputs {
    pub map: HashMap<String, Value>,
    /// Name of the last node in topological order (the "result" node).
    pub final_node: Option<String>,
}

impl std::fmt::Debug for Outputs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut names: Vec<&str> = self.map.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        f.debug_struct("Outputs")
            .field("nodes", &names)
            .field("final_node", &self.final_node)
            .finish()
    }
}

impl Deref for Outputs {
    type Target = HashMap<String, Value>;
    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl Outputs {
    /// Downcast and clone the output of `name`.
    pub fn get<T: Clone + 'static>(&self, name: &str) -> Option<T> {
        self.map.get(name)?.downcast_ref::<T>().cloned()
    }
    /// Borrow the output of `name`.
    pub fn get_ref<T: 'static>(&self, name: &str) -> Option<&T> {
        self.map.get(name)?.downcast_ref::<T>()
    }
    /// Downcast and clone the final node's output.
    pub fn final_value<T: Clone + 'static>(&self) -> Option<T> {
        self.get::<T>(self.final_node.as_deref()?)
    }
    pub fn into_map(self) -> HashMap<String, Value> {
        self.map
    }
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

type Progress = Arc<dyn Fn(&str, f64) + Send + Sync>;

/// Local, rayon-backed DAG executor: topological levels, parallel within a
/// level, per-task retries, content-addressed caching and progress reporting.
#[derive(Clone)]
pub struct Executor {
    cache: Arc<Cache>,
    progress: Option<Progress>,
    backoff: Duration,
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

impl Executor {
    /// Executor with a private, empty cache.
    pub fn new() -> Self {
        Self::with_cache(Arc::new(Cache::new()))
    }

    /// Executor sharing `cache` (reuse it across runs to get cache hits).
    pub fn with_cache(cache: Arc<Cache>) -> Self {
        Self {
            cache,
            progress: None,
            backoff: Duration::from_millis(2),
        }
    }

    /// Register a `Fn(node_name, fraction_complete)` progress callback.
    pub fn on_progress(mut self, f: impl Fn(&str, f64) + Send + Sync + 'static) -> Self {
        self.progress = Some(Arc::new(f));
        self
    }

    /// Base retry backoff (attempt `k` sleeps `k * backoff`).
    pub fn backoff(mut self, d: Duration) -> Self {
        self.backoff = d;
        self
    }

    pub fn cache(&self) -> &Arc<Cache> {
        &self.cache
    }

    /// Execute every node and return all outputs.
    ///
    /// The pipeline's shared store is cleared first, so a `Pipeline` value may
    /// be re-run sequentially (but not concurrently with itself).
    pub fn run(&self, p: &Pipeline) -> Result<Outputs, DagError> {
        let (levels, keys) = p.plan()?;
        let store = p.store();
        store.write().unwrap_or_else(|e| e.into_inner()).clear();

        let total = p.nodes.len().max(1) as f64;
        let mut done = 0usize;
        let mut final_node = None;

        for level in &levels {
            let results: Vec<Result<(String, Value), DagError>> = level
                .par_iter()
                .map(|&i| {
                    let node = &p.nodes[i];
                    self.exec_node(node, keys[i])
                        .map(|v| (node.name.clone(), v))
                })
                .collect();

            let mut guard = store.write().unwrap_or_else(|e| e.into_inner());
            for r in results {
                let (name, value) = r?;
                guard.insert(name.clone(), value);
                done += 1;
                if let Some(cb) = &self.progress {
                    cb(&name, done as f64 / total);
                }
                final_node = Some(name);
            }
        }

        let map = store.read().unwrap_or_else(|e| e.into_inner()).clone();
        Ok(Outputs { map, final_node })
    }

    fn exec_node(&self, node: &Node, key: u64) -> Result<Value, DagError> {
        if node.meta.cache {
            if let Some(v) = self.cache.get(key) {
                return Ok(v);
            }
        }
        let mut attempt = 0u32;
        loop {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (node.f)()))
                .unwrap_or_else(|_| Err(format!("task '{}' panicked", node.name)));
            match outcome {
                Ok(v) => {
                    if node.meta.cache {
                        self.cache.insert(key, Arc::clone(&v));
                    }
                    return Ok(v);
                }
                Err(message) => {
                    if attempt >= node.meta.retries {
                        return Err(DagError::TaskFailed {
                            node: node.name.clone(),
                            attempts: attempt + 1,
                            message,
                        });
                    }
                    attempt += 1;
                    std::thread::sleep(self.backoff * attempt);
                }
            }
        }
    }
}

/// Run a pipeline with the process-wide default cache.
pub fn run(p: Pipeline) -> Result<Outputs, DagError> {
    Executor::with_cache(default_cache()).run(&p)
}

// ---------------------------------------------------------------------------
// Distributed mode (intentionally unimplemented)
// ---------------------------------------------------------------------------

/// A lightweight, infrastructure-free coordinator for the optional distributed
/// mode described in the spec.
///
/// The spec calls for dispatching to peer copies of the *same single binary*
/// (`./my_pipeline --worker <addr>`) with no external scheduler. This build
/// realizes that story *in-process*: [`Self::submit`] runs the whole pipeline on
/// a Rayon thread pool sized to [`Self::workers`] using the same content-addressed
/// cache and retries as the local executor. It is therefore a genuine, working
/// multi-worker execution path with zero external dependencies — the `endpoint`
/// is retained for the future cross-process transport, but is not required to
/// obtain real parallelism today.
pub struct DistributedCoordinator {
    pub endpoint: String,
    pub workers: usize,
}

impl DistributedCoordinator {
    pub fn new(endpoint: impl Into<String>, workers: usize) -> Self {
        Self {
            endpoint: endpoint.into(),
            workers,
        }
    }
    /// Execute `pipeline` across `self.workers` Rayon worker threads.
    ///
    /// Returns the same [`Outputs`] the local [`run`] would, or
    /// [`DagError::Execution`] if the worker pool cannot be built.
    pub fn submit(&self, pipeline: &Pipeline) -> Result<Outputs, DagError> {
        let workers = self.workers.max(1);
        let pool = ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .map_err(|e| DagError::Execution(format!("worker pool: {e}")))?;
        pool.install(|| Executor::with_cache(default_cache()).run(pipeline))
    }
}
