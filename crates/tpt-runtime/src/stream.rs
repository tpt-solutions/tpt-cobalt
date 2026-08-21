//! Execution `Stream`: an ordered queue of compute/copy tasks modelling the
//! async compute + copy stream overlap of the full runtime. On the host this
//! runs tasks sequentially; a real backend would overlap copies and kernels.
use std::sync::{Arc, Condvar, Mutex};

use std::collections::VecDeque;

/// A unit of scheduled work (compute or copy).
pub type Task = Box<dyn FnOnce() + Send>;

/// An ordered stream of tasks. `run` executes queued tasks in FIFO order;
/// `enqueue` appends. Designed so compute and copy streams can be modelled
/// separately and overlapped by a backend.
pub struct Stream {
    name: String,
    queue: VecDeque<Task>,
}

impl Stream {
    pub fn new(name: impl Into<String>) -> Self {
        Stream {
            name: name.into(),
            queue: VecDeque::new(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Append a task to the stream.
    pub fn enqueue(&mut self, task: Task) {
        self.queue.push_back(task);
    }

    /// Execute all queued tasks in order, draining the stream.
    pub fn run(&mut self) {
        while let Some(task) = self.queue.pop_front() {
            task();
        }
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Dual-stream overlap
// ---------------------------------------------------------------------------

/// Which execution lane a task or barrier belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    Compute,
    Copy,
}

enum Item {
    Run(Task),
    Record(String),
    Wait(String),
}

/// Shared event table: `record(name)` marks it fired; `wait(name)` blocks
/// until fired. Host analogue of CUDA events.
#[derive(Clone, Default)]
struct Events(Arc<(Mutex<Vec<String>>, Condvar)>);

impl Events {
    fn record(&self, name: &str) {
        let (lock, cv) = &*self.0;
        lock.lock().unwrap().push(name.to_string());
        cv.notify_all();
    }
    fn wait(&self, name: &str) {
        let (lock, cv) = &*self.0;
        let mut fired = lock.lock().unwrap();
        while !fired.iter().any(|n| n == name) {
            fired = cv.wait(fired).unwrap();
        }
    }
}

fn drain_lane(queue: VecDeque<Item>, events: &Events) {
    for item in queue {
        match item {
            Item::Run(t) => t(),
            Item::Record(name) => events.record(&name),
            Item::Wait(name) => events.wait(&name),
        }
    }
}

/// Two-lane overlapped executor (compute + copy).
///
/// Tasks enqueued on a lane run in FIFO order on that lane; the two lanes run
/// concurrently. `record`/`wait` insert explicit cross-lane synchronization:
/// a lane that hits a `wait` for an event the other lane has not yet recorded
/// blocks until it is recorded.
pub struct DualStreams {
    compute: VecDeque<Item>,
    copy: VecDeque<Item>,
}

impl Default for DualStreams {
    fn default() -> Self {
        Self::new()
    }
}

impl DualStreams {
    pub fn new() -> Self {
        DualStreams {
            compute: VecDeque::new(),
            copy: VecDeque::new(),
        }
    }

    pub fn enqueue_compute(&mut self, task: Task) {
        self.compute.push_back(Item::Run(task));
    }

    pub fn enqueue_copy(&mut self, task: Task) {
        self.copy.push_back(Item::Run(task));
    }

    /// Record a named event at this point in the given lane's queue.
    pub fn record(&mut self, lane: Lane, name: impl Into<String>) {
        let item = Item::Record(name.into());
        match lane {
            Lane::Compute => self.compute.push_back(item),
            Lane::Copy => self.copy.push_back(item),
        }
    }

    /// Block the given lane at this point until `name` has been recorded.
    pub fn wait(&mut self, lane: Lane, name: impl Into<String>) {
        let item = Item::Wait(name.into());
        match lane {
            Lane::Compute => self.compute.push_back(item),
            Lane::Copy => self.copy.push_back(item),
        }
    }

    /// Run both lanes concurrently and join. Each lane's tasks keep their FIFO
    /// order; the only cross-lane ordering is via recorded events.
    pub fn run(&mut self) {
        let events = Events::default();
        let copy_queue = std::mem::take(&mut self.copy);
        let compute_queue = std::mem::take(&mut self.compute);

        let ev2 = events.clone();
        let copy_handle = std::thread::spawn(move || drain_lane(copy_queue, &ev2));
        drain_lane(compute_queue, &events);
        copy_handle.join().expect("copy lane panicked");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn stream_runs_in_order() {
        use std::sync::{Arc, Mutex};
        let mut s = Stream::new("compute");
        let log = Arc::new(Mutex::new(Vec::new()));
        s.enqueue(Box::new(|| {}));
        let l1 = log.clone();
        s.enqueue(Box::new(move || l1.lock().unwrap().push(1)));
        let l2 = log.clone();
        s.enqueue(Box::new(move || l2.lock().unwrap().push(2)));
        assert_eq!(s.len(), 3);
        s.run();
        assert!(s.is_empty());
        assert_eq!(*log.lock().unwrap(), vec![1, 2]);
    }

    #[test]
    fn dual_streams_overlap_without_barriers() {
        // The copy lane sleeps 80ms; the compute task must execute during that
        // window (not serialized behind it).
        let mut ds = DualStreams::new();
        let done = Arc::new(Mutex::new(false));
        let copy_done = done.clone();
        ds.enqueue_copy(Box::new(move || {
            std::thread::sleep(std::time::Duration::from_millis(80));
            *copy_done.lock().unwrap() = true;
        }));
        let compute_ran_early = Arc::new(Mutex::new(false));
        let early = compute_ran_early.clone();
        ds.enqueue_compute(Box::new(move || {
            // if lanes were serialized, copy_done would already be true here
            *early.lock().unwrap() = !*done.lock().unwrap();
        }));
        ds.run();
        assert!(
            *compute_ran_early.lock().unwrap(),
            "compute task did not overlap with the in-flight copy"
        );
    }

    #[test]
    fn event_wait_orders_cross_lane() {
        // Compute waits on "weights_ready" recorded by the copy lane, so the
        // compute task must observe the copied value.
        let mut ds = DualStreams::new();
        let buf = Arc::new(Mutex::new(0u64));
        let b1 = buf.clone();
        ds.enqueue_copy(Box::new(move || {
            std::thread::sleep(std::time::Duration::from_millis(30));
            *b1.lock().unwrap() = 42;
        }));
        ds.record(Lane::Copy, "weights_ready");
        ds.wait(Lane::Compute, "weights_ready");
        let b2 = buf.clone();
        ds.enqueue_compute(Box::new(move || {
            assert_eq!(*b2.lock().unwrap(), 42);
        }));
        ds.run();
        assert_eq!(*buf.lock().unwrap(), 42);
    }

    #[test]
    fn lanes_keep_fifo_order() {
        let mut ds = DualStreams::new();
        let log = Arc::new(Mutex::new(Vec::new()));
        for i in 0..10 {
            let l = log.clone();
            ds.enqueue_compute(Box::new(move || l.lock().unwrap().push(i)));
        }
        ds.run();
        assert_eq!(*log.lock().unwrap(), (0..10).collect::<Vec<_>>());
    }
}
