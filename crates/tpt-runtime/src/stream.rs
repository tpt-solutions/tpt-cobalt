//! Execution `Stream`: an ordered queue of compute/copy tasks modelling the
//! async compute + copy stream overlap of the full runtime. On the host this
//! runs tasks sequentially; a real backend would overlap copies and kernels.

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
