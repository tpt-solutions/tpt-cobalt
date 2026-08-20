//! Command Queue and Scheduler - Priority-based GPU command execution.
use crate::error::{ErrorCode, TptrError, TptrResult};
use crate::kernel::KernelConfig;
use crate::memory::MemoryAllocation;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QueuePriority {
    High = 2,
    Normal = 1,
    Low = 0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandStatus {
    Queued,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventHandle(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueueHandle(pub u64);

#[derive(Debug, Clone)]
pub enum Command {
    Allocate {
        size: u64,
        region: crate::memory::MemoryRegion,
        mem_type: crate::memory::MemType,
        access: crate::memory::MemAccess,
    },
    Free(MemoryAllocation),
    Memcpy {
        dst: MemoryAllocation,
        src: MemoryAllocation,
        size: u64,
        dst_offset: u64,
        src_offset: u64,
    },
    Memset {
        dst: MemoryAllocation,
        value: u8,
        size: u64,
        offset: u64,
    },
    LaunchKernel {
        kernel: String,
        config: KernelConfig,
        args: Vec<Vec<u8>>,
    },
    Barrier,
    WaitEvent(EventHandle),
    SignalEvent(EventHandle),
}

impl Command {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Allocate { .. } => "Allocate",
            Self::Free(..) => "Free",
            Self::Memcpy { .. } => "Memcpy",
            Self::Memset { .. } => "Memset",
            Self::LaunchKernel { .. } => "LaunchKernel",
            Self::Barrier => "Barrier",
            Self::WaitEvent(..) => "WaitEvent",
            Self::SignalEvent(..) => "SignalEvent",
        }
    }
}

#[derive(Debug)]
struct CommandEntry {
    id: u64,
    command: Command,
    #[allow(dead_code)]
    priority: QueuePriority,
    #[allow(dead_code)]
    submit_time: Instant,
    #[allow(dead_code)]
    status: CommandStatus,
}

#[derive(Debug)]
pub struct CommandQueue {
    handle: QueueHandle,
    high_queue: VecDeque<CommandEntry>,
    normal_queue: VecDeque<CommandEntry>,
    low_queue: VecDeque<CommandEntry>,
    next_cmd_id: AtomicU64,
    events: HashMap<EventHandle, bool>,
    aging_counter: u64,
    max_aging: u64,
    capacity: usize,
}

impl CommandQueue {
    pub fn new(handle: QueueHandle, capacity: usize) -> Self {
        Self {
            handle,
            high_queue: VecDeque::with_capacity(capacity),
            normal_queue: VecDeque::with_capacity(capacity),
            low_queue: VecDeque::with_capacity(capacity),
            next_cmd_id: AtomicU64::new(1),
            events: HashMap::new(),
            aging_counter: 0,
            max_aging: 10,
            capacity,
        }
    }
    pub fn handle(&self) -> QueueHandle {
        self.handle
    }
    pub fn submit(&mut self, cmd: Command, pri: QueuePriority) -> TptrResult<u64> {
        let total = self.high_queue.len() + self.normal_queue.len() + self.low_queue.len();
        if total >= self.capacity {
            return Err(TptrError::new(
                ErrorCode::QueueFull,
                format!("capacity {} exhausted", self.capacity),
            ));
        }
        let id = self.next_cmd_id.fetch_add(1, Ordering::SeqCst);
        let entry = CommandEntry {
            id,
            command: cmd,
            priority: pri,
            submit_time: Instant::now(),
            status: CommandStatus::Queued,
        };
        match pri {
            QueuePriority::High => self.high_queue.push_back(entry),
            QueuePriority::Normal => self.normal_queue.push_back(entry),
            QueuePriority::Low => self.low_queue.push_back(entry),
        }
        Ok(id)
    }
    pub fn dequeue(&mut self) -> Option<(u64, Command)> {
        if let Some(e) = self.high_queue.pop_front() {
            return Some((e.id, e.command));
        }
        if let Some(e) = self.normal_queue.pop_front() {
            return Some((e.id, e.command));
        }
        self.aging_counter += 1;
        if self.aging_counter >= self.max_aging && !self.low_queue.is_empty() {
            self.aging_counter = 0;
            if let Some(e) = self.low_queue.pop_front() {
                return Some((e.id, e.command));
            }
        }
        if let Some(e) = self.low_queue.pop_front() {
            return Some((e.id, e.command));
        }
        None
    }
    pub fn peek(&self) -> Option<&Command> {
        self.high_queue
            .front()
            .or_else(|| self.normal_queue.front())
            .or_else(|| self.low_queue.front())
            .map(|e| &e.command)
    }
    pub fn len(&self) -> usize {
        self.high_queue.len() + self.normal_queue.len() + self.low_queue.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn clear(&mut self) {
        self.high_queue.clear();
        self.normal_queue.clear();
        self.low_queue.clear();
    }
    pub fn create_event(&mut self) -> EventHandle {
        let id = self.next_cmd_id.fetch_add(1, Ordering::SeqCst);
        let h = EventHandle(id);
        self.events.insert(h, false);
        h
    }
    pub fn signal_event(&mut self, event: EventHandle) {
        self.events.insert(event, true);
    }
    pub fn is_event_signaled(&self, event: EventHandle) -> bool {
        self.events.get(&event).copied().unwrap_or(false)
    }
}

#[derive(Debug)]
pub struct CommandScheduler {
    queues: HashMap<QueueHandle, CommandQueue>,
    next_queue_id: AtomicU64,
}

impl Default for CommandScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandScheduler {
    pub fn new() -> Self {
        Self {
            queues: HashMap::new(),
            next_queue_id: AtomicU64::new(1),
        }
    }
    pub fn create_queue(&mut self, capacity: usize) -> QueueHandle {
        let id = self.next_queue_id.fetch_add(1, Ordering::SeqCst);
        let handle = QueueHandle(id);
        self.queues
            .insert(handle, CommandQueue::new(handle, capacity));
        handle
    }
    pub fn submit(
        &mut self,
        queue: QueueHandle,
        command: Command,
        priority: QueuePriority,
    ) -> TptrResult<u64> {
        let q = self.queues.get_mut(&queue).ok_or_else(|| {
            TptrError::new(
                ErrorCode::InvalidKernel,
                format!("Queue {:?} not found", queue),
            )
        })?;
        q.submit(command, priority)
    }
    pub fn dequeue_next(&mut self) -> Option<(QueueHandle, u64, Command)> {
        let handles: Vec<QueueHandle> = self.queues.keys().copied().collect();
        for handle in &handles {
            if let Some(q) = self.queues.get_mut(handle) {
                if let Some((id, cmd)) = q.dequeue() {
                    return Some((*handle, id, cmd));
                }
            }
        }
        None
    }
    pub fn queue_mut(&mut self, handle: QueueHandle) -> Option<&mut CommandQueue> {
        self.queues.get_mut(&handle)
    }
    pub fn queue(&self, handle: QueueHandle) -> Option<&CommandQueue> {
        self.queues.get(&handle)
    }
    pub fn remove_queue(&mut self, handle: QueueHandle) -> Option<CommandQueue> {
        self.queues.remove(&handle)
    }
    pub fn total_pending(&self) -> usize {
        self.queues.values().map(|q| q.len()).sum()
    }
    pub fn num_queues(&self) -> usize {
        self.queues.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_submit_dequeue() {
        let mut q = CommandQueue::new(QueueHandle(1), 64);
        let id = q.submit(Command::Barrier, QueuePriority::Normal).unwrap();
        assert_eq!(id, 1);
        let (did, cmd) = q.dequeue().unwrap();
        assert_eq!(did, 1);
        assert!(matches!(cmd, Command::Barrier));
        assert!(q.is_empty());
    }
    #[test]
    fn test_priority() {
        let mut q = CommandQueue::new(QueueHandle(1), 64);
        q.submit(Command::Barrier, QueuePriority::Low).unwrap();
        q.submit(Command::Barrier, QueuePriority::High).unwrap();
        let (_, cmd) = q.dequeue().unwrap();
        assert!(matches!(cmd, Command::Barrier));
    }
    #[test]
    fn test_capacity() {
        let mut q = CommandQueue::new(QueueHandle(1), 2);
        q.submit(Command::Barrier, QueuePriority::Normal).unwrap();
        q.submit(Command::Barrier, QueuePriority::Normal).unwrap();
        assert!(q.submit(Command::Barrier, QueuePriority::Normal).is_err());
    }
    #[test]
    fn test_scheduler() {
        let mut s = CommandScheduler::new();
        let h1 = s.create_queue(64);
        let h2 = s.create_queue(64);
        assert_eq!(s.num_queues(), 2);
        s.submit(h1, Command::Barrier, QueuePriority::Normal)
            .unwrap();
        s.submit(h2, Command::Barrier, QueuePriority::High).unwrap();
        assert_eq!(s.total_pending(), 2);
        let (qh, _, _) = s.dequeue_next().unwrap();
        assert!(qh == h1 || qh == h2);
    }
    #[test]
    fn test_events() {
        let mut q = CommandQueue::new(QueueHandle(1), 64);
        let ev = q.create_event();
        assert!(!q.is_event_signaled(ev));
        q.signal_event(ev);
        assert!(q.is_event_signaled(ev));
    }
}
