# Tutorial 7: Kernel Scheduling

**Estimated Time:** 45 minutes  
**Prerequisites:** Tutorial 6

---

## Introduction

This tutorial covers command queues, priority scheduling, and kernel launch, based on
`crates/tpt-gpu-runtime/src/command/queue.rs` and `src/device/device.rs`.

### Scheduling Architecture

A single `CommandQueue` holds three internal `VecDeque`s (high/normal/low); priority is a
property of each *submitted command*, not of the queue itself. A `CommandScheduler` owns
a map of named `CommandQueue`s and round-robins across them:

```
┌─────────────────────────────────────────────────────────────────┐
│  CommandQueue                                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  high_queue │  │ normal_queue│  │  low_queue  │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         └────────────────┼────────────────┘                    │
│                          ▼  dequeue(): high first, then aged    │
│                             normal/low                          │
└─────────────────────────────────────────────────────────────────┘
        ▲ one or more CommandQueues owned by a CommandScheduler
```

---

## Command Queues

```rust
pub enum QueuePriority {
    High = 2,
    Normal = 1,
    Low = 0,
}

pub struct CommandQueue { /* high_queue, normal_queue, low_queue: VecDeque<CommandEntry>, ... */ }
```

### Creating Queues

```rust
use tpt_gpu_runtime::command::QueuePriority;

// The `priority` argument here is currently unused by the implementation —
// priority is set per-command in `submit()`, not per-queue.
let queue = device.create_queue(QueuePriority::Normal, /* capacity */ 64);
```

---

## Command Types

```rust
pub enum Command {
    Allocate { size: u64, region: MemoryRegion, mem_type: MemType, access: MemAccess },
    Free(MemoryAllocation),
    Memcpy { dst: MemoryAllocation, src: MemoryAllocation, size: u64, dst_offset: u64, src_offset: u64 },
    Memset { dst: MemoryAllocation, value: u8, size: u64, offset: u64 },
    LaunchKernel { kernel: String, config: KernelConfig, args: Vec<Vec<u8>> },
    Barrier,
    WaitEvent(EventHandle),
    SignalEvent(EventHandle),
}
```

Submit a command with an explicit priority:

```rust
let cmd_id = device.submit(queue, Command::Barrier, QueuePriority::High)?;
```

---

## Priority Scheduling

```rust
impl CommandQueue {
    pub fn dequeue(&mut self) -> Option<(u64, Command)> {
        // High always drains first.
        // Normal/low pop next, with an aging counter that forces a low-queue
        // pop every `max_aging` dequeues to avoid starvation.
    }
}
```

`CommandScheduler::dequeue_next()` round-robins across all queues it owns, returning the
next `(QueueHandle, cmd_id, Command)` from whichever queue yields one first.

---

## Synchronization Events

Events are booleans scoped to a single `CommandQueue`, not a separate `Event` type:

```rust
let ev = queue.create_event();          // -> EventHandle
queue.signal_event(ev);
assert!(queue.is_event_signaled(ev));
```

There is no `event.wait(timeout)` or `Event::wait_all` — events are polled via
`is_event_signaled`, and `Command::WaitEvent`/`Command::SignalEvent` exist as queueable
commands but are no-ops in the current `dispatch_command` implementation (see below).

---

## Draining a Queue

```rust
// Executes pending Memcpy commands against the backend/simulated arena;
// Allocate/Free/Memset/LaunchKernel currently no-op when dispatched from a queue.
device.synchronize();
println!("pending: {}", device.pending_commands());
```

Kernel execution today happens through the direct, synchronous API rather than the queue:

```rust
let kernel = device.create_kernel("matmul");
let config = KernelConfig::default(); // grid/block set via its builder methods
let handle = device.launch_kernel(&kernel, &config, &[arg_a_bytes, arg_b_bytes]);
```

`launch_kernel` takes raw argument byte buffers (`&[Vec<u8>]`), not `MemoryAllocation`
references, and returns a `KernelHandle` whose status (`Completed`/`Failed`) is set
synchronously before the call returns.

---

## Example: Queueing Commands by Priority

```rust
fn queue_barriers(device: &mut Device) -> TptrResult<()> {
    let queue = device.create_queue(QueuePriority::Normal, 64);
    device.submit(queue, Command::Barrier, QueuePriority::Low)?;
    device.submit(queue, Command::Barrier, QueuePriority::High)?;
    device.synchronize(); // drains both, high-priority one first
    Ok(())
}
```

---

## Exercises

1. **Priority Experiment**: Submit a mix of high/normal/low commands and trace `dequeue()`'s order
2. **Aging**: Submit enough normal-priority commands to observe the low-queue aging kick in
3. **Events**: Use `create_event`/`signal_event`/`is_event_signaled` to gate a second command's logic

---

## Summary

- ✅ `CommandQueue` holds three internal priority sub-queues; priority is per-command, not per-queue
- ✅ Aging-based low-queue promotion to prevent starvation
- ✅ `CommandScheduler` round-robins across multiple named queues
- ✅ Events are per-queue booleans (`create_event`/`signal_event`/`is_event_signaled`), not a timed-wait primitive
- ✅ Kernel launch is synchronous via `Device::launch_kernel`, separate from the command-queue path

**Next:** [Tutorial 8: GPU Primitives](08_gpu_primitives.md)
