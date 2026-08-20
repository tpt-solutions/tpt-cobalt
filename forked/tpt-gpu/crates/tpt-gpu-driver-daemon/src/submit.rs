// submit.rs — Command ring management and kernel submission
//
// Manages the 256-entry command ring shared with the hardware.
// Provides sequence-number-based completion tracking.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};

use crate::mmio::{regs, Mmio};

const RING_ENTRIES: u32 = 256;
const RING_ENTRY_BYTES: usize = 64;
const RING_SIZE: usize = RING_ENTRIES as usize * RING_ENTRY_BYTES;

/// 64-byte command descriptor (matches tpt_cmd_desc_t in tpt_driver.h).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub struct CmdDesc {
    pub opcode: u32,
    pub flags: u32,
    pub kernel_phys: u64,
    pub grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    pub block_x: u32,
    pub block_y: u32,
    pub block_z: u32,
    pub arg_buf_phys: u64,
    pub arg_buf_size: u32,
    pub shared_mem: u32,
    pub completion_phys: u64,
}

const _: () = assert!(
    std::mem::size_of::<CmdDesc>() == 64,
    "CmdDesc must be exactly 64 bytes"
);

pub const CMD_LAUNCH: u32 = 0x01;
pub const CMD_COPY: u32 = 0x02;
pub const CMD_FENCE: u32 = 0x03;

/// Thread-safe command ring manager.
pub struct CommandRing {
    mmio: Arc<Mmio>,
    ring: Mutex<RingState>,
    seq_issued: AtomicU64,
    seq_completed: AtomicU64,
}

struct RingState {
    /// Host-side copy of ring memory (we write here and update HEAD)
    buf: Box<[u8; RING_SIZE]>,
    head: u32,
}

impl CommandRing {
    pub fn new(mmio: Arc<Mmio>) -> Self {
        let ring = RingState {
            buf: Box::new([0u8; RING_SIZE]),
            head: 0,
        };
        CommandRing {
            mmio,
            ring: Mutex::new(ring),
            seq_issued: AtomicU64::new(0),
            seq_completed: AtomicU64::new(0),
        }
    }

    /// Submit one command descriptor; returns the sequence number.
    pub fn submit(&self, desc: &CmdDesc) -> Result<u64> {
        let mut ring = self.ring.lock().unwrap();

        let tail = self.mmio.read32(regs::CMDRING_TAIL);
        let head = ring.head;

        // Check space: full if head+1 == tail (mod RING_ENTRIES)
        if (head + 1) % RING_ENTRIES == tail % RING_ENTRIES {
            bail!("command ring full (head={head} tail={tail})");
        }

        // Write descriptor into local ring buffer
        let slot = (head as usize) * RING_ENTRY_BYTES;
        let desc_bytes = unsafe {
            std::slice::from_raw_parts(desc as *const CmdDesc as *const u8, RING_ENTRY_BYTES)
        };
        ring.buf[slot..slot + RING_ENTRY_BYTES].copy_from_slice(desc_bytes);

        // Advance head and ring doorbell
        ring.head = (head + 1) % RING_ENTRIES;
        self.mmio.write32(regs::CMDRING_HEAD, ring.head);

        let seq = self.seq_issued.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(seq)
    }

    /// Mark a sequence number as completed (called from IRQ path / poll).
    pub fn mark_completed(&self, seq: u64) {
        self.seq_completed.fetch_max(seq, Ordering::SeqCst);
    }

    /// Poll until `seq` is completed or `timeout_ms` elapses.
    pub fn wait(&self, seq: u64, timeout_ms: u64) -> Result<()> {
        let deadline =
            std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms.max(1));

        loop {
            if self.seq_completed.load(Ordering::Acquire) >= seq {
                return Ok(());
            }
            if std::time::Instant::now() >= deadline {
                bail!("wait timeout for seq={seq}");
            }
            // Poll hardware idle flag as a proxy for completion
            if self.mmio.read32(regs::STATUS) & crate::mmio::status_bits::IDLE != 0 {
                self.seq_completed.fetch_max(seq, Ordering::SeqCst);
                return Ok(());
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    pub fn seq_issued(&self) -> u64 {
        self.seq_issued.load(Ordering::Relaxed)
    }
    pub fn seq_completed(&self) -> u64 {
        self.seq_completed.load(Ordering::Acquire)
    }
}

/// Build a LAUNCH command descriptor.
pub fn make_launch(
    kernel_phys: u64,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    arg_buf_phys: u64,
    arg_buf_size: u32,
    shared_mem: u32,
    completion_phys: u64,
) -> CmdDesc {
    CmdDesc {
        opcode: CMD_LAUNCH,
        flags: 0,
        kernel_phys,
        grid_x: grid.0,
        grid_y: grid.1,
        grid_z: grid.2,
        block_x: block.0,
        block_y: block.1,
        block_z: block.2,
        arg_buf_phys,
        arg_buf_size,
        shared_mem,
        completion_phys,
    }
}
