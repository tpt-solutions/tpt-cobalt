// protocol.rs — Unix socket JSON protocol types

use serde::{Deserialize, Serialize};

/// Request sent by clients to /run/tptd.sock
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    GetInfo,
    AllocMem {
        size: u64,
        flags: u32,
    },
    FreeMem {
        handle: u64,
    },
    SubmitKernel {
        kernel_handle: u64, // VRAM buffer handle for kernel binary
        arg_handle: u64,    // VRAM buffer handle for argument buffer
        arg_size: u32,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
    },
    WaitKernel {
        seq_no: u64,
        timeout_ms: u64,
    },
    QueryPerf,
    Reset,
}

/// Response sent back to clients
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Response {
    Ok(OkPayload),
    Err { error: String, code: i32 },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OkPayload {
    Info {
        version_major: u32,
        version_minor: u32,
        vram_bytes: u64,
        num_sm: u32,
        warp_lanes: u32,
        caps: u32,
    },
    Alloc {
        handle: u64,
        phys_addr: u64,
    },
    Free,
    Submit {
        seq_no: u64,
    },
    Wait {
        status: u32,
    },
    Perf {
        inst_retired: u64,
        core_cycles: u64,
        l1d_misses: u64,
        l2_misses: u64,
    },
    Reset,
}

impl Response {
    pub fn err(msg: impl Into<String>, code: i32) -> Self {
        Response::Err {
            error: msg.into(),
            code,
        }
    }
}
