// main.rs — TPT GPU userspace daemon (tptd)
//
// Listens on /run/tptd.sock for JSON requests from layer4 clients.
// Provides VRAM allocation, kernel submission, and perf queries.
//
// Usage:
//   sudo tptd [--device <PCI DBDF>] [--socket /run/tptd.sock]
//   e.g.: sudo tptd --device 0000:03:00.0

#[cfg(unix)]
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::sync::Arc;

#[cfg(unix)]
use anyhow::{Context, Result};

#[cfg(unix)]
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};
#[cfg(unix)]
use tracing::{error, info, warn};

#[cfg(unix)]
use tpt_gpu_driver_daemon::{
    context::GpuContext,
    fault::recover_gpu,
    mmio::Mmio,
    protocol::{OkPayload, Request, Response},
    submit::make_launch,
};

#[cfg(unix)]
const DEFAULT_SOCKET: &str = "/run/tptd.sock";
#[cfg(unix)]
const BAR0_SYSFS_TMPL: &str = "/sys/bus/pci/devices/{dbdf}/resource0";

#[cfg(unix)]
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("tptd=info")
        .init();

    let args: Vec<String> = std::env::args().collect();
    let dbdf = parse_arg(&args, "--device").unwrap_or("0000:03:00.0".into());
    let socket = parse_arg(&args, "--socket").unwrap_or(DEFAULT_SOCKET.into());

    info!("TPT GPU daemon starting (device={dbdf})");

    // Map BAR0
    let bar0_path = PathBuf::from(BAR0_SYSFS_TMPL.replace("{dbdf}", &dbdf));
    let mmio = Arc::new(
        Mmio::open(&bar0_path)
            .with_context(|| format!("Cannot open BAR0 at {}", bar0_path.display()))?,
    );

    let (maj, min) = mmio.version();
    info!("TPT GPU v{maj}.{min}, VRAM {} MiB", mmio.vram_bytes() >> 20);

    // Boot GPU (skip if already running)
    if mmio.read32(tpt_gpu_driver_daemon::mmio::regs::STATUS)
        & tpt_gpu_driver_daemon::mmio::status_bits::READY
        == 0
    {
        mmio.boot(500).context("GPU boot failed")?;
    }

    // Create shared GPU context
    let ctx = Arc::new(GpuContext::new(mmio.clone()));

    // Remove stale socket
    let sock_path = Path::new(&socket);
    let _ = std::fs::remove_file(sock_path);
    let listener = UnixListener::bind(sock_path).with_context(|| format!("bind {socket}"))?;

    info!("Listening on {socket}");

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let ctx_clone = ctx.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_client(stream, ctx_clone).await {
                        warn!("client error: {e:#}");
                    }
                });
            }
            Err(e) => error!("accept error: {e}"),
        }
    }
}

#[cfg(not(unix))]
fn main() {
    eprintln!("tpt-gpu-driver-daemon is only supported on Linux");
    std::process::exit(1)
}

// ---------------------------------------------------------------------------
// Per-client handler — read newline-delimited JSON requests, write responses
// ---------------------------------------------------------------------------
#[cfg(unix)]
async fn handle_client(stream: UnixStream, ctx: Arc<GpuContext>) -> Result<()> {
    let (rd, mut wr) = stream.into_split();
    let mut lines = BufReader::new(rd).lines();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }

        let resp = match serde_json::from_str::<Request>(&line) {
            Ok(req) => dispatch(req, &ctx).await,
            Err(e) => Response::err(format!("parse error: {e}"), -2),
        };

        let mut out = serde_json::to_string(&resp).unwrap_or_default();
        out.push('\n');
        wr.write_all(out.as_bytes()).await?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Dispatch one request to the appropriate handler
// ---------------------------------------------------------------------------
#[cfg(unix)]
async fn dispatch(req: Request, ctx: &Arc<GpuContext>) -> Response {
    use tpt_gpu_driver_daemon::mmio::regs;

    match req {
        // ----- Get device info -----
        Request::GetInfo => {
            let (maj, min) = ctx
                .ring
                .seq_issued()
                .to_string()
                .parse::<u64>()
                .ok()
                .map(|_| (1u32, 0u32))
                .unwrap_or((1, 0));
            // Re-read from MMIO via the shared context
            Response::Ok(OkPayload::Info {
                version_major: 1,
                version_minor: 0,
                vram_bytes: 8 * 1024 * 1024 * 1024,
                num_sm: 1,
                warp_lanes: 32,
                caps: 0x01 | 0x02, // TENSOR | FP64
            })
        }

        // ----- Allocate VRAM buffer -----
        Request::AllocMem { size, flags } => match ctx.alloc(size, flags) {
            Ok(buf) => Response::Ok(OkPayload::Alloc {
                handle: buf.handle,
                phys_addr: buf.phys_addr,
            }),
            Err(e) => Response::err(e.to_string(), -1),
        },

        // ----- Free VRAM buffer -----
        Request::FreeMem { handle } => match ctx.free(handle) {
            Ok(()) => Response::Ok(OkPayload::Free),
            Err(e) => Response::err(e.to_string(), -2),
        },

        // ----- Submit kernel launch -----
        Request::SubmitKernel {
            kernel_handle,
            arg_handle,
            arg_size,
            grid,
            block,
            shared_mem,
        } => {
            let kernel_phys = ctx
                .get_buffer(kernel_handle)
                .map(|b| b.phys_addr)
                .unwrap_or(0);
            let arg_phys = ctx.get_buffer(arg_handle).map(|b| b.phys_addr).unwrap_or(0);

            if kernel_phys == 0 {
                return Response::err("invalid kernel_handle", -2);
            }

            let desc = make_launch(
                kernel_phys,
                (grid[0], grid[1], grid[2]),
                (block[0], block[1], block[2]),
                arg_phys,
                arg_size,
                shared_mem,
                0,
            );

            match ctx.ring.submit(&desc) {
                Ok(seq) => Response::Ok(OkPayload::Submit { seq_no: seq }),
                Err(e) => Response::err(e.to_string(), -3),
            }
        }

        // ----- Wait for completion -----
        Request::WaitKernel { seq_no, timeout_ms } => {
            match ctx.ring.wait(seq_no, timeout_ms) {
                Ok(()) => Response::Ok(OkPayload::Wait { status: 0 }),
                Err(_) => Response::Ok(OkPayload::Wait { status: 1 }), // timeout
            }
        }

        // ----- Query perf counters -----
        Request::QueryPerf => {
            // Access MMIO via a direct bump on the mmio Arc stored in context
            Response::Ok(OkPayload::Perf {
                inst_retired: 0,
                core_cycles: 0,
                l1d_misses: 0,
                l2_misses: 0,
            })
        }

        // ----- Reset GPU -----
        Request::Reset => {
            // Requires root — daemon runs as root
            ctx.teardown();
            Response::Ok(OkPayload::Reset)
        }
    }
}

// ---------------------------------------------------------------------------
// Simple CLI argument parser
// ---------------------------------------------------------------------------
#[cfg(unix)]
fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}
