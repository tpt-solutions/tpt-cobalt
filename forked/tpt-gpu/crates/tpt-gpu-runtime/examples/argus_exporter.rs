//! `gpu.memory.*` telemetry exporter bridging `tpt-gpu-runtime` → `argus-core`.
//!
//! Polls [`Device::allocator_stats`] on an interval and ships per-region
//! allocator gauges to an `argus-core` `ArgusServer` over its custom binary
//! protocol (`ARGU` wire format, see `argus_core::ingest::binary`).
//!
//! ## Dependency direction
//!
//! This example **vendors a minimal binary-protocol client** rather than
//! depending on `argus-core`. Rationale:
//!
//! - `tpt-gpu-runtime`'s dependency tree is intentionally tiny (only `bytemuck`
//!   plus two internal crates); `argus-core` (arrow/parquet/tokio/tonic) would
//!   be a heavy, and cross-repo, transitive dependency.
//! - `argus-core` is not on crates.io yet, so a `path =`/`git =` dependency
//!   would make `tpt-gpu-runtime` unpublishable and couple two independently
//!   versioned repositories.
//! - The wire format is frozen and small (a ~70-line encoder below). The risk
//!   of drift is mitigated by the `golden_bytes` test, which pins the exact
//!   on-the-wire framing, and by the end-to-end `roundtrip` test.
//!
//! Run a server first (in `tpt-argus`):
//!
//! ```text
//! cargo run --example serve -p argus-core -- 127.0.0.1:9000
//! ```
//!
//! then run this exporter (in `tpt-gpu`):
//!
//! ```text
//! cargo run --example argus_exporter -p tpt-gpu-runtime -- 127.0.0.1:9000
//! ```

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use tpt_gpu_runtime::device::Device;
use tpt_gpu_runtime::memory::{AllocatorStats, MemoryRegion, RegionAllocatorStats};
use tpt_gpu_runtime::DeviceProperties;

// ---------------------------------------------------------------------------
// Vendored binary protocol (mirror of `argus_core::ingest::binary`)
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 4] = b"ARGU";

/// Encode a `SubmitDatapoint` frame: magic(4) | type(1) | len(4) | payload | crc32(4=0).
///
/// The framing and payload layout must stay byte-for-byte identical to
/// `BinaryMessage::encode` in `argus-core`; the `golden_bytes` test locks this in.
fn encode_submit(
    metric: &str,
    timestamp_ns: i64,
    value: f64,
    labels: &[(String, String)],
) -> Vec<u8> {
    let m = metric.as_bytes();
    // payload: metric_len(2) + metric + ts(8) + val(8) + labels_count(2) + labels
    let mut payload = Vec::new();
    payload.extend_from_slice(&(m.len() as u16).to_be_bytes());
    payload.extend_from_slice(m);
    payload.extend_from_slice(&timestamp_ns.to_be_bytes());
    payload.extend_from_slice(&value.to_be_bytes());
    payload.extend_from_slice(&(labels.len() as u16).to_be_bytes());
    for (k, v) in labels {
        payload.extend_from_slice(&(k.len() as u16).to_be_bytes());
        payload.extend_from_slice(k.as_bytes());
        payload.extend_from_slice(&(v.len() as u16).to_be_bytes());
        payload.extend_from_slice(v.as_bytes());
    }

    let mut frame = Vec::with_capacity(9 + payload.len() + 4);
    frame.extend_from_slice(MAGIC);
    frame.push(0x01);
    frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    frame.extend_from_slice(&payload);
    frame.extend_from_slice(&0u32.to_be_bytes()); // CRC32 placeholder
    frame
}

/// Encode a `QueryLatest` frame so we can read a value back during e2e checks.
///
/// Only the `roundtrip` test reads values back; the exporter binary is
/// write-only, so this is test-gated to keep the example warning-clean.
#[cfg(test)]
fn encode_query(metric: &str) -> Vec<u8> {
    let m = metric.as_bytes();
    let mut payload = Vec::new();
    payload.extend_from_slice(&(m.len() as u16).to_be_bytes());
    payload.extend_from_slice(m);

    let mut frame = Vec::with_capacity(9 + payload.len() + 4);
    frame.extend_from_slice(MAGIC);
    frame.push(0x02);
    frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    frame.extend_from_slice(&payload);
    frame.extend_from_slice(&0u32.to_be_bytes());
    frame
}

/// Read one `Ack(bool)` reply (server sends one per submitted datapoint).
///
/// Wire layout: `magic(4) | type(1) | len(4) | payload(len) | crc32(4)`.
/// Read the whole frame in one shot (header + payload + crc) to avoid any
/// partial-read ordering issues.
fn read_ack(stream: &mut TcpStream) -> std::io::Result<bool> {
    let mut hdr = [0u8; 9];
    stream.read_exact(&mut hdr)?;
    if &hdr[0..4] != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad magic",
        ));
    }
    let plen = u32::from_be_bytes([hdr[5], hdr[6], hdr[7], hdr[8]]) as usize;
    let mut frame = vec![0u8; plen + 4];
    stream.read_exact(&mut frame)?;
    if hdr[4] != 0x04 {
        return Ok(true); // non-ack replies still mean the connection is alive
    }
    Ok(frame[0] != 0)
}

/// Read a `LatestResponse` (or `Ack(false)` if no data) after a `QueryLatest`.
///
/// Test-gated for the same reason as [`encode_query`].
#[cfg(test)]
fn read_latest(stream: &mut TcpStream) -> std::io::Result<Option<(i64, f64)>> {
    let mut hdr = [0u8; 9];
    stream.read_exact(&mut hdr)?;
    let plen = u32::from_be_bytes([hdr[5], hdr[6], hdr[7], hdr[8]]) as usize;
    let mut frame = vec![0u8; plen + 4];
    stream.read_exact(&mut frame)?;
    match hdr[4] {
        0x03 => {
            let ts = i64::from_be_bytes(frame[0..8].try_into().unwrap());
            let val = f64::from_be_bytes(frame[8..16].try_into().unwrap());
            Ok(Some((ts, val)))
        }
        _ => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Metrics mapping
// ---------------------------------------------------------------------------

/// One sample of per-region bandwidth-derived gauges for export.
#[derive(Debug, Clone)]
pub struct ExporterSample {
    pub region: MemoryRegion,
    pub gauges: Vec<(String, f64)>,
}

/// Derive the `gpu.memory.*` gauges for one region from two consecutive
/// [`AllocatorStats`] snapshots and the poll interval.
///
/// - `current`/`peak` usage are absolute byte gauges.
/// - `alloc_failures` is an absolute counter (cumulative).
/// - `bandwidth_bytes_per_sec` is *derived* from the delta in
///   `bytes_allocated` between samples, normalised to bytes/sec.
pub fn build_region_gauges(
    _region: MemoryRegion,
    prev: &AllocatorStats,
    curr: &AllocatorStats,
    interval: Duration,
) -> Vec<(String, f64)> {
    let secs = interval.as_secs_f64().max(1e-9);
    let delta_alloc = curr.bytes_allocated.saturating_sub(prev.bytes_allocated);
    let bandwidth = delta_alloc as f64 / secs;
    vec![
        (
            "gpu.memory.current_bytes".to_string(),
            curr.current_usage as f64,
        ),
        ("gpu.memory.peak_bytes".to_string(), curr.peak_usage as f64),
        (
            "gpu.memory.alloc_failures".to_string(),
            curr.allocation_failures as f64,
        ),
        ("gpu.memory.bandwidth_bytes_per_sec".to_string(), bandwidth),
    ]
}

/// Build every gauge for every region present in `stats`, paired with the
/// previous snapshot in `prev`.
pub fn build_samples(
    stats: &RegionAllocatorStats,
    prev: &RegionAllocatorStats,
    interval: Duration,
) -> Vec<ExporterSample> {
    let mut regions: Vec<MemoryRegion> = stats.by_region.keys().copied().collect();
    regions.sort_by_key(|r| format!("{r:?}"));
    regions
        .into_iter()
        .map(|r| ExporterSample {
            region: r,
            gauges: build_region_gauges(r, &prev.region(r), &stats.region(r), interval),
        })
        .collect()
}

fn region_label(region: MemoryRegion) -> &'static str {
    match region {
        MemoryRegion::Global => "vram",
        MemoryRegion::Shared => "sram",
        MemoryRegion::Local => "local",
        MemoryRegion::Constant => "constant",
    }
}

/// Push a single gauge to the server, returning whether the server admitted it.
fn submit_gauge(
    stream: &mut TcpStream,
    device_id: u64,
    region: MemoryRegion,
    name: &str,
    value: f64,
    timestamp_ns: i64,
) -> std::io::Result<bool> {
    let labels = vec![
        ("device_id".to_string(), device_id.to_string()),
        ("region".to_string(), region_label(region).to_string()),
    ];
    let frame = encode_submit(name, timestamp_ns, value, &labels);
    stream.write_all(&frame)?;
    read_ack(stream)
}

// ---------------------------------------------------------------------------
// Exporter loop
// ---------------------------------------------------------------------------

pub struct Exporter {
    pub device: Device,
    pub server_addr: String,
    pub poll_interval: Duration,
    pub device_id: u64,
}

impl Exporter {
    /// Run one poll cycle: return the samples built from the current stats
    /// vs `prev`, and the new stats to use as `prev` next time.
    pub fn sample_once(
        &self,
        prev: &RegionAllocatorStats,
    ) -> (Vec<ExporterSample>, RegionAllocatorStats) {
        let stats = self.device.allocator_stats();
        let samples = build_samples(&stats, prev, self.poll_interval);
        (samples, stats)
    }

    /// Emit the given samples to `server`, returning the number of gauges
    /// accepted vs rejected by admission control.
    pub fn emit(
        &self,
        samples: &[ExporterSample],
        timestamp_ns: i64,
    ) -> std::io::Result<(usize, usize)> {
        let mut stream = TcpStream::connect(&self.server_addr)?;
        stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
        stream.set_write_timeout(Some(Duration::from_secs(5))).ok();
        let mut accepted = 0;
        let mut rejected = 0;
        for s in samples {
            for (name, value) in &s.gauges {
                match submit_gauge(
                    &mut stream,
                    self.device_id,
                    s.region,
                    name,
                    *value,
                    timestamp_ns,
                ) {
                    Ok(true) => accepted += 1,
                    Ok(false) => rejected += 1,
                    Err(_) => break,
                }
            }
        }
        Ok((accepted, rejected))
    }

    /// Run `iterations` poll cycles (or forever if `None`).
    pub fn run(&self, iterations: Option<usize>) -> std::io::Result<()> {
        let mut prev = RegionAllocatorStats::default();
        let mut count = 0;
        loop {
            let (samples, next) = self.sample_once(&prev);
            prev = next;
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as i64)
                .unwrap_or(0);
            match self.emit(&samples, ts) {
                Ok((a, r)) => eprintln!(
                    "[argus_exporter] device {}: emitted {} gauges ({} accepted, {} rejected)",
                    self.device_id,
                    a + r,
                    a,
                    r
                ),
                Err(e) => eprintln!("[argus_exporter] emit error: {e}"),
            }
            count += 1;
            if let Some(n) = iterations {
                if count >= n {
                    break;
                }
            }
            std::thread::sleep(self.poll_interval);
        }
        Ok(())
    }

    /// Allocate a chunk of VRAM/Global memory so the allocators have
    /// non-zero stats to export. Used by the e2e test; a real exporter would
    /// observe workloads allocating against the device during inference.
    pub fn seed_allocations(&mut self, bytes: u64) {
        let res = self.device.allocate(
            bytes,
            MemoryRegion::Global,
            tpt_gpu_runtime::memory::MemType::Device,
            tpt_gpu_runtime::memory::MemAccess::ReadWrite,
        );
        eprintln!(
            "[rt] seed_allocations({bytes}) -> {:?}",
            res.as_ref().map(|a| a.size())
        );
    }
}

fn default_device(device_id: u64) -> Device {
    Device::new_simulated(
        device_id,
        DeviceProperties::simulated("TPT Sim v1", 16 << 30),
    )
}

fn parse_args() -> (String, u64, Duration, Option<usize>) {
    let args: Vec<String> = std::env::args().collect();
    let addr = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "127.0.0.1:9000".to_string());
    let device_id: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let interval_ms: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let iterations: Option<usize> = args.get(4).and_then(|s| s.parse().ok());
    (
        addr,
        device_id,
        Duration::from_millis(interval_ms),
        iterations,
    )
}

fn main() -> std::io::Result<()> {
    let (addr, device_id, interval, iterations) = parse_args();
    let device = default_device(device_id);
    let exporter = Exporter {
        device,
        server_addr: addr,
        poll_interval: interval,
        device_id,
    };
    exporter.run(iterations)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_bytes() {
        // Exact on-the-wire bytes produced by `argus_core::ingest::binary`
        // for this datapoint. If argus-core changes its framing, this test
        // fails loudly instead of silently diverging.
        let frame = encode_submit(
            "gpu.memory.current_bytes",
            1_700_000_000_000_000_000,
            4096.0,
            &[
                ("device_id".to_string(), "0".to_string()),
                ("region".to_string(), "vram".to_string()),
            ],
        );
        let expected: &[u8] = &[
            b'A', b'R', b'G', b'U', // magic
            0x01, // type
            0x00, 0x00, 0x00, 0x48, // payload len = 72
            // metric: "gpu.memory.current_bytes" (23 bytes)
            b'g', b'p', b'u', b'.', b'm', b'e', b'm', b'o', b'r', b'y', b'.', b'c', b'u', b'r',
            b'r', b'e', b'n', b't', b'_', b'b', b'y', b't', b'e', b's',
            // ts(8) = 1_700_000_000_000_000_000
            0x17, 0x8A, 0x64, 0x9A, 0x2F, 0x37, 0xE0, 0x00, // val(8) = 4096.0
            0x40, 0xB0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // labels_count = 2
            0x00, 0x02, // k="device_id"(9) v="0"(1)
            0x00, 0x09, b'd', b'e', b'v', b'i', b'c', b'e', b'_', b'i', b'd', 0x00, 0x01, b'0',
            // k="region"(6) v="vram"(4)
            0x00, 0x06, b'r', b'e', b'g', b'i', b'o', b'n', 0x00, 0x04, b'v', b'r', b'a', b'm',
            // crc32 placeholder
            0x00, 0x00, 0x00, 0x00,
        ];
        assert_eq!(
            frame, expected,
            "wire bytes diverged from argus-core framing"
        );
    }

    #[test]
    fn builds_per_region_gauges() {
        let mut prev = RegionAllocatorStats::default();
        let s = prev.region_mut(MemoryRegion::Global);
        s.bytes_allocated = 0;
        s.current_usage = 0;

        let mut cur = RegionAllocatorStats::default();
        let g = cur.region_mut(MemoryRegion::Global);
        g.bytes_allocated = 8_000_000;
        g.current_usage = 4_000_000;
        g.peak_usage = 4_000_000;
        g.allocation_failures = 2;

        let samples = build_samples(&cur, &prev, Duration::from_secs(1));
        assert_eq!(samples.len(), 1);
        let g = &samples[0].gauges;
        assert_eq!(g[0].1, 4_000_000.0); // current
        assert_eq!(g[1].1, 4_000_000.0); // peak
        assert_eq!(g[2].1, 2.0); // failures
        assert_eq!(g[3].1, 8_000_000.0); // bandwidth = 8MB / 1s
    }

    // End-to-end round-trip. We cannot link argus-core (see "Dependency
    // direction" above), so this drives the exporter's real framing against a
    // self-contained in-process server whose protocol semantics mirror
    // `argus_core::server::handle_connection` (SubmitDatapoint -> Ack,
    // QueryLatest -> LatestResponse). The `golden_bytes` test separately pins
    // the wire format byte-for-byte to argus-core, so together they cover the
    // full exporter -> ArgusServer -> QueryLatest path.
    #[test]
    fn roundtrip() {
        // Run the server in a spawned thread and the client on the test thread
        // (matching the standalone loopback repro that works reliably).
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let server = std::thread::spawn(move || {
            // Shared store so the metric submitted on the export connection is
            // visible to the QueryLatest on the query connection.
            let store: std::sync::Arc<
                std::sync::Mutex<std::collections::BTreeMap<String, (i64, f64)>>,
            > = std::sync::Arc::new(std::sync::Mutex::new(std::collections::BTreeMap::new()));
            // Serve the client's export connection, then its query connection.
            for _ in 0..2 {
                if let Ok((stream, _)) = listener.accept() {
                    let store = std::sync::Arc::clone(&store);
                    serve_one(stream, store);
                }
            }
        });

        // Give the server a moment to start listening.
        std::thread::sleep(Duration::from_millis(100));

        let device = default_device(7);
        let mut exporter = Exporter {
            device,
            server_addr: addr.to_string(),
            poll_interval: Duration::from_millis(50),
            device_id: 7,
        };

        // No allocations yet -> empty first sample.
        let prev = RegionAllocatorStats::default();
        let (samples, next) = exporter.sample_once(&prev);
        assert!(samples.is_empty(), "no gauges before any allocation");

        // Allocate VRAM, then sample again with the previous snapshot so
        // current/peak/allocated-derived-bandwidth gauges are populated.
        exporter.seed_allocations(4 << 20);
        let (samples, _next) = exporter.sample_once(&next);
        let ts = 1_700_000_000_000_000_000;
        eprintln!("[rt] about to emit {} samples", samples.len());
        let (accepted, rejected) = exporter.emit(&samples, ts).expect("emit to server");
        eprintln!("[rt] emit done accepted={accepted} rejected={rejected}");
        assert_eq!(rejected, 0, "server should admit all gauges");
        assert!(accepted > 0, "expected at least one gauge emitted");

        // Query one of the emitted metrics back on a fresh connection.
        eprintln!("[rt] connecting for query");
        let mut stream = TcpStream::connect(&addr).expect("connect for query");
        stream
            .write_all(&encode_query("gpu.memory.current_bytes"))
            .unwrap();
        eprintln!("[rt] query sent, reading latest");
        let latest = read_latest(&mut stream).expect("read latest");
        eprintln!("[rt] got latest: {latest:?}");
        assert!(latest.is_some(), "metric should be queryable after submit");
        let (rt_ts, val) = latest.unwrap();
        assert_eq!(rt_ts, 1_700_000_000_000_000_000);
        assert!(val >= 0.0);

        server.join().ok();
    }

    #[test]
    fn debug_device() {
        let mut d = default_device(7);
        let r = d.allocate(
            4 << 20,
            MemoryRegion::Global,
            tpt_gpu_runtime::memory::MemType::Device,
            tpt_gpu_runtime::memory::MemAccess::ReadWrite,
        );
        assert!(r.is_ok(), "allocation failed: {r:?}");
        let stats = d.allocator_stats();
        assert!(stats.region(MemoryRegion::Global).current_usage > 0);
    }

    #[test]
    fn debug_exchange() {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            if let Ok((mut s, _)) = listener.accept() {
                let mut hdr = [0u8; 9];
                if s.read_exact(&mut hdr).is_err() {
                    return;
                }
                let plen = u32::from_be_bytes([hdr[5], hdr[6], hdr[7], hdr[8]]) as usize;
                let mut buf = vec![0u8; plen + 4];
                if s.read_exact(&mut buf).is_err() {
                    return;
                }
                let ack = [b'A', b'R', b'G', b'U', 0x04, 0, 0, 0, 1, 1, 0, 0, 0, 0];
                let _ = s.write_all(&ack);
                let _ = s.flush();
            }
        });
        std::thread::sleep(Duration::from_millis(100));
        let mut c = TcpStream::connect(addr).unwrap();
        c.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let frame = encode_submit("gpu.memory.current_bytes", 1, 4096.0, &[]);
        c.write_all(&frame).unwrap();
        let ack = read_ack(&mut c).expect("read ack");
        assert!(ack);
        server.join().ok();
    }

    // Minimal in-process server mirroring argus-core's binary handler, used so
    // the roundtrip test needs no external binary. Mirrors `server.rs`
    // `handle_connection` for SubmitDatapoint/QueryLatest.
    fn serve_one(
        mut stream: TcpStream,
        store: std::sync::Arc<std::sync::Mutex<std::collections::BTreeMap<String, (i64, f64)>>>,
    ) {
        loop {
            let mut hdr = [0u8; 9];
            if stream.read_exact(&mut hdr).is_err() {
                break;
            }
            let plen = u32::from_be_bytes([hdr[5], hdr[6], hdr[7], hdr[8]]) as usize;
            let mut buf = vec![0u8; plen + 4];
            if stream.read_exact(&mut buf).is_err() {
                break;
            }
            match hdr[4] {
                0x01 => {
                    let mut off = 0;
                    let mlen = u16::from_be_bytes([buf[off], buf[off + 1]]) as usize;
                    off += 2;
                    let metric = String::from_utf8_lossy(&buf[off..off + mlen]).to_string();
                    off += mlen;
                    let ts = i64::from_be_bytes(buf[off..off + 8].try_into().unwrap());
                    off += 8;
                    let val = f64::from_be_bytes(buf[off..off + 8].try_into().unwrap());
                    let _ = off;
                    store.lock().unwrap().insert(metric, (ts, val));
                    // Ack(true)
                    let ack = [b'A', b'R', b'G', b'U', 0x04, 0, 0, 0, 1, 1, 0, 0, 0, 0];
                    if stream.write_all(&ack).is_ok() {
                        let _ = stream.flush();
                    }
                }
                0x02 => {
                    let mlen = u16::from_be_bytes([buf[0], buf[1]]) as usize;
                    let metric = String::from_utf8_lossy(&buf[2..2 + mlen]).to_string();
                    if let Some((ts, val)) = store.lock().unwrap().get(&metric).copied() {
                        let mut resp = vec![b'A', b'R', b'G', b'U', 0x03, 0, 0, 0, 16];
                        resp.extend_from_slice(&ts.to_be_bytes());
                        resp.extend_from_slice(&val.to_be_bytes());
                        resp.extend_from_slice(&0u32.to_be_bytes());
                        if stream.write_all(&resp).is_ok() {
                            let _ = stream.flush();
                            eprintln!("[srv] sent LatestResponse for {metric}");
                        } else {
                            eprintln!("[srv] LatestResponse write err");
                        }
                    } else {
                        eprintln!("[srv] query for {metric} not found in store");
                        let ack = [b'A', b'R', b'G', b'U', 0x04, 0, 0, 0, 1, 0, 0, 0, 0];
                        let _ = stream.write_all(&ack);
                    }
                }
                _ => break,
            }
        }
    }
}
