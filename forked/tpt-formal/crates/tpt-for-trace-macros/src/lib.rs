#![no_std]
//! Trace / instrumentation macros for verification.
//!
//! A single global tracer function receives formatted trace lines produced by
//! the [`trace_event!`], [`trace_point!`], and [`trace_value!`] macros. The
//! tracer is a plain function pointer, so it works in `no_std` (set it from
//! your HAL or runtime). A `std`-gated [`RingTrace`] collector is provided for
//! capturing traces in tests and tooling.

#[cfg(feature = "std")]
extern crate std;

use core::fmt;
use core::sync::atomic::{AtomicUsize, Ordering};

/// Default capacity of the per-event format buffer.
pub const TRACE_BUF_LEN: usize = 256;

/// A tracer callback: receives a target label and a formatted message.
pub type TraceFn = fn(target: &str, msg: &str);

static TRACE_FN: AtomicUsize = AtomicUsize::new(0);

/// Install a global tracer. Passing a function that does nothing disables
/// tracing; the default (no tracer) simply drops events.
pub fn set_tracer(f: TraceFn) {
    TRACE_FN.store(f as usize, Ordering::Relaxed);
}

/// The currently installed tracer, if any.
pub fn tracer() -> Option<TraceFn> {
    let p = TRACE_FN.load(Ordering::Relaxed);
    if p == 0 {
        None
    } else {
        // Safe: the stored value is a `TraceFn` pointer set via `set_tracer`.
        Some(unsafe { core::mem::transmute::<usize, TraceFn>(p) })
    }
}

/// Emit a trace line to the installed tracer (no-op if none is set).
pub fn emit(target: &str, msg: &str) {
    if let Some(f) = tracer() {
        f(target, msg);
    }
}

/// A `fmt::Write` sink backed by a fixed byte buffer (no allocation).
pub struct ArrayWriter<'a> {
    buf: &'a mut [u8],
    len: usize,
}

impl<'a> ArrayWriter<'a> {
    /// Wrap `buf` for formatted writing.
    pub fn new(buf: &'a mut [u8]) -> Self {
        ArrayWriter { buf, len: 0 }
    }

    /// Number of bytes written so far.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether nothing has been written yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// View the written bytes as a string slice.
    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.buf[..self.len]).unwrap_or("")
    }
}

impl<'a> fmt::Write for ArrayWriter<'a> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let bytes = s.as_bytes();
        let space = self.buf.len().saturating_sub(self.len);
        let take = space.min(bytes.len());
        self.buf[self.len..self.len + take].copy_from_slice(&bytes[..take]);
        self.len += take;
        Ok(())
    }
}

/// Emit a formatted trace event to `target`.
///
/// The message is rendered into a fixed `TRACE_BUF_LEN` buffer; longer
/// messages are truncated.
#[macro_export]
macro_rules! trace_event {
    ($target:expr, $($arg:tt)*) => {{
        let mut __buf = [0u8; $crate::TRACE_BUF_LEN];
        let __len = {
            let mut __w = $crate::ArrayWriter::new(&mut __buf);
            let _ = ::core::fmt::Write::write_fmt(
                &mut __w,
                ::core::format_args!($($arg)*),
            );
            __w.len()
        };
        let __msg = match ::core::str::from_utf8(&__buf[..__len]) {
            ::core::result::Result::Ok(s) => s,
            ::core::result::Result::Err(_) => "",
        };
        $crate::emit($target, __msg);
    }};
}

/// Emit a zero-argument trace point.
#[macro_export]
macro_rules! trace_point {
    ($target:expr) => {
        $crate::trace_event!($target, "")
    };
}

/// Emit a trace line showing the `Debug` representation of `value`.
#[macro_export]
macro_rules! trace_value {
    ($target:expr, $value:expr) => {
        $crate::trace_event!($target, "{:?}", $value)
    };
}

#[cfg(feature = "std")]
mod ring {
    use super::*;
    use std::boxed::Box;
    use std::string::{String, ToString};
    use std::sync::Mutex;
    use std::vec::Vec;

    /// A captured trace record.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Record {
        /// Monotonic sequence number.
        pub seq: u64,
        /// Target label.
        pub target: String,
        /// Formatted message.
        pub msg: String,
    }

    /// The shared, mutex-protected ring state behind the static.
    struct RingState {
        records: Vec<Record>,
        capacity: usize,
        seq: u64,
    }

    static RING: Mutex<Option<RingState>> = Mutex::new(None);

    fn ring_tracer(target: &str, msg: &str) {
        if let Ok(mut guard) = RING.lock() {
            if let Some(state) = guard.as_mut() {
                state.records.push(Record {
                    seq: state.seq,
                    target: target.to_string(),
                    msg: msg.to_string(),
                });
                state.seq += 1;
                if state.records.len() > state.capacity {
                    state.records.remove(0);
                }
            }
        }
    }

    /// A thin handle to the installed global ring tracer. All reads/writes go
    /// through the `Mutex`-protected static, so no `unsafe` and no aliased
    /// `&mut` is ever manufactured — concurrent `trace_event!` calls are safe.
    pub struct RingTrace {
        _private: (),
    }

    impl RingTrace {
        /// Install a ring collector with the given `capacity` as the global
        /// tracer. Returns a handle whose methods read/write the shared state
        /// under a mutex.
        pub fn install(capacity: usize) -> &'static RingTrace {
            {
                let mut guard = RING.lock().expect("ring mutex poisoned");
                *guard = Some(RingState {
                    records: Vec::new(),
                    capacity: capacity.max(1),
                    seq: 0,
                });
            }
            set_tracer(ring_tracer);
            // Leak a marker handle for the static lifetime.
            Box::leak(Box::new(RingTrace { _private: () }))
        }

        /// Snapshot of captured records (owned: a borrow can't outlive the
        /// dropped mutex guard).
        pub fn records(&self) -> Vec<Record> {
            RING.lock()
                .map(|g| g.as_ref().map(|s| s.records.clone()).unwrap_or_default())
                .unwrap_or_default()
        }

        /// Number of captured records.
        pub fn len(&self) -> usize {
            RING.lock()
                .map(|g| g.as_ref().map(|s| s.records.len()).unwrap_or(0))
                .unwrap_or(0)
        }

        /// Whether no records have been captured.
        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }

        /// Remove all captured records.
        pub fn clear(&self) {
            if let Ok(mut guard) = RING.lock() {
                if let Some(state) = guard.as_mut() {
                    state.records.clear();
                    state.seq = 0;
                }
            }
        }
    }

    /// Uninstall the ring tracer, restoring the no-tracer state.
    pub fn uninstall() {
        if let Ok(mut guard) = RING.lock() {
            *guard = None;
        }
        set_tracer(nop_tracer);
    }

    fn nop_tracer(_target: &str, _msg: &str) {}
}

#[cfg(feature = "std")]
pub use ring::{uninstall, Record, RingTrace};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_writer_truncates() {
        let mut buf = [0u8; 8];
        let mut w = ArrayWriter::new(&mut buf);
        let _ = core::fmt::Write::write_str(&mut w, "hello world");
        assert_eq!(w.as_str(), "hello wo");
    }

    #[cfg(feature = "std")]
    #[test]
    fn ring_collects_events() {
        let ring = RingTrace::install(8);
        trace_event!("test", "value = {}", 42);
        trace_value!("test", [1, 2, 3]);
        assert_eq!(ring.len(), 2);
        assert_eq!(ring.records()[0].msg, "value = 42");
        uninstall();
        assert!(RingTrace::install(1).records().is_empty());
    }
}
