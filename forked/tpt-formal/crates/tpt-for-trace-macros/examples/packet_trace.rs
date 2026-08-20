//! Example: instrument a small state machine with the global tracer.
//!
//! The macros `trace_event!` / `trace_point!` / `trace_value!` all funnel through
//! one global `TraceFn`, so they work in `no_std`. Here a fake packet processor
//! emits lifecycle events; a custom tracer prints them, and (under `std`) a
//! `RingTrace` captures them for post-hoc analysis. `ArrayWriter` shows the
//! allocation-free, truncating buffer used by the macros themselves.

use core::fmt::Write;
use tpt_for_trace_macros::{
    emit, set_tracer, trace_event, trace_point, trace_value, tracer, ArrayWriter, TraceFn,
};

/// Mimic a packet processor advancing through states, emitting a trace line at
/// each step. In a real HAL this would be a `no_std` driver.
fn process(state: &mut u32) {
    trace_event!("pkt", "enter process, state = {}", *state);
    *state += 1;
    trace_value!("pkt", *state);
    trace_point!("pkt::done");
}

fn main() {
    // 1) A custom, allocation-free tracer (exactly what a `no_std` runtime
    //    installs). `set_tracer` stores a plain function pointer; `tracer()`
    //    lets callers confirm what is installed.
    let custom: TraceFn = |target, msg| println!("[{target}] {msg}");
    set_tracer(custom);
    println!("installed tracer present: {}", tracer().is_some());
    process(&mut 0);

    // 2) `emit` drives the same global tracer directly, without a macro.
    emit("manual", "emitted without a macro");

    // 3) `ArrayWriter` renders formatted text into a fixed buffer with no
    //    allocation and truncates once the buffer is full (no_std-friendly).
    let mut buf = [0u8; 16];
    let mut w = ArrayWriter::new(&mut buf);
    let _ = write!(&mut w, "status=ok, extra ignored");
    println!("array writer (cap 16): {:?} (len {})", w.as_str(), w.len());
    assert_eq!(w.as_str(), "status=ok, extra");

    // 4) Under `std`, capture events into a bounded ring for analysis.
    #[cfg(feature = "std")]
    {
        use tpt_for_trace_macros::{uninstall, RingTrace};
        let ring = RingTrace::install(8);
        process(&mut 10);
        trace_event!("pkt", "post-analysis marker");
        // RingTrace records are queryable by target/seq.
        println!("ring captured {} record(s)", ring.len());
        let pkt_count = ring.records().iter().filter(|r| r.target == "pkt").count();
        println!("records tagged 'pkt': {pkt_count}");
        let first = &ring.records()[0];
        println!(
            "first record: #{} [{}] {}",
            first.seq, first.target, first.msg
        );
        assert!(ring.len() >= 2);
        uninstall();
    }
}
