//! Execution streams: FIFO ordering on a single stream and genuine
//! compute/copy overlap with named event barriers on `DualStreams`.
//!
//! Run with: `cargo run -p tpt-runtime --example dual_streams`

use std::sync::{Arc, Mutex};
use tpt_runtime::{DualStreams, Lane, Stream};

fn main() {
    // --- Single stream: strict FIFO execution -------------------------------
    let mut stream = Stream::new("compute");
    let log = Arc::new(Mutex::new(Vec::new()));

    for i in 0..3 {
        let log = log.clone();
        stream.enqueue(Box::new(move || log.lock().unwrap().push(i)));
    }
    assert_eq!(stream.len(), 3);
    stream.run();
    assert!(stream.is_empty());
    println!("single stream order: {:?}", *log.lock().unwrap());

    // --- Dual streams: compute overlaps a slow copy -------------------------
    let mut dual = DualStreams::new();

    let copy_done = Arc::new(Mutex::new(false));
    let flag = copy_done.clone();
    dual.enqueue_copy(Box::new(move || {
        std::thread::sleep(std::time::Duration::from_millis(50));
        *flag.lock().unwrap() = true;
    }));

    let early = Arc::new(Mutex::new(false));
    let probe = early.clone();
    let copy_done2 = copy_done.clone();
    dual.enqueue_compute(Box::new(move || {
        // If lanes were serialized, the 50 ms copy would already be done.
        *probe.lock().unwrap() = !*copy_done2.lock().unwrap();
    }));

    // Event barrier: the compute lane waits until the copy lane records.
    dual.record(Lane::Copy, "copy_finished");
    let synced = Arc::new(Mutex::new(false));
    let after = synced.clone();
    dual.wait(Lane::Compute, "copy_finished");
    dual.enqueue_compute(Box::new(move || {
        *after.lock().unwrap() = *copy_done.lock().unwrap();
    }));

    dual.run();

    assert!(*early.lock().unwrap(), "compute should overlap the copy");
    assert!(
        *synced.lock().unwrap(),
        "barrier should order compute after copy"
    );
    println!("overlap + event barrier both verified");
}
