//! Cross-process tensor mailbox: publish with atomic rename, wait, load.
//! The same pattern works across real processes sharing a directory.
//!
//! Run with: `cargo run -p tpt-hub --example ipc_tensor_mailbox`

use std::path::PathBuf;
use std::time::Duration;

use tpt_hub::{ipc_available, ipc_publish, ipc_wait_for};
use tpt_tensor::Tensor;

fn main() {
    // Any shared directory works — use a per-run temp dir for the demo.
    let dir =
        PathBuf::from(std::env::temp_dir().join(format!("tpt-hub-mailbox-{}", std::process::id())));

    let activation = Tensor::from_typed(vec![0.1_f64, 0.5, -0.3])
        .reshape(&[3])
        .unwrap();

    // Producer side: publish atomically (readers never see partial data).
    // Note: names are flat file names (no `/`).
    let path = ipc_publish(&dir, "layer0_activations", &activation).unwrap();
    println!("published to {}", path.display());

    // Consumer side: block until the tensor appears (no-op here since it
    // already exists), then load it.
    let received = ipc_wait_for(&dir, "layer0_activations", Duration::from_secs(5)).unwrap();
    assert_eq!(received.to_vec::<f64>().unwrap(), vec![0.1, 0.5, -0.3]);
    println!("wait_for + load OK");

    // Directories can hold any number of named tensors.
    let names = ipc_available(&dir);
    assert_eq!(names, vec!["layer0_activations".to_string()]);
    println!("available tensors in mailbox: {names:?}");

    // Clean up the demo directory.
    std::fs::remove_dir_all(&dir).ok();
}
