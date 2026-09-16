//! SafeTensors round-trip plus the TPTB container and JSON debug format.
//!
//! Run with: `cargo run -p tpt-hub --example safetensors_roundtrip`

use std::collections::HashMap;

use tpt_hub::{
    load_safetensors, load_tptb, save_safetensors, save_tptb, tensor_from_json_debug,
    tensor_to_json_debug,
};
use tpt_tensor::Tensor;

fn main() {
    let w = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0])
        .reshape(&[2, 2])
        .unwrap();

    // --- SafeTensors: interop with PyTorch / HuggingFace -------------------
    let bytes = save_safetensors(&[("weights", &w)]);
    println!("safetensors payload: {} bytes", bytes.len());

    let loaded: HashMap<String, Tensor> = load_safetensors(&bytes).unwrap();
    assert_eq!(loaded["weights"].shape(), &[2, 2]);
    assert_eq!(
        loaded["weights"].to_vec::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    println!("safetensors round-trip OK");

    // --- TPTB: minimal self-describing binary container --------------------
    let tptb = save_tptb(&w);
    let restored = load_tptb(&tptb).unwrap();
    assert_eq!(restored.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    println!("tptb round-trip OK ({} bytes)", tptb.len());

    // --- JSON debug format: human-readable inspection -----------------------
    let json = tensor_to_json_debug(&w);
    println!("json debug:\n{json}");
    let rebuilt = tensor_from_json_debug(&json).unwrap();
    assert_eq!(rebuilt.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}
