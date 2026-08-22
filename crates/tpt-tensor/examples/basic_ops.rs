//! Basic tensor ops: construction, reshape, element-wise math, matmul,
//! reductions, and metadata inspection.
//!
//! Run with: `cargo run -p tpt-tensor --example basic_ops`

use tpt_tensor::{Device, DType, Tensor};

fn main() {
    // Build from flat data and reshape (row-major).
    let a = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0])
        .reshape(&[2, 2])
        .unwrap();
    let b = Tensor::ones(&[2, 2], Device::Cpu);

    // Element-wise ops.
    let sum = a.add(&b);
    let prod = a.mul(&b);
    println!("a + 1    = {:?}", sum.to_vec::<f64>().unwrap());
    println!("a * 1    = {:?}", prod.to_vec::<f64>().unwrap());

    // Matrix multiply and transpose.
    let m = a.matmul(&b);
    println!("a @ 1    = {:?} (shape {:?})", m.to_vec::<f64>().unwrap(), m.shape());
    println!("a^T      = {:?}", a.transpose().to_vec::<f64>().unwrap());

    // Reductions.
    println!("sum(a)   = {}", a.sum_all().to_vec::<f64>().unwrap()[0]);
    println!("mean(a)  = {}", a.mean_all().to_vec::<f64>().unwrap()[0]);

    // Metadata.
    assert_eq!(a.dtype(), DType::F64);
    assert_eq!(a.shape(), &[2, 2]);
    assert_eq!(a.ndim(), 2);
    assert_eq!(a.numel(), 4);
    assert_eq!(a.device(), Device::Cpu);
    println!("meta     = dtype F64, shape {:?}, {} elements", a.shape(), a.numel());

    // Trainable leaves carry an autograd node slot for tpt-autograd.
    let w = Tensor::from_typed(vec![0.5_f64]).with_autograd();
    assert!(w.requires_grad());
    println!("w is trainable: {}", w.requires_grad());
}
