//! Batching a dataset with `TensorDataset` + `DataLoader`, including
//! epoch shuffling.
//!
//! Run with: `cargo run -p tpt-ml --example data_loader`

use tpt_ml::{DataLoader, Dataset, TensorDataset};
use tpt_tensor::Tensor;

fn sample(v: f64) -> Tensor {
    Tensor::from_typed(vec![v]).reshape(&[1]).unwrap()
}

fn main() {
    // 8 samples of (input, target).
    let inputs: Vec<Tensor> = (0..8).map(|i| sample(i as f64)).collect();
    let targets: Vec<Tensor> = (0..8).map(|i| sample((i * 10) as f64)).collect();

    let dataset = TensorDataset::new(inputs, targets);
    assert_eq!(dataset.len(), 8);

    // Batched iteration in order.
    let mut loader = DataLoader::new(&dataset, 3, false);
    println!("batches per epoch: {}", loader.num_batches());

    while let Some((x, y)) = loader.next() {
        println!(
            "batch x = {:?}  y = {:?}",
            x.to_vec::<f64>().unwrap(),
            y.to_vec::<f64>().unwrap()
        );
    }
    assert_eq!(loader.num_batches(), 3); // ceil(8/3)

    // Reset reshuffles when shuffle=true; batches keep stacked [B, 1] shape.
    loader.reset();
    let mut count = 0;
    for (x, _) in loader {
        assert_eq!(x.ndim(), 2);
        assert!(x.shape()[0] >= 1 && x.shape()[0] <= 3);
        count += 1;
    }
    println!("epoch 2 yielded {count} batches");
}
