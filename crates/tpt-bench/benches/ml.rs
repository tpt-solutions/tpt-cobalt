//! Criterion micro-benchmarks for the ML stack: Linear fwd+bwd, transformer
//! block forward, and the TPT-Script `train_step` interpreter path.
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tpt_ml::Module;
use tpt_tensor::Tensor;

fn rand_mat(rows: usize, cols: usize) -> Tensor {
    let mut s: u64 = 0xBF58476D1CE4E5B9;
    let data: Vec<f64> = (0..rows * cols)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 2000) as f64 / 1000.0 - 1.0
        })
        .collect();
    Tensor::from_typed(data).reshape(&[rows, cols]).unwrap()
}

fn bench_linear(c: &mut Criterion) {
    let x = rand_mat(128, 256);
    let w = rand_mat(256, 128).with_autograd();
    c.bench_function("linear_128x256x128_fwd_bwd", |b| {
        b.iter(|| {
            let w = w.clone();
            let x = black_box(x.clone()).with_autograd();
            let y = tpt_autograd::matmul(&x, &w);
            tpt_autograd::backward(&y);
        })
    });
}

fn bench_transformer(c: &mut Criterion) {
    let mut block = tpt_ml::attention::TransformerBlock::new(64, 4, 128);
    // [B, T, D] per the block's contract
    let input = rand_mat(2 * 8, 64).reshape(&[2, 8, 64]).unwrap();
    c.bench_function("transformer_block_b8_t8_d64_fwd", |b| {
        b.iter(|| black_box(block.forward(black_box(&input))))
    });
}

fn bench_train_step(c: &mut Criterion) {
    let mut interp = tpt_lang::Interpreter::new();
    interp
        .run("let net = mlp(8, 32, 4)\nlet xs = ones([16, 8])\nlet ys = ones([16, 4])\n")
        .unwrap();
    c.bench_function("tpt_script_train_step_b16", |b| {
        b.iter(|| black_box(interp.run("train_step(net, xs, ys, 0.01)").unwrap()))
    });
}

criterion_group!(benches, bench_linear, bench_transformer, bench_train_step);
criterion_main!(benches);
