//! Criterion micro-benchmarks for the matmul family (f64, CPU path).
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_tensor::Tensor;

fn rand_mat(rows: usize, cols: usize) -> Tensor {
    let mut s: u64 = 0x9E3779B97F4A7C15;
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

fn bench_matmul(c: &mut Criterion) {
    let mut g = c.benchmark_group("matmul_f64");
    for &n in &[64usize, 128, 256] {
        let a = rand_mat(n, n);
        let b = rand_mat(n, n);
        g.throughput(criterion::Throughput::Elements((n * n) as u64));
        g.bench_with_input(format!("{n}x{n}"), &(n), |bencher, _| {
            bencher.iter(|| black_box(a.matmul(black_box(&b))))
        });
    }
    g.finish();
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
