//! Criterion benchmark harness for the LBM solver.
//!
//! Run `cargo bench -p tpt-phys-cfd` for long-term tracking.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use tpt_phys_cfd::Lbm2D;

fn bench_lbm(c: &mut Criterion) {
    let mut group = c.benchmark_group("lbm_step");
    for &(nx, ny) in &[(128usize, 128usize), (256usize, 128usize)] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{nx}x{ny}")),
            &(nx, ny),
            |b, &(nx, ny)| {
                let mut lat = Lbm2D::new(nx, ny, 0.53);
                b.iter(|| lat.step([0.0, 0.0]));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_lbm);
criterion_main!(benches);
