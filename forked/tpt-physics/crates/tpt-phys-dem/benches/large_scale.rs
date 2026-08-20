//! Criterion benchmark harness for the DEM parallel stepper.
//!
//! Replaces the old `eprintln!`-timing example with a proper `criterion`
//! harness (run `cargo bench -p tpt-phys-dem` for long-term tracking).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use tpt_phys_dem::particle::Particle;
use tpt_phys_dem::world::World;

fn make(n: usize) -> World {
    let r = 0.25;
    let l = (n as f64).sqrt().ceil() * 1.0 + 4.0;
    let mut rng = Lcg::new(0xBEEF);
    let mut particles = Vec::with_capacity(n);
    for _ in 0..n {
        let x = r + rng.next_f64() * (l - 2.0 * r);
        let y = r + rng.next_f64() * (l - 2.0 * r);
        let z = r + rng.next_f64() * (l - 2.0 * r);
        particles.push(Particle::new([x, y, z], [0.0; 3], r, 1000.0));
    }
    let mut w = World::new(particles, 1e-4);
    w.e_star = 5e7;
    w.max_speed = 15.0;
    w
}

fn bench_step_par(c: &mut Criterion) {
    let mut group = c.benchmark_group("dem_step_par");
    for &n in &[10_000usize, 100_000usize] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut world = make(n);
            b.iter(|| world.step_par());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_step_par);
criterion_main!(benches);

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64
    }
}
