//! Parallel contact sweep: `World::step` vs `World::step_par`.
//!
//! [`World::step`] is a straightforward sequential semi-implicit Euler step.
//! [`World::step_par`] instead computes each particle's force independently
//! (summing its own neighbour contacts from the spatial-hash neighbour lists),
//! so the *narrow phase* distributes across the `rayon` thread pool with no
//! cross-particle write races. This is the CPU path used for very large particle
//! counts — the crate's `large_scale` test drives >100k particles through it.
//!
//! This example checks two things:
//!
//! 1. **Parity.** Because the pairwise Hertz–Mindlin contact force is exactly
//!    antisymmetric, both steppers produce the same physics; only the
//!    floating-point summation order differs.
//! 2. **Scaling.** It times both steppers across several particle counts so you
//!    can see where the parallel path starts to pay off on *your* machine.
//!
//! The scene is deliberately floor-free, obstacle-free and bond-free so the
//! comparison is apples to apples (the two steppers differ in boundary
//! handling — see the note printed at the end).
//!
//! Run with (release is essential for a meaningful timing):
//!
//! ```text
//! cargo run --release --example parallel_step -p tpt-phys-dem
//! ```

use std::time::{Duration, Instant};

use tpt_phys_dem::particle::Particle;
use tpt_phys_dem::world::World;

const R: f64 = 0.05;
/// Slight initial overlap so every particle carries real contact work.
const OVERLAP: f64 = 1.0e-3;
const STEPS: usize = 50;

/// A dense cubic cloud of `nx * ny * nz` mutually-overlapping grains.
fn build_world(nx: usize, ny: usize, nz: usize) -> World {
    let spacing = 2.0 * R - OVERLAP;
    let mut particles = Vec::with_capacity(nx * ny * nz);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                particles.push(Particle::new(
                    [i as f64 * spacing, j as f64 * spacing, k as f64 * spacing],
                    [0.0; 3],
                    R,
                    1000.0,
                ));
            }
        }
    }
    let mut world = World::new(particles, 1e-5);
    world.e_star = 1e7; // soft contacts so the explicit step stays stable
    world.drag = 5.0; // damp the expanding cloud
    world.max_speed = 2.0;
    // No floor, no obstacles, no bonds ⇒ the two steppers are directly
    // comparable (see the note at the end of the output).
    world.floor_y = -1.0e3;
    world
}

fn time_steps(mut world: World, parallel: bool) -> (Duration, f64) {
    let t = Instant::now();
    for _ in 0..STEPS {
        if parallel {
            world.step_par();
        } else {
            world.step();
        }
    }
    (t.elapsed(), world.kinetic_energy())
}

fn main() {
    println!("DEM contact sweep: sequential vs. rayon-parallel");
    println!("  steps per measurement : {STEPS}");
    println!(
        "  threads available     : {}",
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    );
    println!();
    println!(
        "  {:>9} {:>12} {:>12} {:>9} {:>12}",
        "particles", "step()", "step_par()", "ratio", "KE delta"
    );
    println!("  {}", "-".repeat(58));

    for &(nx, ny, nz) in &[(20, 6, 20), (30, 8, 30), (40, 12, 40)] {
        let n = nx * ny * nz;

        let (seq_time, ke_seq) = time_steps(build_world(nx, ny, nz), false);
        let (par_time, ke_par) = time_steps(build_world(nx, ny, nz), true);

        let seq_rate = (n * STEPS) as f64 / seq_time.as_secs_f64();
        let par_rate = (n * STEPS) as f64 / par_time.as_secs_f64();
        let rel_ke = (ke_seq - ke_par).abs() / ke_seq.abs().max(1e-30);

        println!(
            "  {n:>9} {:>12.3?} {:>12.3?} {:>8.2}x {rel_ke:>12.2e}",
            seq_time,
            par_time,
            par_rate / seq_rate
        );

        assert!(
            ke_seq.is_finite() && ke_par.is_finite(),
            "simulation diverged at n = {n}"
        );
        assert!(
            rel_ke < 1e-6,
            "parallel sweep disagreed with the sequential one by {rel_ke} at n = {n}"
        );
    }

    println!();
    println!("Parity: both steppers agree on kinetic energy to round-off at every size,");
    println!("confirming the antisymmetric pairwise contact law is applied identically.");
    println!();
    println!("Reading the `ratio` column (>1.0x means step_par is faster). On a uniform");
    println!("cloud like this one, step_par is typically *not* a win, for two structural");
    println!("reasons:");
    println!("  1. Only the narrow phase is parallelised. The broad phase");
    println!("     (`SpatialHash::build`, plus building per-particle neighbour lists)");
    println!("     runs sequentially in both paths and dominates a uniform scene.");
    println!("  2. The parallel path is race-free precisely because each particle sums");
    println!("     its own contacts, so every pair's contact force is evaluated *twice*");
    println!("     (once for i, once for j). The sequential path evaluates each pair once");
    println!("     and applies equal-and-opposite forces.");
    println!();
    println!("So `step_par` trades ~2x arithmetic plus per-particle allocation for thread");
    println!("parallelism; it needs enough cores (and enough contacts per particle) to");
    println!("come out ahead. Measure on your own scene and hardware before switching:");
    println!("`cargo bench -p tpt-phys-dem` tracks `step_par` at 10k and 100k particles.");
    println!();
    println!("Boundary caveat: `step` additionally applies an *inelastic floor* — it");
    println!("zeroes any remaining downward velocity of a floor-contacting particle —");
    println!("which `step_par` does not, so a scene with an active floor settles");
    println!("slightly differently between the two steppers. Fixed-obstacle");
    println!("de-penetration *is* applied by both.");
}
