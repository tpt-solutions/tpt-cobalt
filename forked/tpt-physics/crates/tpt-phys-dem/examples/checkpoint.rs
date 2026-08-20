//! Checkpoint and resume a long DEM run.
//!
//! Long granular runs need to survive interruption. [`World`] is fully
//! `serde`-serializable, so the entire simulation state — particle positions,
//! velocities, temperatures, obstacles, bonds and every solver parameter —
//! round-trips through a compact `bincode` buffer:
//!
//! * [`World::to_checkpoint`] / [`World::from_checkpoint`] — in-memory bytes,
//! * [`World::save_checkpoint`] / [`World::load_checkpoint`] — file-backed.
//!
//! Resuming is *exact*: this example proves that a run interrupted at step 500
//! and resumed from disk follows bit-for-bit the same trajectory as an
//! uninterrupted run.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example checkpoint -p tpt-phys-dem
//! ```

use tpt_phys_dem::particle::Particle;
use tpt_phys_dem::world::World;
use tpt_phys_dem::Obstacle;

/// A modest, reproducible scene: a grid of grains poured around a spacer.
fn build_world() -> World {
    let r = 0.08;
    let mut particles = Vec::new();
    for i in 0..6 {
        for j in 0..3 {
            for k in 0..6 {
                particles.push(Particle::new(
                    [
                        -0.9 + i as f64 * 0.36,
                        0.5 + j as f64 * 0.2,
                        -0.9 + k as f64 * 0.36,
                    ],
                    [0.0; 3],
                    r,
                    1500.0,
                ));
            }
        }
    }
    let spacer = Obstacle::Cylinder {
        center: [0.0, 0.75, 0.0],
        axis: [0.0, 1.0, 0.0],
        radius: 0.25,
        half_height: 0.75,
    };
    let mut world = World::with_obstacles(particles, 2e-4, vec![spacer]);
    world.e_star = 5e7;
    world.restitution = 0.05;
    world.drag = 2.0;
    world.max_speed = 5.0;
    world
}

/// Max absolute positional difference between two worlds.
fn max_position_error(a: &World, b: &World) -> f64 {
    a.particles
        .iter()
        .zip(&b.particles)
        .flat_map(|(p, q)| (0..3).map(move |k| (p.position[k] - q.position[k]).abs()))
        .fold(0.0_f64, f64::max)
}

fn main() {
    const BEFORE: usize = 500;
    const AFTER: usize = 500;

    // --- Reference run: BEFORE + AFTER steps, never interrupted. ------------
    let mut reference = build_world();
    for _ in 0..BEFORE + AFTER {
        reference.step();
    }

    // --- Interrupted run: BEFORE steps, checkpoint to disk, then resume. ----
    let mut interrupted = build_world();
    for _ in 0..BEFORE {
        interrupted.step();
    }

    // In-memory checkpoint (what you would push to object storage).
    let bytes = interrupted.to_checkpoint().expect("serialize world");

    // File-backed checkpoint (what a batch job would write between stages).
    let path = std::env::temp_dir().join("tpt_dem_example_checkpoint.bin");
    interrupted
        .save_checkpoint(&path)
        .expect("write checkpoint");
    let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

    let mut resumed = World::load_checkpoint(&path).expect("read checkpoint");
    let _ = std::fs::remove_file(&path);

    println!("DEM checkpoint / resume");
    println!("  particles              : {}", reference.particles.len());
    println!("  obstacles              : {}", reference.obstacles.len());
    println!("  steps before / after   : {BEFORE} / {AFTER}");
    println!(
        "  in-memory checkpoint   : {} bytes ({:.1} bytes/particle)",
        bytes.len(),
        bytes.len() as f64 / reference.particles.len() as f64
    );
    println!("  on-disk checkpoint     : {file_size} bytes");

    // State immediately after restore must match exactly.
    let restore_error = max_position_error(&interrupted, &resumed);
    println!("  restore position error : {restore_error:.3e} m (expect 0)");

    // ...and the resumed run must continue along the same trajectory.
    for _ in 0..AFTER {
        resumed.step();
    }
    let trajectory_error = max_position_error(&reference, &resumed);
    println!("  post-resume divergence : {trajectory_error:.3e} m (expect ~0)");
    println!(
        "  kinetic energy         : reference {:.4e} J vs resumed {:.4e} J",
        reference.kinetic_energy(),
        resumed.kinetic_energy()
    );

    assert!(restore_error == 0.0, "restore must be bit-exact");
    assert!(
        trajectory_error < 1e-12,
        "resumed trajectory diverged by {trajectory_error}"
    );
    println!();
    println!("OK: an interrupted run resumed from disk reproduced the reference trajectory.");
}
