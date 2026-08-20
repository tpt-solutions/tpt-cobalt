//! Combined DEM + CFD (resolved, one-way CFD→DEM) coupling demo.
//!
//! The LBM fluid (`tpt_phys_cfd::Lbm2D`) is driven by a body force so it
//! develops a mean flow in `+x`. Each coupling step the fluid's mean x-velocity
//! is sampled and injected into the DEM granular bed (`tpt_phys_dem::World`) as
//! a uniform drag body acceleration via [`World::external_accel`] — the
//! coupling hook a partitioned CFD-DEM solver uses to push the granular phase
//! with the fluid traction. This is a one-way (fluid → granular) coupling;
//! two-way back-coupling (granular drag on the fluid) is a follow-up.

use tpt_phys_cfd::Lbm2D;
use tpt_phys_dem::particle::Particle;
use tpt_phys_dem::world::World;

fn main() {
    // --- DEM granular bed -------------------------------------------------
    let mut particles = Vec::new();
    let r = 0.08;
    let rho = 2600.0;
    let (nx, ny, nz) = (8usize, 6, 8);
    let gap = 2.05 * r;
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                particles.push(Particle::new(
                    [i as f64 * gap, r + j as f64 * gap, k as f64 * gap - 0.3],
                    [0.0; 3],
                    r,
                    rho,
                ));
            }
        }
    }
    let mut dem = World::new(particles, 2e-4);
    dem.drag = 1.0; // fluid drag damping inside the bed
    dem.max_speed = 6.0;

    // --- LBM fluid --------------------------------------------------------
    let mut lbm = Lbm2D::new(120, 40, 0.6);
    lbm.set_horizontal_walls();
    lbm.initialise(1.0, [0.0, 0.0]);
    let fx = 1e-5; // driving body force → mean +x flow

    let coupling_drag = 2.0; // gain converting fluid speed → granular accel
    let n_steps = 600;
    for step in 0..n_steps {
        // Fluid sub-step.
        lbm.step([fx, 0.0]);
        // Sample mean fluid x-velocity (the traction the fluid applies).
        let mean_u: f64 = lbm.ux.iter().sum::<f64>() / lbm.ux.len() as f64;
        // One-way coupling: inject fluid drag as a uniform body acceleration.
        dem.external_accel = [coupling_drag * mean_u, 0.0, 0.0];
        dem.step();

        if step % 150 == 0 || step == n_steps - 1 {
            let mean_vx = dem.particles.iter().map(|p| p.velocity[0]).sum::<f64>()
                / dem.particles.len() as f64;
            println!(
                "step {:4} | fluid u_mean={:+.4} | DEM KE={:.3} J | DEM <vx>={:+.4} m/s",
                step,
                mean_u,
                dem.kinetic_energy(),
                mean_vx
            );
        }
    }

    assert!(dem.kinetic_energy().is_finite(), "DEM diverged");
    let mean_vx =
        dem.particles.iter().map(|p| p.velocity[0]).sum::<f64>() / dem.particles.len() as f64;
    println!("final mean granular vx (fluid-driven drift) = {mean_vx:.4} m/s");
}
