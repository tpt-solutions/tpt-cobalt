//! Flow past a circular cylinder — bluff-body wake and vortex shedding.
//!
//! A uniform stream (`XBoundary::Inlet`) is driven through a channel (no-slip
//! top/bottom walls) past a stationary circular obstacle. The example develops
//! the steady wake, reports the size of the recirculation bubble, then probes
//! the vertical velocity at a downstream point to detect periodic vortex
//! shedding (the von Kármán street) and estimate the Strouhal number.
//!
//! Run with: `cargo run --release --example flow_past_cylinder -p tpt-phys-cfd`

use tpt_phys_cfd::{Lbm2D, XBoundary};

fn main() {
    let nx = 200;
    let ny = 60;
    let tau = 0.53; // ν = cs²(τ−½) ≈ 0.01
    let u0 = 0.08;

    // Cylinder centred in the channel, one diameter from the inlet.
    let cx = 60.0;
    let cy = ny as f64 / 2.0;
    let r = 8.0;

    let mut sim = Lbm2D::new(nx, ny, tau);
    sim.set_x_boundary(XBoundary::Inlet(u0));
    sim.set_horizontal_walls(); // channel top/bottom no-slip walls
    sim.add_circle(cx, cy, r);
    sim.initialise(1.0, [u0, 0.0]);

    // Develop the steady wake.
    let warmup = 5_000;
    for _ in 0..warmup {
        sim.step([0.0, 0.0]);
    }

    // Recirculation bubble: most-negative u in a band just behind the cylinder
    // (the standing reverse-flow region of the wake).
    let mut min_ux = f64::INFINITY;
    let mut wake_max_uy = 0.0_f64;
    let x0 = (cx + 0.75 * r) as usize;
    let x1 = (cx + 2.5 * r) as usize;
    for iy in 1..ny - 1 {
        for ix in x0..=x1 {
            let i = sim.idx(ix, iy);
            min_ux = min_ux.min(sim.ux[i]);
        }
        wake_max_uy = wake_max_uy.max(sim.uy[sim.idx((cx + 2.0 * r) as usize, iy)].abs());
    }

    // Vortex shedding: track uy at a probe in the near-wake and count zero
    // crossings of the alternating vertical velocity.
    let probe = sim.idx((cx + 3.0 * r) as usize, cy as usize);
    let mut prev = sim.uy[probe];
    let mut crossings = 0usize;
    let shed_steps = 12_000;
    for t in 0..shed_steps {
        sim.step([0.0, 0.0]);
        let uy = sim.uy[probe];
        if t > 1000 && prev * uy < 0.0 {
            crossings += 1;
        }
        prev = uy;
    }

    let d = 2.0 * r;
    let cycles = crossings as f64 / 2.0;
    let freq = cycles / (shed_steps as f64); // 1 / lattice time
    let re = u0 * d / sim.viscosity();
    let st = freq * d / u0; // Strouhal number

    println!("Flow past a cylinder (D2Q9 LBM)");
    println!("  lattice             : {nx} x {ny}  (tau = {tau})");
    println!("  inlet velocity u0   : {u0}");
    println!("  cylinder D          : {d} cells, Re ≈ {re:.0}");
    println!("  after {warmup} + {shed_steps} steps:");
    println!("    recirculation u   : {min_ux:.4} (negative ⇒ reverse flow behind body)");
    println!("    shear-layer |uy|  : {wake_max_uy:.4}");
    println!("    shedding cycles   : {cycles:.1} over {shed_steps} steps");
    println!("    Strouhal St       : {st:.3}  (classical ≈ 0.2 at these Re)");

    assert!(
        min_ux < -0.002,
        "no recirculation bubble formed (min u = {min_ux})"
    );
    assert!(re > 40.0, "Re too low for shedding");
    println!();
    println!("OK: steady wake + periodic von Kármán shedding detected.");
}
