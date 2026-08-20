//! Hello-world for `tpt-phys-cfd`: a lid-driven cavity.
//!
//! A fully-enclosed square cavity with its top lid moving in `+x` drives a
//! recirculating flow. This example runs the LBM solver and reports the
//! developed flow statistics. (The cavity primary-vortex benchmark is
//! *experimental* — see the crate docs — so this prints diagnostics rather
//! than asserting a target vortex location.)
//!
//! Run with: `cargo run --example cavity --release`

use tpt_phys_cfd::{Lbm2D, XBoundary};

fn main() {
    let nx = 64;
    let ny = 64;
    let tau = 0.53; // low viscosity ⇒ moderate Reynolds number
    let u_lid = 0.1;

    let mut lbm = Lbm2D::new(nx, ny, tau);
    lbm.set_box_walls(); // enclose all four sides
    lbm.set_moving_lid(ny - 1, u_lid); // top row moves in +x
    lbm.set_x_boundary(XBoundary::Periodic); // periodic left/right (channel-like)

    let steps = 2000;
    for _ in 0..steps {
        lbm.step([0.0, 0.0]);
    }

    // Report max speed and the location of the fastest interior node.
    let mut vmax = 0.0_f64;
    let mut vmax_ix = 0;
    let mut vmax_iy = 0;
    for iy in 0..ny {
        for ix in 0..nx {
            let i = lbm.idx(ix, iy);
            if lbm.solid[i] {
                continue;
            }
            let v = (lbm.ux[i].powi(2) + lbm.uy[i].powi(2)).sqrt();
            if v > vmax {
                vmax = v;
                vmax_ix = ix;
                vmax_iy = iy;
            }
        }
    }

    println!("Lid-driven cavity ({nx}x{ny}, tau={tau}, u_lid={u_lid}):");
    println!("  steps run            : {steps}");
    println!("  max speed            : {vmax:.4} (lid = {u_lid})");
    println!("  fastest node at      : ({vmax_ix}, {vmax_iy})");
    println!("  (cavity primary-vortex benchmark is experimental)");
}
