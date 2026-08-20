//! CFD validation: flow past a circular cylinder (vortex shedding).
//!
//! A uniform stream of speed `U` past a fixed circular cylinder of diameter
//! `D` at Reynolds number `Re = U D / ν` exhibits, for `Re ≳ 47`, the
//! periodic von Kármán vortex street: the transverse (lift) velocity in the
//! wake oscillates and changes sign as alternate vortices detach. At low `Re`
//! (≈ 20) the wake is steady and symmetric with a small recirculation bubble
//! behind the cylinder. This validation checks both regimes:
//!
//! * **steady, low-Re**: the wake is left–right symmetric and a reversed-flow
//!   recirculation region exists just behind the cylinder;
//! * **unsteady, moderate-Re**: the transverse velocity at a wake station
//!   oscillates through zero (sign change) — the shed-vortex signature.

use tpt_phys_cfd::{Lbm2D, XBoundary};

fn run_cylinder(u_inf: f64, steps: usize) -> Lbm2D {
    let nx = 120;
    let ny = 48;
    let tau = 0.55;
    let cx = 30.0;
    let cy = (ny as f64) / 2.0;
    let r = 6.0; // D = 12

    let mut sim = Lbm2D::new(nx, ny, tau);
    sim.set_horizontal_walls();
    sim.set_x_boundary(XBoundary::Inlet(u_inf));
    sim.add_circle(cx, cy, r);
    sim.initialise(1.0, [u_inf, 0.0]);

    for _ in 0..steps {
        sim.step([0.0, 0.0]);
    }
    sim
}

#[test]
fn cylinder_steady_symmetric_wake() {
    let u_inf = 0.03; // Re = 0.03*12/0.0167 ≈ 21.6 (steady)
    let sim = run_cylinder(u_inf, 4000);
    let ny = sim.ny;
    let cx = 30;
    let cy = ny / 2;

    // Free-stream carries fluid in +x.
    let free = sim.ux[sim.idx(cx + 40, cy)];
    assert!(free > 0.0, "free-stream must flow in +x: {free}");

    // Reversed-flow recirculation just behind the cylinder.
    let behind = sim.ux[sim.idx(cx + 8, cy)];
    assert!(
        behind < 0.0,
        "expected wake recirculation (u_x<0): {behind}"
    );

    // Left–right symmetry: the wake transverse velocity should be ~0 along the
    // centre-line (no lift at steady state).
    let lift_center = sim.uy[sim.idx(cx + 8, cy)];
    assert!(
        lift_center.abs() < 0.01,
        "steady wake must be symmetric (|u_y| small at centre): {}",
        lift_center
    );

    eprintln!(
        "cylinder steady wake OK: free-stream u_x={:.4}, behind-cylinder u_x={:.4}, centre-line u_y={:.5}",
        free, behind, lift_center
    );
}

#[test]
fn cylinder_vortex_shedding() {
    let u_inf = 0.1; // Re = 0.1*12/0.0167 ≈ 72 (Kármán shedding)
    let nx = 120;
    let ny = 48;
    let cx = 30.0;
    // Slightly off-centre cylinder + a brief transverse kick break the
    // perfect fore-aft/up-down symmetry so the unstable von Kármán mode can
    // grow (a perfectly symmetric setup can remain steady indefinitely).
    let cy = (ny as f64) / 2.0 + 0.5;
    let mut sim = Lbm2D::new(nx, ny, 0.55);
    sim.set_horizontal_walls();
    sim.set_x_boundary(XBoundary::Inlet(u_inf));
    sim.add_circle(cx, cy, 6.0);
    sim.initialise(1.0, [u_inf, 0.0]);

    // Warm up, with a short transverse kick to seed the asymmetry.
    for n in 0..3000 {
        let f = if n < 300 { [0.0, 5e-5] } else { [0.0, 0.0] };
        sim.step(f);
    }

    // Sample the transverse velocity at a wake station over time.
    let wx = (cx + 14.0) as usize;
    let wy = cy as usize;
    let mut uy_history = Vec::with_capacity(3000);
    for _ in 0..3000 {
        sim.step([0.0, 0.0]);
        uy_history.push(sim.uy[sim.idx(wx, wy)]);
    }

    let min = uy_history.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = uy_history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(min.is_finite() && max.is_finite());
    // Shedding signature: the transverse velocity swings through zero.
    assert!(
        min < 0.0 && max > 0.0,
        "no vortex shedding: uy in [{min}, {max}]"
    );
    assert!(
        (max - min) > 0.005,
        "wake transverse velocity barely varies: range {}",
        max - min
    );

    eprintln!(
        "cylinder shedding OK: wake u_y range [{:.4}, {:.4}] over {} samples",
        min,
        max,
        uy_history.len()
    );
}
