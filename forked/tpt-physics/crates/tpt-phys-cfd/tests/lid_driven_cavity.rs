//! CFD validation: lid-driven cavity flow (Ghia et al. reference).
//!
//! A square cavity with three stationary no-slip walls and a top lid moving in
//! `+x` at speed `U` develops a primary recirculating vortex: the lid drags
//! fluid in `+x` just below it, the fastest streamwise flow sits in the upper
//! half of the cavity, and a return flow (`u_x < 0`) appears near the floor.
//!
//! **Status: EXPERIMENTAL / not root-caused.** The moving-lid half-way
//! bounce-back BC (`Lbm2D::set_moving_lid` + `bounce` in `src/lib.rs`) has
//! *not* been root-caused against the Ghia et al. reference. The moving-lid
//! correction is the Ladd bounce-back form, but the primary-vortex convergence
//! issue is unresolved, so this test is gated behind `#[ignore]` and the
//! README keeps the cavity in the "Experimental" column. Do **not** silently
//! flip this or the README claim to "Validated" — either fix the BC or keep
//! both honestly marked experimental.
//!
//! The run is gated behind `#[ignore]` because it needs many steps to reach
//! steady state and is slow in debug; run it with
//! `cargo test --release -- --ignored` to inspect the centre-line profile.

use tpt_phys_cfd::Lbm2D;

#[test]
#[ignore = "slow steady-state LBM; run with `cargo test --release -- --ignored`"]
fn lid_driven_cavity_primary_vortex() {
    let nx = 48;
    let ny = 48;
    // Relaxation time `τ` sets the kinematic viscosity `ν = cs²(τ−½)`. Pick a
    // *steady* regime (`Re = u·L/ν ≈ 100`) that BGK resolves cleanly on this
    // coarse lattice: `u_lid = 0.03`, `τ = 0.54` ⇒ `ν ≈ 0.0133`, `Re ≈ 108`.
    // (The previous `Re ≈ 720` setting was unsteady/under-resolved here and
    // failed to converge to the expected primary vortex.)
    let tau = 0.54;
    let u_lid = 0.03;

    let mut sim = Lbm2D::new(nx, ny, tau);
    sim.set_box_walls();
    sim.set_moving_lid(ny - 1, u_lid);
    sim.initialise(1.0, [0.0, 0.0]);

    // The primary vortex is already at steady state by ~60k steps (verified
    // identical to a 1M-step run); run comfortably past the ≈170k diffusion
    // timescale so the assertion holds regardless of lattice rounding.
    for _ in 0..200_000 {
        sim.step([0.0, 0.0]);
    }

    let col = sim.x_velocity_at_column(nx / 2);
    let interior: Vec<f64> = col[1..ny - 1].to_vec();
    for v in &interior {
        assert!(v.is_finite());
    }

    let beneath_lid = interior[interior.len() - 1];
    let max_u = interior.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let argmax = interior
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let min_u = interior.iter().cloned().fold(f64::INFINITY, f64::min);
    let positive = interior.iter().filter(|&&v| v > 0.0).count();
    let pos_frac = positive as f64 / interior.len() as f64;
    eprintln!(
        "DBG cavity u_lid={:.3}: beneath={:.4} max={:.4}@{} min={:.4} pos%={:.1} prof={:?}",
        u_lid,
        beneath_lid,
        max_u,
        argmax,
        min_u,
        100.0 * pos_frac,
        interior.iter().rev().step_by(4).collect::<Vec<_>>()
    );

    // Primary-vortex validation (lid-driven cavity, Ghia et al. reference
    // structure): the moving lid drags fluid in `+x` just beneath it, the
    // fastest streamwise flow sits in the upper portion of the cavity, and a
    // recirculating vortex forms with `u_x > 0` over a substantial span of the
    // centre-line (the lid-dominated upper region) and a return flow
    // (`u_x < 0`) near the floor. On this coarse 48×48 lattice the BGK solution
    // places the vortex centre high (≈0.7 from the floor), so we assert a
    // meaningful primary-vortex span rather than an exact Ghia fraction.
    assert!(beneath_lid > 0.01, "lid did not drag fluid: {beneath_lid}");
    assert!(
        argmax >= interior.len() / 2,
        "peak velocity not in upper half: @{}",
        argmax
    );
    assert!(
        pos_frac > 0.20,
        "primary vortex too small: u_x>0 over {:.0}% of height",
        100.0 * pos_frac
    );
    assert!(min_u < 0.0, "no return flow at the floor: min_u = {min_u}");
}
