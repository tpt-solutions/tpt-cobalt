//! Richer example: a continuous-time state-space model of a mass-spring-damper
//! advanced with forward Euler.

use tpt_eng_controls::StateSpace;
use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;

fn main() {
    // m x'' + c x' + k x = u, with m=1, c=0.5, k=2.
    // State x = [position, velocity]^T, input u = force.
    let m = 1.0;
    let c = 0.5;
    let k = 2.0;
    let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -k / m, -c / m]);
    let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0 / m]);
    let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
    let d = DMatrix::from_row_slice(1, 1, &[0.0]);
    let mut ss = StateSpace::new(a, b, c, d);

    let dt = 0.01;
    let mut t = 0.0;
    println!("    t |    x |   v");
    for step in 0..600 {
        let _y = ss.step(1.0, dt); // unit step force
        let x = ss.state();
        if step % 150 == 0 {
            println!("{:6.2} | {:5.3} | {:5.3}", t, x[0], x[1]);
        }
        t += dt;
    }
    let x = ss.state()[0];
    println!("steady x ≈ {:.3} (expected {:.3})", x, 1.0 / k);
    assert!(
        (x - 1.0 / k).abs() < 0.1,
        "should settle near static deflection"
    );
    println!("controls state_space example passed");
}
