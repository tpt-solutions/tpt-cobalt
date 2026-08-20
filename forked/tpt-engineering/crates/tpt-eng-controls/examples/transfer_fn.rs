//! Richer example: a discrete transfer-function model of a first-order lag and
//! its step response.

use tpt_eng_controls::TransferFunction;

fn main() {
    // G(s) = 1/(s+1) discretised (zero-order hold, dt=0.1):
    //   y[n] = e^-0.1 y[n-1] + (1 - e^-0.1) u[n-1]
    let dt = 0.1;
    let a1 = (-0.1_f64).exp();
    let b1 = 1.0 - a1;
    let mut tf = TransferFunction::new(&[0.0, b1], &[1.0, -a1]);

    let mut y = 0.0;
    println!("    t |    y");
    for step in 0..60 {
        y = tf.step(1.0); // unit step input
        let t = step as f64 * dt;
        if step % 15 == 0 {
            println!("{:6.2} | {:5.3}", t, y);
        }
    }
    println!("final y = {:.3} (expected 1.000)", y);
    assert!((y - 1.0).abs() < 0.05, "should approach unit DC gain");
    println!("controls transfer_fn example passed");
}
