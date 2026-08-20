//! Runnable example: PID step response onto a simple first-order plant.

use tpt_eng_controls::{Pid, PidGains};

fn main() {
    let gains = PidGains::new(2.0, 1.0, 0.5);
    let mut pid = Pid::new(gains).with_output_limit(10.0);
    pid.set_setpoint(1.0);

    let dt = 0.1;
    let mut y = 0.0f64;
    println!("  t |   u   |   y");
    for step in 0..100 {
        let u = pid.update(y, dt);
        y += u * dt * 0.2; // first-order plant: dy/dt = 0.2·u
        if step % 20 == 0 {
            println!("{:4.1} | {:5.3} | {:5.3}", step as f64 * dt, u, y);
        }
    }
    assert!(
        (y - 1.0).abs() < 0.1,
        "controller should converge to setpoint"
    );
    println!("pid step example passed");
}
