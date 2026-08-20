//! Basic example: a discrete PID controller tracking a setpoint on a plant.

use tpt_eng_controls::{Pid, PidGains};

fn main() {
    // Kp=3, Ki=1.5, Kd=0.5, saturate output at ±15.
    let mut pid = Pid::new(PidGains::new(3.0, 1.5, 0.5)).with_output_limit(15.0);
    pid.set_setpoint(10.0);

    let dt = 0.05;
    let mut y = 0.0f64;
    println!("  t |    u |    y");
    for step in 0..120 {
        let u = pid.update(y, dt);
        y += u * dt * 0.5; // first-order plant with gain 0.5
        if step % 30 == 0 {
            println!("{:5.2} | {:5.3} | {:5.3}", step as f64 * dt, u, y);
        }
    }
    println!("final y = {:.3} (setpoint 10.000)", y);
    assert!(
        (y - 10.0).abs() < 0.5,
        "controller should track the setpoint"
    );
    println!("controls basic (PID) example passed");
}
