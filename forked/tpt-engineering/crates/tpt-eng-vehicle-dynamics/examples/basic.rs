// Basic runnable example for `tpt-eng-vehicle-dynamics`.
//
// Demonstrates the three core public-API areas: Pacejka "magic formula" tire
// forces, quadratic aerodynamic drag/lift, and the kinematic roll-center
// calculation for a double-wishbone suspension.

use tpt_eng_geometry::Point3;
use tpt_eng_vehicle_dynamics::{
    drag_force, lift_force, pacejka_lateral, pacejka_longitudinal, roll_center_height,
};

fn main() {
    // Pacejka "magic formula" coefficients (B, C, D, E) for a passenger-car tire.
    let (b, c, d, e) = (10.0, 1.65, 1000.0, 0.8);

    // Lateral force at a 4° slip angle.
    let fy = pacejka_lateral(4.0, b, c, d, e);
    println!("Lateral force @ 4°   : {:.1} N", fy);

    // Longitudinal (traction) force at 5% slip.
    let fx = pacejka_longitudinal(0.05, b, c, d, e);
    println!("Longit. force @ 5%   : {:.1} N", fx);

    // Aerodynamic drag at 30 m/s (rho = 1.225, Cd = 0.30, A = 2.2 m²).
    let drag = drag_force(1.225, 0.30, 2.2, 30.0);
    println!("Aero drag @ 30 m/s   : {:.1} N", drag);

    // Downforce (negative lift) at the same speed (Cl = -0.9).
    let down = lift_force(1.225, -0.9, 2.2, 30.0);
    println!("Downforce @ 30 m/s   : {:.1} N", down);

    // Roll-center height of a double-wishbone front suspension (front view).
    let rc = roll_center_height(
        Point3::new(0.10, 0.0, 0.48),   // lower inner pivot
        Point3::new(0.80, 0.0, 0.20),   // lower outer pivot
        Point3::new(0.15, 0.0, 0.3625), // upper inner pivot
        Point3::new(0.70, 0.0, 0.50),   // upper outer pivot
        Point3::new(0.80, 0.0, 0.0),    // tire contact patch (left track edge)
        1.6,                            // track width
    );
    println!(
        "Roll-center height    : {:.3} m",
        rc.expect("roll center is defined for this geometry")
    );
}
