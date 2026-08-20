//! Cohesive (bonded) particles: agglomerates that hold together and debond.
//!
//! DEM particles can be linked by elastic [`Bond`](tpt_phys_dem::world::Bond)s
//! that resist both tension and compression about a rest length, so a cluster
//! behaves like a fragile agglomerate (wet powder, a lightly-cemented grain
//! cluster, a green body). When the bond force exceeds its `strength` the bond
//! debonds and is dropped from the force balance.
//!
//! This example contrasts three particles:
//!   * a **bonded, unbreakable** chain that stays together under a pull,
//!   * an identical chain with a **finite bond strength** that snaps, and
//!   * an **unbonded** pair that flies apart ballistically.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example cohesive_bonds -p tpt-phys-dem
//! ```

use tpt_phys_dem::particle::Particle;
use tpt_phys_dem::world::World;

/// Two particles a unit apart on the x-axis, each pulled outward.
fn separating_pair() -> Vec<Particle> {
    vec![
        Particle::new([0.0, 0.0, 0.0], [-3.0, 0.0, 0.0], 0.3, 1000.0),
        Particle::new([1.0, 0.0, 0.0], [3.0, 0.0, 0.0], 0.3, 1000.0),
    ]
}

fn separation(w: &World) -> f64 {
    let a = w.particles[0].position;
    let b = w.particles[1].position;
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Build a zero-gravity world so the only forces are contact + bond.
fn floating_world(ps: Vec<Particle>) -> World {
    let mut w = World::new(ps, 1e-3);
    w.gravity = [0.0; 3];
    w
}

fn main() {
    const STEPS: usize = 200;

    // 1. Strong, unbreakable bond.
    let mut strong = floating_world(separating_pair());
    strong.bond_stiffness = 1e4;
    strong.bond_strength = 0.0; // 0 ⇒ never breaks
    strong.create_bonds(0.5);

    // 2. Finite-strength bond that will snap under the pull.
    let mut fragile = floating_world(separating_pair());
    fragile.bond_stiffness = 1e4;
    fragile.bond_strength = 20.0; // N
    fragile.create_bonds(0.5);

    // 3. No bonds at all.
    let mut free = floating_world(separating_pair());

    let d0 = separation(&strong);
    for _ in 0..STEPS {
        strong.step();
        fragile.step();
        free.step();
    }

    println!("Cohesive bonds vs. free particles ({STEPS} steps, initial separation {d0:.3} m)");
    println!(
        "  strong bond (∞ strength) : separation {:.3} m, active bonds {}",
        separation(&strong),
        strong.active_bonds()
    );
    println!(
        "  fragile bond (20 N)      : separation {:.3} m, active bonds {}  (snapped!)",
        separation(&fragile),
        fragile.active_bonds()
    );
    println!(
        "  no bond (ballistic)      : separation {:.3} m",
        separation(&free)
    );

    println!();
    println!("The unbreakable bond keeps the pair close; the fragile bond debonds and");
    println!("lets it drift; the unbonded pair separates fastest of all.");

    assert_eq!(strong.active_bonds(), 1, "strong bond must survive");
    assert_eq!(fragile.active_bonds(), 0, "fragile bond must debond");
    assert!(
        separation(&free) > separation(&strong),
        "free pair must separate most"
    );
}
