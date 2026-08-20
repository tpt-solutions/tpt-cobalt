//! Silo / hopper discharge and the Beverloo trend.
//!
//! A converging funnel is filled with grains and drained through an orifice at
//! the bottom. Below roughly one particle diameter the flow **arches** and
//! stops; above it, the steady discharge rate grows with orifice size (the
//! classic Beverloo law, where rate scales with the orifice opening to a
//! power). This example sweeps the half-orifice and prints the measured steady
//! discharge rate for each.
//!
//! Run with (release strongly recommended — this steps a few thousand times):
//!
//! ```text
//! cargo run --release --example hopper_discharge -p tpt-phys-dem
//! ```

use tpt_phys_dem::scenarios::{hopper_discharge, HopperParams};

fn main() {
    println!("Hopper discharge vs. orifice size (Beverloo trend)");
    println!("  {:>12}  {:>18}", "half-orifice", "rate [particles/s]");
    println!("  {}", "-".repeat(34));

    let mut previous = None;
    for &half in &[0.2_f64, 0.5, 1.0, 1.5, 2.0] {
        let params = HopperParams {
            orifice_half: half,
            ..Default::default()
        };
        let rate = hopper_discharge(&params);
        let arrow = match previous {
            Some(prev) if rate > prev + 1e-9 => "  (faster)",
            Some(_) => "",
            None => "",
        };
        println!("  {half:>12.2}  {rate:>18.2}{arrow}");
        previous = Some(rate);
    }

    println!();
    println!("A ~0.2 half-orifice arches (near-zero rate); larger orifices");
    println!("discharge progressively faster — the qualitative Beverloo behaviour.");
}
