//! Example: compare worst-case, RSS, and Monte-Carlo stack-up of two
//! dimensions whose nominal values nearly cancel.

use rand::SeedableRng;
use rand::rngs::StdRng;
use tpt_eng_tolerance::{DimTol, monte_carlo, rss, worst_case};

fn main() {
    let dims = vec![
        DimTol::new("shaft", 50.0, 0.05),
        DimTol::new("bore", -50.0, 0.05),
    ];

    let (wlo, whi) = worst_case(&dims);
    let (rlo, rhi) = rss(&dims);
    let mut rng = StdRng::seed_from_u64(42);
    let mc = monte_carlo(&dims, 20_000, Some((wlo, whi)), &mut rng);

    println!(
        "nominal stack-up      = {:.4}",
        dims.iter().map(|d| d.nominal).sum::<f64>()
    );
    println!("worst-case interval    = [{:.4}, {:.4}]", wlo, whi);
    println!("RSS (3-sigma) interval = [{:.4}, {:.4}]", rlo, rhi);
    println!(
        "Monte-Carlo: mean = {:.4}, std = {:.4}, yield = {:.3}",
        mc.mean,
        mc.std,
        mc.yield_fraction.unwrap_or(0.0)
    );
}
