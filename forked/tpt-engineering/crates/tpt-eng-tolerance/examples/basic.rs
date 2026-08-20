//! Basic `tpt-eng-tolerance` usage: build toleranced dimensions and compare the
//! three standard stack-up methods (worst case, RSS, Monte Carlo), then do the
//! same with the signed [`Stackup`] model.
//!
//! All dimensions are millimetres.
//!
//! Run with `cargo run -p tpt-eng-tolerance --example basic`.

use rand::SeedableRng;
use rand::rngs::StdRng;
use tpt_eng_tolerance::{
    DimTol, Stackup, StackupMember, monte_carlo, rss, rss_contributions, worst_case,
};

fn main() {
    // A three-part shim stack: two symmetric dimensions and one asymmetric
    // machined feature (+0.00 / -0.05).
    let dims = vec![
        DimTol::new("plate A", 12.00, 0.05),
        DimTol::new("plate B", 8.00, 0.03),
        DimTol::asymmetric("machined boss", 25.00, 0.00, 0.05),
    ];

    println!(
        "{:<16}{:>10}{:>10}{:>10}",
        "dimension", "nominal", "min", "max"
    );
    for d in &dims {
        println!(
            "{:<16}{:>10.3}{:>10.3}{:>10.3}",
            d.name,
            d.nominal,
            d.min(),
            d.max()
        );
    }
    let nominal: f64 = dims.iter().map(|d| d.nominal).sum();
    println!("nominal stack = {nominal:.3} mm");

    // --- Worst case: every dimension simultaneously at its limit ------------
    let (wc_lo, wc_hi) = worst_case(&dims);
    println!();
    println!(
        "worst case  = [{wc_lo:.3}, {wc_hi:.3}] mm  (spread {:.3} mm)",
        wc_hi - wc_lo
    );

    // --- RSS: statistical combination, each +/-tol treated as +/-3 sigma ----
    let (rss_lo, rss_hi) = rss(&dims);
    println!(
        "RSS (3 sigma)= [{rss_lo:.3}, {rss_hi:.3}] mm  (spread {:.3} mm)",
        rss_hi - rss_lo
    );
    println!(
        "RSS is {:.3} % of the worst-case spread",
        100.0 * (rss_hi - rss_lo) / (wc_hi - wc_lo)
    );

    // --- Monte Carlo: sample each dimension inside its interval ------------
    // A seeded generator keeps the example reproducible.
    let mut rng = StdRng::seed_from_u64(2026);
    let spec = (44.90, 45.10); // assembly specification limits
    let mc = monte_carlo(&dims, 50_000, Some(spec), &mut rng);
    println!();
    println!(
        "Monte Carlo ({} samples, spec [{:.3}, {:.3}]):",
        mc.n, spec.0, spec.1
    );
    println!("  mean = {:.4} mm, std = {:.4} mm", mc.mean, mc.std);
    println!("  observed range = [{:.4}, {:.4}] mm", mc.min, mc.max);
    println!(
        "  yield = {:.3} % of assemblies within spec",
        100.0 * mc.yield_fraction.unwrap_or(0.0)
    );

    // Variance share of each dimension (all tolerances squared, normalised).
    println!();
    println!("RSS variance contributions:");
    for (d, share) in dims.iter().zip(rss_contributions(&dims)) {
        println!("  {:<16}{:>8.3} %", d.name, 100.0 * share);
    }

    // --- The signed `Stackup` model: members add or subtract ---------------
    // Gap = housing bore 40.00 - shaft 25.00 - spacer 14.80.
    let gap = Stackup::new(vec![
        StackupMember::symmetric(40.00, 0.05, 1.0),
        StackupMember::symmetric(25.00, 0.02, -1.0),
        StackupMember::symmetric(14.80, 0.03, -1.0),
    ]);
    let (gap_lo, gap_hi) = gap.worst_case();
    let (gap_rss_lo, gap_rss_hi) = gap.rss();
    let gap_mc = gap.monte_carlo(100_000, 7);
    println!();
    println!("Signed gap stack-up:");
    println!("  nominal gap   = {:.3} mm", gap.nominal());
    println!("  worst case    = [{gap_lo:.3}, {gap_hi:.3}] mm");
    println!("  RSS           = [{gap_rss_lo:.3}, {gap_rss_hi:.3}] mm");
    println!(
        "  Monte Carlo   = mean {:.4} mm, std {:.4} mm, 3 sigma band [{:.4}, {:.4}] mm",
        gap_mc.mean, gap_mc.std_dev, gap_mc.lower_3sigma, gap_mc.upper_3sigma
    );
    println!("  minimum gap never closes = {}", gap_lo > 0.0);
}
