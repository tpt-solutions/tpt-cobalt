//! Richer scenario: diagnose and fix a failing bearing-preload gap.
//!
//! A five-part axial stack must leave a 0.10..0.30 mm running clearance. The
//! example evaluates the stack, ranks the contributors, estimates sensitivities
//! by Monte Carlo, then re-tightens only the dominant dimension and shows the
//! yield recovered — the usual cost-driven tolerance-allocation loop.
//!
//! All dimensions are millimetres.
//!
//! Run with `cargo run -p tpt-eng-tolerance --example sensitivity`.

use rand::SeedableRng;
use rand::rngs::StdRng;
use tpt_eng_tolerance::{
    DimTol, StackupResult, monte_carlo, monte_carlo_sensitivities, rank_contributors, rss,
    rss_contributions, worst_case,
};

/// Required running clearance at the bearing, mm.
const SPEC: (f64, f64) = (0.10, 0.30);

/// The as-drawn stack: housing depth minus everything fitted inside it.
fn as_drawn() -> Vec<DimTol> {
    vec![
        // Positive contributors: the space available.
        DimTol::new("housing bore depth", 82.00, 0.20),
        // Negative contributors: the parts that fill it.
        DimTol::new("bearing outer race", -20.00, 0.05),
        DimTol::new("spacer sleeve", -31.60, 0.08),
        DimTol::new("shoulder shim", -6.00, 0.03),
        DimTol::asymmetric("retaining cap seat", -24.20, 0.02, 0.06),
    ]
}

/// Evaluate one candidate stack and print the summary line block.
fn evaluate(label: &str, dims: &[DimTol], seed: u64) -> StackupResult {
    let nominal: f64 = dims.iter().map(|d| d.nominal).sum();
    let (wc_lo, wc_hi) = worst_case(dims);
    let (rss_lo, rss_hi) = rss(dims);
    let mut rng = StdRng::seed_from_u64(seed);
    let mc = monte_carlo(dims, 100_000, Some(SPEC), &mut rng);

    println!("--- {label} ---");
    println!("  nominal clearance = {nominal:.4} mm");
    println!(
        "  worst case        = [{wc_lo:.4}, {wc_hi:.4}] mm  (spread {:.4})",
        wc_hi - wc_lo
    );
    println!(
        "  RSS 3 sigma       = [{rss_lo:.4}, {rss_hi:.4}] mm  (spread {:.4})",
        rss_hi - rss_lo
    );
    println!(
        "  Monte Carlo       = mean {:.4}, std {:.4}, range [{:.4}, {:.4}]",
        mc.mean, mc.std, mc.min, mc.max
    );
    println!(
        "  yield vs spec [{:.3}, {:.3}] = {:.3} %",
        SPEC.0,
        SPEC.1,
        100.0 * mc.yield_fraction.unwrap_or(0.0)
    );
    let wc_ok = wc_lo >= SPEC.0 && wc_hi <= SPEC.1;
    println!(
        "  worst-case acceptance = {}",
        if wc_ok {
            "all parts interchangeable"
        } else {
            "some assemblies out of spec"
        }
    );
    mc
}

fn main() {
    let dims = as_drawn();
    println!(
        "Bearing preload gap: {} contributing dimensions\n",
        dims.len()
    );
    println!(
        "{:<22}{:>10}{:>10}{:>10}",
        "dimension", "nominal", "min", "max"
    );
    for d in &dims {
        println!(
            "{:<22}{:>10.3}{:>10.3}{:>10.3}",
            d.name,
            d.nominal,
            d.min(),
            d.max()
        );
    }
    println!();

    let before = evaluate("as drawn", &dims, 11);

    // --- Where does the variation come from? -------------------------------
    println!();
    println!("Contributor ranking (RSS variance share):");
    let shares = rss_contributions(&dims);
    for (rank, (idx, share)) in rank_contributors(&dims).iter().enumerate() {
        println!(
            "  {}. {:<22} tol +/-{:.3} -> {:>7.3} % of the variance",
            rank + 1,
            dims[*idx].name,
            dims[*idx].tol,
            100.0 * share
        );
    }
    println!(
        "  (shares sum to {:.3}, confirming a complete variance budget)",
        shares.iter().sum::<f64>()
    );

    // Monte-Carlo sensitivity: the linear correlation of each input with the
    // resulting clearance. Sign shows the direction of influence.
    let mut rng = StdRng::seed_from_u64(99);
    let corr = monte_carlo_sensitivities(&dims, 50_000, &mut rng);
    println!();
    println!("Monte-Carlo sensitivities (Pearson correlation with the gap):");
    for (d, c) in dims.iter().zip(&corr) {
        println!("  {:<22}{:>8.3}", d.name, c);
    }

    // --- Tolerance allocation: tighten only the dominant dimension ---------
    let worst = rank_contributors(&dims)[0].0;
    println!();
    println!(
        "Tightening `{}` from +/-{:.3} to +/-{:.3} mm (finish bore instead of rough bore)",
        dims[worst].name,
        dims[worst].tol,
        dims[worst].tol / 4.0
    );
    let mut revised = as_drawn();
    revised[worst] = DimTol::new(
        dims[worst].name.clone(),
        dims[worst].nominal,
        dims[worst].tol / 4.0,
    );
    println!();
    let after = evaluate("revised", &revised, 11);

    // --- Outcome -----------------------------------------------------------
    let y_before = before.yield_fraction.unwrap_or(0.0);
    let y_after = after.yield_fraction.unwrap_or(0.0);
    println!();
    println!("Result of the single tolerance change:");
    println!(
        "  yield        {:.3} % -> {:.3} %",
        100.0 * y_before,
        100.0 * y_after
    );
    println!(
        "  variation    std {:.4} -> {:.4} mm ({:.3} % reduction)",
        before.std,
        after.std,
        100.0 * (1.0 - after.std / before.std)
    );
    println!(
        "  scrap rate   {:.3} % -> {:.3} % of assemblies",
        100.0 * (1.0 - y_before),
        100.0 * (1.0 - y_after)
    );
    println!();
    println!("Contributor ranking after the change:");
    for (rank, (idx, share)) in rank_contributors(&revised).iter().enumerate().take(3) {
        println!(
            "  {}. {:<22}{:>7.3} %",
            rank + 1,
            revised[*idx].name,
            100.0 * share
        );
    }
}
