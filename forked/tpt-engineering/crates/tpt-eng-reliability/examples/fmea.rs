//! Example: rank FMEA items and report a Weibull life metric.

use tpt_eng_reliability::{FmeaItem, rank_by_rpn, weibull_b_life, weibull_reliability};

fn main() {
    let items = vec![
        FmeaItem::new("A", "pump", "leak", "seal wear", "fluid loss", 7, 3, 4),
        FmeaItem::new("B", "pump", "stall", "bearing seize", "no flow", 9, 2, 5),
        FmeaItem::new("C", "valve", "stick", "corrosion", "reduced flow", 4, 2, 2),
    ];

    println!("FMEA ranking (highest RPN first):");
    for it in rank_by_rpn(&items) {
        println!(
            "  {}  RPN={}  (S{} O{} D{})",
            it.id,
            it.rpn(),
            it.severity,
            it.occurrence,
            it.detection
        );
    }

    // Weibull life: eta = 1000 h, beta = 2.0.
    let r = weibull_reliability(100.0, 1000.0, 2.0).unwrap();
    let b10 = weibull_b_life(10.0, 1000.0, 2.0).unwrap();
    println!("Weibull R(100 h) = {:.4}, B10 life = {:.1} h", r, b10);
}
