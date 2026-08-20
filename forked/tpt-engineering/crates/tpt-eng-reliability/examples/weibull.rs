//! Richer example: Weibull life analysis of an engineered component.
//!
//! Models a wind-turbine main-bearing fleet with a Weibull life distribution
//! and reports the engineering quantities a reliability engineer needs: the
//! reliability and failure-rate curves over a service life, the B10 life, and
//! the survival probability through a warranty period. Two candidate designs
//! (different shape/scale) are compared side by side.

use tpt_eng_reliability::{
    weibull_b_life, weibull_failure_rate, weibull_mean, weibull_pdf, weibull_reliability,
};

#[derive(Debug)]
struct Bearing {
    name: &'static str,
    eta: f64,
    beta: f64,
}

fn main() {
    let designs = [
        Bearing {
            name: "Std",
            eta: 40_000.0,
            beta: 1.6,
        },
        Bearing {
            name: "Premium",
            eta: 60_000.0,
            beta: 2.2,
        },
    ];

    let horizon = 20_000.0; // hours of service life to report over
    let step = 2_500.0;
    let warranty = 5_000.0; // hours

    for d in &designs {
        println!(
            "\n=== {} bearing: eta={} h, beta={} ===",
            d.name, d.eta, d.beta
        );
        println!(
            "  mean life        = {:.1} h",
            weibull_mean(d.eta, d.beta).unwrap()
        );
        println!(
            "  B10 life         = {:.1} h",
            weibull_b_life(10.0, d.eta, d.beta).unwrap()
        );
        println!(
            "  R(warranty {:.0} h) = {:.3}",
            warranty,
            weibull_reliability(warranty, d.eta, d.beta).unwrap()
        );

        println!("  t (h)   R(t)    f(t)    h(t)/1e3");
        let mut t = 0.0;
        while t <= horizon {
            let r = weibull_reliability(t, d.eta, d.beta).unwrap();
            let f = weibull_pdf(t, d.eta, d.beta).unwrap();
            let h = weibull_failure_rate(t, d.eta, d.beta).unwrap();
            // Print R/f only when positive to keep the table readable.
            if r > 1e-4 {
                println!("  {:6.0}  {:.3}  {:.3e}  {:.3e}", t, r, f, h / 1e3);
            }
            t += step;
        }
    }
}
