//! Example: runtime monitoring of a message bus.
//!
//! Property: every `req` must eventually be followed by an `ack`:
//!   G (req → F ack)
//! We feed an event stream to a `Monitor` and print its `Verdict`, showing a
//! Satisfied trace, a Violated trace, and an Inconclusive prefix, plus the
//! `Verdict` lattice combinators and one-shot `Trace` checking.
use tpt_for_runtime_verify::{Formula, Monitor, Step, Trace, Verdict};

/// G (req → F ack): every request is eventually acknowledged.
fn req_ack() -> Formula {
    Formula::globally(Formula::implies(
        Formula::atom("req"),
        Formula::eventually(Formula::atom("ack")),
    ))
}

fn main() {
    // Safety property: a FINITE prefix can refute but never confirm `G …`.
    // So even a clean req→ack prefix stays Inconclusive.
    let mut m = Monitor::new(req_ack());
    m.observe(&Step::from_names(["req"]));
    println!("after req alone:      {:?}", m.verdict()); // Inconclusive
    m.observe(&Step::from_names(["ack"]));
    println!("after req→ack:        {:?}", m.verdict()); // still Inconclusive (G)

    // A property that CAN be satisfied: "ack eventually happens" → Satisfied.
    let mut m2 = Monitor::new(Formula::eventually(Formula::atom("ack")));
    m2.observe(&Step::from_names(["req"]));
    m2.observe(&Step::from_names(["ack"]));
    println!("F ack (req→ack):      {:?}", m2.verdict()); // Satisfied

    // A violation: "ok holds globally" — a single bad step breaks it.
    let mut m3 = Monitor::new(Formula::globally(Formula::atom("ok")));
    m3.observe(&Step::from_names(["ok"]));
    println!("G ok (so far ok):     {:?}", m3.verdict()); // Inconclusive
    m3.observe(&Step::from_names(["bad"]));
    println!("G ok (then bad):      {:?}", m3.verdict()); // Violated

    // Combine verdicts with the lattice combinators.
    println!(
        "Violated ∧ Satisfied = {:?}",
        Verdict::Violated.and(Verdict::Satisfied)
    );
    println!(
        "Inconclusive ∨ Satisfied = {:?}",
        Verdict::Inconclusive.or(Verdict::Satisfied)
    );

    // One-shot check of a whole trace and inspecting individual steps.
    let mut trace = Trace::new();
    trace.push(Step::from_names(["req"]));
    trace.push(Step::from_names(["ack"]));
    trace.push(Step::from_names(["req"]));
    trace.push(Step::from_names(["nack"]));
    println!("trace length = {}", trace.len());
    println!("step 0 has 'req'? {}", trace.steps()[0].has("req"));
    println!(
        "one-shot F ack verdict = {:?}",
        Monitor::check(&Formula::eventually(Formula::atom("ack")), &trace)
    );
}
