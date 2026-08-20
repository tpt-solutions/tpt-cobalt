//! Example: deterministic, seedable property-based testing.
//!
//! The whole point: a failing property yields a concrete counterexample you can
//! replay *exactly* from its seed — no OS entropy, no flakiness. We (1) verify a
//! property holds, (2) find a real bug and print its counterexample, and (3) prove
//! that re-running with the same seed reproduces the identical counterexample.
use tpt_for_det_proptest::{
    assert_prop, check_prop, CounterExample, DeterministicRng, Gen, Outcome,
};

/// Property: `clamp(x, 10, 100)` always lands inside [10, 100].
fn clamp_ok(x: u64) -> bool {
    let lo = 10u64;
    let hi = 100u64;
    let r = x.clamp(lo, hi);
    r >= lo && r <= hi
}

/// A custom strategy: pairs of u64, demonstrating the `Gen` trait.
struct PairGen;
impl Gen<(u64, u64)> for PairGen {
    fn generate(&self, rng: &mut DeterministicRng) -> (u64, u64) {
        (rng.next_u64(), rng.next_u64())
    }
}

fn main() {
    // --- 1. A passing property --------------------------------------------
    let ok = check_prop(&(), 500, 1, |x: u64| clamp_ok(x));
    match ok {
        Outcome::Passed { cases, .. } => println!("clamp property held over {cases} cases"),
        Outcome::Failed(ce) => println!("unexpected failure: {ce:?}"),
    }

    // --- 2. A failing property → concrete, replayable counterexample ------
    // Buggy "all numbers are even" property; the first odd value the seeded stream
    // produces becomes the counterexample.
    let bad = check_prop(&(), 500, 7, |x: u64| x % 2 == 0);
    let first_ce: CounterExample<u64> = match &bad {
        Outcome::Failed(ce) => {
            println!(
                "counterexample (seed {}): input {} failed at case {}",
                ce.seed, ce.input, ce.case
            );
            ce.clone()
        }
        Outcome::Passed { .. } => unreachable!("the 'even' property is constructed to fail"),
    };

    // --- 3. Determinism: same seed → identical counterexample -------------
    let replay = check_prop(&(), 500, 7, |x: u64| x % 2 == 0);
    let replay_ce: CounterExample<u64> = match replay {
        Outcome::Failed(ce) => ce,
        Outcome::Passed { .. } => unreachable!("the 'even' property is constructed to fail"),
    };
    assert_eq!(first_ce, replay_ce);
    println!(
        "replay with seed {} reproduced the same counterexample (input {}, case {})",
        replay_ce.seed, replay_ce.input, replay_ce.case
    );

    // --- 4. Custom `Gen` + the RNG primitives ------------------------------
    let pair = PairGen.generate(&mut DeterministicRng::new(99));
    let mut rng = DeterministicRng::new(123);
    let flip = rng.gen_bool();
    let roll = rng.gen_range(6);
    let mut buf = [0u8; 4];
    rng.fill_bytes(&mut buf);
    println!("custom PairGen(99) -> {pair:?}");
    println!("gen_bool={flip}, gen_range(6)={roll}, fill_bytes={buf:?}");

    // `assert_prop` is test-friendly: it panics with the counterexample on failure.
    // Here the property always holds, so the run completes silently.
    assert_prop(&(), 50, 3, |x: u64| clamp_ok(x));
    println!("assert_prop ran 50 seeded cases with no counterexample");
}
