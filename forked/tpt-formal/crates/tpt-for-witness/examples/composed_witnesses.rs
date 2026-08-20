//! Example: bundle values with compile-time proofs that a predicate holds, and
//! compose predicates.
//!
//! Construction is gated by a runtime check, so a `Witness<P, T>` is only built
//! when `P` actually holds. We show single predicates, a non-empty slice, the
//! `And` combinator (with which side failed), and `map` preserving the proof.

use tpt_for_witness::{And, NonEmpty, NonNegative, NonZero, Positive, Witness};

fn main() {
    // --- A single predicate: strictly positive. ----------------------------
    match Witness::<Positive, i64>::try_new(42) {
        Ok(w) => println!("Positive(42) built: value = {}", w.into_inner()),
        Err(()) => println!("Positive(42) rejected"),
    }
    match Witness::<Positive, i64>::try_new(-3) {
        Ok(w) => println!("Positive(-3) built: {}", w.into_inner()),
        Err(()) => println!("Positive(-3) rejected (not > 0)"),
    }

    // --- A non-empty slice, verified at construction. ----------------------
    let data = [1u8, 2, 3];
    let w = Witness::<NonEmpty, _>::try_new(&data[..]).expect("non-empty");
    println!("NonEmpty slice has {} elements", w.as_ref().len());
    let empty: &[u8] = &[];
    println!(
        "NonEmpty empty slice rejected? {}",
        Witness::<NonEmpty, _>::try_new(empty).is_err()
    );

    // --- Compose predicates with And, and inspect WHICH side failed. -------
    type NonNegNonZero = And<NonNegative, NonZero>;
    for v in [5i32, 0, -1] {
        match Witness::<NonNegNonZero, i32>::try_new(v) {
            Ok(w) => println!("NonNegNonZero({v}) ok -> {}", w.into_inner()),
            Err(e) => println!(
                "NonNegNonZero({v}) rejected: non_negative_failed={}, non_zero_failed={}",
                e.p.is_some(),
                e.q.is_some()
            ),
        }
    }

    // --- map preserves the predicate: 5 (positive) doubled is still positive.
    let w = Witness::<Positive, i64>::try_new(5).unwrap();
    // SAFETY: doubling a positive integer stays positive, so P is preserved.
    let w2 = unsafe { w.map(|x| x * 2) };
    println!("mapped Positive(5) -> {} (still positive)", w2.into_inner());
}
