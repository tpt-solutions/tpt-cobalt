//! Example: refinement types guarantee a value satisfies a predicate.
//!
//! We refine a port number to the valid `1..=65535` range, and refine a payload
//! to be non-empty using an `Invariant` from `tpt-for-contract`. A rejected value
//! yields a `RefineError` we print rather than `unwrap()` away, and we show the
//! proof-preserving `map` and the value-recovering `into_inner`.
use tpt_for_contract::Invariant;
use tpt_for_refinement::{Predicate, Refined};

/// Port numbers must be in the dynamic/private range 1..=65535.
struct ValidPort;
impl Predicate<u16> for ValidPort {
    fn check(v: &u16) -> bool {
        (1..=65535).contains(v)
    }
}

/// A buffer that must never be empty (its own `Invariant`).
struct NonEmptyBuf {
    data: Vec<u8>,
}
impl Invariant for NonEmptyBuf {
    fn check(&self) -> bool {
        !self.data.is_empty()
    }
}

fn main() {
    // Valid construction: the value is known to satisfy the predicate.
    let port = Refined::<u16, ValidPort>::new(8080).expect("8080 is a valid port");
    println!("refined port = {}", port.get());

    // Rejected value: construction returns Err describing the failing predicate.
    match Refined::<u16, ValidPort>::new(0) {
        Ok(_) => println!("unexpected: 0 is not a valid port"),
        Err(e) => println!("rejected 0: {:?} (predicate `{}`)", e, e.predicate),
    }

    // An `Invariant` type composes directly as a refinement predicate (blanket impl).
    let buf = Refined::<NonEmptyBuf, NonEmptyBuf>::new(NonEmptyBuf {
        data: vec![1, 2, 3],
    })
    .expect("buffer is non-empty");
    println!("refined buffer has {} bytes", buf.get().data.len());

    // `map` preserves the predicate (unsafe: caller vouches the mapping keeps
    // the value valid). It consumes the refined value, returning a new one.
    let doubled = unsafe { port.map(|v| v.wrapping_mul(2)) };
    println!("mapped port = {}", doubled.get());

    // `into_inner` recovers the raw value, abandoning the proof wrapper.
    let raw = Refined::<u16, ValidPort>::new(443).unwrap().into_inner();
    println!("raw port = {}", raw);
}
