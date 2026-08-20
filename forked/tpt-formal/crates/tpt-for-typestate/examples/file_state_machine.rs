//! Example: encode program invariants in the type system so illegal states are
//! unrepresentable, plus runtime range checks and safe numeric casts.
//!
//! Scenario: a `File` that must be `Open` before reading, and an integer
//! `Percent` that must stay in `0..=100`. The typestate transitions show the
//! valid ordering (`open` then `read`); the invalid ordering is impossible to
//! even write. The runtime/checked fallbacks catch mistakes the type system
//! cannot (e.g. a value arriving from I/O).

use tpt_for_typestate::bounded::{Bound, Bounded, Checked};
use tpt_for_typestate::ghost::{State, Stateful, Token};
use tpt_for_typestate::newtype::Newtype;
use tpt_for_typestate::phantom::{invariant, Contravariant, Covariant};
use tpt_for_typestate::safe_cast::{SafeCast, TrySafeCast};

// ---- Typestate: a file is either `Closed` or `Open` ----
#[derive(Clone, Copy)]
struct Closed;
#[derive(Clone, Copy)]
struct Open;
impl State for Closed {}
impl State for Open {}

/// Open the file: consumes proof that it is `Closed`, yields proof it is `Open`.
/// The transition is checked entirely at compile time (zero runtime cost).
fn open(_t: Token<Closed>) -> Token<Open> {
    Token::new()
}
/// Read requires proof the file is `Open`; calling it with a `Closed` token is a
/// compile error, so the invalid transition is unrepresentable.
fn read(_t: Token<Open>) -> usize {
    42
}

// A type-level bound: percentages live in 0..=100.
struct Percent;
impl Bound for Percent {
    type Value = u8;
    const MIN: u8 = 0;
    const MAX: u8 = 100;
}

tpt_for_typestate::define_newtype!(Meters, u64);

fn main() {
    // 1) Typestate transition (valid path). We *validate* the raw handle once,
    //    then the type system guarantees every later operation matches its
    //    required state. `open`/`read` cannot be called out of order.
    let handle: Stateful<Closed, u32> = unsafe { Stateful::<Closed, u32>::new_unchecked(7) };
    let _raw = handle.into_inner();
    let closed = Token::<Closed>::new();
    let opened = open(closed);
    println!("read returned {} bytes", read(opened));

    // 2) Runtime range check: rejects out-of-range input, clamps when saturating.
    let b = Bounded::new(5u8, 0, 10).expect("in range");
    let sat = b.saturate(20).get();
    let rej = Bounded::new(11u8, 0, 10).is_none();
    println!("bounded = {}, saturate(20) = {sat}", b.get());
    println!("out-of-range rejected: {rej}");

    // 3) Type-level named bound: the range is part of the type.
    let pct = Checked::<Percent>::new(50);
    let pct_rej = Checked::<Percent>::new(101).is_none();
    println!(
        "percent 50 ok = {}, 101 rejected = {pct_rej}",
        pct.is_some()
    );

    // 4) Transparent newtype with `From`/`Into` glue and `Deref`.
    let m = Meters::from_inner(3);
    println!("meters = {} (via deref)", *m);

    // 5) Safe casts: widening is infallible; narrowing fails instead of truncating.
    let wide: u64 = 200u8.safe_cast();
    let narrow: Result<u8, _> = 300u16.try_safe_cast();
    println!("200u8 -> u64 = {wide}; 300u16 -> u8 = {narrow:?}");

    // 6) Variance-controlled phantom markers (zero-sized building blocks for
    //    precise typestate/variance encodings).
    let _inv = invariant::<u32>();
    let _cov: Covariant<u32> = Covariant::new();
    let _con: Contravariant<u8> = Contravariant::new();
    println!(
        "phantom markers are zero-sized: {}",
        core::mem::size_of::<Contravariant<u8>>() == 0
    );
}
