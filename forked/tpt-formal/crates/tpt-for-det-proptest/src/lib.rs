//! Deterministic, seedable property-based testing.
//!
//! Unlike fuzzing harnesses that draw entropy from the OS, this crate drives
//! case generation from an explicit 64-bit seed via a fixed PRNG. The same
//! seed always produces the same sequence of cases, so a failing property can
//! be reproduced exactly.

use core::num::Wrapping;

/// A small, fast, fully deterministic PRNG (xorshift64*).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeterministicRng {
    state: Wrapping<u64>,
}

impl DeterministicRng {
    /// Create a generator from a seed. A zero seed is mapped to a non-zero
    /// state so the generator is always well-defined.
    #[inline]
    pub fn new(seed: u64) -> Self {
        DeterministicRng {
            state: Wrapping(if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            }),
        }
    }

    /// Advance the generator and return the next 64-bit value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = Wrapping(x);
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Produce a `u64` in `[0, n)`.
    #[inline]
    pub fn gen_range(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        // Lemire's method — unbiased and branch-light.
        let m = (self.next_u64() as u128) * (n as u128);
        (m >> 64) as u64
    }

    /// Produce a `bool`.
    #[inline]
    pub fn gen_bool(&mut self) -> bool {
        (self.next_u64() & 1) == 1
    }

    /// Fill a byte buffer.
    #[inline]
    pub fn fill_bytes(&mut self, buf: &mut [u8]) {
        let mut i = 0;
        while i < buf.len() {
            let word = self.next_u64();
            for b in word.to_le_bytes() {
                if i >= buf.len() {
                    break;
                }
                buf[i] = b;
                i += 1;
            }
        }
    }
}

/// A strategy that produces values of `T` from a deterministic RNG.
pub trait Gen<T> {
    /// Generate the next value.
    fn generate(&self, rng: &mut DeterministicRng) -> T;
}

impl Gen<u64> for () {
    #[inline]
    fn generate(&self, rng: &mut DeterministicRng) -> u64 {
        rng.next_u64()
    }
}

impl Gen<u32> for () {
    #[inline]
    fn generate(&self, rng: &mut DeterministicRng) -> u32 {
        rng.next_u64() as u32
    }
}

impl Gen<i64> for () {
    #[inline]
    fn generate(&self, rng: &mut DeterministicRng) -> i64 {
        rng.next_u64() as i64
    }
}

impl Gen<bool> for () {
    #[inline]
    fn generate(&self, rng: &mut DeterministicRng) -> bool {
        rng.gen_bool()
    }
}

/// A counterexample captured when a property fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CounterExample<T> {
    /// The input that failed the property.
    pub input: T,
    /// The seed the generator was started with.
    pub seed: u64,
    /// The zero-based case index that failed.
    pub case: u64,
}

/// The outcome of running a property.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome<T> {
    /// All generated cases satisfied the property.
    Passed {
        /// Number of cases checked.
        cases: u64,
        /// Seed used.
        seed: u64,
    },
    /// A case violated the property.
    Failed(CounterExample<T>),
}

/// Run `prop` over `cases` generated values from `gen`, seeded by `seed`.
///
/// Returns [`Outcome::Passed`] if every case satisfies `prop`, or the first
/// [`Outcome::Failed`] counterexample otherwise. The run is fully determined
/// by `(gen, cases, seed)`.
pub fn check_prop<T, G, F>(gen: &G, cases: u64, seed: u64, prop: F) -> Outcome<T>
where
    T: Clone,
    G: Gen<T>,
    F: Fn(T) -> bool,
{
    let mut rng = DeterministicRng::new(seed);
    for case in 0..cases {
        let input = gen.generate(&mut rng);
        if !prop(input.clone()) {
            return Outcome::Failed(CounterExample { input, seed, case });
        }
    }
    Outcome::Passed { cases, seed }
}

/// Check a property and panic on the first failure (test-friendly).
///
/// # Panics
///
/// Panics with the counterexample if the property is violated.
pub fn assert_prop<T, G, F>(gen: &G, cases: u64, seed: u64, prop: F)
where
    G: Gen<T>,
    F: Fn(T) -> bool,
    T: core::fmt::Debug + Clone,
{
    match check_prop(gen, cases, seed, prop) {
        Outcome::Passed { .. } => {}
        Outcome::Failed(ce) => panic!(
            "property failed at case {} (seed {}): {:?}",
            ce.case, ce.seed, ce.input
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic() {
        let mut a = DeterministicRng::new(42);
        let mut b = DeterministicRng::new(42);
        assert_eq!(a.next_u64(), b.next_u64());
        assert_eq!(a.gen_range(100), b.gen_range(100));
    }

    #[test]
    fn property_passes() {
        let out = check_prop(&(), 100, 1, |x: u64| x.wrapping_add(1) != x.wrapping_sub(1));
        assert!(matches!(out, Outcome::Passed { .. }));
    }

    #[test]
    fn property_finds_counterexample() {
        let out = check_prop(&(), 100, 7, |x: u64| x % 2 == 0);
        match out {
            Outcome::Failed(ce) => assert!(ce.input % 2 != 0),
            Outcome::Passed { .. } => panic!("expected failure"),
        }
    }
}
