#![no_std]
//! Redundancy / fault-tolerance primitives.
//!
//! These are the building blocks of fault-tolerant systems: take several
//! copies of a value or computation and *vote* on the correct result so a
//! single faulty replica cannot corrupt the output. `no_std`, core-only.

#[cfg(test)]
extern crate std;

/// Return the value agreed on by a strict majority of `votes`, if any.
///
/// With an odd number of votes there is always a strict majority; with an
/// even number, a tie yields `None`.
pub fn majority_vote<T: PartialEq + Copy>(votes: &[T]) -> Option<T> {
    let n = votes.len();
    if n == 0 {
        return None;
    }
    let threshold = n / 2 + 1;
    for i in 0..n {
        let mut count = 0usize;
        for j in 0..n {
            if votes[i] == votes[j] {
                count += 1;
            }
        }
        if count >= threshold {
            return Some(votes[i]);
        }
    }
    None
}

/// Triple Modular Redundancy: three copies voted pairwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tmr<T> {
    a: T,
    b: T,
    c: T,
}

impl<T: PartialEq + Copy> Tmr<T> {
    /// Store three replicas.
    pub fn new(a: T, b: T, c: T) -> Self {
        Tmr { a, b, c }
    }

    /// The agreed value, or `None` if all three differ.
    pub fn vote(&self) -> Option<T> {
        if self.a == self.b || self.a == self.c {
            Some(self.a)
        } else if self.b == self.c {
            Some(self.b)
        } else {
            None
        }
    }
}

/// `N` replicated copies of a value, voted by majority.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Redundant<T, const N: usize> {
    copies: [T; N],
}

impl<T: Copy + PartialEq, const N: usize> Redundant<T, N> {
    /// Store `N` copies. `N` must be at least 1.
    pub fn new(copies: [T; N]) -> Self {
        Redundant { copies }
    }

    /// The number of replicas.
    pub fn len(&self) -> usize {
        N
    }

    /// Whether any replicas are stored (always `true` for `N >= 1`).
    pub fn is_empty(&self) -> bool {
        N == 0
    }

    /// The majority-agreed value, or `None` on a tie with no strict majority.
    pub fn majority(&self) -> Option<T> {
        majority_vote(&self.copies)
    }
}

impl<T, const N: usize> core::ops::Deref for Redundant<T, N> {
    type Target = [T; N];
    #[inline]
    fn deref(&self) -> &[T; N] {
        &self.copies
    }
}

/// A Hamming-style single-bit parity check over a byte slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Parity(u8);

impl Parity {
    /// Compute even parity (1 if the number of set bits is odd).
    pub fn compute(data: &[u8]) -> Self {
        let mut bits = 0u8;
        for &b in data {
            bits ^= b.count_ones() as u8;
        }
        Parity(bits & 1)
    }

    /// The parity bit (0 or 1).
    pub fn bit(&self) -> u8 {
        self.0
    }

    /// Whether `other` matches this parity.
    pub fn matches(&self, other: &Parity) -> bool {
        self.0 == other.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn majority_basic() {
        assert_eq!(majority_vote(&[1u8, 1, 2]), Some(1));
        assert_eq!(majority_vote(&[1u8, 2]), None);
        assert_eq!(majority_vote::<u8>(&[]), None);
    }

    #[test]
    fn tmr_votes() {
        let ok = Tmr::new(10u32, 10, 9).vote();
        assert_eq!(ok, Some(10));
        assert_eq!(Tmr::new(1u32, 2, 3).vote(), None);
    }

    #[test]
    fn redundant_majority() {
        let r = Redundant::<u8, 5>::new([1, 1, 1, 2, 3]);
        assert_eq!(r.majority(), Some(1));
        assert_eq!(r.len(), 5);
    }

    #[test]
    fn parity_detects_change() {
        let p = Parity::compute(&[0b1010_1010, 0b0101_0101]);
        assert!(p.matches(&Parity::compute(&[0b1010_1010, 0b0101_0101])));
        assert!(!p.matches(&Parity::compute(&[0b1010_1011, 0b0101_0101])));
    }
}
