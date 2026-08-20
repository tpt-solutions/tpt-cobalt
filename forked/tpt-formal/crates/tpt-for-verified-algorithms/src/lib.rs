//! Verified algorithm implementations.
//!
//! Each routine is guarded by [`tpt_for_contract`] pre/post/invariant checks,
//! so calling it with inputs that violate its contract panics instead of
//! producing a silently-wrong result. The checks are written against the crate's
//! MSRV (1.75), avoiding newer std helpers.

use tpt_for_contract::{ensures, invariant, requires};

/// Greatest common divisor of `a` and `b` (Euclid's algorithm).
///
/// # Preconditions
///
/// `a` and `b` are not both zero.
///
/// # Postconditions
///
/// The result is strictly positive and divides both inputs.
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    requires!(a != 0 || b != 0, "at least one argument must be non-zero");
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    let result = a;
    ensures!(result > 0);
    result
}

/// Clamp `x` into the inclusive range `[lo, hi]`.
///
/// # Preconditions
///
/// `lo <= hi`.
///
/// # Postconditions
///
/// The result lies within `[lo, hi]` and between `x` and the nearer bound.
pub fn clamp(x: i64, lo: i64, hi: i64) -> i64 {
    requires!(lo <= hi);
    let result = if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    };
    ensures!(result >= lo && result <= hi);
    ensures!(result == x || result == lo || result == hi);
    result
}

/// Binary search `target` in a sorted `slice`, returning its index or `None`.
///
/// # Preconditions
///
/// `slice` is sorted in non-decreasing order.
///
/// # Postconditions
///
/// If `Some(i)` is returned, `slice[i] == target`.
pub fn binary_search(slice: &[i64], target: i64) -> Option<usize> {
    requires!(is_sorted(slice));
    let mut lo = 0usize;
    let mut hi = slice.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match slice[mid].cmp(&target) {
            core::cmp::Ordering::Equal => {
                ensures!(slice[mid] == target);
                return Some(mid);
            }
            core::cmp::Ordering::Less => lo = mid + 1,
            core::cmp::Ordering::Greater => hi = mid,
        }
    }
    None
}

/// Insertion sort: returns a non-decreasing permutation of `input`.
///
/// # Postconditions
///
/// The result is sorted and has the same length as `input`.
pub fn insertion_sort(input: &[i64]) -> Vec<i64> {
    let mut out: Vec<i64> = Vec::with_capacity(input.len());
    for &x in input {
        let pos = out.iter().position(|&y| y > x).unwrap_or(out.len());
        out.insert(pos, x);
        invariant!(is_sorted(&out));
    }
    ensures!(is_sorted(&out));
    ensures!(out.len() == input.len());
    out
}

/// MSRV-safe ascending-sorted check (avoids `slice::is_sorted`, stable in 1.82).
fn is_sorted(slice: &[i64]) -> bool {
    slice.windows(2).all(|w| w[0] <= w[1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_for_contract::debug_requires;

    #[test]
    fn gcd_works() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(17, 0), 17);
    }

    #[test]
    #[should_panic(expected = "precondition")]
    fn gcd_rejects_both_zero() {
        gcd(0, 0);
    }

    #[test]
    fn clamp_works() {
        assert_eq!(clamp(5, 0, 10), 5);
        assert_eq!(clamp(-3, 0, 10), 0);
        assert_eq!(clamp(99, 0, 10), 10);
    }

    #[test]
    fn binary_search_find_and_miss() {
        let data = [1, 3, 5, 7, 9];
        assert_eq!(binary_search(&data, 5), Some(2));
        assert_eq!(binary_search(&data, 4), None);
    }

    #[test]
    fn binary_search_respects_precondition() {
        // The precondition accepts a sorted slice.
        let sorted = [1, 2, 3];
        debug_requires!(is_sorted(&sorted));
        assert_eq!(binary_search(&sorted, 2), Some(1));
    }

    #[test]
    fn insertion_sort_sorts() {
        let r = insertion_sort(&[3, 1, 2, 5, 4]);
        assert_eq!(r, vec![1, 2, 3, 4, 5]);
    }
}
