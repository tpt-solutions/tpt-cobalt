//! Example: the contract-guarded algorithms, including their failure paths.
//!
//! Every routine is wrapped in `tpt-for-contract` precondition / postcondition
//! checks. The happy path returns a correctly-computed result; the failure path
//! is a *panic* — the contract refusing bad input instead of producing a
//! silently-wrong answer. We surface that panic explicitly rather than unwrap it
//! away.

use std::panic;
use tpt_for_verified_algorithms::{binary_search, clamp, gcd, insertion_sort};

fn main() {
    // --- gcd: Euclid's algorithm (panics if both args are zero). -----------
    println!("gcd(12, 18) = {}  (expect 6)", gcd(12, 18));
    println!("gcd(17, 0)  = {}  (expect 17)", gcd(17, 0));

    // --- clamp: saturate into the inclusive range [lo, hi]. ---------------
    println!("clamp(5, 0, 10)  = {}  (expect 5)", clamp(5, 0, 10));
    println!("clamp(-3, 0, 10) = {}  (expect 0)", clamp(-3, 0, 10));
    println!("clamp(99, 0, 10) = {}  (expect 10)", clamp(99, 0, 10));

    // --- binary_search: requires a SORTED slice. --------------------------
    let data = [1, 3, 5, 7, 9];
    println!(
        "binary_search({data:?}, 5) = {:?}  (expect Some(2))",
        binary_search(&data, 5)
    );
    println!(
        "binary_search({data:?}, 4) = {:?}  (expect None)",
        binary_search(&data, 4)
    );

    // --- insertion_sort: returns a sorted permutation. ---------------------
    let input = [3, 1, 2, 5, 4];
    let sorted = insertion_sort(&input);
    let ok_sorted = sorted.windows(2).all(|w| w[0] <= w[1]);
    println!("insertion_sort({input:?}) = {sorted:?}  (sorted? {ok_sorted})");

    // --- Failure path: the contracts panic rather than be silently wrong. --
    // gcd(0, 0) violates "at least one argument non-zero".
    let res = panic::catch_unwind(|| gcd(0, 0));
    println!("gcd(0,0) rejected by precondition? {}", res.is_err());

    // An unsorted slice also trips binary_search's precondition.
    let unsorted = [3, 1, 2];
    let res = panic::catch_unwind(|| binary_search(&unsorted, 2));
    println!(
        "binary_search on unsorted slice rejected by precondition? {}",
        res.is_err()
    );
}
