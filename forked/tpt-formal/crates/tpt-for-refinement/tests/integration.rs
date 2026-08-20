//! Cross-crate integration: a `Refined` value must satisfy both its refinement
//! predicate and an arbitrary `tpt-for-contract` invariant, demonstrating the two
//! pillars composing without `tpt-for-contract` being a direct dependency of the
//! test (it is reachable via `tpt-for-refinement`).

use tpt_for_contract::invariant;
use tpt_for_refinement::{Predicate, Refined};

struct Positive;
impl Predicate<f64> for Positive {
    fn check(value: &f64) -> bool {
        *value > 0.0
    }
}

#[test]
fn refined_value_satisfies_contract_invariant() {
    let r = Refined::<f64, Positive>::new(3.0).unwrap();
    // The refinement predicate holds (enforced at construction)...
    assert!(*r.get() > 0.0);
    // ...and so does an arbitrary design-by-contract invariant on the value.
    invariant!(*r.get() < 10.0);
    assert_eq!(*r.get(), 3.0);
}

#[test]
fn rejected_value_is_not_refined() {
    assert!(Refined::<f64, Positive>::new(-1.0).is_err());
}
