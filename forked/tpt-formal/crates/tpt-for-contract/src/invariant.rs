//! Struct invariants.

/// A value that carries a checkable invariant.
///
/// Implement this for a type to declare the condition that must hold for every
/// valid instance. Use [`invariant!`](crate::invariant!) or
/// [`check_invariant`](crate::invariant::check_invariant) to assert it.
///
/// ```
/// use tpt_for_contract::Invariant;
///
/// struct NonEmptyVec<T> { data: Vec<T> }
/// impl<T> Invariant for NonEmptyVec<T> {
///     fn check(&self) -> bool { !self.data.is_empty() }
/// }
/// ```
pub trait Invariant {
    /// Return `true` if this value satisfies its invariant.
    fn check(&self) -> bool;
}

/// Check `value`'s invariant, panicking if it does not hold.
///
/// # Panics
///
/// Panics (via [`invariant!`](crate::invariant!)) if
/// `value.check()` returns `false`.
#[macro_export]
macro_rules! check_invariant {
    ($value:expr $(,)?) => {{
        let v = &$value;
        $crate::invariant!($crate::Invariant::check(v));
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate as tpt_for_contract;

    struct Bounded {
        lo: i32,
        hi: i32,
    }
    impl Invariant for Bounded {
        fn check(&self) -> bool {
            self.lo <= self.hi
        }
    }

    #[test]
    fn preconditions_pass() {
        let x = 5;
        tpt_for_contract::requires!(x > 0);
        tpt_for_contract::requires!(x < 10);
    }

    #[test]
    #[should_panic(expected = "precondition")]
    fn precondition_fails() {
        let x = -1;
        tpt_for_contract::requires!(x > 0);
    }

    #[test]
    fn postcondition_and_invariant() {
        let b = Bounded { lo: 0, hi: 10 };
        tpt_for_contract::invariant!(b.check());
        tpt_for_contract::check_invariant!(b);
    }

    #[test]
    #[should_panic(expected = "invariant")]
    fn invariant_fails() {
        let b = Bounded { lo: 10, hi: 0 };
        tpt_for_contract::check_invariant!(b);
    }

    #[test]
    fn loop_invariant_holds() {
        let mut sum = 0i32;
        let mut i = 0i32;
        tpt_for_contract::loop_invariant!(sum == i * (i - 1) / 2 || i == 0);
        while i < 5 {
            tpt_for_contract::loop_invariant!(sum >= 0);
            sum += i;
            i += 1;
        }
        assert_eq!(sum, 10);
    }
}
