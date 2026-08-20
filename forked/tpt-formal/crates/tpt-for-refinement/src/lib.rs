#![no_std]
//! Refinement types.
//!
//! A [`Refined<T, P>`] is a value of type `T` that is known to satisfy the
//! predicate `P`. Construction is the *only* way to create one, and it refuses
//! to wrap a value that fails `P::check`, so any `Refined` in scope is a proof
//! that its predicate holds.
//!
//! Refinement predicates compose with [`tpt_for_contract`]'s [`Invariant`]
//! trait: a type implementing `Invariant` can be used directly as a refinement
//! predicate via the blanket [`Predicate`] impl below.

use core::marker::PhantomData;
use tpt_for_contract::Invariant;

/// A predicate that a refinement type requires of its inner value.
pub trait Predicate<T> {
    /// Return `true` if `value` satisfies this refinement.
    fn check(value: &T) -> bool;
}

/// Blanket: any type's own [`Invariant`] is a valid refinement predicate.
impl<T: Invariant> Predicate<T> for T {
    #[inline]
    fn check(value: &T) -> bool {
        value.check()
    }
}

/// A value of type `T` refined by predicate `P`.
#[derive(Clone, Copy)]
pub struct Refined<T, P> {
    value: T,
    _marker: PhantomData<fn() -> P>,
}

/// Error returned when a value fails its refinement predicate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefineError {
    /// Source text of the failing predicate (best-effort).
    pub predicate: &'static str,
}

impl<T, P: Predicate<T>> Refined<T, P> {
    /// Refine `value`, returning `Err` if `P::check(&value)` is false.
    #[inline]
    pub fn new(value: T) -> Result<Self, RefineError> {
        if P::check(&value) {
            Ok(Refined {
                value,
                _marker: PhantomData,
            })
        } else {
            Err(RefineError {
                predicate: core::any::type_name::<P>(),
            })
        }
    }

    /// Construct without checking. Prefer [`Refined::new`].
    ///
    /// # Safety
    ///
    /// `value` must satisfy `P`.
    #[inline]
    pub unsafe fn new_unchecked(value: T) -> Self {
        Refined {
            value,
            _marker: PhantomData,
        }
    }

    /// Borrow the refined value (still satisfies `P`).
    #[inline]
    pub fn get(&self) -> &T {
        &self.value
    }

    /// Consume the refinement and recover the inner value.
    #[inline]
    pub fn into_inner(self) -> T {
        self.value
    }

    /// Map the inner value, preserving the refinement `P`.
    ///
    /// # Safety
    ///
    /// `f` must map a `P`-satisfying value to another `P`-satisfying value.
    #[inline]
    pub unsafe fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Refined<U, P>
    where
        P: Predicate<U>,
    {
        Refined {
            value: f(self.value),
            _marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_for_contract::Invariant;

    struct Positive;
    impl Predicate<i64> for Positive {
        fn check(value: &i64) -> bool {
            *value > 0
        }
    }

    struct NonEmpty<T> {
        data: T,
    }
    impl<T: AsRef<[u8]>> NonEmpty<T> {
        fn len(&self) -> usize {
            self.data.as_ref().len()
        }
    }
    impl<T: AsRef<[u8]>> Invariant for NonEmpty<T> {
        fn check(&self) -> bool {
            !self.data.as_ref().is_empty()
        }
    }

    #[test]
    fn refines_when_predicate_holds() {
        let r = Refined::<i64, Positive>::new(5).unwrap();
        assert_eq!(*r.get(), 5);
        assert!(Refined::<i64, Positive>::new(-1).is_err());
    }

    #[test]
    fn invariant_as_predicate() {
        let ne = NonEmpty { data: [1u8, 2, 3] };
        let r = Refined::<NonEmpty<[u8; 3]>, NonEmpty<[u8; 3]>>::new(ne).unwrap();
        assert_eq!(r.get().len(), 3);
    }
}
