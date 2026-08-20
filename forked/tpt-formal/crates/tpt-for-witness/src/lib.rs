#![no_std]
//! Witness types.
//!
//! A *witness* bundles a value with a zero-sized proof that some [`Predicate`]
//! holds of it. Construction is gated by the predicate's runtime or trusted
//! check, so a function receiving a `Witness<P, T>` can rely on `P` being true
//! without re-checking.
//!
//! Built on [`tpt_for_typestate`] for variance-correct phantom storage.

use core::marker::PhantomData;
use tpt_for_typestate::phantom::Contravariant;

/// A predicate over `T`.
///
/// Implementors describe how to *verify* the property they stand for. A
/// witness is only constructible when [`Predicate::check`] succeeds (or via a
/// trusted `unsafe` constructor for code that has already proven it).
pub trait Predicate<T: ?Sized>: Sized {
    /// Error returned when the predicate does not hold.
    type Error;
    /// Verify the predicate holds for `value`.
    fn check(value: &T) -> Result<(), Self::Error>;
}

/// A value `T` together with a compile-time proof that `P: Predicate<T>` holds.
#[derive(Clone, Copy)]
pub struct Witness<P, T> {
    _pred: Contravariant<P>,
    value: T,
}

impl<P: Predicate<T>, T> Witness<P, T> {
    /// Construct by *verifying* the predicate at runtime.
    ///
    /// Returns the predicate's error if it does not hold.
    #[inline]
    pub fn try_new(value: T) -> Result<Self, P::Error> {
        P::check(&value)?;
        // Safe: we just verified P holds.
        Ok(unsafe { Self::new_unchecked(value) })
    }

    /// Construct without checking. Prefer [`Witness::try_new`].
    ///
    /// # Safety
    ///
    /// `value` must satisfy `P`.
    #[inline]
    pub unsafe fn new_unchecked(value: T) -> Self {
        Witness {
            _pred: Contravariant::new(),
            value,
        }
    }

    /// Consume the witness and recover the inner value.
    #[inline]
    pub fn into_inner(self) -> T {
        self.value
    }

    /// Transform the inner value while the predicate is preserved.
    ///
    /// # Safety
    ///
    /// `f` must map a `P`-satisfying value to another `P`-satisfying value.
    #[inline]
    pub unsafe fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Witness<P, U> {
        Witness {
            _pred: Contravariant::new(),
            value: f(self.value),
        }
    }
}

impl<P, T> AsRef<T> for Witness<P, T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &self.value
    }
}

/// Logical AND of two predicates.
pub struct And<P, Q>(PhantomData<(P, Q)>);

impl<P, Q, T> Predicate<T> for And<P, Q>
where
    P: Predicate<T>,
    Q: Predicate<T>,
{
    type Error = AndError<P::Error, Q::Error>;
    #[inline]
    fn check(value: &T) -> Result<(), Self::Error> {
        match (P::check(value), Q::check(value)) {
            (Ok(()), Ok(())) => Ok(()),
            (e1, e2) => Err(AndError {
                p: e1.err(),
                q: e2.err(),
            }),
        }
    }
}

/// Error combining the failures of two [`And`] predicates.
pub struct AndError<P, Q> {
    /// Failure of the left predicate, if any.
    pub p: Option<P>,
    /// Failure of the right predicate, if any.
    pub q: Option<Q>,
}

// --- Concrete predicates -------------------------------------------------

/// Predicate: a number is strictly positive (`> 0`).
pub struct Positive;

/// Predicate: a number is non-negative (`>= 0`).
pub struct NonNegative;

/// Predicate: a number is non-zero.
pub struct NonZero;

impl<T: PartialOrd + core::cmp::PartialEq + Copy + crate::num::Zero> Predicate<T> for Positive {
    type Error = ();
    #[inline]
    fn check(value: &T) -> Result<(), ()> {
        if *value > T::zero() {
            Ok(())
        } else {
            Err(())
        }
    }
}

impl<T: PartialOrd + core::cmp::PartialEq + Copy + crate::num::Zero> Predicate<T> for NonNegative {
    type Error = ();
    #[inline]
    fn check(value: &T) -> Result<(), ()> {
        if *value >= T::zero() {
            Ok(())
        } else {
            Err(())
        }
    }
}

impl<T: core::cmp::PartialEq + Copy + crate::num::Zero> Predicate<T> for NonZero {
    type Error = ();
    #[inline]
    fn check(value: &T) -> Result<(), ()> {
        if *value != T::zero() {
            Ok(())
        } else {
            Err(())
        }
    }
}

/// Predicate: a slice/collection is non-empty.
pub struct NonEmpty;

/// Minimal zero-value helper for the numeric predicates.
pub mod num {
    /// A type that has a zero value.
    pub trait Zero {
        /// The additive identity.
        fn zero() -> Self;
    }
    impl Zero for u8 {
        fn zero() -> Self {
            0
        }
    }
    impl Zero for u16 {
        fn zero() -> Self {
            0
        }
    }
    impl Zero for u32 {
        fn zero() -> Self {
            0
        }
    }
    impl Zero for u64 {
        fn zero() -> Self {
            0
        }
    }
    impl Zero for u128 {
        fn zero() -> Self {
            0
        }
    }
    impl Zero for i8 {
        fn zero() -> Self {
            0
        }
    }
    impl Zero for i16 {
        fn zero() -> Self {
            0
        }
    }
    impl Zero for i32 {
        fn zero() -> Self {
            0
        }
    }
    impl Zero for i64 {
        fn zero() -> Self {
            0
        }
    }
    impl Zero for i128 {
        fn zero() -> Self {
            0
        }
    }
    impl Zero for f32 {
        fn zero() -> Self {
            0.0
        }
    }
    impl Zero for f64 {
        fn zero() -> Self {
            0.0
        }
    }
}

impl<T> Predicate<[T]> for NonEmpty {
    type Error = ();
    #[inline]
    fn check(value: &[T]) -> Result<(), ()> {
        if value.is_empty() {
            Err(())
        } else {
            Ok(())
        }
    }
}

impl<T> Predicate<&[T]> for NonEmpty {
    type Error = ();
    #[inline]
    fn check(value: &&[T]) -> Result<(), ()> {
        if value.is_empty() {
            Err(())
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_witness() {
        let w = Witness::<Positive, i32>::try_new(5).unwrap();
        assert_eq!(*w.as_ref(), 5);
        assert!(Witness::<Positive, i32>::try_new(-1).is_err());
    }

    #[test]
    fn non_empty_witness() {
        let data = [1u8, 2, 3];
        let w = Witness::<NonEmpty, _>::try_new(&data[..]).unwrap();
        assert_eq!(w.as_ref().len(), 3);
        let empty: &[u8] = &[];
        assert!(Witness::<NonEmpty, _>::try_new(empty).is_err());
    }

    #[test]
    fn and_combines() {
        type NonNegNonZero = And<NonNegative, NonZero>;
        assert!(Witness::<NonNegNonZero, i32>::try_new(5).is_ok());
        assert!(Witness::<NonNegNonZero, i32>::try_new(0).is_err());
        assert!(Witness::<NonNegNonZero, i32>::try_new(-1).is_err());
    }

    #[test]
    fn non_zero_witness() {
        assert!(Witness::<NonZero, u32>::try_new(1).is_ok());
        assert!(Witness::<NonZero, u32>::try_new(0).is_err());
    }
}
