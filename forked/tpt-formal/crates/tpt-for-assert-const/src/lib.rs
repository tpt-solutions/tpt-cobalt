#![no_std]
//! Compile-time assertions.
//!
//! Assert invariants while compiling rather than while running:
//!
//! - [`const_assert!`], [`const_assert_eq!`], [`const_assert_ne!`] evaluate a
//!   boolean (or equality) at compile time and fail the build if it is false.
//! - The [`Const`]/[`IsTrue`]/`[`Same`]` type-level machinery lets a trait
//!   bound *require* a boolean or type equality to hold.
//!
//! Everything is `no_std`.

/// Assert a boolean expression at compile time.
///
/// ```
/// tpt_for_assert_const::const_assert!(2 + 2 == 4);
/// tpt_for_assert_const::const_assert!(core::mem::size_of::<u32>() == 4);
/// ```
#[macro_export]
macro_rules! const_assert {
    ($cond:expr $(,)?) => {
        const _: () = core::assert!($cond);
    };
}

/// Assert two expressions are equal at compile time.
#[macro_export]
macro_rules! const_assert_eq {
    ($left:expr, $right:expr $(,)?) => {
        const _: () = core::assert!($left == $right);
    };
}

/// Assert two expressions are not equal at compile time.
#[macro_export]
macro_rules! const_assert_ne {
    ($left:expr, $right:expr $(,)?) => {
        const _: () = core::assert!($left != $right);
    };
}

/// A compile-time boolean carried at the type level.
pub struct Const<const N: bool>;

/// Type-level "this boolean is `true`".
///
/// Implemented only for [`Const`]`<true>`, so a bound `Const<B>: IsTrue`
/// forces `B` to be `true` at compile time.
pub trait IsTrue {
    /// The boolean value (always `true` for implementors).
    const VALUE: bool;
}

impl IsTrue for Const<true> {
    const VALUE: bool = true;
}

/// Type equality used as a compile-time assertion.
///
/// `Same` is implemented for every `T` with `Output = T`, so a bound
/// `T: Same<Output = U>` requires `T` and `U` to be the same type.
pub trait Same {
    /// The type this resolves to (always `Self`).
    type Output;
}

impl<T> Same for T {
    type Output = T;
}

#[cfg(test)]
#[allow(clippy::eq_op)]
mod tests {
    use super::*;

    const_assert!(1 + 1 == 2);
    const_assert_eq!(core::mem::size_of::<u64>(), 8);
    const_assert_ne!(1u8, 2u8);

    #[test]
    fn is_true_holds_for_true() {
        fn requires_true<B: IsTrue>() {}
        requires_true::<Const<true>>();
    }

    #[test]
    fn same_requires_type_equality() {
        fn requires_same<T, U>()
        where
            T: Same<Output = U>,
        {
        }
        requires_same::<u32, u32>();
    }
}
