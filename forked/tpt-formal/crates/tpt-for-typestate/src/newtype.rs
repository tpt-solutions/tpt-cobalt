//! Newtype wrappers with `From`/`Into` glue over an inner type.
//!
//! A newtype is the cheapest way to give a distinct type to a value that
//! would otherwise be a bare primitive, gaining the type system's help in
//! preventing accidental mixing. [`Newtype`] provides the conversion trait
//! surface; [`define_newtype!`] generates a full implementation.

/// Conversion surface between a newtype and its inner representation.
pub trait Newtype: Sized {
    /// The wrapped representation type.
    type Inner;

    /// Wrap an inner value. This is infallible — the newtype's invariant, if
    /// any, is the caller's responsibility (use a constructor for checked
    /// variants).
    fn from_inner(inner: Self::Inner) -> Self;

    /// Unwrap to the inner value.
    fn into_inner(self) -> Self::Inner;
}

/// Define a transparent newtype around `Inner` with `Debug`/`Clone`/`Copy`/
/// `PartialEq`/`Eq`/`Hash`/`Ord` derived and [`Newtype`] implemented.
///
/// ```
/// use tpt_for_typestate::newtype::Newtype;
/// tpt_for_typestate::define_newtype!(Meters, u64);
/// let m = Meters::from_inner(3);
/// assert_eq!(m.into_inner(), 3);
/// ```
#[macro_export]
macro_rules! define_newtype {
    ($(#[$meta:meta])* $name:ident, $inner:ty) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name($inner);

        impl $crate::newtype::Newtype for $name {
            type Inner = $inner;
            #[inline]
            fn from_inner(inner: $inner) -> Self { $name(inner) }
            #[inline]
            fn into_inner(self) -> $inner { self.0 }
        }

        impl From<$inner> for $name {
            #[inline]
            fn from(inner: $inner) -> Self { $name(inner) }
        }

        impl From<$name> for $inner {
            #[inline]
            fn from(wrapper: $name) -> $inner { wrapper.0 }
        }

        impl core::ops::Deref for $name {
            type Target = $inner;
            #[inline]
            fn deref(&self) -> &$inner { &self.0 }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    define_newtype!(UserId, u64);
    define_newtype!(Tag, u32);

    #[test]
    fn newtype_is_distinct() {
        let a = UserId::from_inner(1);
        let b = Tag::from_inner(1);
        // The following would not compile — types are distinct:
        // let _: UserId = b;
        assert_eq!(a.into_inner(), 1u64);
        assert_eq!(b.into_inner(), 1u32);
    }

    #[test]
    fn newtype_orders_and_hashes() {
        let a = UserId::from_inner(2);
        let b = UserId::from_inner(3);
        assert!(a < b);
        let mut set = std::collections::HashSet::new();
        set.insert(a);
        assert!(set.contains(&a));
        assert!(!set.contains(&b));
    }
}
