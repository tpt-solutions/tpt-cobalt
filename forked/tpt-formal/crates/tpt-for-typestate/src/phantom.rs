//! Variance-controlled phantom markers and the sealed-trait pattern.
//!
//! `PhantomData<T>` is covariant over `T`. When a typestate encoding needs
//! invariant or contravariant ownership of a type parameter, reach for
//! [`Invariant`] / [`Contravariant`] instead.

use core::marker::PhantomData;

/// Phantom marker that is **invariant** over `T`.
///
/// Useful when a typestate wrapper must not be freely coerced between
/// distinct state type parameters.
pub struct Invariant<T>(PhantomData<fn() -> T>);

/// Phantom marker that is **covariant** over `T` (like `PhantomData<T>`).
pub struct Covariant<T>(PhantomData<T>);

/// Phantom marker that is **contravariant** over `T`.
pub struct Contravariant<T>(PhantomData<fn(T)>);

impl<T> Invariant<T> {
    /// Construct a zero-sized invariant marker.
    #[inline]
    pub const fn new() -> Self {
        Invariant(PhantomData)
    }
}

impl<T> Covariant<T> {
    /// Construct a zero-sized covariant marker.
    #[inline]
    pub const fn new() -> Self {
        Covariant(PhantomData)
    }
}

impl<T> Contravariant<T> {
    /// Construct a zero-sized contravariant marker.
    #[inline]
    pub const fn new() -> Self {
        Contravariant(PhantomData)
    }
}

/// Construct an [`Invariant`] marker for `T`.
#[inline]
pub const fn invariant<T>() -> Invariant<T> {
    Invariant::new()
}

/// Construct a [`Covariant`] marker for `T`.
#[inline]
pub const fn covariant<T>() -> Covariant<T> {
    Covariant::new()
}

/// Construct a [`Contravariant`] marker for `T`.
#[inline]
pub const fn contravariant<T>() -> Contravariant<T> {
    Contravariant::new()
}

macro_rules! impl_marker_traits {
    ($($t:ty),*) => {
        $(
            impl<T> Clone for $t {
                #[inline]
                fn clone(&self) -> Self { *self }
            }
            impl<T> Copy for $t {}
            impl<T> Default for $t {
                #[inline]
                fn default() -> Self { Self::new() }
            }
            impl<T> core::fmt::Debug for $t {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    f.write_str(core::any::type_name::<Self>())
                }
            }
            impl<T> PartialEq for $t {
                #[inline]
                fn eq(&self, _other: &Self) -> bool { true }
            }
            impl<T> Eq for $t {}
            impl<T> core::hash::Hash for $t {
                #[inline]
                fn hash<H: core::hash::Hasher>(&self, _state: &mut H) {}
            }
        )*
    };
}

impl_marker_traits!(Invariant<T>, Covariant<T>, Contravariant<T>);

/// Sealing trait for the sealed-trait pattern.
///
/// Implement this in the same crate that owns the public trait you want to
/// keep non-implementable downstream, then have that public trait have
/// `Sealed` as a supertrait.
pub trait Sealed {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn markers_are_zero_sized() {
        assert_eq!(core::mem::size_of::<Invariant<u8>>(), 0);
        assert_eq!(core::mem::size_of::<Covariant<u8>>(), 0);
        assert_eq!(core::mem::size_of::<Contravariant<u8>>(), 0);
    }

    #[test]
    fn markers_hash_and_eq() {
        let mut set = HashSet::new();
        set.insert(invariant::<u32>());
        assert!(set.contains(&invariant::<u32>()));
    }
}
