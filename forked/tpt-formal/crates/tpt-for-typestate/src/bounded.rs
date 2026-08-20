//! Range-checked numeric values.
//!
//! Two flavours are provided:
//!
//! - [`Bounded`] — runtime-checked bounds attached to a value.
//! - [`Checked`] — type-level named bounds via the [`Bound`] trait, so a
//!   value's valid range is part of its type.
//!
//! Both use only `core`, so they are usable in `no_std`.

use core::fmt;

/// Runtime range-checked value.
///
/// Holds `value` together with inclusive bounds `min`/`max`. Construction
/// rejects out-of-range values rather than truncating them.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bounded<T> {
    value: T,
    min: T,
    max: T,
}

impl<T: Copy + PartialOrd> Bounded<T> {
    /// Construct a bounded value, returning `None` if `value` is outside
    /// `[min, max]` (or if `min > max`).
    #[inline]
    pub fn new(value: T, min: T, max: T) -> Option<Self> {
        if min <= max && value >= min && value <= max {
            Some(Bounded { value, min, max })
        } else {
            None
        }
    }

    /// Construct without checking. Prefer [`Bounded::new`] unless the bound
    /// has already been validated by other means.
    ///
    /// # Safety
    ///
    /// `value` must satisfy `min <= value <= max` and `min <= max`.
    #[inline]
    pub unsafe fn new_unchecked(value: T, min: T, max: T) -> Self {
        Bounded { value, min, max }
    }

    /// The contained value.
    #[inline]
    pub fn get(&self) -> T {
        self.value
    }

    /// The inclusive lower bound.
    #[inline]
    pub fn min(&self) -> T {
        self.min
    }

    /// The inclusive upper bound.
    #[inline]
    pub fn max(&self) -> T {
        self.max
    }

    /// Clamp `value` into this bounded range and return a new bounded value.
    #[inline]
    pub fn saturate(&self, value: T) -> Self {
        let v = if value < self.min {
            self.min
        } else if value > self.max {
            self.max
        } else {
            value
        };
        // Safe: v is within [min, max] by construction.
        unsafe { Bounded::new_unchecked(v, self.min, self.max) }
    }
}

impl<T: Copy + PartialOrd + fmt::Debug> fmt::Debug for Bounded<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bounded")
            .field("value", &self.value)
            .field("min", &self.min)
            .field("max", &self.max)
            .finish()
    }
}

/// A named, type-level bound for [`Checked`].
///
/// Implementors associate a `Value` type together with its inclusive valid
/// range. This lets a value's permitted interval be part of its type.
pub trait Bound {
    /// The underlying numeric type.
    type Value: Copy + PartialOrd;
    /// Inclusive lower bound.
    const MIN: Self::Value;
    /// Inclusive upper bound.
    const MAX: Self::Value;
}

/// A value of a [`Bound`] type, guaranteed in range.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Checked<B: Bound> {
    value: B::Value,
}

impl<B: Bound> Checked<B> {
    /// Construct, returning `None` if `value` is outside `[B::MIN, B::MAX]`.
    #[inline]
    pub fn new(value: B::Value) -> Option<Self> {
        if value >= B::MIN && value <= B::MAX {
            Some(Checked { value })
        } else {
            None
        }
    }

    /// Construct without checking. Prefer [`Checked::new`].
    ///
    /// # Safety
    ///
    /// `value` must satisfy `B::MIN <= value <= B::MAX`.
    #[inline]
    pub unsafe fn new_unchecked(value: B::Value) -> Self {
        Checked { value }
    }

    /// The contained value.
    #[inline]
    pub fn get(&self) -> B::Value {
        self.value
    }
}

impl<B: Bound> fmt::Debug for Checked<B>
where
    B::Value: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Checked")
            .field("value", &self.value)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_rejects_out_of_range() {
        assert!(Bounded::new(5u8, 0, 10).is_some());
        assert!(Bounded::new(11u8, 0, 10).is_none());
        assert!(Bounded::new(5u8, 10, 0).is_none());
    }

    #[test]
    fn bounded_saturates() {
        let b = Bounded::new(5u8, 0, 10).unwrap();
        assert_eq!(b.saturate(20).get(), 10);
        assert_eq!(b.saturate(0).get(), 0);
    }

    struct Pct;
    impl Bound for Pct {
        type Value = u8;
        const MIN: u8 = 0;
        const MAX: u8 = 100;
    }

    #[test]
    fn checked_uses_named_bounds() {
        assert!(Checked::<Pct>::new(50).is_some());
        assert!(Checked::<Pct>::new(101).is_none());
        let c = unsafe { Checked::<Pct>::new_unchecked(75) };
        assert_eq!(c.get(), 75);
    }
}
