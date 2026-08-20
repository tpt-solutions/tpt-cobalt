//! Safe numeric casts.
//!
//! - [`SafeCast`] is an *infallible* widening cast (e.g. `u8 -> u32`) that can
//!   never lose information.
//! - [`TrySafeCast`] is a *fallible* narrowing or cross-kind cast (e.g.
//!   `u32 -> u8`, `f64 -> f32`) that returns [`CastError`] instead of
//!   silently truncating.

use core::fmt;

/// Error returned by a failed [`TrySafeCast`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CastError {
    from: &'static str,
    to: &'static str,
}

impl CastError {
    /// The type name of the source value.
    pub fn from_type(&self) -> &'static str {
        self.from
    }

    /// The type name of the target value.
    pub fn to_type(&self) -> &'static str {
        self.to
    }
}

impl fmt::Display for CastError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "safe cast from {} to {} out of range or lossy",
            self.from, self.to
        )
    }
}

/// Infallible widening cast. Implement only for casts that cannot lose
/// information.
pub trait SafeCast<T>: Sized {
    /// Perform the widening cast.
    fn safe_cast(self) -> T;
}

/// Fallible narrowing / cross-kind cast.
pub trait TrySafeCast<T>: Sized {
    /// Error type on failure.
    type Error;
    /// Attempt the cast, returning an error instead of truncating.
    fn try_safe_cast(self) -> Result<T, Self::Error>;
}

macro_rules! safe_casts {
    ($($from:ty => $($to:ty),*);* $(;)?) => {
        $(
            $(
                #[allow(
                    clippy::cast_lossless,
                    clippy::cast_possible_truncation,
                    clippy::cast_possible_wrap,
                    clippy::cast_precision_loss,
                    clippy::cast_sign_loss
                )]
                impl SafeCast<$to> for $from {
                    #[inline]
                    fn safe_cast(self) -> $to {
                        self as $to
                    }
                }
            )*
        )*
    };
}

safe_casts! {
    u8 => u16, u32, u64, u128, i16, i32, i64, i128, usize, isize, f32, f64;
    u16 => u32, u64, u128, i32, i64, i128, usize, f32, f64;
    u32 => u64, u128, i64, i128, f64;
    u64 => u128, i128;
    i8 => i16, i32, i64, i128, f32, f64;
    i16 => i32, i64, i128, f32, f64;
    i32 => i64, i128, f64;
    i64 => i128;
    f32 => f64;
}

macro_rules! try_casts {
    ($($from:ty => $($to:ty),*);* $(;)?) => {
        $(
            $(
                #[allow(
                    clippy::cast_lossless,
                    clippy::cast_possible_truncation,
                    clippy::cast_possible_wrap,
                    clippy::cast_precision_loss,
                    clippy::cast_sign_loss
                )]
                impl TrySafeCast<$to> for $from {
                    type Error = CastError;
                    #[inline]
                    fn try_safe_cast(self) -> Result<$to, CastError> {
                        let err = CastError {
                            from: stringify!($from),
                            to: stringify!($to),
                        };
                        if self < (<$to>::MIN as $from) || self > (<$to>::MAX as $from) {
                            return Err(err);
                        }
                        Ok(self as $to)
                    }
                }
            )*
        )*
    };
}

try_casts! {
    u16 => u8;
    u32 => u8, u16;
    u64 => u8, u16, u32;
    u128 => u8, u16, u32, u64;
    usize => u8, u16, u32;
    i16 => i8;
    i32 => i8, i16;
    i64 => i8, i16, i32;
    i128 => i8, i16, i32, i64;
    isize => i8, i16, i32;
    f64 => f32;
}

/// Convert `T` into `U` via [`SafeCast`].
#[inline]
pub fn safe_cast<T, U>(value: T) -> U
where
    T: SafeCast<U>,
{
    value.safe_cast()
}

/// Convert `T` into `U` via [`TrySafeCast`].
#[inline]
pub fn try_safe_cast<T, U>(value: T) -> Result<U, <T as TrySafeCast<U>>::Error>
where
    T: TrySafeCast<U>,
{
    value.try_safe_cast()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widening_is_infallible() {
        let x: u32 = 200u8.safe_cast();
        assert_eq!(x, 200u32);
        let y: f64 = 3u32.safe_cast();
        assert_eq!(y, 3.0f64);
    }

    #[test]
    fn narrowing_rejects_out_of_range() {
        assert_eq!(
            try_safe_cast::<u16, u8>(300),
            Err(CastError {
                from: "u16",
                to: "u8"
            })
        );
        assert_eq!(try_safe_cast::<u16, u8>(10), Ok(10u8));
        assert_eq!(
            try_safe_cast::<u32, u16>(70000),
            Err(CastError {
                from: "u32",
                to: "u16"
            })
        );
    }

    #[test]
    fn free_functions_work() {
        assert_eq!(safe_cast::<u8, u64>(5), 5u64);
        assert_eq!(try_safe_cast::<u32, u8>(255), Ok(255u8));
    }
}
