//! # tpt-approx — Clean-room floating-point approximate comparison
//!
//! Provides `relative_eq!` / `abs_diff_eq!` predicates plus `assert_*` macro
//! forms with `epsilon = ...`, `max_relative = ...` style named arguments.
//!
//! ```ignore
//! use tpt_approx::{assert_relative_eq, assert_abs_diff_eq};
//!
//! let a: f64 = 0.1 + 0.2;
//! assert_relative_eq!(a, 0.3, epsilon = 1e-12);
//! assert_abs_diff_eq!(3.0_f64.sqrt(), 1.7320508075688772, epsilon = 1e-12);
//! ```

/// Core relative-equality predicate.
#[macro_export]
macro_rules! relative_eq {
    ($a:expr, $b:expr) => {
        $crate::__impl_relative_eq!($a, $b, f64::EPSILON, f64::EPSILON * 4.0)
    };
    ($a:expr, $b:expr, epsilon = $eps:expr) => {
        $crate::__impl_relative_eq!($a, $b, $eps, f64::INFINITY)
    };
    ($a:expr, $b:expr, max_relative = $max_rel:expr) => {
        $crate::__impl_relative_eq!($a, $b, f64::EPSILON * 4.0, $max_rel)
    };
    ($a:expr, $b:expr, epsilon = $eps:expr, max_relative = $max_rel:expr) => {
        $crate::__impl_relative_eq!($a, $b, $eps, $max_rel)
    };
    ($a:expr, $b:expr, max_relative = $max_rel:expr, epsilon = $eps:expr) => {
        $crate::__impl_relative_eq!($a, $b, $eps, $max_rel)
    };
}

/// Core absolute-difference predicate.
#[macro_export]
macro_rules! abs_diff_eq {
    ($a:expr, $b:expr) => {
        (($a as f64) - ($b as f64)).abs() <= f64::EPSILON
    };
    ($a:expr, $b:expr, epsilon = $eps:expr) => {
        (($a as f64) - ($b as f64)).abs() <= ($eps as f64)
    };
}

/// Implementation detail of [`relative_eq!`].
#[doc(hidden)]
#[macro_export]
macro_rules! __impl_relative_eq {
    ($a:expr, $b:expr, $eps:expr, $max_rel:expr) => {{
        let (a, b) = (&$a, &$b);
        let (a, b) = (*a as f64, *b as f64);
        if a == b {
            true
        } else {
            let abs_diff = (a - b).abs();
            let abs_max = a.abs().max(b.abs());
            abs_diff <= ($eps as f64) || (abs_diff / abs_max) <= ($max_rel as f64)
        }
    }};
}

/// Panic unless two values are relatively equal.
#[macro_export]
macro_rules! assert_relative_eq {
    ($($arg:tt)*) => {
        if !$crate::relative_eq!($($arg)*) {
            panic!("assertion failed: values are not approximately equal");
        }
    };
}

/// Panic unless two values are absolutely equal within tolerance.
#[macro_export]
macro_rules! assert_abs_diff_eq {
    ($a:expr, $b:expr) => {
        assert!(crate::abs_diff_eq!($a, $b), "values are not approximately equal");
    };
    ($a:expr, $b:expr, epsilon = $eps:expr) => {
        assert!((($a as f64) - ($b as f64)).abs() <= ($eps as f64), "values are not approximately equal");
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn relative_equality() {
        assert!(relative_eq!(0.1 + 0.2, 0.3, epsilon = 1e-12));
        assert!(!relative_eq!(1.0, 2.0));
        assert_relative_eq!(1.0, 1.0000000000000002, max_relative = 1e-15);
    }

    #[test]
    fn absolute_equality() {
        assert!(abs_diff_eq!(1.0, 1.0 + 1e-16));
        assert_abs_diff_eq!(2.0, 2.0, epsilon = 1e-9);
    }
}
