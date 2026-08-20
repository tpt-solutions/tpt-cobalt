//! # tpt-omni — The Unified Arrow-Native Data Engine
//!
//! `tpt-omni` is the foundation of the TPT stack. It backs a single
//! [`OmniFrame`] type with Apache Arrow columnar memory and exposes that same
//! memory as a **table**, an **N-D tensor**, or a **sparse** matrix with
//! *zero-copy* views. Every element-wise and reduction operation is transparently
//! parallelized with Rayon.
//!
//! ```ignore
//! use tpt_omni::prelude::*;
//!
//! let frame = OmniFrame::from_columns(vec![]); // ...
//! let table = frame.as_table();
//! let adults = table.filter(col("age").gt(18));
//! let tensor = frame.as_tensor::<f64>("data", &[10, 5, 3]);
//! let normalized = (&tensor - tensor.mean()) / tensor.std();
//! let subset = slice![normalized, 0..5, All, 2];
//! ```

pub mod error;
pub mod frame;
pub mod mmap;
pub mod parallel;
pub mod sparse;
pub mod table;
pub mod tensor;

pub use error::OmniError;
pub use frame::OmniFrame;
pub use mmap::MmapedFrame;
pub use sparse::Sparse;
pub use table::{Column, Expr, Table};
pub use tensor::{Tensor, TensorView};

pub use ndarray;

/// NumPy-style slicing for tensors and tensor views.
///
/// `slice![tensor, 0..5, All, 2]` returns a zero-copy `ArrayView`. `All` maps to
/// a full-range slice.
#[macro_export]
macro_rules! slice {
    (@acc $data:expr, [$($acc:expr),*]) => {{
        let elems: Vec<$crate::ndarray::SliceInfoElem> = vec![$($acc),*];
        let info: $crate::tensor::Slice = $crate::ndarray::SliceInfo::try_from(elems)
            .expect("invalid slice specification");
        $data.slice(&info)
    }};
    (@acc $data:expr, [$($acc:expr),*] All, $($rest:tt)*) => {
        slice!(@acc $data, [$($acc,)* $crate::ndarray::SliceInfoElem::from(..)] $($rest)*)
    };
    (@acc $data:expr, [$($acc:expr),*] $e:expr, $($rest:tt)*) => {
        slice!(@acc $data, [$($acc,)* $crate::ndarray::SliceInfoElem::from($e)] $($rest)*)
    };
    (@acc $data:expr, [$($acc:expr),*] All) => {
        slice!(@acc $data, [$($acc,)* $crate::ndarray::SliceInfoElem::from(..)])
    };
    (@acc $data:expr, [$($acc:expr),*] $e:expr) => {
        slice!(@acc $data, [$($acc,)* $crate::ndarray::SliceInfoElem::from($e)])
    };
    (@acc $data:expr, [$($acc:expr),*] ,) => {
        slice!(@acc $data, [$($acc),*])
    };
    ($data:expr) => { $data };
    ($data:expr, $($rest:tt)*) => {
        slice!(@acc $data, [] $($rest)*)
    };
}

/// Re-exports for ergonomic use.
pub mod prelude {
    pub use crate::error::OmniError;
    pub use crate::frame::OmniFrame;
    pub use crate::ndarray;
    pub use crate::slice;
    pub use crate::sparse::Sparse;
    pub use crate::table::{col, Column, Expr, Table};
    pub use crate::tensor::{Slice, Tensor, TensorView};
}
