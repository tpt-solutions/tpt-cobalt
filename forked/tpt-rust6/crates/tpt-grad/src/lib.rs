//! # tpt-grad — autodiff for the TPT stack
//!
//! Two layers, one op vocabulary:
//!
//! 1. **Runtime reverse-mode AD** ([`tape`], [`scalar`]) — a classic Wengert
//!    tape over [`tpt_omni::Tensor<f64>`] ([`Tape`] / [`Variable`]) and over
//!    plain `f64` ([`ScalarTape`] / [`Value`]).
//! 2. **Macro-driven AD** ([`derive_grad`], [`derive_vmap`], [`derive_jit`]) —
//!    attribute macros that read a function body at compile time and emit
//!    companion functions. `#[derive_grad] fn f(..)` emits `f_grad`,
//!    `#[derive_vmap]` emits `f_vmap`, `#[derive_jit]` emits `f_jit`. The
//!    original function is always re-emitted unchanged.
//!
//! ## Supported subset (enforced at compile time)
//!
//! The macros deliberately handle a **curated subset** of Rust rather than
//! arbitrary code. A function is accepted when:
//!
//! * it is free-standing (no `self`, no generics, no `where`-clause, not
//!   `async`/`unsafe`);
//! * every argument is `Tensor<f64>` (differentiated / batched) or `f64`
//!   (passed through, never differentiated);
//! * the return type is `Tensor<f64>` or `f64`;
//! * the body is a sequence of `let <ident> = <expr>;` bindings followed by a
//!   tail expression, where every `<expr>` is built only from:
//!
//! | Construct | Notes |
//! |---|---|
//! | `a + b`, `a - b`, `a * b`, `a / b` | element-wise, broadcasting |
//! | `-a` | unary negation (see the caveat below) |
//! | `(a)` | parentheses |
//! | identifiers, float literals | e.g. `x`, `2.0` (integer literals are rejected) |
//! | `.clone()` | no-op on the tape (`Variable` is `Copy`) |
//! | `.powf(c)` | `c` a float literal or an `f64` argument |
//! | `.exp()`, `.ln()`, `.sqrt()` | element-wise |
//! | `.sum()`, `.mean()` | reduce to a rank-0 value |
//!
//! Anything else (loops, `if`, indexing, calls to other functions, integer
//! literals, `&` borrows, ...) is rejected with a `compile_error!` naming the
//! offending construct. Use the runtime [`Tape`] directly for those cases.
//!
//! Two caveats come from `tpt_omni::Tensor` itself, which has neither `Clone`
//! nor `Neg`:
//!
//! * [`TensorClone`] and [`TensorOps`] (both in the [`prelude`]) add
//!   `.clone()`, `.powf()`, `.exp()`, `.ln()` and `.sqrt()` to plain tensors so
//!   the *original* function body type-checks; the tape and the fused kernel
//!   provide the same names.
//! * the orphan rule forbids an inherent `Neg` for tensors, so write
//!   `t * -1.0` rather than `-t` in a tensor-typed body. Unary `-` on scalars
//!   is fine everywhere.
//!
//! ## Generated functions
//!
//! * `f_grad(args..) -> (Ret, Vec<Tensor<f64>>)` — the value plus one gradient
//!   per `Tensor<f64>` argument, in declaration order. The body is re-run on a
//!   [`Tape`], so the gradients are exact, not finite differences.
//! * `f_vmap(args..) -> Tensor<f64>` — maps `f` over dim 0 of every
//!   `Tensor<f64>` argument (`f64` arguments are shared) and stacks the results
//!   along a new dim 0. See [`vmap`].
//! * `f_jit(args..) -> Ret` — for a purely element-wise body, a *fused* single
//!   pass with no intermediate tensors (see [`jit`]); this is a **fusion hint**,
//!   not native codegen. Bodies containing reductions fall back to calling `f`.
//!
//! ```
//! use tpt_grad::prelude::*;
//!
//! #[derive_grad]
//! #[derive_vmap]
//! #[derive_jit]
//! fn f(x: Tensor<f64>, y: Tensor<f64>) -> Tensor<f64> {
//!     (x.clone() * y.clone()) + x.clone()
//! }
//!
//! let (v, g) = f_grad(tensor(&[2], &[1.0, 2.0]), tensor(&[2], &[3.0, 4.0]));
//! assert_eq!(v.to_vec(), vec![4.0, 10.0]);
//! assert_eq!(g[0].to_vec(), vec![4.0, 5.0]); // dz/dx = y + 1
//! assert_eq!(g[1].to_vec(), vec![1.0, 2.0]); // dz/dy = x
//! ```

pub mod jit;
pub mod scalar;
pub mod tape;
pub mod vmap;

pub use scalar::{ScalarGrads, ScalarTape, Value};
pub use tape::{full, scalar, tensor, FromVar, Grads, Tape, TensorClone, TensorOps, Variable};

pub use tpt_grad_macro::{derive_grad, derive_jit, derive_vmap};

/// Implementation detail: stable paths for macro-generated code. Not a public
/// API — use [`prelude`] instead.
#[doc(hidden)]
pub mod __rt {
    pub use crate::jit;
    pub use crate::tape::{FromVar, Tape, TensorClone, TensorOps, Variable};
    pub use crate::vmap;
    pub use crate::vmap::VmapOut;
    pub use tpt_omni::Tensor;
}

/// Ergonomic re-exports: the runtime engine, the attribute macros, and
/// `tpt_omni::Tensor`.
pub mod prelude {
    pub use crate::jit;
    pub use crate::scalar::{ScalarGrads, ScalarTape, Value};
    pub use crate::tape::{
        full, scalar, tensor, FromVar, Grads, Tape, TensorClone, TensorOps, Variable,
    };
    pub use crate::vmap;
    /// Attribute macros. See the crate docs for the supported subset.
    pub use tpt_grad_macro::{derive_grad, derive_jit, derive_vmap};
    pub use tpt_omni::Tensor;
}
