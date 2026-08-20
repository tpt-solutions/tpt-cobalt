//! # tpt-lab — reactive, strictly-typed notebook environment
//!
//! A [`Notebook`] holds named **cells**. A cell is either a *constant* value or
//! a *derived* value produced by a closure that reads other cells. Dependencies
//! form a DAG; when a cell changes, only its **transitive dependents** are
//! re-executed, in topological order.
//!
//! ```
//! use tpt_lab::prelude::*;
//!
//! let mut nb = Notebook::new();
//! nb.set("A", 1i64)?;
//! nb.set_expr("B", "A + 1", &["A"], |nb: &Notebook| Ok(nb.try_get::<i64>("A")? + 1))?;
//! nb.set_expr("C", "B * 2", &["B"], |nb: &Notebook| Ok(nb.try_get::<i64>("B")? * 2))?;
//! assert_eq!(nb.get::<i64>("C"), Some(&4));
//!
//! nb.set("A", 10i64)?;              // B and C re-run, nothing else does
//! assert_eq!(nb.get::<i64>("C"), Some(&22));
//! assert_eq!(nb.type_of("B"), Some("i64"));
//! # Ok::<(), tpt_lab::LabError>(())
//! ```
//!
//! ## Scope
//!
//! * **Dependency capture is explicit.** Cells are runtime closures, not parsed
//!   source, so dependencies are declared via `set_expr(.., deps, ..)`,
//!   [`Notebook::depends_on`], or the [`cell!`] macro (which lifts bare
//!   identifiers into dependency names). Inferring them by parsing Rust source
//!   would require a proc-macro over the notebook document and is future work.
//! * **Type checking is runtime.** Every cell records `std::any::type_name` of
//!   its value ([`Notebook::type_of`]); [`Notebook::try_get`] reports
//!   [`LabError::TypeMismatch`] / [`LabError::Undefined`], and re-defining a
//!   cell with a different type is refused unless you call
//!   [`Notebook::redefine`]. Compile-time cross-cell checking is future work.
//! * **`export_rust` is best-effort**: derived cells emit the recorded
//!   expression text, constants emit a literal when one exists.
//! * Rendering covers scalars, [`tpt_omni::Table`] (Markdown) and
//!   [`tpt_omni::Tensor<f64>`] (text grid, or an SVG heatmap with
//!   `--features viz`). Other types render as `<type-name>` unless registered
//!   with [`Notebook::set_rendered`].

pub mod notebook;
pub mod render;
pub mod value;

#[cfg(feature = "lsp")]
pub mod analysis;
#[cfg(feature = "lsp")]
pub mod lsp;

pub use notebook::{Cell, Notebook};
pub use render::Render;
pub use value::Value;

use thiserror::Error;

/// Errors produced by notebook operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LabError {
    /// A cell name that has never been defined was referenced.
    #[error("cell '{0}' is not defined")]
    Undefined(String),

    /// A cell declared a dependency on a cell that does not exist.
    #[error("cell '{cell}' depends on undefined cell '{dep}'")]
    MissingDependency {
        /// The cell declaring the dependency.
        cell: String,
        /// The missing dependency's name.
        dep: String,
    },

    /// A cell was read at the wrong type.
    #[error("cell '{cell}' has type `{actual}`, but `{expected}` was requested")]
    TypeMismatch {
        /// The cell that was read.
        cell: String,
        /// The requested type.
        expected: String,
        /// The type actually stored.
        actual: String,
    },

    /// A redefinition would change a cell's type.
    #[error("cell '{cell}' has type `{was}`; assigning `{now}` would change it (use `redefine`)")]
    TypeChanged {
        /// The cell being redefined.
        cell: String,
        /// The existing type.
        was: String,
        /// The proposed new type.
        now: String,
    },

    /// The requested edges would make the dependency graph cyclic.
    #[error("dependency cycle among cells: {0}")]
    Cycle(String),

    /// A derived cell's closure failed.
    #[error("cell '{cell}' failed to evaluate: {message}")]
    Eval {
        /// The failing cell.
        cell: String,
        /// The failure description.
        message: String,
    },
}

/// Define notebook cells with identifier-captured dependency names.
///
/// ```
/// use tpt_lab::prelude::*;
///
/// let mut nb = Notebook::new();
/// cell!(nb, A = 2i64)?;
/// cell!(nb, B = [A] "A * 3", |nb: &Notebook| Ok(nb.try_get::<i64>("A")? * 3))?;
/// assert_eq!(nb.get::<i64>("B"), Some(&6));
/// # Ok::<(), tpt_lab::LabError>(())
/// ```
#[macro_export]
macro_rules! cell {
    ($nb:expr, $name:ident = [$($dep:ident),* $(,)?] $expr:expr, $f:expr) => {
        $nb.set_expr(
            stringify!($name),
            $expr,
            &[$(stringify!($dep)),*],
            $f,
        )
    };
    ($nb:expr, $name:ident = $value:expr) => {
        $nb.set(stringify!($name), $value)
    };
}

/// `use tpt_lab::prelude::*;`
pub mod prelude {
    pub use crate::cell;
    pub use crate::notebook::{Cell, Notebook};
    pub use crate::render::Render;
    pub use crate::value::Value;
    pub use crate::LabError;
}
