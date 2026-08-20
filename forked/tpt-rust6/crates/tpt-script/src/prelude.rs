//! `use tpt_script::prelude::*;` — the whole stack, Pythonic names.
//!
//! | name | is |
//! |------|----|
//! | `read` / `write` / `read_glob` | [`tpt_io`] entry points |
//! | `omni` / `io` / `stat` / `viz` / `sym` | the crates themselves |
//! | `Table`, `OmniFrame`, `Tensor`, `col`, `slice!` | `tpt_omni::prelude` |
//! | `Plot`, `Scatter`, `Line`, `Gradient` | `tpt_viz::prelude` |
//! | `TableExt`, `Grouped`, `script!`, `run_script` | this crate |

// I/O verbs.
pub use tpt_io::{read, read_glob, write};

// Whole crates under short names.
pub use tpt_io as io;
pub use tpt_omni as omni;
pub use tpt_stat as stat;
pub use tpt_sym as sym;
pub use tpt_viz as viz;

// Data + plotting vocabulary.
pub use tpt_omni::prelude::*;
pub use tpt_viz::prelude::*;

// Script layer.
pub use crate::error::{Res, ScriptError};
pub use crate::runner::{run, run_script, run_script_err};
pub use crate::table::{f64s, i64s, strs, table_of, Grouped, TableExt, Value};
pub use crate::{script, script_main, script_stmt};
