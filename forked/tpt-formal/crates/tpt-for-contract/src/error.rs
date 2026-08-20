//! Contract violation reporting.

use core::fmt;

/// A failed contract check.
///
/// Carries the *kind* of contract (precondition, postcondition, invariant,
/// loop-invariant), the source text of the failing condition, and its
/// location. Constructed by the contract macros; you normally only see this
/// when [`report`] panics with it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ContractError {
    /// Which contract facet failed.
    pub kind: &'static str,
    /// Source text of the failing condition (may include a trailing message).
    pub expr: &'static str,
    /// Source file of the failure.
    pub file: &'static str,
    /// Source line of the failure.
    pub line: u32,
}

impl fmt::Debug for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContractError")
            .field("kind", &self.kind)
            .field("expr", &self.expr)
            .field("file", &self.file)
            .field("line", &self.line)
            .finish()
    }
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} violated at {}:{}: {}",
            self.kind, self.file, self.line, self.expr
        )
    }
}

/// Report a contract violation by panicking.
///
/// Called by the [`requires!`](crate::requires!),
/// [`ensures!`](crate::ensures!), [`invariant!`](crate::invariant!), and
/// [`loop_invariant!`](crate::loop_invariant!) macros. Marked `#[track_caller]`
/// so the panic points at the call site, not this function.
#[track_caller]
pub fn report(err: ContractError) -> ! {
    panic!("{}", err);
}
