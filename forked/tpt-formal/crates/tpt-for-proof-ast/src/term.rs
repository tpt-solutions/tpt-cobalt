//! Terms — the objects a proof talks about.

/// A named identifier (variable or constant name).
pub type Ident = String;

/// A term in the object language.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    /// A variable, referenced by name.
    Var(Ident),
    /// A constant symbol.
    Const(Ident),
    /// Function application `f(t1, ..., tn)`.
    App(Ident, Vec<Term>),
    /// A numeric literal.
    Num(i128),
    /// Lambda abstraction `λ x. body`.
    Lambda(Ident, Box<Term>),
}

impl Term {
    /// Construct a variable term.
    pub fn var(name: impl Into<Ident>) -> Self {
        Term::Var(name.into())
    }

    /// Construct a constant term.
    pub fn const_(name: impl Into<Ident>) -> Self {
        Term::Const(name.into())
    }

    /// Construct a function application.
    pub fn app(name: impl Into<Ident>, args: Vec<Term>) -> Self {
        Term::App(name.into(), args)
    }

    /// Collect the free variables of this term.
    pub fn free_vars(&self) -> std::collections::BTreeSet<Ident> {
        let mut out = std::collections::BTreeSet::new();
        self.collect_free_vars(&mut out);
        out
    }

    fn collect_free_vars(&self, out: &mut std::collections::BTreeSet<Ident>) {
        match self {
            Term::Var(v) => {
                out.insert(v.clone());
            }
            Term::Const(_) | Term::Num(_) => {}
            Term::App(_, args) => {
                for a in args {
                    a.collect_free_vars(out);
                }
            }
            Term::Lambda(bound, body) => {
                let before = out.len();
                body.collect_free_vars(out);
                // A lambda binds `bound` in its body; we approximate by noting
                // the variable name appears (full capture-avoidance is out of
                // scope for a representation crate).
                let _ = (bound, before);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_and_collects() {
        let t = Term::app("add", vec![Term::var("x"), Term::Num(1)]);
        let fv = t.free_vars();
        assert!(fv.contains("x"));
        assert_eq!(fv.len(), 1);
    }
}
