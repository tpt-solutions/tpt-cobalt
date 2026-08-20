//! Sound, one-directional boolean-abstraction decision tier.
//!
//! When the built-in ground evaluator cannot decide a problem (free variables
//! remain), we Tseitin-encode the *boolean structure* of the asserted formula
//! into CNF, treating each distinct non-boolean-connective leaf (a `Bool`/`Var`
//! atom or an `Eq`/`Lt`/`Le`/`Gt`/`Ge` comparison) as an opaque propositional
//! variable matched by structural equality. The resulting CNF is decided by the
//! from-scratch CDCL solver in `tpt-for-sat`.
//!
//! Soundness: the abstraction only ever *adds* constraints (the Tseitin clauses
//! capture the boolean structure exactly, and atoms are left unconstrained).
//! Any real model of the formula induces a satisfying boolean assignment of the
//! abstraction, so if the abstraction is UNSAT the real formula is UNSAT too.
//! We therefore only ever upgrade `Unknown -> Unsat` — we never unsoundly
//! downgrade to `Sat`.

use std::collections::HashMap;

use tpt_for_sat::{Cnf, SatResult as CdclResult, Solver};

use crate::{Problem, SatResult, Term};

/// A DIMACS literal: `var + 1` for a positive literal of (0-based) `var`.
type Lit = i32;

/// Convert a 0-based SAT variable to its positive DIMACS literal.
fn lit(v: usize) -> Lit {
    (v as Lit) + 1
}

/// Tseitin encoder: assigns each sub-term a SAT variable and emits the clauses
/// that make it equivalent to the sub-term.
struct Encoder {
    atoms: HashMap<Term, usize>,
    clauses: Vec<Vec<Lit>>,
    next: usize,
}

impl Encoder {
    fn new() -> Encoder {
        Encoder {
            atoms: HashMap::new(),
            clauses: Vec::new(),
            next: 0,
        }
    }

    /// Return the (0-based) SAT variable standing for the whole sub-term,
    /// emitting the Tseitin clauses that make it equivalent to the sub-term.
    fn encode(&mut self, t: &Term) -> usize {
        match t {
            Term::Bool(b) => {
                // Boolean constants get a dedicated variable whose truth value
                // is fixed by a unit clause, so the abstraction reflects them
                // exactly (otherwise `true` would be a free, unconstrained atom).
                let v = self.fresh();
                self.clause(&[if *b { lit(v) } else { -lit(v) }]);
                v
            }
            Term::And(a, b) => {
                let va = self.encode(a);
                let vb = self.encode(b);
                let v = self.fresh();
                // v <-> (va && vb)
                self.clause(&[-lit(v), lit(va)]);
                self.clause(&[-lit(v), lit(vb)]);
                self.clause(&[-lit(va), -lit(vb), lit(v)]);
                v
            }
            Term::Or(a, b) => {
                let va = self.encode(a);
                let vb = self.encode(b);
                let v = self.fresh();
                // v <-> (va || vb)
                self.clause(&[-lit(v), lit(va), lit(vb)]);
                self.clause(&[-lit(va), lit(v)]);
                self.clause(&[-lit(vb), lit(v)]);
                v
            }
            Term::Not(a) => {
                let va = self.encode(a);
                let v = self.fresh();
                // v <-> !va  :  (v => !va) and (!va => v)
                self.clause(&[-lit(v), -lit(va)]);
                self.clause(&[lit(va), lit(v)]);
                v
            }
            Term::Implies(a, b) => {
                let va = self.encode(a);
                let vb = self.encode(b);
                let v = self.fresh();
                // v <-> (!va || vb)
                self.clause(&[-lit(v), -lit(va), lit(vb)]); // v => (!va || vb)
                self.clause(&[lit(va), lit(v)]); // va => v
                self.clause(&[-lit(vb), lit(v)]); // !vb => v
                v
            }
            _ => {
                // An opaque atom: any term that is not a boolean connective.
                if let Some(&v) = self.atoms.get(t) {
                    v
                } else {
                    let v = self.fresh();
                    self.atoms.insert(t.clone(), v);
                    v
                }
            }
        }
    }

    fn fresh(&mut self) -> usize {
        let v = self.next;
        self.next += 1;
        v
    }

    fn clause(&mut self, lits: &[Lit]) {
        self.clauses.push(lits.to_vec());
    }
}

/// Decide `problem` via boolean abstraction. Returns `Unsat` only when the
/// abstraction is provably unsatisfiable; otherwise `Unknown`.
pub(crate) fn decide(problem: &Problem) -> SatResult {
    if problem.assertions().is_empty() {
        return SatResult::Unknown;
    }
    let mut enc = Encoder::new();
    for a in problem.assertions() {
        let va = enc.encode(a);
        // The assertion must hold, so force its variable true.
        enc.clause(&[lit(va)]);
    }
    let refs: Vec<&[Lit]> = enc.clauses.iter().map(|c| c.as_slice()).collect();
    let cnf = match Cnf::from_lits(enc.next, &refs) {
        Some(c) => c,
        None => return SatResult::Unknown,
    };
    let mut solver = Solver::new(cnf);
    match solver.solve() {
        CdclResult::Unsat => SatResult::Unsat,
        CdclResult::Sat => SatResult::Unknown,
    }
}
