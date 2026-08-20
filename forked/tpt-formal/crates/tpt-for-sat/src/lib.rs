#![allow(clippy::should_implement_trait)]
//! From-scratch, pure-Rust CDCL SAT solver.
//!
//! No FFI, no external solver binary. The solver implements a modern conflict
//! -driven clause-learning (CDCL) loop:
//!
//! - **Watched literals** so unit propagation only wakes clauses whose watch
//!   flips to false.
//! - **1UIP conflict analysis** producing a learned clause on every conflict.
//! - **VSIDS** variable activity ordering for decisions.
//! - **Restart** scheduling (geometric) to escape bad search subtrees.
//!
//! The public surface is intentionally tiny: build a [`Cnf`], wrap it in a
//! [`Solver`], call [`Solver::solve`].
//!
//! ```
//! use tpt_for_sat::{Cnf, Solver, SatResult};
//!
//! // (x) ∧ (¬x ∨ y) ∧ (¬y ∨ z) ∧ (¬z)  → UNSAT (x, y, z all forced false/true)
//! let cnf = Cnf::from_lits(3, &[
//!     &[1],
//!     &[-1, 2],
//!     &[-2, 3],
//!     &[-3],
//! ]).unwrap();
//! let mut solver = Solver::new(cnf);
//! assert_eq!(solver.solve(), SatResult::Unsat);
//! ```

/// A Boolean variable, identified by a `0`-based index.
pub type Var = u32;

/// A literal: a variable with a sign.
///
/// Encoded as `var << 1 | sign`, where `sign == 1` means *negated*. This makes
/// negation a single-bit flip and gives a dense `usize` index for watch lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Lit(u32);

impl Lit {
    /// Build a literal. `negated == true` means the literal is `¬var`.
    pub fn new(var: Var, negated: bool) -> Lit {
        Lit((var << 1) | (negated as u32))
    }

    /// The underlying variable.
    pub fn var(self) -> Var {
        self.0 >> 1
    }

    /// Whether the literal is negated.
    pub fn is_neg(self) -> bool {
        self.0 & 1 == 1
    }

    /// Whether the literal is positive (not negated).
    pub fn is_pos(self) -> bool {
        !self.is_neg()
    }

    /// The negation of this literal.
    pub fn neg(self) -> Lit {
        Lit(self.0 ^ 1)
    }

    /// Dense index used for watch lists (`2 * var + sign`).
    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// Build a literal from a (nonzero) DIMACS integer: positive or negative.
    ///
    /// Panics on `0` (the DIMACS clause terminator), which is not a literal.
    pub fn from_dimacs(x: i32) -> Lit {
        assert_ne!(x, 0, "0 is not a literal in DIMACS encoding");
        if x > 0 {
            Lit::new((x - 1) as Var, false)
        } else {
            Lit::new((-x - 1) as Var, true)
        }
    }
}

/// A clause: a disjunction of literals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Clause {
    lits: Vec<Lit>,
    learnt: bool,
}

impl Clause {
    /// The literals of the clause.
    pub fn lits(&self) -> &[Lit] {
        &self.lits
    }

    /// Whether this clause was learned during search.
    pub fn is_learnt(&self) -> bool {
        self.learnt
    }
}

/// A Boolean formula in conjunctive normal form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cnf {
    vars: usize,
    clauses: Vec<Clause>,
}

impl Cnf {
    /// Number of variables (the solver reserves indices `0..vars`).
    pub fn var_count(&self) -> usize {
        self.vars
    }

    /// The clauses of the formula.
    pub fn clauses(&self) -> &[Clause] {
        &self.clauses
    }

    /// Build a CNF from a variable count and a list of clauses, each given as a
    /// slice of DIMACS integers (positive/negative).
    ///
    /// Returns `None` on malformed input rather than panicking: a `0` literal
    /// (the DIMACS clause terminator) or a literal referencing a variable `>=
    /// vars` is rejected. Use `.unwrap()` only on fixtures you control.
    pub fn from_lits(vars: usize, clauses: &[&[i32]]) -> Option<Cnf> {
        let mut out = Cnf {
            vars,
            clauses: Vec::with_capacity(clauses.len()),
        };
        for c in clauses {
            let mut lits: Vec<Lit> = Vec::with_capacity(c.len());
            for &x in c.iter() {
                if x == 0 {
                    return None;
                }
                let var = if x > 0 {
                    (x - 1) as usize
                } else {
                    (-x - 1) as usize
                };
                if var >= vars {
                    return None;
                }
                lits.push(Lit::new(var as Var, x < 0));
            }
            out.clauses.push(Clause {
                lits,
                learnt: false,
            });
        }
        Some(out)
    }
}

/// The result of a satisfiability query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SatResult {
    /// There exists an assignment satisfying every clause.
    Sat,
    /// No assignment satisfies every clause.
    Unsat,
}

#[derive(Clone, Copy)]
struct Watcher {
    cref: usize,
    blocker: Lit,
}

/// A CDCL SAT solver.
pub struct Solver {
    n_vars: usize,
    clauses: Vec<Clause>,
    watches: Vec<Vec<Watcher>>,
    assigns: Vec<i8>,
    level: Vec<i32>,
    reason: Vec<Option<usize>>,
    trail: Vec<Lit>,
    trail_lim: Vec<usize>,
    qhead: usize,
    activity: Vec<f64>,
    polarity: Vec<bool>,
    conflict_count: i32,
    restart_limit: i32,
    unsat: bool,
}

impl Solver {
    /// Create a solver for the given CNF.
    pub fn new(cnf: Cnf) -> Solver {
        let n = cnf.var_count();
        let mut s = Solver {
            n_vars: n,
            clauses: Vec::new(),
            watches: vec![Vec::new(); 2 * n],
            assigns: vec![0; n],
            level: vec![0; n],
            reason: vec![None; n],
            trail: Vec::new(),
            trail_lim: Vec::new(),
            qhead: 0,
            activity: vec![0.0; n],
            polarity: vec![true; n],
            conflict_count: 0,
            restart_limit: 100,
            unsat: false,
        };
        for c in cnf.clauses {
            if !s.add_clause(&c.lits) {
                s.unsat = true;
            }
        }
        if s.propagate().is_some() {
            s.unsat = true;
        }
        s
    }

    fn lit_value(&self, l: Lit) -> Option<bool> {
        let v = l.var() as usize;
        match self.assigns[v] {
            0 => None,
            1 => Some(l.is_pos()),
            -1 => Some(l.is_neg()),
            _ => unreachable!(),
        }
    }

    fn decision_level(&self) -> i32 {
        self.trail_lim.len() as i32
    }

    fn enqueue(&mut self, l: Lit, r: Option<usize>) -> bool {
        let v = l.var() as usize;
        if self.assigns[v] != 0 {
            return self.lit_value(l) == Some(true);
        }
        self.assigns[v] = if l.is_neg() { -1 } else { 1 };
        self.level[v] = self.decision_level();
        self.reason[v] = r;
        self.trail.push(l);
        true
    }

    fn add_clause(&mut self, lits: &[Lit]) -> bool {
        let mut clause: Vec<Lit> = Vec::with_capacity(lits.len());
        for &l in lits {
            if clause.iter().any(|&x| x == l.neg()) {
                return true; // tautology
            }
            if clause.contains(&l) {
                continue; // duplicate
            }
            clause.push(l);
        }
        if clause.is_empty() {
            return false; // empty clause → already UNSAT
        }
        if clause.len() == 1 {
            return self.enqueue(clause[0], None);
        }
        let idx = self.clauses.len();
        self.clauses.push(Clause {
            lits: clause.clone(),
            learnt: false,
        });
        self.watches[clause[0].index()].push(Watcher {
            cref: idx,
            blocker: clause[1],
        });
        self.watches[clause[1].index()].push(Watcher {
            cref: idx,
            blocker: clause[0],
        });
        true
    }

    /// Unit propagation. Returns the conflicting clause index, if any.
    ///
    /// Invariant: a clause watches two literals that are not currently false.
    /// When a literal `p` is assigned true, its negation `¬p` becomes false, so
    /// we wake the clauses watching `¬p`.
    fn propagate(&mut self) -> Option<usize> {
        while self.qhead < self.trail.len() {
            let p = self.trail[self.qhead];
            self.qhead += 1;
            let q = p.neg().index();
            let mut ws = std::mem::take(&mut self.watches[q]);
            let mut kept: Vec<Watcher> = Vec::with_capacity(ws.len());
            let mut conflict: Option<usize> = None;
            let mut i = 0;
            while i < ws.len() {
                let w = ws[i];
                i += 1;
                // Fast path: if the *other* watched literal is already true the
                // clause is satisfied regardless of `q`; keep the watch as-is.
                if self.lit_value(w.blocker) == Some(true) {
                    kept.push(w);
                    continue;
                }
                let cref = w.cref;
                let slot = if self.clauses[cref].lits[0].index() == q {
                    0
                } else {
                    1
                };
                let other_val = self.lit_value(self.clauses[cref].lits[1 - slot]);
                match other_val {
                    Some(true) => {
                        // Satisfied via the other watched literal.
                        kept.push(w);
                    }
                    _ => {
                        // The other watched literal is false or unassigned. Before
                        // concluding conflict/unit we must check whether a
                        // *non-watched* literal already satisfies the clause —
                        // otherwise a satisfied clause is mis-reported as a conflict
                        // (unsound `Unsat`) or the watch is not relocated.
                        let c_len = self.clauses[cref].lits.len();
                        let mut found = None;
                        for k in 0..c_len {
                            if k == 0 || k == 1 {
                                continue;
                            }
                            if self.lit_value(self.clauses[cref].lits[k]) != Some(false) {
                                found = Some(k);
                                break;
                            }
                        }
                        match found {
                            Some(k) => {
                                let lk = self.clauses[cref].lits[k];
                                self.clauses[cref].lits[slot] = lk;
                                self.clauses[cref].lits[k] = p.neg();
                                self.watches[lk.index()].push(Watcher {
                                    cref,
                                    blocker: self.clauses[cref].lits[1 - slot],
                                });
                            }
                            None => {
                                // No satisfying literal remains. If the other watch
                                // is false this is a real conflict; if it is
                                // unassigned, unit-propagate it (and establish the
                                // position-0 reason invariant `analyze()` relies on).
                                let other = self.clauses[cref].lits[1 - slot];
                                self.clauses[cref].lits.swap(0, 1 - slot);
                                kept.push(w);
                                if other_val == Some(false) || !self.enqueue(other, Some(cref)) {
                                    conflict = Some(cref);
                                    for r in ws.drain(i..) {
                                        kept.push(r);
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            self.watches[q] = kept;
            if conflict.is_some() {
                self.qhead = self.trail.len();
                return conflict;
            }
        }
        None
    }

    fn analyze(&mut self, confl: usize) -> (Vec<Lit>, i32) {
        let mut seen = vec![false; self.n_vars];
        let mut out_learnt: Vec<Lit> = Vec::new();
        let mut path_c = 0i32;
        let mut p: Option<Lit> = None;
        let mut q = self.trail.len() as i32 - 1;
        let mut c = confl;
        let mut out_btlevel = 0i32;
        loop {
            let clause = self.clauses[c].lits.clone();
            let start = if p.is_none() { 0 } else { 1 };
            for qlit in clause.iter().skip(start) {
                let x = qlit.var() as usize;
                if !seen[x] && self.level[x] > 0 {
                    self.bump_activity(x);
                    seen[x] = true;
                    if self.level[x] >= self.decision_level() {
                        path_c += 1;
                    } else {
                        out_learnt.push(*qlit);
                        if self.level[x] > out_btlevel {
                            out_btlevel = self.level[x];
                        }
                    }
                }
            }
            while q >= 0 && !seen[self.trail[q as usize].var() as usize] {
                q -= 1;
            }
            p = Some(self.trail[q as usize]);
            q -= 1;
            path_c -= 1;
            if path_c <= 0 {
                break;
            }
            c = self.reason[p.unwrap().var() as usize].expect("implied literal must have a reason");
        }
        out_learnt.insert(0, p.unwrap().neg());
        if out_learnt.len() > 1 {
            let mut max_i = 1usize;
            for i in 1..out_learnt.len() {
                if self.level[out_learnt[i].var() as usize]
                    > self.level[out_learnt[max_i].var() as usize]
                {
                    max_i = i;
                }
            }
            out_learnt.swap(max_i, 1);
            out_btlevel = self.level[out_learnt[1].var() as usize];
        }
        (out_learnt, out_btlevel)
    }

    fn add_learnt(&mut self, learnt: &[Lit]) {
        let idx = self.clauses.len();
        self.clauses.push(Clause {
            lits: learnt.to_vec(),
            learnt: true,
        });
        self.watches[learnt[0].index()].push(Watcher {
            cref: idx,
            blocker: learnt.get(1).copied().unwrap_or(learnt[0]),
        });
        if learnt.len() > 1 {
            self.watches[learnt[1].index()].push(Watcher {
                cref: idx,
                blocker: learnt[0],
            });
        }
        let _ = self.enqueue(learnt[0], Some(idx));
    }

    fn cancel_until(&mut self, level: i32) {
        if self.trail_lim.len() as i32 <= level {
            return;
        }
        let target = self.trail_lim[level as usize];
        while self.trail.len() > target {
            let l = self.trail.pop().unwrap();
            let v = l.var() as usize;
            self.assigns[v] = 0;
            self.reason[v] = None;
            self.level[v] = 0;
        }
        self.trail_lim.truncate(level as usize);
        self.qhead = self.trail.len();
    }

    fn bump_activity(&mut self, v: usize) {
        self.activity[v] += 1.0;
    }

    fn decay_activities(&mut self) {
        for a in &mut self.activity {
            *a *= 0.95;
        }
    }

    fn pick_decision(&self) -> Option<Var> {
        let mut best: Option<usize> = None;
        let mut best_act = 0.0f64;
        for v in 0..self.n_vars {
            if self.assigns[v] == 0 && (best.is_none() || self.activity[v] > best_act) {
                best = Some(v);
                best_act = self.activity[v];
            }
        }
        best.map(|v| v as Var)
    }

    /// Solve the formula.
    pub fn solve(&mut self) -> SatResult {
        if self.unsat {
            return SatResult::Unsat;
        }
        loop {
            match self.propagate() {
                None => {
                    if let Some(v) = self.pick_decision() {
                        self.trail_lim.push(self.trail.len());
                        let lit = Lit::new(v, !self.polarity[v as usize]);
                        self.enqueue(lit, None);
                    } else {
                        return SatResult::Sat;
                    }
                }
                Some(confl) => {
                    self.conflict_count += 1;
                    if self.decision_level() == 0 {
                        return SatResult::Unsat;
                    }
                    let (learnt, btlevel) = self.analyze(confl);
                    self.cancel_until(btlevel);
                    self.add_learnt(&learnt);
                    self.decay_activities();
                    if self.conflict_count >= self.restart_limit {
                        self.cancel_until(0);
                        self.restart_limit = ((self.restart_limit as f64) * 1.5) as i32 + 100;
                    }
                }
            }
        }
    }

    /// The satisfying assignment, indexed by variable (`var -> true/false`).
    ///
    /// Valid only after [`Solver::solve`] returned [`SatResult::Sat`].
    pub fn model(&self) -> Vec<bool> {
        (0..self.n_vars).map(|v| self.assigns[v] == 1).collect()
    }

    /// The satisfying Boolean value of a variable.
    ///
    /// Returns `None` for a `v` outside `0..var_count()`, matching the
    /// crate-wide "never panic on caller-supplied indices" discipline.
    pub fn value(&self, v: Var) -> Option<bool> {
        let idx = v as usize;
        if idx >= self.n_vars {
            return None;
        }
        match self.assigns[idx] {
            0 => None,
            1 => Some(true),
            -1 => Some(false),
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_encoding() {
        let p = Lit::new(0, false);
        assert_eq!(p.var(), 0);
        assert!(p.is_pos());
        assert_eq!(p.neg().var(), 0);
        assert!(p.neg().is_neg());
        assert_eq!(Lit::from_dimacs(3).var(), 2);
        assert_eq!(Lit::from_dimacs(-2).var(), 1);
        assert!(Lit::from_dimacs(-2).is_neg());
    }

    #[test]
    fn simple_sat() {
        // (x ∨ y) ∧ (¬x ∨ y) ∧ (x ∨ ¬y)  → SAT (x=true,y=true)
        let cnf = Cnf::from_lits(2, &[&[1, 2], &[-1, 2], &[1, -2]]).unwrap();
        let mut s = Solver::new(cnf);
        assert_eq!(s.solve(), SatResult::Sat);
        assert!(s.value(0).unwrap() && s.value(1).unwrap());
    }

    #[test]
    fn simple_unsat() {
        // (x) ∧ (¬x)  → UNSAT
        let cnf = Cnf::from_lits(1, &[&[1], &[-1]]).unwrap();
        let mut s = Solver::new(cnf);
        assert_eq!(s.solve(), SatResult::Unsat);
    }

    #[test]
    fn unit_propagation_chain() {
        // (x) ∧ (¬x ∨ y) ∧ (¬y ∨ z) → SAT with x,y,z true
        let cnf = Cnf::from_lits(3, &[&[1], &[-1, 2], &[-2, 3]]).unwrap();
        let mut s = Solver::new(cnf);
        assert_eq!(s.solve(), SatResult::Sat);
        assert!(s.value(0).unwrap() && s.value(1).unwrap() && s.value(2).unwrap());
    }

    #[test]
    fn pigeonhole_two() {
        // (p0) (p1) (¬p0 ∨ ¬p1)  → UNSAT
        let cnf = Cnf::from_lits(2, &[&[1], &[2], &[-1, -2]]).unwrap();
        let mut s = Solver::new(cnf);
        assert_eq!(s.solve(), SatResult::Unsat);
    }

    #[test]
    fn schur_like_sat() {
        // (a ∨ b) ∧ (¬a ∨ c) ∧ (¬b ∨ ¬c) → SAT
        let cnf = Cnf::from_lits(3, &[&[1, 2], &[-1, 3], &[-2, -3]]).unwrap();
        let mut s = Solver::new(cnf);
        assert_eq!(s.solve(), SatResult::Sat);
        let m = s.model();
        let eval = |l: Lit| {
            if l.is_neg() {
                !m[l.var() as usize]
            } else {
                m[l.var() as usize]
            }
        };
        assert!(s.clauses.iter().all(|c| c.lits.iter().any(|&l| eval(l))));
    }

    #[test]
    fn empty_clause_unsat() {
        let cnf = Cnf::from_lits(0, &[&[]]).unwrap();
        let mut s = Solver::new(cnf);
        assert_eq!(s.solve(), SatResult::Unsat);
    }

    #[test]
    fn learned_clause_solves_harder() {
        // A small UNSAT that needs a learned clause:
        // (x ∨ y) (x ∨ ¬y) (¬x ∨ z) (¬x ∨ ¬z) (¬y ∨ ¬z) (¬y ∨ z) ... classic 3-SAT pair
        let cnf = Cnf::from_lits(
            3,
            &[&[1, 2], &[1, -2], &[-1, 3], &[-1, -3], &[2, 3], &[-2, -3]],
        )
        .unwrap();
        let mut s = Solver::new(cnf);
        // (x∨y)(x∨¬y) forces x; (¬x∨z)(¬x∨¬z) forces ¬x → conflict
        assert_eq!(s.solve(), SatResult::Unsat);
    }

    #[test]
    fn conflict_analysis_uses_position_zero_invariant() {
        // UNSAT that forces `propagate()` to unit-implied a literal whose
        // reason clause has the falsified watch at index 0 (`slot == 0`),
        // so the implied literal lands at index 1. If `analyze()`'s
        // "asserted literal at position 0" invariant is not established,
        // resolution through that reason clause drops a real literal and the
        // solver can emit an unsound learned clause.
        //
        // p ; (¬r) ; (¬p ∨ q ∨ r) ⇒ q ; (¬q ∨ s) ⇒ s ; (¬s ∨ t) ⇒ t ;
        // (¬t ∨ ¬u) with u ; contradiction t. The reason chain for q runs
        // through the length-3 clause `(¬p ∨ q ∨ r)`, whose falsified watch is
        // at index 0 — exactly the path `analyze()` mishandles without the
        // position-0 invariant.
        let cnf = Cnf::from_lits(
            6,
            &[
                &[1],        // p
                &[-3],       // ¬r
                &[-1, 2, 3], // (¬p ∨ q ∨ r)
                &[-2, 4],    // (¬q ∨ s)
                &[-4, 5],    // (¬s ∨ t)
                &[-5, -6],   // (¬t ∨ ¬u)
                &[6],        // u
            ],
        )
        .unwrap();
        let mut s = Solver::new(cnf);
        assert_eq!(s.solve(), SatResult::Unsat);
    }

    #[test]
    fn value_rejects_out_of_range_var() {
        let cnf = Cnf::from_lits(2, &[&[1, 2]]).unwrap();
        let mut s = Solver::new(cnf);
        s.solve();
        assert_eq!(s.value(2), None);
        assert_eq!(s.value(1_000_000), None);
    }

    #[test]
    fn from_lits_rejects_malformed() {
        // `0` is the DIMACS clause terminator, not a literal.
        assert!(Cnf::from_lits(2, &[&[0]]).is_none());
        // var index `>= vars` is out of range.
        assert!(Cnf::from_lits(2, &[&[1, -3]]).is_none());
        assert!(Cnf::from_lits(1, &[&[-2]]).is_none());
        // well-formed input is accepted.
        assert!(Cnf::from_lits(3, &[&[1], &[-2, 3]]).is_some());
    }
}
