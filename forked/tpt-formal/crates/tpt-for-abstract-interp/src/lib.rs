#![allow(clippy::should_implement_trait)]
//! Generic abstract-interpretation framework.
//!
//! Provides the three ingredients of abstract interpretation:
//!
//! 1. An [`AbstractDomain`] trait (the elements of an abstract lattice with
//!    `join`/`meet`/`widen`).
//! 2. A [`Interval`] domain over `i64` — the canonical first domain.
//! 3. A fixpoint [`analyze`] engine over a control-flow graph ([`Cfg`]) of
//!    assignment/assume basic blocks, using a worklist with widening to
//!    guarantee termination on loops.
//!
//! ```
//! use tpt_for_abstract_interp::{AbstractDomain, Cfg, Stmt, Expr, CmpOp, Interval, analyze};
//!
//! // x = 0; while (x < 10) { x = x + 1; }   →  x ∈ [10, +∞) at exit
//! let mut cfg = Cfg::new(1);
//! cfg.block(&[Stmt::assign(0, Expr::const_(0))], &[1]);
//! cfg.block(&[], &[2, 3]);                                   // loop head
//! cfg.block(
//!     &[Stmt::assume(0, CmpOp::Lt, 10),
//!        Stmt::assign(0, Expr::add(Expr::var(0), Expr::const_(1)))],
//!     &[1],
//! );
//! cfg.block(&[Stmt::assume(0, CmpOp::Ge, 10)], &[4]);
//! cfg.block(&[], &[]);                                       // exit
//!
//! let states = analyze(&cfg, 0, vec![Interval::bottom()]).unwrap();
//! let exit = states[4][0];
//! assert_eq!(exit.lo(), 10);
//! assert_eq!(exit.hi(), i64::MAX);
//! ```

/// A variable identifier (a `0`-based index into the abstract state vector).
pub type VarId = usize;

/// The contract every abstract domain must satisfy.
///
/// `join` (least upper bound) over-approximates disjunction (merging two
/// execution paths); `meet` is the greatest lower bound; `widen` is the
/// widening operator that enforces termination of ascending fixpoint
/// iteration; `bottom` is the empty set (unreachable / contradictory state).
pub trait AbstractDomain: Clone + PartialEq {
    /// The top element: no information (all behaviors possible).
    fn top() -> Self;
    /// The bottom element: the empty set of concrete states.
    fn bottom() -> Self;
    /// Whether this element is bottom.
    fn is_bottom(&self) -> bool;
    /// Least upper bound (over-approximation of disjunction).
    fn join(&self, other: &Self) -> Self;
    /// Greatest lower bound (over-approximation of conjunction).
    fn meet(&self, other: &Self) -> Self;
    /// Widening operator used to force fixpoint termination.
    fn widen(&self, other: &Self) -> Self;
}

/// A numeric interval `[lo, hi]` over `i64`.
///
/// `lo > hi` represents the bottom (empty) domain element. `top` is
/// `[i64::MIN, i64::MAX]`. Arithmetic is saturating so it never panics and
/// always over-approximates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Interval {
    lo: i64,
    hi: i64,
}

impl Interval {
    /// Construct `[lo, hi]`. A range with `lo > hi` is the bottom element.
    pub fn new(lo: i64, hi: i64) -> Interval {
        Interval { lo, hi }
    }

    /// The lower bound.
    pub fn lo(&self) -> i64 {
        self.lo
    }

    /// The upper bound.
    pub fn hi(&self) -> i64 {
        self.hi
    }

    /// Whether the interval contains the concrete value `x`.
    pub fn contains(&self, x: i64) -> bool {
        !self.is_bottom() && self.lo <= x && x <= self.hi
    }

    /// Whether this interval is the top element (no information).
    pub fn is_top(&self) -> bool {
        !self.is_bottom() && self.lo == i64::MIN && self.hi == i64::MAX
    }

    fn sat_add(a: i64, b: i64) -> i64 {
        a.saturating_add(b)
    }

    fn sat_sub(a: i64, b: i64) -> i64 {
        a.saturating_sub(b)
    }

    fn sat_mul(a: i64, b: i64) -> i64 {
        a.saturating_mul(b)
    }
}

impl AbstractDomain for Vec<Interval> {
    fn top() -> Vec<Interval> {
        unreachable!("state vectors are built from a known variable count")
    }

    fn bottom() -> Vec<Interval> {
        Vec::new()
    }

    fn is_bottom(&self) -> bool {
        self.iter().all(Interval::is_bottom)
    }

    fn join(&self, other: &Vec<Interval>) -> Vec<Interval> {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a.join(b))
            .collect()
    }

    fn meet(&self, other: &Vec<Interval>) -> Vec<Interval> {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a.meet(b))
            .collect()
    }

    fn widen(&self, other: &Vec<Interval>) -> Vec<Interval> {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a.widen(b))
            .collect()
    }
}

impl AbstractDomain for Interval {
    fn top() -> Interval {
        Interval {
            lo: i64::MIN,
            hi: i64::MAX,
        }
    }

    fn bottom() -> Interval {
        Interval { lo: 1, hi: 0 }
    }

    fn is_bottom(&self) -> bool {
        self.lo > self.hi
    }

    fn join(&self, other: &Interval) -> Interval {
        if self.is_bottom() {
            return *other;
        }
        if other.is_bottom() {
            return *self;
        }
        Interval {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    fn meet(&self, other: &Interval) -> Interval {
        if self.is_bottom() || other.is_bottom() {
            return Interval::bottom();
        }
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo > hi {
            Interval::bottom()
        } else {
            Interval { lo, hi }
        }
    }

    fn widen(&self, other: &Interval) -> Interval {
        if self.is_bottom() {
            return *other;
        }
        if other.is_bottom() {
            return *self;
        }
        let lo = if other.lo < self.lo {
            i64::MIN
        } else {
            self.lo
        };
        let hi = if other.hi > self.hi {
            i64::MAX
        } else {
            self.hi
        };
        Interval { lo, hi }
    }
}

/// A numeric expression over program variables, used by the transfer
/// functions of the fixpoint engine.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// A constant integer.
    Const(i64),
    /// Reading a variable.
    Var(VarId),
    /// Addition.
    Add(Box<Expr>, Box<Expr>),
    /// Subtraction.
    Sub(Box<Expr>, Box<Expr>),
    /// Multiplication.
    Mul(Box<Expr>, Box<Expr>),
    /// Negation.
    Neg(Box<Expr>),
}

impl Expr {
    /// A constant expression.
    pub fn const_(c: i64) -> Expr {
        Expr::Const(c)
    }

    /// A variable read.
    pub fn var(v: VarId) -> Expr {
        Expr::Var(v)
    }

    /// `a + b`.
    pub fn add(a: Expr, b: Expr) -> Expr {
        Expr::Add(Box::new(a), Box::new(b))
    }

    /// `a - b`.
    pub fn sub(a: Expr, b: Expr) -> Expr {
        Expr::Sub(Box::new(a), Box::new(b))
    }

    /// `a * b`.
    pub fn mul(a: Expr, b: Expr) -> Expr {
        Expr::Mul(Box::new(a), Box::new(b))
    }

    /// `-a`.
    pub fn neg(a: Expr) -> Expr {
        Expr::Neg(Box::new(a))
    }

    /// Evaluate the expression against an abstract state of intervals.
    ///
    /// Returns `Interval::bottom` if any referenced variable is bottom.
    pub fn eval(&self, state: &[Interval]) -> Interval {
        match self {
            Expr::Const(c) => Interval::new(*c, *c),
            Expr::Var(v) => {
                // Out-of-range variable: fall back to top (unknown), consistent
                // with the crate's "unknown → top" philosophy. Never panics on
                // malformed caller-supplied `VarId`s.
                if *v >= state.len() {
                    return Interval::top();
                }
                let i = state[*v];
                if i.is_bottom() {
                    Interval::bottom()
                } else {
                    i
                }
            }
            Expr::Add(a, b) => a.eval(state).add(b.eval(state)),
            Expr::Sub(a, b) => a.eval(state).sub(b.eval(state)),
            Expr::Mul(a, b) => a.eval(state).mul(b.eval(state)),
            Expr::Neg(a) => a.eval(state).neg(),
        }
    }
}

impl Interval {
    fn add(self, o: Interval) -> Interval {
        if self.is_bottom() || o.is_bottom() {
            return Interval::bottom();
        }
        Interval::new(
            Interval::sat_add(self.lo, o.lo),
            Interval::sat_add(self.hi, o.hi),
        )
    }

    fn sub(self, o: Interval) -> Interval {
        if self.is_bottom() || o.is_bottom() {
            return Interval::bottom();
        }
        Interval::new(
            Interval::sat_sub(self.lo, o.hi),
            Interval::sat_sub(self.hi, o.lo),
        )
    }

    fn mul(self, o: Interval) -> Interval {
        if self.is_bottom() || o.is_bottom() {
            return Interval::bottom();
        }
        let candidates = [
            Interval::sat_mul(self.lo, o.lo),
            Interval::sat_mul(self.lo, o.hi),
            Interval::sat_mul(self.hi, o.lo),
            Interval::sat_mul(self.hi, o.hi),
        ];
        let lo = candidates.iter().copied().min().unwrap();
        let hi = candidates.iter().copied().max().unwrap();
        Interval::new(lo, hi)
    }

    fn neg(self) -> Interval {
        if self.is_bottom() {
            return Interval::bottom();
        }
        Interval::new(Interval::sat_sub(0, self.hi), Interval::sat_sub(0, self.lo))
    }
}

/// A comparison operator used by [`Stmt::Assume`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    /// `var < k`.
    Lt,
    /// `var <= k`.
    Le,
    /// `var > k`.
    Gt,
    /// `var >= k`.
    Ge,
    /// `var == k`.
    Eq,
}

/// A single statement in a basic block.
#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    /// `v = expr`.
    Assign(VarId, Expr),
    /// Refine `v`'s interval by the constraint `v <op> k` (sound over-approx).
    Assume(VarId, CmpOp, i64),
}

impl Stmt {
    /// Build an assignment statement.
    pub fn assign(v: VarId, expr: Expr) -> Stmt {
        Stmt::Assign(v, expr)
    }

    /// Build an assume (narrowing) statement.
    pub fn assume(v: VarId, op: CmpOp, k: i64) -> Stmt {
        Stmt::Assume(v, op, k)
    }
}

/// A basic block: a straight-line sequence of statements with successor edges.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    stmts: Vec<Stmt>,
    succ: Vec<usize>,
}

impl Block {
    /// The statements of the block.
    pub fn stmts(&self) -> &[Stmt] {
        &self.stmts
    }

    /// The successor node indices.
    pub fn succ(&self) -> &[usize] {
        &self.succ
    }
}

/// A control-flow graph: a list of basic blocks indexed by node id.
#[derive(Clone, Debug)]
pub struct Cfg {
    blocks: Vec<Block>,
    nvars: usize,
}

impl Cfg {
    /// Create a CFG with `nvars` variables and `nblocks` empty blocks.
    pub fn new(nvars: usize) -> Cfg {
        Cfg {
            blocks: Vec::new(),
            nvars,
        }
    }

    /// Number of variables in the abstract state.
    pub fn nvars(&self) -> usize {
        self.nvars
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Whether the CFG has no nodes.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Append a block with the given statements and successors; returns its id.
    pub fn block(&mut self, stmts: &[Stmt], succ: &[usize]) -> usize {
        self.blocks.push(Block {
            stmts: stmts.to_vec(),
            succ: succ.to_vec(),
        });
        self.blocks.len() - 1
    }

    /// Overwrite block `id` (used when ids must be deterministic).
    pub fn set_block(&mut self, id: usize, stmts: &[Stmt], succ: &[usize]) {
        self.blocks[id] = Block {
            stmts: stmts.to_vec(),
            succ: succ.to_vec(),
        };
    }

    fn transfer(&self, node: usize, input: &[Interval]) -> Vec<Interval> {
        let mut state = input.to_vec();
        for s in &self.blocks[node].stmts {
            match s {
                Stmt::Assign(v, e) => {
                    // Write to an out-of-range variable is a no-op (never panics
                    // on malformed caller input); reads already fall back to top.
                    if *v < state.len() {
                        state[*v] = e.eval(&state);
                    }
                }
                Stmt::Assume(v, op, k) => {
                    if *v < state.len() {
                        state[*v] = narrow(state[*v], *op, *k);
                    }
                }
            }
        }
        state
    }
}

fn narrow(i: Interval, op: CmpOp, k: i64) -> Interval {
    if i.is_bottom() {
        return Interval::bottom();
    }
    let (lo, hi) = match op {
        CmpOp::Lt => (i.lo, i.hi.min(k.saturating_sub(1))),
        CmpOp::Le => (i.lo, i.hi.min(k)),
        CmpOp::Gt => (i.lo.max(k.saturating_add(1)), i.hi),
        CmpOp::Ge => (i.lo.max(k), i.hi),
        CmpOp::Eq => (i.lo.max(k), i.hi.min(k)),
    };
    if lo > hi {
        Interval::bottom()
    } else {
        Interval::new(lo, hi)
    }
}

/// Analyze a CFG, returning the entry abstract state for every node.
///
/// Uses a worklist fixpoint with widening at every merge, which is sound and
/// guaranteed to terminate for domains of finite ascending height (such as
/// [`Interval`]). Returns `None` if `entry` is out of bounds.
pub fn analyze(cfg: &Cfg, entry: usize, init: Vec<Interval>) -> Option<Vec<Vec<Interval>>> {
    if entry >= cfg.blocks.len() || init.len() != cfg.nvars {
        return None;
    }
    let n = cfg.blocks.len();
    let bottom = (0..cfg.nvars)
        .map(|_| Interval::bottom())
        .collect::<Vec<_>>();
    let mut state: Vec<Vec<Interval>> = vec![bottom.clone(); n];
    state[entry] = init.clone();

    // Worklist fixpoint: seed the worklist at the entry only and rely solely
    // on incremental push propagation. Each node's entry state (`state[node]`)
    // is the value accumulated by predecessor pushes; `widen` is monotonic in
    // the sense `widen(a, b) ⊇ a` always, so the accumulated value only ever
    // grows. We deliberately do NOT recompute `state[node]` from predecessors
    // on every visit: `widen` is not monotonic in its *first* argument, so a
    // recompute can be strictly smaller than the value already established and
    // break the ascending-chain guarantee the algorithm's soundness rests on.
    let mut work: Vec<usize> = vec![entry];
    while let Some(node) = work.pop() {
        let in_state = state[node].clone();
        let exit = cfg.transfer(node, &in_state);
        for &s in &cfg.blocks[node].succ {
            if s >= n {
                continue;
            }
            let merged = state[s].widen(&exit);
            if merged != state[s] {
                state[s] = merged;
                work.push(s);
            }
        }
    }
    Some(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_lattice_ops() {
        let a = Interval::new(0, 5);
        let b = Interval::new(3, 10);
        assert_eq!(a.join(&b), Interval::new(0, 10));
        assert_eq!(a.meet(&b), Interval::new(3, 5));
        // Widening is *not* join: it jumps the upper bound straight to top.
        assert_eq!(a.widen(&b), Interval::new(0, i64::MAX));
        let big = Interval::new(0, 5);
        let wide = big.widen(&Interval::new(i64::MIN, i64::MAX));
        assert!(wide.is_top());
        assert!(Interval::bottom().is_bottom());
        assert!(Interval::top().is_top());
    }

    #[test]
    fn interval_arithmetic() {
        let x = Interval::new(1, 3);
        let y = Interval::new(10, 20);
        assert_eq!(x.add(y), Interval::new(11, 23));
        assert_eq!(x.mul(y), Interval::new(10, 60));
        assert_eq!(x.sub(y), Interval::new(-19, -7));
        assert_eq!(x.neg(), Interval::new(-3, -1));
    }

    #[test]
    fn expr_eval() {
        // (x + 1) * 2  with x ∈ [1,3]  → [4,8]
        let e = Expr::mul(Expr::add(Expr::var(0), Expr::const_(1)), Expr::const_(2));
        let state = vec![Interval::new(1, 3)];
        assert_eq!(e.eval(&state), Interval::new(4, 8));
    }

    #[test]
    fn loop_bound_analysis() {
        // x = 0; while (x < 10) { x = x + 1; }  →  x ∈ [10, +∞) at exit
        let mut cfg = Cfg::new(1);
        cfg.block(&[Stmt::assign(0, Expr::const_(0))], &[1]);
        cfg.block(&[], &[2, 3]);
        cfg.block(
            &[
                Stmt::assume(0, CmpOp::Lt, 10),
                Stmt::assign(0, Expr::add(Expr::var(0), Expr::const_(1))),
            ],
            &[1],
        );
        cfg.block(&[Stmt::assume(0, CmpOp::Ge, 10)], &[4]);
        cfg.block(&[], &[]);

        let states = analyze(&cfg, 0, vec![Interval::bottom()]).unwrap();
        let exit = states[4][0];
        assert_eq!(exit.lo(), 10);
        assert_eq!(exit.hi(), i64::MAX);
    }

    #[test]
    fn unreachable_branch_is_bottom() {
        // x = 5; assume(x < 0)  → bottom (contradiction)
        let mut cfg = Cfg::new(1);
        cfg.block(&[Stmt::assign(0, Expr::const_(5))], &[1]);
        cfg.block(&[Stmt::assume(0, CmpOp::Lt, 0)], &[2]);
        cfg.block(&[], &[]);
        let states = analyze(&cfg, 0, vec![Interval::bottom()]).unwrap();
        assert!(states[2][0].is_bottom());
    }

    #[test]
    fn two_variable_analysis() {
        // x = 0; y = x + 7;  → x∈[0,0], y∈[7,7]
        let mut cfg = Cfg::new(2);
        cfg.block(
            &[
                Stmt::assign(0, Expr::const_(0)),
                Stmt::assign(1, Expr::add(Expr::var(0), Expr::const_(7))),
            ],
            &[1],
        );
        cfg.block(&[], &[]);
        let states = analyze(&cfg, 0, vec![Interval::bottom(), Interval::bottom()]).unwrap();
        assert_eq!(states[1][0], Interval::new(0, 0));
        assert_eq!(states[1][1], Interval::new(7, 7));
    }

    #[test]
    fn diamond_merge_never_shrinks() {
        // Two predecessors merge into one node with no back-edge:
        //   entry: x = 0
        //   a:     x = 1 ; b: x = 2
        //   merge:  y = x
        // The merge node's x must be the widening of both (a sound over-
        // approximation), and must never shrink across fixpoint iterations even
        // though `widen` is non-monotonic in its first argument.
        let mut cfg = Cfg::new(2);
        cfg.block(&[Stmt::assign(0, Expr::const_(0))], &[1, 2]); // entry
        cfg.block(&[Stmt::assign(0, Expr::const_(1))], &[3]); // a
        cfg.block(&[Stmt::assign(0, Expr::const_(2))], &[3]); // b
        cfg.block(&[Stmt::assign(1, Expr::var(0))], &[]); // merge
        let states = analyze(&cfg, 0, vec![Interval::bottom(), Interval::bottom()]).unwrap();
        // The merge node's entry `x` is the widening of the two predecessor
        // exits {1, 2}: a sound over-approximation that contains both values.
        // (analyze returns *entry* states; `y = x` lives in the merge exit and
        // only matters for successors, of which there are none here.)
        let x = states[3][0];
        assert!(x.lo() <= 1);
        assert!(x.hi() >= 2 || x.is_top());
        // The ascending chain must hold: nothing the fixpoint settles on can be
        // a strict under-approximation of either predecessor's single value.
        assert!(x.lo() <= 2);
    }

    #[test]
    fn out_of_range_varid_is_safe() {
        // A statement referencing a variable beyond `nvars` must not panic;
        // reads fall back to top and the write is a no-op. The effect shows up
        // in the successor's entry state.
        let mut cfg = Cfg::new(1);
        cfg.block(
            &[
                Stmt::assign(0, Expr::add(Expr::var(0), Expr::var(5))),
                Stmt::assign(7, Expr::const_(9)),
            ],
            &[1],
        );
        cfg.block(&[], &[]);
        let states = analyze(&cfg, 0, vec![Interval::new(3, 3)]).unwrap();
        // entry var 0 = 3 + top(var 5) = a sound over-approximation (the unknown
        // read widens the lower bound below 3); no panic on the out-of-range
        // read or the out-of-range write.
        assert!(!states[1][0].is_bottom());
        assert!(states[1][0].lo() < 3);
        assert!(states[1][0].hi() >= 3);
    }
}
