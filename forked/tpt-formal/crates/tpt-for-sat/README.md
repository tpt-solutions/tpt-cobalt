# tpt-for-sat

From-scratch, pure-Rust CDCL SAT solver (no FFI): CNF, watched literals, 1UIP clause learning, restarts.

No external solver binary, no FFI. Implements a modern conflict-driven
clause-learning (CDCL) loop: watched literals for cheap unit propagation, 1UIP
conflict analysis producing a learned clause on every conflict, VSIDS activity
ordering for decisions, and geometric restart scheduling. The public surface is
tiny: build a `Cnf`, wrap it in a `Solver`, call `solve`.

## Features

- `Lit` — a Boolean literal (`var << 1 | sign`), with `new`, `var`, `is_neg`,
  `is_pos`, `neg`, `index`, and `from_dimacs` (DIMACS integer → literal).
- `Clause` — a disjunction of literals, with `lits` and `is_learnt` accessors.
- `Cnf` — a formula in conjunctive normal form, with `var_count`, `clauses`,
  and `from_lits(vars, &[&[i32]])` (returns `None` on malformed / out-of-range input).
- `SatResult` — `Sat` / `Unsat`.
- `Solver` — a full CDCL engine: `new(cnf)`, `solve()` returning `SatResult`,
  `model()` (full assignment by variable), and `value(var)` (per-variable
  assignment, `None` for out-of-range variables).
- Conflict-driven clause learning implemented in-tree with zero dependencies:
  watched literals for cheap unit propagation, 1UIP conflict analysis that
  learns a clause on every conflict, VSIDS variable-activity ordering for
  decisions, and geometric restart scheduling.

## Example

Encode "is this graph 2-colorable?" as a CNF and solve it. Each vertex `v` is a
variable `x_v`; an edge `(u, v)` becomes `(x_u ∨ x_v) ∧ (¬x_u ∨ ¬x_v)`. A path
(`0—1—2`) is bipartite → `Sat` (read the coloring from the model); a triangle
(`0—1—2—0`) is an odd cycle → `Unsat`. The literal API and `from_lits`'s
rejection of malformed input are also shown.

```rust
use tpt_for_sat::{Cnf, Lit, SatResult, Solver};

// 3-vertex path 0—1—2: a valid coloring is x0=true, x1=false, x2=true.
let mut path = vec![vec![1, 2], vec![-1, -2], vec![2, 3], vec![-2, -3]];
let refs: Vec<&[i32]> = path.iter().collect();
let cnf = Cnf::from_lits(3, &refs).unwrap();
let mut solver = Solver::new(cnf);
assert_eq!(solver.solve(), SatResult::Sat);
// Adjacent vertices differ in the model.
assert!(solver.value(0) != solver.value(1) && solver.value(1) != solver.value(2));

// Triangle 0—1—2—0: odd cycle → Unsat.
let mut tri = vec![vec![1, 2], vec![-1, -2], vec![2, 3], vec![-2, -3], vec![1, 3], vec![-1, -3]];
let refs: Vec<&[i32]> = tri.iter().collect();
let cnf = Cnf::from_lits(3, &refs).unwrap();
assert_eq!(Solver::new(cnf).solve(), SatResult::Unsat);

// Literal API and malformed-input rejection.
let a_pos = Lit::from_dimacs(1);
let a_neg = Lit::from_dimacs(-1);
assert_eq!(a_pos.neg(), a_neg);
assert!(Cnf::from_lits(2, &[&[0]]).is_none());
```

Run it with `cargo run --example sat_basic -p tpt-for-sat`.

## Cargo features

No optional features. Pure `std` (no external dependencies), single-file CDCL
implementation.

## Integration

The Boolean core that higher-level solvers build on; `tpt-for-smt-lite` and
`tpt-for-vcgen` provide the richer theories and obligations layered above
satisfiability.

## Status

This crate is in early (0.1.0) development. APIs may change between releases.

## License

Licensed under either of [MIT](https://opensource.org/licenses/MIT) or
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) at your option.
