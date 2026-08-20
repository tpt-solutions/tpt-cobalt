# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- `Lit` — a Boolean literal (`var << 1 | sign`) with `new`, `var`, `is_neg`,
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
