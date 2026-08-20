# tpt-for-proof-ast

Proof AST: representation of terms, formulas, and proof trees.

A small, serialization-agnostic AST for first-order proofs. It models *terms*
(the objects proofs talk about), *formulas* (the statements), and *proofs*
(trees built from inference rules). It intentionally contains only the
structure — soundness checking is left to downstream tools.

## Features

- `Term` — variables, constants, applications, numeric literals, and lambdas,
  with `free_vars()`.
- `Formula` — predicates plus the usual connectives and `∀`/`∃` quantifiers,
  with `and`/`or`/`implies`/`forall` builders and `predicates()`.
- `Proof` — a conclusion plus the rule and sub-proofs that derive it, with
  `axiom` / `assumption` / `step` constructors, and `size` / `depth` /
  `conclusions()` traversals.
- `Rule` — `Axiom`, `Assumption`, `ModusPonens`, `ForallIntro`/`ForallElim`,
  `AndIntro`, and `Named` justifications.

## Example

```rust
use tpt_for_proof_ast::proof::Rule;
use tpt_for_proof_ast::{Term, Formula, Proof};

// A function application and its free variables.
let t = Term::app("add", vec![Term::var("x"), Term::Num(1)]);
assert_eq!(t.free_vars().len(), 1);

// A universally-quantified implication ∀n. even(n) → halvable(n).
let prop = Formula::pred("even", vec![Term::var("n")])
    .implies(Formula::pred("halvable", vec![Term::var("n")]));
assert_eq!(prop.clone().forall("n").predicates().len(), 2);

// A modus-ponens proof tree: (P→Q), P ⊢ Q
let imp = Formula::pred("P", vec![]).implies(Formula::pred("Q", vec![]));
let proof = Proof::step(
    Formula::pred("Q", vec![]),
    Rule::ModusPonens,
    vec![
        Proof::axiom("ax1", imp),
        Proof::axiom("ax2", Formula::pred("P", vec![])),
    ],
);
assert_eq!(proof.size(), 3);
assert_eq!(proof.depth(), 2);
assert_eq!(proof.conclusions().len(), 3);
```

## Cargo features

No optional features. Pure `std` (no external dependencies).

## Integration

The shared proof data model for the verification crates in the workspace; pairs
with `tpt-for-vcgen` (which produces the obligations) and
`tpt-for-model-check` / `tpt-for-symbolic-exec` (which explore them).

## Status

This crate is in early (0.1.0) development. APIs may change between releases.

## License

Licensed under either of [MIT](https://opensource.org/licenses/MIT) or
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) at your option.