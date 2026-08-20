//! Example: build a small first-order proof and inspect its structure.
//!
//! We use three layers of the AST: `Term`s (the objects proofs talk about),
//! `Formula`s (the statements), and `Proof` trees (how conclusions are derived
//! by inference rules). We also exercise the structure traversals (`free_vars`,
//! `predicates`, `size`, `depth`, `conclusions`).
use tpt_for_proof_ast::proof::Rule;
use tpt_for_proof_ast::{Formula, Proof, Term};

fn main() {
    // 1. Terms: free-variable collection over a function application `add(x, 1)`.
    let term = Term::app("add", vec![Term::var("x"), Term::Num(1)]);
    println!("term add(x, 1) free vars = {:?}", term.free_vars());

    // 2. Formulas: the universally-quantified implication `∀n. even(n) → halvable(n)`.
    let prop = Formula::pred("even", vec![Term::var("n")])
        .implies(Formula::pred("halvable", vec![Term::var("n")]));
    let quantified = prop.clone().forall("n");
    println!("formula predicates = {:?}", quantified.predicates());

    // 3. Modus ponens: from (P → Q) and P, derive Q.
    let imp = Formula::pred("P", vec![]).implies(Formula::pred("Q", vec![]));
    let mp = Proof::step(
        Formula::pred("Q", vec![]),
        Rule::ModusPonens,
        vec![
            Proof::axiom("ax-imp", imp),
            Proof::axiom("ax-p", Formula::pred("P", vec![])),
        ],
    );

    // 4. Conjunction introduction: from P and Q, derive P ∧ Q.
    let conj = Proof::step(
        Formula::pred("P", vec![]).and(Formula::pred("Q", vec![])),
        Rule::AndIntro,
        vec![
            Proof::axiom("ax-p2", Formula::pred("P", vec![])),
            Proof::axiom("ax-q2", Formula::pred("Q", vec![])),
        ],
    );

    // 5. Quantifier introduction: wrap a lemma as `∀n. even(n) → halvable(n)`.
    let quantified_proof = Proof::step(
        quantified.clone(),
        Rule::ForallIntro("n".to_string()),
        vec![Proof::axiom("even-impl-halvable", prop)],
    );

    println!(
        "modus-ponens proof: size={}, depth={}",
        mp.size(),
        mp.depth()
    );
    println!(
        "conjunction proof:   size={}, depth={}",
        conj.size(),
        conj.depth()
    );
    println!(
        "quantified proof:    size={}, depth={}",
        quantified_proof.size(),
        quantified_proof.depth()
    );

    // 6. Traverse every concluded formula in pre-order.
    print!("modus-ponens conclusions: ");
    for c in mp.conclusions() {
        print!("{c:?} ");
    }
    println!();
}
