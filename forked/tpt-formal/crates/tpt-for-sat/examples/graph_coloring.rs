//! Example: use the built-in CDCL solver to 2-color graphs.
//!
//! We encode "can this graph be colored with 2 colors so that adjacent
//! vertices differ?" Each vertex `v` is a Boolean variable `x_v` (color 0/1).
//! An edge (u, v) becomes two clauses forcing the endpoints to differ:
//! `(x_u ∨ x_v) ∧ (¬x_u ∨ ¬x_v)`. If the solver returns `Sat`, the model tells
//! us a valid coloring; if `Unsat`, no 2-coloring exists (the graph is not
//! bipartite). Both outcomes are demonstrated below, along with the literal API
//! and `from_lits`'s rejection of malformed input.

use tpt_for_sat::{Cnf, Lit, SatResult, Solver};

/// Add the "u and v must differ" edge constraint to `clauses`.
/// Each `i32` is the 1-based DIMACS literal for a vertex (negative = `¬x_v`).
fn differ(clauses: &mut Vec<Vec<i32>>, u: i32, v: i32) {
    clauses.push(vec![u, v]);
    clauses.push(vec![-u, -v]);
}

fn main() {
    // ---- Scenario 1: a 3-vertex path  0—1—2  (bipartite → SAT) ----
    // Edges: (0,1) and (1,2). A valid coloring is x0=true, x1=false, x2=true.
    let mut path: Vec<Vec<i32>> = Vec::new();
    differ(&mut path, 1, 2);
    differ(&mut path, 2, 3);
    let refs: Vec<&[i32]> = path.iter().map(|v| v.as_slice()).collect();
    let cnf = Cnf::from_lits(3, &refs).expect("well-formed CNF");
    let mut solver = Solver::new(cnf);
    let result = solver.solve();
    println!("path graph 2-colorable: {result:?}");
    assert_eq!(result, SatResult::Sat);

    // Read the satisfying coloring back from the model (vertex -> color 0/1).
    let c0 = u8::from(solver.value(0).unwrap());
    let c1 = u8::from(solver.value(1).unwrap());
    let c2 = u8::from(solver.value(2).unwrap());
    println!("  coloring: vertex0={c0} vertex1={c1} vertex2={c2}");
    // Adjacent vertices must differ in any valid coloring.
    assert!(
        c0 != c1 && c1 != c2,
        "adjacent vertices differ in the model"
    );
    // The full model is also available indexed by variable.
    println!("  full model: {:?}", solver.model());

    // ---- Scenario 2: a triangle  0—1—2—0  (odd cycle → UNSAT) ----
    // Adding edge (0,2) to the path makes a 3-cycle, which is not 2-colorable.
    let mut triangle: Vec<Vec<i32>> = Vec::new();
    differ(&mut triangle, 1, 2);
    differ(&mut triangle, 2, 3);
    differ(&mut triangle, 1, 3);
    let refs: Vec<&[i32]> = triangle.iter().map(|v| v.as_slice()).collect();
    let cnf = Cnf::from_lits(3, &refs).expect("well-formed CNF");
    let mut solver = Solver::new(cnf);
    let result = solver.solve();
    println!("triangle graph 2-colorable: {result:?}");
    assert_eq!(result, SatResult::Unsat);

    // ---- Literal API: DIMACS encoding and negation ----
    // `Lit::from_dimacs` turns a signed integer literal into a `Lit`.
    let a_pos = Lit::from_dimacs(1);
    let a_neg = Lit::from_dimacs(-1);
    println!(
        "lit +1 -> var {}, negated var {} (is_neg={})",
        a_pos.var(),
        a_neg.var(),
        a_neg.is_neg()
    );
    // `neg()` flips the sign; `from_lits` rejects malformed input (returns None).
    assert_eq!(a_pos.neg(), a_neg);
    println!(
        "malformed '0' literal rejected: {}",
        Cnf::from_lits(2, &[&[0]]).is_none()
    );
    println!(
        "out-of-range var rejected:     {}",
        Cnf::from_lits(2, &[&[1, -3]]).is_none()
    );
}
