//! Example: generate and discharge verification conditions for tiny programs.
//!
//! We model small imperative programs in the weakest-precondition (WP) calculus
//! and ask vcgen whether each one establishes its postcondition. The interesting
//! case is the FALSIFIED program: we print the generated verification condition
//! so you can see exactly what the solver was asked to prove (and failed to).

use tpt_for_vcgen::{generate_vc, verify, wp, BExpr, Expr, Spec, Stmt, VcResult};

fn main() {
    // ===== Program A: VERIFIED =============================================
    //   x := 1;  x := x + 1;     post: x == 2
    let spec_a = Spec {
        pre: BExpr::bool(true),
        post: BExpr::eq(Expr::var("x"), Expr::const_(2)),
    };
    let body_a = vec![
        Stmt::assign("x", Expr::const_(1)),
        Stmt::assign("x", Expr::add(Expr::var("x"), Expr::const_(1))),
    ];
    // wp(body_a, x==2) reduces to (1 + 1) == 2 = true, so the VC `pre ∧ wp`
    // is trivially true and its negation is UNSAT -> Verified.
    println!(
        "Program A (x:=1; x:=x+1; post x==2): {:?}",
        verify(&spec_a, &body_a)
    );
    println!("  wp(body, post) = {:?}", wp(&body_a, &spec_a.post));

    // ===== Program B: FALSIFIED ============================================
    //   x := 1;     post: x == 2
    let spec_b = Spec {
        pre: BExpr::bool(true),
        post: BExpr::eq(Expr::var("x"), Expr::const_(2)),
    };
    let body_b = vec![Stmt::assign("x", Expr::const_(1))];
    let res_b = verify(&spec_b, &body_b);
    println!("Program B (x:=1; post x==2): {res_b:?}");
    if res_b == VcResult::Falsified {
        // Print the actual VC the solver was asked to prove (negated, so it is
        // SAT). This is the obligation that failed to hold.
        let vc = generate_vc(&spec_b, &body_b);
        println!("  generated VC (SMT-LIB2):\n{}", vc.to_smtlib2());
    }

    // ===== Program C: `assume` as a guard ==================================
    //   assume(x > 0);  x := x + 1;     post: x > 1
    // wp turns the assume into a hypothesis: (x>0) -> (x+1 > 1), which is true
    // for every x. The free variable `x` remains, so the built-in ground
    // evaluator reports Inconclusive — an external SMT solver is needed for the
    // definitive UNSAT verdict.
    let spec_c = Spec {
        pre: BExpr::bool(true),
        post: BExpr::gt(Expr::var("x"), Expr::const_(1)),
    };
    let body_c = vec![
        Stmt::assume(BExpr::gt(Expr::var("x"), Expr::const_(0))),
        Stmt::assign("x", Expr::add(Expr::var("x"), Expr::const_(1))),
    ];
    println!(
        "Program C (assume x>0; x:=x+1; post x>1): {:?} (needs external solver)",
        verify(&spec_c, &body_c)
    );

    // ===== Direct WP demonstration =========================================
    // wp(x := x*2, x == 4) must substitute to (x*2 == 4).
    let wp_d = wp(
        &[Stmt::assign(
            "x",
            Expr::mul(Expr::var("x"), Expr::const_(2)),
        )],
        &BExpr::eq(Expr::var("x"), Expr::const_(4)),
    );
    println!("wp(x := x*2, x == 4) = {wp_d:?}");
}
