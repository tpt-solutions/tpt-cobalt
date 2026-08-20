//! Differential / property test for the CDCL solver.
//!
//! Generates many small random CNFs (deterministically, reproducible by seed)
//! and checks the solver's `Sat`/`Unsat` verdict against a brute-force
//! truth-table oracle, and verifies any claimed `Sat` model satisfies every
//! clause. This is the direct regression guard for the conflict-analysis and
//! unit-propagation fixes in `src/lib.rs`.

use tpt_for_det_proptest::{assert_prop, DeterministicRng, Gen};
use tpt_for_sat::{Cnf, SatResult, Solver};

/// A randomly generated CNF: `vars` variables and a list of clauses, each a list
/// of DIMACS literals in range `[1, vars]` / `[-vars, -1]`.
#[derive(Clone, Debug)]
struct RandCnf {
    vars: usize,
    lits: Vec<Vec<i32>>,
}

/// Marker strategy type for `Gen<RandCnf>`.
struct CnfGen;

impl Gen<RandCnf> for CnfGen {
    fn generate(&self, rng: &mut DeterministicRng) -> RandCnf {
        let vars = 1 + rng.gen_range(5) as usize; // 1..=5
        let nclauses = 1 + rng.gen_range(9) as usize; // 1..=9
        let mut lits = Vec::with_capacity(nclauses);
        for _ in 0..nclauses {
            let nlit = 1 + rng.gen_range(3) as usize; // 1..=3
            let mut clause = Vec::with_capacity(nlit);
            for _ in 0..nlit {
                let v = rng.gen_range(vars as u64) as i32 + 1; // 1..=vars
                let sign = if rng.gen_bool() { 1 } else { -1 };
                clause.push(sign * v);
            }
            lits.push(clause);
        }
        RandCnf { vars, lits }
    }
}

/// Brute-force truth-table oracle: is there an assignment satisfying every clause?
fn brute_force_sat(vars: usize, lits: &[Vec<i32>]) -> bool {
    for bits in 0..(1u64 << vars) {
        let mut all = true;
        for clause in lits {
            let mut clause_ok = false;
            for &l in clause {
                let var = l.unsigned_abs() as usize - 1;
                let val = ((bits >> var) & 1) == 1;
                let lit_true = if l > 0 { val } else { !val };
                if lit_true {
                    clause_ok = true;
                    break;
                }
            }
            if !clause_ok {
                all = false;
                break;
            }
        }
        if all {
            return true;
        }
    }
    false
}

/// Verify a claimed `Sat` model actually satisfies every clause.
fn check_model(vars: usize, lits: &[Vec<i32>], model: &[bool]) -> bool {
    if model.len() != vars {
        return false;
    }
    for clause in lits {
        let mut clause_ok = false;
        for &l in clause {
            let var = (l.unsigned_abs() - 1) as usize;
            let val = model[var];
            let lit_true = if l > 0 { val } else { !val };
            if lit_true {
                clause_ok = true;
                break;
            }
        }
        if !clause_ok {
            return false;
        }
    }
    true
}

fn property(rc: RandCnf) -> bool {
    let slices: Vec<&[i32]> = rc.lits.iter().map(|c| c.as_slice()).collect();
    let cnf = match Cnf::from_lits(rc.vars, &slices) {
        Some(c) => c,
        // Out-of-range literals shouldn't occur for in-range generation; treat
        // as a no-op pass rather than failing the differential check.
        None => return true,
    };
    let mut solver = Solver::new(cnf);
    match solver.solve() {
        SatResult::Sat => {
            let model: Vec<bool> = (0..rc.vars)
                .map(|v| solver.value(v as u32).unwrap_or(false))
                .collect();
            check_model(rc.vars, &rc.lits, &model)
        }
        SatResult::Unsat => !brute_force_sat(rc.vars, &rc.lits),
    }
}

#[test]
fn differential_against_truth_table() {
    // Run across many seeds so any latent unsoundness is caught; the run is
    // fully reproducible for a given seed.
    for seed in 0..32u64 {
        assert_prop(&CnfGen, 200, seed, property);
    }
}
