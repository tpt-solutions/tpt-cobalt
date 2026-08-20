//! Example: explicit-state safety checking of two transition systems.
//!
//! We model (1) a bounded counter that must never reach its error threshold, and
//! (2) a naive mutual-exclusion protocol for two processes. Each is checked for a
//! SAFE instance and an UNSAFE one; in the unsafe case we print the concrete
//! counterexample trace, the diagnostic that makes model checking useful.
use std::fmt::Write;

use tpt_for_model_check::{check_safety, reachable, Counterexample, Model, SafetyResult};

// ---- Model 1: a counter bounded by `cap`; `max` is the forbidden threshold. ----
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Counter {
    cap: u32,
    max: u32,
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Count(u32);
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CountAct {
    Inc,
}

impl Model for Counter {
    type State = Count;
    type Action = CountAct;
    fn initials(&self) -> Vec<Count> {
        vec![Count(0)]
    }
    fn actions(&self, s: &Count) -> Vec<CountAct> {
        if s.0 < self.cap {
            vec![CountAct::Inc]
        } else {
            vec![]
        }
    }
    fn step(&self, s: &Count, _: &CountAct) -> Count {
        Count(s.0 + 1)
    }
    fn is_error(&self, s: &Count) -> bool {
        s.0 >= self.max
    }
}

// ---- Model 2: two processes, each may enter/exit its critical section. ----
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ME {
    p0: bool,
    p1: bool,
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum MEAct {
    P0Enter,
    P0Exit,
    P1Enter,
    P1Exit,
}

impl Model for ME {
    type State = ME;
    type Action = MEAct;
    fn initials(&self) -> Vec<ME> {
        vec![ME {
            p0: false,
            p1: false,
        }]
    }
    fn actions(&self, s: &ME) -> Vec<MEAct> {
        let mut v = Vec::new();
        if !s.p0 {
            v.push(MEAct::P0Enter);
        } else {
            v.push(MEAct::P0Exit);
        }
        if !s.p1 {
            v.push(MEAct::P1Enter);
        } else {
            v.push(MEAct::P1Exit);
        }
        v
    }
    fn step(&self, s: &ME, a: &MEAct) -> ME {
        match a {
            MEAct::P0Enter => ME { p0: true, p1: s.p1 },
            MEAct::P0Exit => ME {
                p0: false,
                p1: s.p1,
            },
            MEAct::P1Enter => ME { p0: s.p0, p1: true },
            MEAct::P1Exit => ME {
                p0: s.p0,
                p1: false,
            },
        }
    }
    fn is_error(&self, s: &ME) -> bool {
        s.p0 && s.p1
    }
}

/// Pretty-print a counterexample: `s0 --Inc--> s1 --Inc--> ... --err--> sN`.
fn show_ce(ce: &Counterexample<Count, CountAct>) {
    let mut line = String::new();
    for (i, (state, action)) in ce.steps.iter().enumerate() {
        if i > 0 {
            match action {
                Some(CountAct::Inc) => {
                    let _ = write!(line, " --Inc--> ");
                }
                None => {
                    let _ = write!(line, " --err--> ");
                }
            }
        }
        let _ = write!(line, "{}", state.0);
    }
    println!("  counterexample trace: {line}");
}

fn main() {
    // Counter, cap=2: it can only ever reach {0,1,2}; error (>=3) is unreachable.
    let safe = Counter { cap: 2, max: 3 };
    match check_safety(&safe) {
        SafetyResult::Safe => {
            println!(
                "counter (cap=2): SAFE — {} reachable states",
                reachable(&safe).len()
            )
        }
        SafetyResult::Violated(_) => println!("counter (cap=2): unexpectedly UNSAFE"),
    }

    // Counter, cap=3: from 2 it can Inc to 3, the error state → counterexample.
    let unsafe_m = Counter { cap: 3, max: 3 };
    match check_safety(&unsafe_m) {
        SafetyResult::Safe => println!("counter (cap=3): unexpectedly SAFE"),
        SafetyResult::Violated(ce) => {
            println!("counter (cap=3): UNSAFE");
            show_ce(&ce);
        }
    }

    // Mutual exclusion: with no protocol both processes can enter at once.
    let me = ME {
        p0: false,
        p1: false,
    };
    match check_safety(&me) {
        SafetyResult::Safe => println!("mutex: unexpectedly SAFE"),
        SafetyResult::Violated(ce) => {
            let last = &ce.steps.last().unwrap().0;
            println!(
                "mutex: UNSAFE — both processes in critical section (p0={}, p1={})",
                last.p0, last.p1
            );
        }
    }
}
