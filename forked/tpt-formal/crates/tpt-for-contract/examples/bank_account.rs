//! Example: design-by-contract on a tiny bank account.
//!
//! The point of the crate: a contract breach is reported with the offending
//! condition's source text, file, and line — not just a bare panic. We show the
//! *happy path* (every contract holds) and the *failure path* (we catch the panic
//! and print the structured error) so you can see exactly what a violation looks
//! like. The two caught panics below are intentional demonstrations.
use std::panic;

use tpt_for_contract::{
    check_invariant, debug_requires, ensures, invariant, loop_invariant, requires, ContractError,
    Invariant,
};

/// A bank account whose balance must never go negative.
struct Account {
    balance: i64,
}

impl Invariant for Account {
    fn check(&self) -> bool {
        self.balance >= 0
    }
}

impl Account {
    fn deposit(&mut self, amt: i64) -> i64 {
        requires!(amt >= 0, "deposit amount must be non-negative");
        self.balance = self.balance.saturating_add(amt);
        ensures!(self.balance >= 0);
        self.balance
    }

    fn withdraw(&mut self, amt: i64) -> i64 {
        requires!(amt >= 0, "withdraw amount must be non-negative");
        requires!(amt <= self.balance, "cannot overdraw");
        self.balance -= amt;
        ensures!(self.balance >= 0);
        self.balance
    }
}

/// Sum the first `n` non-negative integers, keeping a running loop invariant.
fn total(n: i64) -> i64 {
    requires!(n >= 0);
    debug_requires!(n <= 1_000_000, "n must stay within demo limits");
    let mut sum = 0i64;
    let mut i = 0;
    while i < n {
        loop_invariant!(sum >= 0);
        loop_invariant!(sum == i * (i - 1) / 2);
        sum = sum.saturating_add(i);
        i += 1;
    }
    ensures!(sum == n * (n - 1) / 2);
    sum
}

fn main() {
    // --- happy path: every contract holds ---------------------------------
    let mut acc = Account { balance: 100 };
    check_invariant!(acc); // Account::check() => balance >= 0
    acc.deposit(50);
    acc.withdraw(30);
    check_invariant!(acc);
    println!("after deposit(50)/withdraw(30), balance = {}", acc.balance);
    invariant!(acc.balance == 120);
    println!("sum 0..10 = {}", total(10));

    // --- failure path 1: violated precondition -----------------------------
    // Caught so the demo keeps running and we can *print* the structured error.
    let result = panic::catch_unwind(|| {
        let mut a = Account { balance: 10 };
        a.withdraw(20); // overdraw -> requires!(amt <= self.balance) fails
    });
    if let Err(payload) = result {
        let msg = match payload.downcast_ref::<String>() {
            Some(s) => s.as_str(),
            None => "non-String panic payload",
        };
        println!("caught violation 1 (precondition): {msg}");
    }

    // --- failure path 2: violated struct invariant -------------------------
    let bad = Account { balance: -5 };
    let result = panic::catch_unwind(|| {
        check_invariant!(bad); // balance >= 0 is false
    });
    if let Err(payload) = result {
        let msg = match payload.downcast_ref::<String>() {
            Some(s) => s.as_str(),
            None => "non-String panic payload",
        };
        println!("caught violation 2 (invariant): {msg}");
    }

    // --- the structured error type itself ----------------------------------
    let err = ContractError {
        kind: "demo",
        expr: "balance >= 0",
        file: "examples/basic.rs",
        line: 1,
    };
    println!("ContractError Display: {err}");
}
