# tpt-for-contract

Design-by-contract: preconditions, postconditions, invariants, loop invariants.

Consolidates the four facets of one contract concern into a single coherent
surface. All checks are `no_std` and panic with the offending condition's source
text, file, and line. A debug-only variant is provided for checks that should be
skipped in release builds.

## Features

- `requires!(cond)` / `requires!(cond, msg)` — precondition, checked at function
  entry.
- `ensures!(cond)` / `ensures!(cond, msg)` — postcondition, checked at return.
- `invariant!(cond)` — a value's invariant, checked in place.
- `loop_invariant!(cond)` — verified on every loop iteration.
- `debug_requires!` / `debug_ensures!` / `debug_invariant!` /
  `debug_loop_invariant!` — compile to nothing under
  `cfg(not(debug_assertions))`.
- `Invariant` trait + `check_invariant!` macro — declare and assert a value's
  own invariant (`Invariant::check`).
- `ContractError` / `report` — the structured failure type (kind/expr/file/line)
  and the reporter hook called by every macro.

## Example

```rust
use tpt_for_contract::{check_invariant, ensures, invariant, loop_invariant, requires, Invariant};

struct Account { balance: i64 }
impl Invariant for Account {
    fn check(&self) -> bool { self.balance >= 0 }
}
impl Account {
    fn withdraw(&mut self, amt: i64) -> i64 {
        requires!(amt >= 0, "withdraw amount must be non-negative");
        requires!(amt <= self.balance, "cannot overdraw");
        self.balance -= amt;
        ensures!(self.balance >= 0);
        self.balance
    }
}

fn total(n: i64) -> i64 {
    requires!(n >= 0);
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
    let mut acc = Account { balance: 100 };
    check_invariant!(acc);
    acc.withdraw(30);
    check_invariant!(acc);
    // A *violated* contract panics with the condition's source text, file, line:
    //   acc.withdraw(20); // -> "precondition violated ... amt <= self.balance"
}
```

Run it with `cargo run --example contract_basic -p tpt-for-contract`.

## Cargo features

The crate is `#![no_std]` and uses only `core`. The `std` and `alloc` features
are declared for downstream convenience.

## Integration

This is the contract surface consumed by `tpt-for-refinement`,
`tpt-for-verified-algorithms`, and `tpt-for-verified-ode`, which guard their
logic with these macros.

## Status

This crate is in early (0.1.0) development. APIs may change between releases.

## License

Licensed under either of [MIT](https://opensource.org/licenses/MIT) or
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) at your option.
