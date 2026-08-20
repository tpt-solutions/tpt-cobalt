# tpt-for-model-check

Explicit-state model checking (clean-room implementation).

Answers a *safety* question — "is a bad state reachable from an initial
state?" — by exploring the reachable state graph. When the property is violated
it returns a concrete counterexample trace, the diagnostic that makes model
checking useful in practice.

## Features

- `Model` — describe a (finite, reachable) transition system: `initials`,
  `actions`, `step`, and `is_error`.
- `check_safety(model)` — BFS over the state graph; returns `Safe` or a
  `Violated` counterexample (shortest witness trace).
- `reachable(model)` — enumerate every reachable state (coverage / debugging).
- Clean-room implementation; `stateright` (MIT) is documented as a possible future
  backend behind a feature flag, mirroring `tpt-for-smt-lite`'s handling of
  `rsmt2`/`z3`.

## Example

```rust
use tpt_for_model_check::{check_safety, reachable, Model, SafetyResult};

// A counter bounded by `cap`; `max` is the (forbidden) error threshold.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Counter { cap: u32, max: u32 }
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Count(u32);
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CountAct { Inc }

impl Model for Counter {
    type State = Count;
    type Action = CountAct;
    fn initials(&self) -> Vec<Count> { vec![Count(0)] }
    fn actions(&self, s: &Count) -> Vec<CountAct> {
        if s.0 < self.cap { vec![CountAct::Inc] } else { vec![] }
    }
    fn step(&self, s: &Count, _: &CountAct) -> Count { Count(s.0 + 1) }
    fn is_error(&self, s: &Count) -> bool { s.0 >= self.max }
}

// Safe: cap=2 means the counter can only reach {0,1,2}; `max` (>=3) is unreachable.
let safe = Counter { cap: 2, max: 3 };
assert!(matches!(check_safety(&safe), SafetyResult::Safe));
println!("reachable states: {}", reachable(&safe).len());

// Unsafe: from 2 we can `Inc` to 3, which is the error state → counterexample.
let bad = Counter { cap: 3, max: 3 };
if let SafetyResult::Violated(ce) = check_safety(&bad) {
    println!("counterexample length: {}", ce.steps.len());
}
```

Run it with `cargo run --example model_check_basic -p tpt-for-model-check`.

## Cargo features

No optional features. Pure `std` (no external dependencies).

## Integration

The state-exploration engine for the workspace's verification story; pairs with
`tpt-for-proof-ast` (obligations) and `tpt-for-trace-macros` (trace capture).

## Status

This crate is in early (0.1.0) development. APIs may change between releases.

## License

Licensed under either of [MIT](https://opensource.org/licenses/MIT) or
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) at your option.