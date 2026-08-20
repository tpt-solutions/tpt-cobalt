# tpt-for-det-proptest

Deterministic, seedable property-based testing.

Unlike fuzzing harnesses that draw entropy from the OS, this crate drives case
generation from an explicit 64-bit seed via a fixed PRNG. The same seed always
produces the same sequence of cases, so a failing property can be reproduced
exactly — crucial for CI and regression tracking.

## Features

- `DeterministicRng` — a fast xorshift64* generator; `new`, `next_u64`,
  `gen_range`, `gen_bool`, `fill_bytes` (a zero seed is remapped to a non-zero
  state).
- `Gen<T>` — a strategy producing `T` from a deterministic RNG; blanket impls
  for `()` over `u64` / `u32` / `i64` / `bool`.
- `check_prop` — run a property over `N` generated cases, returning
  `Outcome::Passed` or a `CounterExample` (input, seed, case index).
- `assert_prop` — like `check_prop` but panics with the counterexample on
  failure, for use inside `#[test]`s.
- `Outcome<T>` / `CounterExample<T>` — the result and replayable failure types.

## Example

```rust
use tpt_for_det_proptest::{check_prop, CounterExample, DeterministicRng, Gen, Outcome};

// A passing property over a determined run (gen, cases, seed):
let ok = check_prop(&(), 500, 1, |x: u64| x.clamp(10, 100) >= 10);
assert!(matches!(ok, Outcome::Passed { .. }));

// A failing property returns a concrete, replayable counterexample:
let bad = check_prop(&(), 500, 7, |x: u64| x % 2 == 0);
let first: CounterExample<u64> = match &bad {
    Outcome::Failed(ce) => ce.clone(),
    Outcome::Passed { .. } => unreachable!(),
};
// Re-running with the same seed reproduces the identical counterexample:
let replay = check_prop(&(), 500, 7, |x: u64| x % 2 == 0);
let replay_ce: CounterExample<u64> = match replay {
    Outcome::Failed(ce) => ce,
    Outcome::Passed { .. } => unreachable!(),
};
assert_eq!(first, replay_ce);

// Custom strategies implement `Gen<T>` over a `DeterministicRng`:
struct PairGen;
impl Gen<(u64, u64)> for PairGen {
    fn generate(&self, rng: &mut DeterministicRng) -> (u64, u64) {
        (rng.next_u64(), rng.next_u64())
    }
}
```

Run it with `cargo run --example det_proptest_basic -p tpt-for-det-proptest`.

## Cargo features

No optional features. `core`-only (`no_std`), no dependencies.

## Integration

The deterministic-engine counterpart to `tpt-for-deterministic-sim`; both share
the same xorshift64* PRNG design so simulation and testing stay reproducible.

## Status

This crate is in early (0.1.0) development. APIs may change between releases.

## License

Licensed under either of [MIT](https://opensource.org/licenses/MIT) or
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) at your option.
