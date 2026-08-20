# tpt-for-verified-algorithms

Verified algorithm implementations guarded by design-by-contract checks.

Each routine is wrapped in `tpt-for-contract` precondition / postcondition /
loop-invariant checks, so calling it with inputs that violate its contract
panics instead of producing a silently-wrong result. Checks are written against
the workspace MSRV (1.75), avoiding newer `std` helpers.

## Features

- `gcd(a, b)` — Euclid's GCD; panics if both inputs are zero.
- `clamp(x, lo, hi)` — saturating clamp into `[lo, hi]`.
- `binary_search(slice, target)` — requires a sorted slice, returns the index.
- `insertion_sort(input)` — returns a sorted permutation of the input.
- All functions document their contracts in their doc comments and enforce them
  with the `requires!` / `ensures!` / `invariant!` macros.

## Example

```rust
use std::panic;
use tpt_for_verified_algorithms::{binary_search, clamp, gcd, insertion_sort};

// Happy path: each routine returns a correctly-computed, contract-checked result.
assert_eq!(gcd(12, 18), 6);
assert_eq!(gcd(17, 0), 17);
assert_eq!(clamp(5, 0, 10), 5);
assert_eq!(clamp(-3, 0, 10), 0);
assert_eq!(clamp(99, 0, 10), 10);
assert_eq!(binary_search(&[1, 3, 5, 7, 9], 5), Some(2));
assert_eq!(binary_search(&[1, 3, 5, 7, 9], 4), None);
let sorted = insertion_sort(&[3, 1, 2, 5, 4]);
assert!(sorted.windows(2).all(|w| w[0] <= w[1]));

// Failure path: the contracts panic rather than be silently wrong.
// gcd(0, 0) violates "at least one argument non-zero".
assert!(panic::catch_unwind(|| gcd(0, 0)).is_err());
// An unsorted slice also trips binary_search's precondition.
assert!(panic::catch_unwind(|| binary_search(&[3, 1, 2], 2)).is_err());
```

See `examples/algorithms_demo.rs` (`cargo run --example verified_algorithms_basic -p
tpt-for-verified-algorithms`) for the printable version, including the panic
messages the contracts emit on bad input.

## Cargo features

No optional features. Depends on `tpt-for-contract`.

## Integration

A showcase consumer of `tpt-for-contract`: the same contract surface is used by
`tpt-for-verified-ode` for numerically-sensitive integrators.

## Status

This crate is in early (0.1.0) development. APIs may change between releases.

## License

Licensed under either of [MIT](https://opensource.org/licenses/MIT) or
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) at your option.