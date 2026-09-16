# tpt-approx

Clean-room floating-point approximate-comparison macros for the
`tpt-cobalt` workspace: `relative_eq!` / `abs_diff_eq!` predicates plus
`assert_*` forms with `epsilon = ...` / `max_relative = ...` named
arguments.

Floating-point results should almost never be compared with `==`: rounding
makes exact equality wrong for transitive math (`0.1 + 0.2 != 0.3`) and for
any reordered but mathematically identical expression. This crate gives the
workspace one small, dependency-free vocabulary for "equal up to rounding".

## Features

- **Two predicates** — `relative_eq!(a, b)` scales the tolerance with the
  magnitude of the operands; `abs_diff_eq!(a, b)` uses an absolute
  tolerance. Both return `bool` and compose in conditions, not just
  assertions.
- **Assert forms** — `assert_relative_eq!` / `assert_abs_diff_eq!` panic
  with a message showing both operands on failure, so test failures are
  readable.
- **Named tolerances** — `epsilon = 1e-12`, `max_relative = 1e-9` style
  arguments, with sensible defaults (machine epsilon and a small multiple).
- **Macro-only, zero dependencies** — nothing to pull into a dependency
  tree; the crate compiles nothing but macro definitions.
- **Clean-room** — implemented from the mathematics of floating-point
  comparison, not copied from any existing crate (no code, doc text, or
  API surface was transcribed from third-party sources).

## Usage

```rust
use tpt_approx::{assert_relative_eq, assert_abs_diff_eq, relative_eq};

let a: f64 = 0.1 + 0.2;
assert!(relative_eq!(a, 0.3));                       // default tolerances
assert_relative_eq!(a, 0.3, epsilon = 1e-12);
assert_abs_diff_eq!(3.0_f64.sqrt(), 1.7320508075688772, epsilon = 1e-12);
```

## Testing

```sh
cargo test -p tpt-approx
```

## Status

Stable at 0.1.x. Part of the first-party `tpt-cobalt` workspace; consumed
by the numerical test suites of the scientific crates.
