# tpt-for-assert-const

Compile-time assertions (`const_assert` / `const_assert_eq` / type-level checks).

Catch invariant violations *while compiling* instead of at runtime. Everything is
`no_std`.

## Features

- `const_assert!(cond)` — fail the build if `cond` is false at compile time.
- `const_assert_eq!(a, b)` / `const_assert_ne!(a, b)` — compile-time equality
  and inequality checks.
- `Const<bool>` — a compile-time boolean carried at the type level.
- `IsTrue` — a trait implemented only for `Const<true>`, so a bound
  `Const<B>: IsTrue` forces `B` to be `true` at compile time.
- `Same` — a trait with `Output = Self` for every `T`, so a bound
  `T: Same<Output = U>` requires `T` and `U` to be the same type.

## Example

```rust
use tpt_for_assert_const::{const_assert, const_assert_eq, const_assert_ne, Const, IsTrue, Same};

// A ring buffer whose masking index math needs a power-of-two capacity:
const CAPACITY: usize = 64;
const_assert!(CAPACITY.is_power_of_two());
const_assert_eq!(CAPACITY & (CAPACITY - 1), 0);

// A wire-format header whose exact byte size must never change by accident:
#[repr(C)]
struct Header { version: u8, flags: u8, length: u16, tag: u32 }
const_assert_eq!(core::mem::size_of::<Header>(), 8);
const_assert_ne!(1u8, 2u8);

// `IsTrue` forces a compile-time boolean; `Same` forces two types to be equal:
fn requires_tiny_index<B: IsTrue>() {}
requires_tiny_index::<Const<true>>();
fn identity_only<T, U>()
where
    T: Same<Output = U>,
{
}
identity_only::<u32, u32>();
```

Run it with `cargo run --example assert_const_basic -p tpt-for-assert-const`.

## Cargo features

The crate is `#![no_std]` and uses only `core`. The `std` and `alloc` features
are declared for downstream convenience; the crate has no `std`-only code paths.

## Integration

Used wherever a compile-time invariant must be guaranteed before codegen — for
example to guard generic implementations in the rest of the `tpt-formal`
workspace.

## Status

This crate is in early (0.1.0) development. APIs may change between releases.

## License

Licensed under either of [MIT](https://opensource.org/licenses/MIT) or
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) at your option.
