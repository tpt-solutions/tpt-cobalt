# tpt-for-refinement

Refinement types: values carrying a compile-time refinement predicate.

A `Refined<T, P>` is a value of type `T` known to satisfy the predicate `P`.
Construction is the *only* way to create one, and it refuses to wrap a value
that fails `P::check`, so any `Refined` in scope is a proof that its predicate
holds.

## Features

- `Refined<T, P>` — a value of type `T` refined by predicate `P`.
- `Predicate<T>` — the refinement predicate trait (`check(&T) -> bool`).
- Blanket impl: any type implementing `tpt-for-contract`'s `Invariant` is a valid
  refinement predicate, so contract-checked types compose directly.
- `new` / `new_unchecked` (trusted) / `get` / `into_inner` / `map` — ergonomic,
  proof-preserving accessors.

## Example

```rust
use tpt_for_contract::Invariant;
use tpt_for_refinement::{Predicate, Refined};

// A value-level predicate.
struct Positive;
impl Predicate<i64> for Positive {
    fn check(value: &i64) -> bool { *value > 0 }
}

let r = Refined::<i64, Positive>::new(5).unwrap();
assert_eq!(*r.get(), 5);
assert!(Refined::<i64, Positive>::new(-1).is_err()); // rejected with RefineError

// An `Invariant` type composes directly as a refinement predicate (blanket impl).
struct NonEmpty { data: Vec<u8> }
impl Invariant for NonEmpty {
    fn check(&self) -> bool { !self.data.is_empty() }
}
let buf = Refined::<NonEmpty, NonEmpty>::new(NonEmpty { data: vec![1] }).unwrap();
assert_eq!(buf.get().data.len(), 1);
```

Any type that implements `tpt_for_contract::Invariant` can be used as `P`
without writing a separate `Predicate` impl.

## Cargo features

The crate is `#![no_std]` and uses only `core`. The `std` and `alloc` features
are declared for downstream convenience.

## Integration

Depends on `tpt-for-contract` (reuses its `Invariant` trait) and complements
`tpt-for-witness`, offering value-level refinement alongside witness-style
proofs.

## Status

This crate is in early (0.1.0) development. APIs may change between releases.

## License

Licensed under either of [MIT](https://opensource.org/licenses/MIT) or
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) at your option.