# tpt-for-redundancy

Redundancy / fault-tolerance primitives (voting, TMR, N-copy).

The building blocks of fault-tolerant systems: take several copies of a value or
computation and *vote* on the correct result so a single faulty replica cannot
corrupt the output. `no_std`, core-only.

## Features

- `majority_vote(votes)` — return the value agreed on by a strict majority
  (ties / empty → `None`).
- `Tmr<T>` — Triple Modular Redundancy: three replicas voted pairwise.
- `Redundant<T, N>` — `N` replicas with `majority()` voting and `Deref` to the
  backing array.
- `Parity` — a Hamming-style single-bit even-parity check over byte slices.

## Example

```rust
use tpt_for_redundancy::{majority_vote, Parity, Redundant, Tmr};

// Triple Modular Redundancy masks a single faulty replica.
let tmr = Tmr::new(50u8, 50, 70);
assert_eq!(tmr.vote(), Some(50));
assert_eq!(Tmr::new(1u8, 2, 3).vote(), None); // all differ

// N-copy majority voting.
let r = Redundant::<u8, 5>::new([1, 1, 1, 0, 0]);
assert_eq!(r.majority(), Some(1));
assert_eq!(r.len(), 5);

// A straight majority vote can tie on an even count.
assert_eq!(majority_vote(&[1u8, 1, 2, 2]), None);

// Single-bit parity detects a flipped bit.
let p = Parity::compute(&[0b1010_1010]);
let mut c = [0b1010_1010];
c[0] ^= 0b0000_1000;
assert!(!p.matches(&Parity::compute(&c)));
```

## Cargo features

The crate is `#![no_std]` and uses only `core`. The `std` and `alloc` features
are declared for downstream convenience.

## Integration

A practical component used alongside the analysis crates
(`tpt-for-model-check`, `tpt-for-deterministic-sim`) when modelling and
validating fault-tolerant behaviour.

## Status

This crate is in early (0.1.0) development. APIs may change between releases.

## License

Licensed under either of [MIT](https://opensource.org/licenses/MIT) or
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) at your option.