# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- `phantom` — variance-controlled markers `Invariant` / `Covariant` /
  `Contravariant` (more precise than `PhantomData<T>`) with constructors
  `invariant` / `covariant` / `contravariant`, plus a `Sealed` trait for the
  sealed-trait pattern.
- `ghost` — `State` trait, `Token<S>` typestate tokens, `Stateful<S, T>`
  state-annotated values, and `Ghost<P, T>` predicate-carrying ghost values.
- `newtype` — the `Newtype` conversion trait and the `define_newtype!` macro
  generating transparent newtypes with `From`/`Into` glue and derived traits.
- `bounded` — `Bounded<T>` runtime range checks and `Checked<B>` type-level
  named bounds via the `Bound` trait.
- `safe_cast` — `SafeCast` (infallible widening) and `TrySafeCast` (fallible
  narrowing / cross-kind) traits, the `CastError` type, and the `safe_cast` /
  `try_safe_cast` free functions.
- `no_std`, core-only, zero-dependency implementation; `std` and `alloc`
  features declared for downstream convenience.
