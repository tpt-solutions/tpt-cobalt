# Contributing to tpt-engineering

## Policy: reports only, no external code contributions

`tpt-engineering` is developed internally by TPT Solutions. To keep the
engineering primitives coherent and auditable, **we do not accept external
pull requests or code contributions.**

That said, we welcome two kinds of input from outside the core team:

- **Bug reports** — open a [GitHub Issue](https://github.com/tpt-solutions/tpt-engineering/issues)
  with a minimal reproduction, the affected crate(s), and the commit/tag or
  `main` revision you are on.
- **Feature requests** — open a [GitHub Issue](https://github.com/tpt-solutions/tpt-engineering/issues)
  describing the engineering use case, the expected API shape, and any
  reference (standard, textbook, or spec) the behaviour should follow.

Please do **not** open a PR proposing code changes. Issues are triaged by the
maintainers, who implement accepted changes internally.

## Reporting a vulnerability

**Do not open a public issue for security vulnerabilities.** See
[`SECURITY.md`](SECURITY.md) for the private reporting path and what to include
in a report.

## Local workflow (for maintainers)

```sh
cargo xtask check      # fmt --check + clippy -D warnings + cargo-deny
cargo xtask test       # full workspace test suite
cargo xtask doctest    # documentation tests
cargo xtask no-std-matrix   # bare-metal build of the no_std props crates
```

A root `justfile` mirrors these (`just check`, `just test`, `just ci`).
