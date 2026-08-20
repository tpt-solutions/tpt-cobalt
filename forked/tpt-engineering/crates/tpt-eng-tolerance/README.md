# tpt-eng-tolerance

Tolerance analysis for dimension and parameter stack-up in the TPT ecosystem: worst-case, root-sum-square (RSS), and Monte-Carlo evaluation, together with sensitivity and contributor-ranking helpers.

## Features

- Worst-case and RSS stack-up intervals for bilateral / asymmetric dimensions.
- Monte-Carlo stack-up evaluation with an optional yield estimate.
- 1-D stack-up model (`Stackup` / `StackupMember`) with worst-case, RSS, and Monte-Carlo (`MonteCarloResult`) methods.
- Sensitivity (Pearson correlation) and RSS contributor ranking.

## Installation

```toml
[dependencies]
tpt-eng-tolerance = "0.1"
```

## Quick start

```rust
use tpt_eng_tolerance::{monte_carlo, rss, worst_case, DimTol};

let dims = vec![
    DimTol::new("a", 10.0, 0.1),
    DimTol::new("b", 20.0, 0.2),
    DimTol::new("c", 5.0, 0.1),
];

let (wc_lo, wc_hi) = worst_case(&dims);
let (rss_lo, rss_hi) = rss(&dims);
assert!(rss_lo > wc_lo && rss_hi < wc_hi);

let mut rng = rand::thread_rng();
let result = monte_carlo(&dims, 50_000, None, &mut rng);
assert!((result.mean - 35.0).abs() < 0.05);
```

## Crate modules

| Module | Purpose |
| --- | --- |
| (root) | Stack-up functions and 1-D `Stackup` / `StackupMember` / `MonteCarloResult` model. |

## Related crates

- [tpt-eng-gdt](../tpt-eng-gdt/) — re-exports the 1-D stack-up types and adds GD&T zone / datum conformance.
- [tpt-eng-report](../tpt-eng-report/) — record tolerance calculations in a report.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
