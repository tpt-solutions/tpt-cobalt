# tpt-eng-reliability

Reliability and life analysis for the TPT ecosystem: fatigue, life prediction, failure rates, FMEA structures, and probabilistic-design helpers.

## Features

- Fatigue models: Basquin S–N relation and Miner's-rule cumulative damage.
- Life distributions: Weibull and exponential reliability, pdf, failure rate, mean life, and B-life.
- FMEA data structures for failure-mode documentation.
- Probabilistic design: reliability-index / normal-integration helpers.

## Installation

```toml
[dependencies]
tpt-eng-reliability = "0.1"
```

## Quick start

```rust
use tpt_eng_reliability::fatigue::miners_rule;
use tpt_eng_reliability::life::{weibull_b_life, weibull_reliability};

// A component at 1000 h with a Weibull shape beta=1.5, scale eta=5000 h.
let r = weibull_reliability(1000.0, 5000.0, 1.5).unwrap();
assert!(r > 0.9);

// B10 life: time by which 10% of the population has failed.
let b10 = weibull_b_life(10.0, 5000.0, 1.5).unwrap();

// Miner's-rule damage for two stress blocks.
let damage = miners_rule(&[(2_000.0, 10_000.0), (500.0, 5_000.0)]);
assert!(damage < 1.0);
```

## Crate modules

| Module | Purpose |
| --- | --- |
| `fatigue` | Basquin S–N and Miner's-rule cumulative damage. |
| `life` | Weibull / exponential distributions, B-life. |
| `fmea` | Failure-mode effects analysis structures. |
| `probabilistic` | Reliability-index / normal-integration helpers. |

## Related crates

- [tpt-eng-structural](../tpt-eng-structural/) — structural checks that consume life-prediction results.
- [tpt-eng-report](../tpt-eng-report/) — record reliability calculations in a report.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
