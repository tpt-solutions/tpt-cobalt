# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-15

### Added
- Criterion benchmarks: f64 matmul (64/128/256), Linear forward+backward
  through the tape, transformer-block forward, and the TPT-Script
  `train_step` interpreter path.
- `tpt-bench-report` wall-clock report generator (Markdown + JSON,
  deterministic LCG-seeded inputs, 10-run warmup rule).
- `benches/PYTORCH_PROTOCOL.md`: the fixed PyTorch comparison recipe and
  the by-design differences to quote.
- README with benchmark list, usage, and status.

[0.1.0]: https://github.com/tpt-solutions/tpt-cobalt/releases/tag/tpt-bench-v0.1.0
