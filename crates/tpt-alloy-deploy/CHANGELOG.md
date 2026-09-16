# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-14

### Added
- RP2040 UF2 block generation from raw flash images (`to_uf2`), spec-verified
  byte layout with known-vector tests.
- ESP32 ROM-UART SLIP framing, command packets (Sync/FlashBegin/FlashData/
  FlashEnd) with XOR-0xEF checksums, and `EspFlasher` over a `Transport`
  trait; in-memory mock transport drives the tests.
- Fleet OTA: sha256-digested `NodeArtifact`s, `UpdateManifest` validation
  (duplicates, foreign releases, digest tampering), staged two-phase
  `Rollout` state machine with abort-on-failure and skip modes, and
  image-stripping JSON export.

### Fixed
- Response parsing reads the data section after the full 9-byte header
  (direction 1 + command 2 + size 2 + checksum 4); an earlier off-by-one
  happened to pass the first status checks.
- `Rollout::begin` now actually stages nodes in ascending node-id order.

[0.1.0]: https://github.com/tpt-solutions/tpt-cobalt/releases/tag/tpt-alloy-deploy-v0.1.0
