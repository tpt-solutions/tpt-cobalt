# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-10

### Added
- Initial release of the TPT-UIR textual format.
- Hand-written lexer and recursive-descent parser (no `nom`/`pest`).
- `to_text` / `parse_text` for serialization and parsing.
- `Pretty` wrapper implementing `Display` for pretty-printing.
- Round-trip support for nested regions, `Quantization` attributes, and
  `dialect_version` attributes.
