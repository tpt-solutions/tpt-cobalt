# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-15

### Added
- Pure analysis layer: `analyze` (checker + syntax diagnostics),
  `hover_at` (values with tensor shapes, function signatures, module
  listings, dotted-path resolution), `completions_at` (state-aware
  globals/keywords/natives/module members).
- tower-lsp stdio server (`tpt-lsp` binary) with full-document sync,
  hover, and completion handlers.
- README with architecture, testing, and documented limitations.

[0.1.0]: https://github.com/tpt-solutions/tpt-cobalt/releases/tag/tpt-lsp-v0.1.0
