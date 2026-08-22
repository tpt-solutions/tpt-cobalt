# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-22

### Added
- Dynamic Value model: Nil, Bool, Num, Str, List, Dict, and a native first-class Tensor variant.
- Truthiness trait: empty containers falsy; tensors truthy iff every element is nonzero.
- Operators alue_add, alue_sub, alue_mul, alue_div, alue_eq with numeric promotion (scalar broadcast onto tensors, both operand orders) and LangError diagnostics including division-by-zero detection.
- Environment: lexically scoped variable bindings with parent-chain lookup and assignment.
- Two runnable examples (alues_and_truthiness, scoped_env) and a comprehensive README.

### Fixed
- Crate previously failed to compile: lib.rs declared mod env / mod ops / mod value but the module files were missing. All three are now implemented.

### Notes
- Deliberate divergence from the spec sketch: List/Dict use Arc<Mutex<..>> instead of a tracing GC; cycles leak until a GC lands.

[0.1.0]: https://github.com/tpt-solutions/tpt-crucible/releases/tag/tpt-lang-v0.1.0
