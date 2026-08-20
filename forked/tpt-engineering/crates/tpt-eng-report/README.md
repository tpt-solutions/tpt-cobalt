# tpt-eng-report

Calculation-report data model and exporters for the TPT ecosystem. Provides a serializable `Report` model plus Markdown, HTML, and JSON (calculation-trail) exporters, with validation helpers that annotate results with a pass / warn / fail status.

## Features

- Serializable `Report` model: assumptions, results, tables, headings, paragraphs.
- Markdown, HTML, and JSON exporters (to string or file).
- Validation helpers (`validate_min` / `validate_max` / `validate_range`) that assign a `ValidationStatus`.
- JSON calculation trail that round-trips through `serde`.

## Installation

```toml
[dependencies]
tpt-eng-report = "0.1"
```

## Quick start

```rust
use tpt_eng_report::{
    to_markdown, NamedValue, Report, ResultEntry, ValidationStatus,
};

let report = Report::new("Beam check")
    .with_summary("Simply supported beam (load <= 10 kN).")
    .assumptions(vec![NamedValue::new("Length", 5.0).with_unit("m")])
    .results(vec![ResultEntry::with_limits(
        "Max moment",
        12.5,
        Some("kNm".to_string()),
        Some(0.0),
        Some(15.0),
    )]);

let md = to_markdown(&report);
assert!(md.contains("# Beam check"));
assert_eq!(report.overall_status(), ValidationStatus::Pass);
```

## Crate modules

| Module | Purpose |
| --- | --- |
| `model` | `Report`, `Section`, `Table`, `NamedValue`, `ResultEntry`, `ValidationStatus`. |
| `markdown` | Markdown renderer. |
| `html` | HTML renderer (with escaping). |
| `json` | JSON calculation-trail renderer. |
| `validate` | Min / max / range validation helpers. |
| `error` | Crate `Error` / `Result` types. |

## Related crates

- [tpt-eng-plot](../tpt-eng-plot/) — render report results as charts.
- [tpt-eng-cli](../tpt-eng-cli/) — CLI front-end that writes reports from calculations.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
