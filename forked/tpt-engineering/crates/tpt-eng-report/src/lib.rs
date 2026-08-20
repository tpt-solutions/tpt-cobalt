//! Calculation-report data model and exporters for the TPT ecosystem.
//!
//! Provides a serializable [`Report`] data model plus Markdown, HTML, and JSON (calculation
//! trail) exporters, and validation helpers used to annotate results with a pass/warn/fail status.

pub mod error;
pub mod html;
pub mod json;
pub mod markdown;
pub mod model;
pub mod validate;

pub use error::{Error, Result};
pub use model::{NamedValue, Report, ResultEntry, Section, Table, ValidationStatus};
pub use validate::{validate_max, validate_min, validate_range};

use std::path::Path;

/// Render the report to a Markdown string.
pub fn to_markdown(report: &Report) -> String {
    markdown::render(report)
}

/// Render the report to an HTML string.
pub fn to_html(report: &Report) -> String {
    html::render(report)
}

/// Render the report to a JSON calculation-trail string.
///
/// # Errors
///
/// Returns [`crate::Error`] if the report cannot be serialized to JSON.
pub fn to_json(report: &Report) -> Result<String> {
    json::render(report)
}

/// Write the report as Markdown to a file.
///
/// # Errors
///
/// Returns [`crate::Error`] if the file cannot be created or written to.
pub fn write_markdown<P: AsRef<Path>>(report: &Report, path: P) -> Result<()> {
    std::fs::write(path, markdown::render(report))?;
    Ok(())
}

/// Write the report as HTML to a file.
///
/// # Errors
///
/// Returns [`crate::Error`] if the file cannot be created or written to.
pub fn write_html<P: AsRef<Path>>(report: &Report, path: P) -> Result<()> {
    std::fs::write(path, html::render(report))?;
    Ok(())
}

/// Write the report as a JSON calculation trail to a file.
///
/// # Errors
///
/// Returns [`crate::Error`] if the report cannot be serialized or the file
/// cannot be created or written to.
pub fn write_json<P: AsRef<Path>>(report: &Report, path: P) -> Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    std::fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::*;

    fn sample_report() -> Report {
        Report::new("Beam check")
            .with_author("TPT")
            .with_date("2026-08-14")
            .with_summary("Quick sanity check of a simply supported beam (load <= 10 kN).")
            .assumptions(vec![
                NamedValue::new("Length", 5.0).with_unit("m"),
                NamedValue::new("Load", 10.0).with_unit("kN"),
            ])
            .results(vec![
                ResultEntry::with_limits(
                    "Max moment",
                    12.5,
                    Some("kNm".to_string()),
                    Some(0.0),
                    Some(15.0),
                ),
                ResultEntry::with_limits(
                    "Max deflection",
                    18.0,
                    Some("mm".to_string()),
                    Some(0.0),
                    Some(20.0),
                ),
            ])
            .heading("Notes")
            .paragraph("All checks within permissible limits.")
    }

    #[test]
    fn test_markdown_contains_sections() {
        let md = to_markdown(&sample_report());
        assert!(md.contains("# Beam check"));
        assert!(md.contains("## Assumptions"));
        assert!(md.contains("## Results"));
        assert!(md.contains("PASS"));
        assert!(md.contains("Max moment"));
    }

    #[test]
    fn test_html_contains_sections() {
        let html = to_html(&sample_report());
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<h1>Beam check</h1>"));
        assert!(html.contains("PASS"));
        // Ensure special characters are escaped.
        assert!(html.contains("&lt;"));
    }

    #[test]
    fn test_json_roundtrip() {
        let report = sample_report();
        let json = to_json(&report).unwrap();
        let loaded: Report = serde_json::from_str(&json).unwrap();
        assert_eq!(report, loaded);
        assert_eq!(loaded.overall_status(), ValidationStatus::Pass);
    }

    #[test]
    fn test_validation_fail() {
        let entry = ResultEntry::with_limits(
            "Stress",
            250.0,
            Some("MPa".to_string()),
            Some(0.0),
            Some(200.0),
        );
        assert_eq!(entry.status, ValidationStatus::Fail);
    }

    #[test]
    fn test_write_files() {
        let dir = std::env::temp_dir();
        let md = dir.join("tpt_report_test.md");
        let htm = dir.join("tpt_report_test.html");
        let json = dir.join("tpt_report_test.json");
        let report = sample_report();
        write_markdown(&report, &md).unwrap();
        write_html(&report, &htm).unwrap();
        write_json(&report, &json).unwrap();
        assert!(md.exists() && htm.exists() && json.exists());
        let _ = std::fs::remove_file(md);
        let _ = std::fs::remove_file(htm);
        let _ = std::fs::remove_file(json);
    }
}
