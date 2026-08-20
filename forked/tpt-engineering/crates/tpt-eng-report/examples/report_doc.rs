//! tpt-eng-report: a richer design-report document with a results table and
//! pass/warn/fail validation, exported to Markdown/HTML/JSON.
//!
//! Run with: `cargo run --example report_doc -p tpt-eng-report`

use tpt_eng_report::{
    NamedValue, Report, ResultEntry, Table, ValidationStatus, to_html, to_markdown, write_html,
    write_json, write_markdown,
};

fn main() {
    let report = Report::new("Simply supported beam — design check")
        .with_subtitle("Eurocode-style verification")
        .with_author("TPT Engineering")
        .with_date("2026-08-14")
        .with_summary(
            "Verification of a 5 m simply supported beam under a 10 kN UDL against \
             bending, shear, and deflection limits.",
        )
        .assumptions(vec![
            NamedValue::new("Span L", 5.0).with_unit("m"),
            NamedValue::new("Load w", 10.0).with_unit("kN/m"),
            NamedValue::new("f_y", 355.0)
                .with_unit("MPa")
                .with_description("yield strength"),
        ])
        .results(vec![
            ResultEntry::with_limits(
                "Bending moment M",
                31.25,
                Some("kNm".into()),
                Some(0.0),
                Some(45.0),
            ),
            ResultEntry::with_limits(
                "Shear force V",
                25.0,
                Some("kN".into()),
                Some(0.0),
                Some(40.0),
            ),
            // Within the limit but close to it -> flagged WARN explicitly.
            ResultEntry::new(
                "Deflection",
                19.5,
                Some("mm".into()),
                ValidationStatus::Warn,
                Some("within 2.5% of the 20.0 mm limit".into()),
            ),
        ])
        .table(
            Table::new(
                vec![
                    "Quantity".into(),
                    "Value".into(),
                    "Limit".into(),
                    "Status".into(),
                ],
                vec![
                    vec![
                        "M".into(),
                        "31.25 kNm".into(),
                        "45.0 kNm".into(),
                        "PASS".into(),
                    ],
                    vec![
                        "V".into(),
                        "25.0 kN".into(),
                        "40.0 kN".into(),
                        "PASS".into(),
                    ],
                    vec![
                        "defl".into(),
                        "19.5 mm".into(),
                        "20.0 mm".into(),
                        "WARN".into(),
                    ],
                ],
            )
            .with_caption("Summary of results"),
        )
        .heading("Conclusion")
        .paragraph(
            "All demands satisfy their limits; the deflection check is within 2.5% of \
             the allowable and is flagged for monitoring.",
        );

    let status = report.overall_status();
    println!(
        "overall status: {} (expected {})",
        status.as_symbol(),
        ValidationStatus::Warn.as_symbol()
    );

    let md = to_markdown(&report);
    println!("{md}");

    let dir = std::env::temp_dir();
    let md_path = dir.join("tpt_report_doc.md");
    let html_path = dir.join("tpt_report_doc.html");
    let json_path = dir.join("tpt_report_doc.json");
    write_markdown(&report, &md_path).unwrap();
    write_html(&report, &html_path).unwrap();
    write_json(&report, &json_path).unwrap();
    println!(
        "wrote {}, {}, {}",
        md_path.display(),
        html_path.display(),
        json_path.display()
    );
    // Sanity check: HTML must escape and contain the title.
    let html = to_html(&report);
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("WARN"));
}
