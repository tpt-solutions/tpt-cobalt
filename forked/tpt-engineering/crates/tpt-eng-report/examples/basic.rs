//! tpt-eng-report basic usage: build a report and export Markdown/HTML/JSON.
//!
//! Run with: `cargo run --example basic -p tpt-eng-report`

use tpt_eng_report::{
    NamedValue, Report, ResultEntry, to_markdown, write_html, write_json, write_markdown,
};

fn main() {
    let report = Report::new("Beam check")
        .with_author("TPT")
        .with_date("2026-08-14")
        .with_summary("Simply supported beam: load <= 10 kN.")
        .assumptions(vec![
            NamedValue::new("Length", 5.0).with_unit("m"),
            NamedValue::new("Load", 10.0).with_unit("kN"),
        ])
        .results(vec![
            ResultEntry::with_limits(
                "Max moment",
                12.5,
                Some("kNm".into()),
                Some(0.0),
                Some(15.0),
            ),
            ResultEntry::with_limits(
                "Max deflection",
                18.0,
                Some("mm".into()),
                Some(0.0),
                Some(20.0),
            ),
        ])
        .heading("Notes")
        .paragraph("All checks within permissible limits.");

    println!("{}", to_markdown(&report));

    let dir = std::env::temp_dir();
    let md = dir.join("tpt_report_basic.md");
    let html = dir.join("tpt_report_basic.html");
    let json = dir.join("tpt_report_basic.json");
    write_markdown(&report, &md).unwrap();
    write_html(&report, &html).unwrap();
    write_json(&report, &json).unwrap();
    println!(
        "wrote {}, {}, {}",
        md.display(),
        html.display(),
        json.display()
    );
    println!("overall status: {}", report.overall_status().as_symbol());
}
