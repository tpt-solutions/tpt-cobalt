//! Markdown exporter for [`Report`].

use crate::model::{NamedValue, Report, ResultEntry, Section, Table};

/// Render a report to a Markdown string.
pub fn render(report: &Report) -> String {
    let mut out = String::new();

    out.push_str(&format!("# {}\n\n", report.title));

    if let Some(subtitle) = &report.subtitle {
        out.push_str(&format!("_{}_\n\n", subtitle));
    }

    let mut meta: Vec<String> = Vec::new();
    if let Some(author) = &report.author {
        meta.push(format!("**Author:** {author}"));
    }
    if let Some(date) = &report.date {
        meta.push(format!("**Date:** {date}"));
    }
    if !meta.is_empty() {
        out.push_str(&meta.join("  |  "));
        out.push_str("\n\n");
    }

    if let Some(summary) = &report.summary {
        out.push_str(&format!("{summary}\n\n"));
    }

    for section in &report.sections {
        match section {
            Section::Heading(text) => {
                out.push_str(&format!("## {text}\n\n"));
            }
            Section::Paragraph(text) => {
                out.push_str(&format!("{text}\n\n"));
            }
            Section::Table(table) => {
                out.push_str(&render_table(table));
                out.push('\n');
            }
            Section::Assumptions(values) => {
                out.push_str("## Assumptions\n\n");
                out.push_str(&render_named_values(values));
                out.push('\n');
            }
            Section::Results(entries) => {
                out.push_str("## Results\n\n");
                out.push_str(&render_results(entries));
                out.push('\n');
            }
        }
    }

    out
}

fn render_table(table: &Table) -> String {
    let mut out = String::new();
    if let Some(caption) = &table.caption {
        out.push_str(&format!("_{caption}_\n\n"));
    }
    out.push('|');
    for header in &table.headers {
        out.push_str(&format!(" {header} |"));
    }
    out.push_str("\n|");
    for _ in &table.headers {
        out.push_str(" --- |");
    }
    out.push('\n');
    for row in &table.rows {
        out.push('|');
        for cell in row {
            out.push_str(&format!(" {cell} |"));
        }
        out.push('\n');
    }
    out
}

fn render_named_values(values: &[NamedValue]) -> String {
    let mut out = String::new();
    out.push_str("| Name | Value | Unit | Description |\n");
    out.push_str("| --- | --- | --- | --- |\n");
    for v in values {
        let value = format!("{}", v.value);
        let unit = v.unit.as_deref().unwrap_or("");
        let description = v.description.as_deref().unwrap_or("");
        out.push_str(&format!(
            "| {} | {} | {unit} | {description} |\n",
            v.name, value
        ));
    }
    out
}

fn render_results(entries: &[ResultEntry]) -> String {
    let mut out = String::new();
    out.push_str("| Name | Value | Unit | Status | Detail |\n");
    out.push_str("| --- | --- | --- | --- | --- |\n");
    for e in entries {
        let value = format!("{}", e.value);
        let unit = e.unit.as_deref().unwrap_or("");
        let detail = e.detail.as_deref().unwrap_or("");
        out.push_str(&format!(
            "| {} | {value} | {unit} | {} | {detail} |\n",
            e.name,
            e.status.as_symbol()
        ));
    }
    out
}
