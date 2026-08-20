//! HTML exporter for [`Report`].

use crate::model::{NamedValue, Report, ResultEntry, Section, Table};

/// Render a report to an HTML string.
pub fn render(report: &Report) -> String {
    let mut body = String::new();

    body.push_str(&format!("<h1>{}</h1>\n", escape(&report.title)));

    if let Some(subtitle) = &report.subtitle {
        body.push_str(&format!(
            "<p class=\"subtitle\"><em>{}</em></p>\n",
            escape(subtitle)
        ));
    }

    let mut meta: Vec<String> = Vec::new();
    if let Some(author) = &report.author {
        meta.push(format!("<strong>Author:</strong> {}", escape(author)));
    }
    if let Some(date) = &report.date {
        meta.push(format!("<strong>Date:</strong> {}", escape(date)));
    }
    if !meta.is_empty() {
        body.push_str(&format!("<p>{}</p>\n", meta.join(" &nbsp;|&nbsp; ")));
    }

    if let Some(summary) = &report.summary {
        body.push_str(&format!("<p>{}</p>\n", escape(summary)));
    }

    for section in &report.sections {
        match section {
            Section::Heading(text) => {
                body.push_str(&format!("<h2>{}</h2>\n", escape(text)));
            }
            Section::Paragraph(text) => {
                body.push_str(&format!("<p>{}</p>\n", escape(text)));
            }
            Section::Table(table) => {
                body.push_str(&render_table(table));
            }
            Section::Assumptions(values) => {
                body.push_str("<h2>Assumptions</h2>\n");
                body.push_str(&render_named_values(values));
            }
            Section::Results(entries) => {
                body.push_str("<h2>Results</h2>\n");
                body.push_str(&render_results(entries));
            }
        }
    }

    format!(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <title>{}</title>\n</head>\n<body>\n{}</body>\n</html>\n",
        escape(&report.title),
        body
    )
}

fn render_table(table: &Table) -> String {
    let mut out = String::new();
    out.push_str("<table>\n");
    if let Some(caption) = &table.caption {
        out.push_str(&format!("<caption>{}</caption>\n", escape(caption)));
    }
    out.push_str("<thead><tr>");
    for header in &table.headers {
        out.push_str(&format!("<th>{}</th>", escape(header)));
    }
    out.push_str("</tr></thead>\n<tbody>\n");
    for row in &table.rows {
        out.push_str("<tr>");
        for cell in row {
            out.push_str(&format!("<td>{}</td>", escape(cell)));
        }
        out.push_str("</tr>\n");
    }
    out.push_str("</tbody>\n</table>\n");
    out
}

fn render_named_values(values: &[NamedValue]) -> String {
    let mut out = String::from(
        "<table>\n<thead><tr><th>Name</th><th>Value</th><th>Unit</th><th>Description</th></tr></thead>\n<tbody>\n",
    );
    for v in values {
        let unit = v.unit.as_deref().unwrap_or("");
        let description = v.description.as_deref().unwrap_or("");
        out.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{unit}</td><td>{description}</td></tr>\n",
            escape(&v.name),
            v.value
        ));
    }
    out.push_str("</tbody>\n</table>\n");
    out
}

fn render_results(entries: &[ResultEntry]) -> String {
    let mut out = String::from(
        "<table>\n<thead><tr><th>Name</th><th>Value</th><th>Unit</th><th>Status</th><th>Detail</th></tr></thead>\n<tbody>\n",
    );
    for e in entries {
        let unit = e.unit.as_deref().unwrap_or("");
        let detail = e.detail.as_deref().unwrap_or("");
        out.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{unit}</td><td>{}</td><td>{detail}</td></tr>\n",
            escape(&e.name),
            e.value,
            escape(e.status.as_symbol())
        ));
    }
    out.push_str("</tbody>\n</table>\n");
    out
}

/// Escape the five significant HTML characters.
fn escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}
