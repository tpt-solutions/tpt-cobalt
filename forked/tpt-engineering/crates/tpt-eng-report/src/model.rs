//! The calculation-report data model.

use serde::{Deserialize, Serialize};

/// Status used to annotate a result after validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationStatus {
    /// Result is within acceptable limits.
    Pass,
    /// Result is within limits but close to a boundary.
    Warn,
    /// Result is outside acceptable limits.
    Fail,
    /// No acceptance criteria were applied.
    Info,
}

impl ValidationStatus {
    /// Short human-readable symbol for the status.
    pub fn as_symbol(&self) -> &'static str {
        match self {
            ValidationStatus::Pass => "PASS",
            ValidationStatus::Warn => "WARN",
            ValidationStatus::Fail => "FAIL",
            ValidationStatus::Info => "INFO",
        }
    }
}

/// A named input value or assumption.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamedValue {
    /// Human-readable name.
    pub name: String,
    /// Numeric value.
    pub value: f64,
    /// Unit of measure.
    pub unit: Option<String>,
    /// Optional description.
    pub description: Option<String>,
}

impl NamedValue {
    /// Create a new named value with no unit or description.
    pub fn new(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            unit: None,
            description: None,
        }
    }

    /// Attach a unit.
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }

    /// Attach a description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// A calculated output value annotated with a validation status.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResultEntry {
    /// Human-readable name.
    pub name: String,
    /// Numeric value.
    pub value: f64,
    /// Unit of measure.
    pub unit: Option<String>,
    /// Validation status.
    pub status: ValidationStatus,
    /// Optional detail (e.g. the acceptance criterion that was applied).
    pub detail: Option<String>,
}

impl ResultEntry {
    /// Create a result entry with an explicit status.
    pub fn new(
        name: impl Into<String>,
        value: f64,
        unit: Option<String>,
        status: ValidationStatus,
        detail: Option<String>,
    ) -> Self {
        Self {
            name: name.into(),
            value,
            unit,
            status,
            detail,
        }
    }

    /// Create a result entry validated against an inclusive `[min, max]` range.
    ///
    /// `min`/`max` of `None` mean the bound is open. The `detail` string records the applied
    /// criterion.
    pub fn with_limits(
        name: impl Into<String>,
        value: f64,
        unit: Option<String>,
        min: Option<f64>,
        max: Option<f64>,
    ) -> Self {
        let status = crate::validate::validate_range(value, min, max);
        let detail = match (min, max) {
            (Some(lo), Some(hi)) => Some(format!("limit: {lo} <= x <= {hi}")),
            (Some(lo), None) => Some(format!("limit: x >= {lo}")),
            (None, Some(hi)) => Some(format!("limit: x <= {hi}")),
            (None, None) => None,
        };
        Self::new(name, value, unit, status, detail)
    }
}

/// A tabular block of string cells.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Table {
    /// Optional table caption.
    pub caption: Option<String>,
    /// Column headers.
    pub headers: Vec<String>,
    /// Rows of cells.
    pub rows: Vec<Vec<String>>,
}

impl Table {
    /// Create a new table from headers and rows.
    pub fn new(headers: Vec<String>, rows: Vec<Vec<String>>) -> Self {
        Self {
            caption: None,
            headers,
            rows,
        }
    }

    /// Attach a caption.
    pub fn with_caption(mut self, caption: impl Into<String>) -> Self {
        self.caption = Some(caption.into());
        self
    }
}

/// A content block within a report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Section {
    /// A level-2 heading.
    Heading(String),
    /// A paragraph of prose.
    Paragraph(String),
    /// A generic table.
    Table(Table),
    /// A list of assumptions / inputs.
    Assumptions(Vec<NamedValue>),
    /// A list of validated results.
    Results(Vec<ResultEntry>),
}

/// A calculation report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    /// Report title.
    pub title: String,
    /// Optional subtitle.
    pub subtitle: Option<String>,
    /// Optional author.
    pub author: Option<String>,
    /// Optional date (free-form string).
    pub date: Option<String>,
    /// Optional executive summary.
    pub summary: Option<String>,
    /// Content sections.
    pub sections: Vec<Section>,
}

impl Report {
    /// Create a new, empty report with the given title.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            subtitle: None,
            author: None,
            date: None,
            summary: None,
            sections: Vec::new(),
        }
    }

    /// Attach a subtitle.
    pub fn with_subtitle(mut self, subtitle: impl Into<String>) -> Self {
        self.subtitle = Some(subtitle.into());
        self
    }

    /// Attach an author.
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Attach a date.
    pub fn with_date(mut self, date: impl Into<String>) -> Self {
        self.date = Some(date.into());
        self
    }

    /// Attach an executive summary.
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    /// Append a section and return `self` for chaining.
    pub fn add_section(mut self, section: Section) -> Self {
        self.sections.push(section);
        self
    }

    /// Append a heading section.
    pub fn heading(mut self, text: impl Into<String>) -> Self {
        self.sections.push(Section::Heading(text.into()));
        self
    }

    /// Append a paragraph section.
    pub fn paragraph(mut self, text: impl Into<String>) -> Self {
        self.sections.push(Section::Paragraph(text.into()));
        self
    }

    /// Append an assumptions section.
    pub fn assumptions(mut self, values: Vec<NamedValue>) -> Self {
        self.sections.push(Section::Assumptions(values));
        self
    }

    /// Append a results section.
    pub fn results(mut self, values: Vec<ResultEntry>) -> Self {
        self.sections.push(Section::Results(values));
        self
    }

    /// Append a table section.
    pub fn table(mut self, table: Table) -> Self {
        self.sections.push(Section::Table(table));
        self
    }

    /// Overall worst-case status across all result entries (useful for a verdict line).
    pub fn overall_status(&self) -> ValidationStatus {
        use ValidationStatus::*;
        let rank = |s: ValidationStatus| match s {
            Fail => 3,
            Warn => 2,
            Pass => 1,
            Info => 0,
        };
        let mut worst = Info;
        for section in &self.sections {
            if let Section::Results(entries) = section {
                for entry in entries {
                    if rank(entry.status) > rank(worst) {
                        worst = entry.status;
                    }
                }
            }
        }
        worst
    }
}
