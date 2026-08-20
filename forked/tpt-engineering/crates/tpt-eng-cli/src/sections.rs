//! Cross-section catalogue and property calculations for the CLI `sections inspect`
//! command. The actual geometry is delegated to [`tpt_eng_sections`]; this module
//! is a thin CLI-facing wrapper that maps the command-line shapes onto the
//! canonical section types.

use tpt_eng_sections::{Circle, ISection, Rectangle, Section as SectionsTrait};

/// A standard cross-section shape (CLI-facing enum; delegates to
/// [`tpt_eng_sections`]).
#[derive(Debug, Clone)]
pub enum Section {
    /// Rectangle of breadth `b` and height `h` (about centroidal axes).
    Rectangle { b: f64, h: f64 },
    /// Solid circle of diameter `d`.
    Circle { d: f64 },
    /// Symmetric I-beam: total depth `d`, flange width `bf`, flange thickness `tf`, web thickness `tw`.
    IBeam { d: f64, bf: f64, tf: f64, tw: f64 },
}

impl Section {
    /// Cross-sectional area.
    pub fn area(&self) -> f64 {
        match self {
            Section::Rectangle { b, h } => Rectangle::new(*b, *h).area(),
            Section::Circle { d } => Circle::new(*d).area(),
            Section::IBeam { d, bf, tf, tw } => ISection::new(*d, *bf, *tf, *tw).area(),
        }
    }

    /// Second moment of area about the strong (x) centroidal axis.
    pub fn second_moment_x(&self) -> f64 {
        match self {
            Section::Rectangle { b, h } => Rectangle::new(*b, *h).second_moments().0,
            Section::Circle { d } => Circle::new(*d).second_moments().0,
            Section::IBeam { d, bf, tf, tw } => ISection::new(*d, *bf, *tf, *tw).second_moments().0,
        }
    }

    /// Second moment of area about the weak (y) centroidal axis.
    pub fn second_moment_y(&self) -> f64 {
        match self {
            Section::Rectangle { b, h } => Rectangle::new(*b, *h).second_moments().1,
            Section::Circle { d } => Circle::new(*d).second_moments().1,
            Section::IBeam { d, bf, tf, tw } => ISection::new(*d, *bf, *tf, *tw).second_moments().1,
        }
    }

    /// Section modulus about the strong axis (`Ix / (h/2)`).
    pub fn section_modulus_x(&self) -> f64 {
        match self {
            Section::Rectangle { b, h } => Rectangle::new(*b, *h).section_modulus().0,
            Section::Circle { d } => Circle::new(*d).section_modulus().0,
            Section::IBeam { d, bf, tf, tw } => {
                ISection::new(*d, *bf, *tf, *tw).section_modulus().0
            }
        }
    }
}

/// Format a section as a multi-line property listing.
pub fn describe(section: &Section) -> String {
    let kind = match section {
        Section::Rectangle { .. } => "Rectangle",
        Section::Circle { .. } => "Circle",
        Section::IBeam { .. } => "I-beam",
    };
    format!(
        "Section: {kind}\n  area:               {:.4e}\n  I_x (strong):        {:.4e}\n  I_y (weak):         {:.4e}\n  section modulus W_x: {:.4e}",
        section.area(),
        section.second_moment_x(),
        section.second_moment_y(),
        section.section_modulus_x(),
    )
}
