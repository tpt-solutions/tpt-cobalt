//! Widget value types introspected by `#[tpt_app]`.

use crate::view::Node;
use std::fmt;

/// A numeric range widget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slider<T> {
    /// Lower bound.
    pub min: T,
    /// Upper bound.
    pub max: T,
    /// Increment between steps.
    pub step: T,
    /// Current value.
    pub value: T,
}

impl<T> Slider<T> {
    /// Create a slider from explicit bounds.
    pub fn new(min: T, max: T, step: T, value: T) -> Self {
        Slider {
            min,
            max,
            step,
            value,
        }
    }
}

impl<T: Copy> Slider<T> {
    /// Current value.
    pub fn value(&self) -> T {
        self.value
    }
}

macro_rules! slider_default {
    ($($t:ty),*) => {$(
        impl Default for Slider<$t> {
            fn default() -> Self { Slider { min: 0.0, max: 1.0, step: 0.01, value: 0.0 } }
        }
    )*};
}
slider_default!(f32, f64);

/// A single-choice list widget over borrowed string choices.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dropdown<'a> {
    /// Available choices.
    pub choices: &'a [&'a str],
    /// Index of the selected choice.
    pub selected: usize,
}

impl<'a> Dropdown<'a> {
    /// Create a dropdown over `choices` with `selected` index.
    pub fn new(choices: &'a [&'a str], selected: usize) -> Self {
        Dropdown { choices, selected }
    }

    /// The selected choice, or `""` when out of range.
    pub fn selected_str(&self) -> &'a str {
        self.choices.get(self.selected).copied().unwrap_or("")
    }
}

/// Placeholder for an in-browser file drop target.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FileUpload {
    /// Name of the uploaded file (empty when nothing was uploaded).
    pub name: &'static str,
}

impl FileUpload {
    /// Create a file upload holding `name`.
    pub fn new(name: &'static str) -> Self {
        FileUpload { name }
    }

    /// `true` when a file has been supplied.
    pub fn is_empty(&self) -> bool {
        self.name.is_empty()
    }
}

/// An RGB colour picker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColorPicker {
    /// Current RGB triple.
    pub value: (u8, u8, u8),
}

impl ColorPicker {
    /// Create a picker from an RGB triple.
    pub fn new(value: (u8, u8, u8)) -> Self {
        ColorPicker { value }
    }

    /// `#rrggbb` representation of the current colour.
    pub fn hex(&self) -> String {
        let (r, g, b) = self.value;
        format!("#{r:02x}{g:02x}{b:02x}")
    }
}

/// Renders a widget value into the view tree.
pub trait Widget {
    /// Build the widget's node, labelled/named by the argument `name`.
    fn widget_node(&self, name: &str) -> Node;
}

fn labelled(name: &str, class: &str, input: Node) -> Node {
    Node::element(
        "label",
        [("class", class.to_string()), ("for", name.to_string())],
        vec![Node::text(name), input],
    )
}

impl<T: fmt::Display> Widget for Slider<T> {
    fn widget_node(&self, name: &str) -> Node {
        let input = Node::element(
            "input",
            vec![
                ("type", "range".to_string()),
                ("name", name.to_string()),
                ("id", name.to_string()),
                ("min", self.min.to_string()),
                ("max", self.max.to_string()),
                ("step", self.step.to_string()),
                ("value", self.value.to_string()),
            ],
            vec![],
        );
        labelled(name, "tpt-slider", input)
    }
}

impl Widget for Dropdown<'_> {
    fn widget_node(&self, name: &str) -> Node {
        let options = self
            .choices
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut attrs = vec![("value", i.to_string())];
                if i == self.selected {
                    attrs.push(("selected", "selected".to_string()));
                }
                Node::element("option", attrs, vec![Node::text(*c)])
            })
            .collect();
        let select = Node::element(
            "select",
            [("name", name.to_string()), ("id", name.to_string())],
            options,
        );
        labelled(name, "tpt-dropdown", select)
    }
}

impl Widget for FileUpload {
    fn widget_node(&self, name: &str) -> Node {
        let input = Node::element(
            "input",
            vec![
                ("type", "file".to_string()),
                ("name", name.to_string()),
                ("id", name.to_string()),
                ("data-file", self.name.to_string()),
            ],
            vec![],
        );
        labelled(name, "tpt-file", input)
    }
}

impl Widget for ColorPicker {
    fn widget_node(&self, name: &str) -> Node {
        let input = Node::element(
            "input",
            vec![
                ("type", "color".to_string()),
                ("name", name.to_string()),
                ("id", name.to_string()),
                ("value", self.hex()),
            ],
            vec![],
        );
        labelled(name, "tpt-color", input)
    }
}

/// Fallback widget for argument types without a dedicated control.
pub fn text_widget(name: &str) -> Node {
    let input = Node::element(
        "input",
        [
            ("type", "text".to_string()),
            ("name", name.to_string()),
            ("id", name.to_string()),
        ],
        vec![],
    );
    labelled(name, "tpt-text", input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::render_to_html;

    #[test]
    fn slider_renders_range_input() {
        let html = render_to_html(&Slider::<f32>::new(0.0, 1.0, 0.1, 0.5).widget_node("t"));
        assert!(html.contains("<input"));
        assert!(html.contains("type=\"range\""));
        assert!(html.contains("value=\"0.5\""));
    }

    #[test]
    fn dropdown_marks_selection() {
        let choices = ["PCA", "t-SNE"];
        let d = Dropdown::new(&choices, 1);
        assert_eq!(d.selected_str(), "t-SNE");
        let html = render_to_html(&d.widget_node("m"));
        assert!(html.contains("<option value=\"1\" selected=\"selected\">t-SNE</option>"));
    }

    #[test]
    fn color_and_file_widgets() {
        assert_eq!(ColorPicker::new((255, 0, 16)).hex(), "#ff0010");
        assert!(render_to_html(&FileUpload::default().widget_node("d")).contains("type=\"file\""));
        assert!(render_to_html(&ColorPicker::default().widget_node("c")).contains("type=\"color\""));
        assert!(Dropdown::default().selected_str().is_empty());
    }
}
