//! Backend-agnostic view tree and HTML serialisation.

/// A minimal, backend-agnostic DOM node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Node {
    /// An element with a tag, attributes and children.
    Element {
        /// Tag name, e.g. `"div"`.
        tag: String,
        /// Attribute name/value pairs.
        attrs: Vec<(String, String)>,
        /// Child nodes.
        children: Vec<Node>,
    },
    /// A text node (escaped on render).
    Text(String),
}

impl Node {
    /// Build an element node.
    pub fn element<T, A, B, I>(tag: T, attrs: I, children: Vec<Node>) -> Node
    where
        T: Into<String>,
        A: Into<String>,
        B: Into<String>,
        I: IntoIterator<Item = (A, B)>,
    {
        Node::Element {
            tag: tag.into(),
            attrs: attrs
                .into_iter()
                .map(|(a, b)| (a.into(), b.into()))
                .collect(),
            children,
        }
    }

    /// Build a text node.
    pub fn text(s: impl Into<String>) -> Node {
        Node::Text(s.into())
    }

    /// Render this node to an HTML fragment.
    pub fn to_html(&self) -> String {
        render_to_html(self)
    }
}

impl Default for Node {
    fn default() -> Self {
        Node::Text(String::new())
    }
}

/// Anything that can produce a [`Node`]; implemented by `#[tpt_app]` structs.
pub trait View {
    /// Build the view tree.
    fn view(&self) -> Node;
}

impl View for Node {
    fn view(&self) -> Node {
        self.clone()
    }
}

impl View for String {
    fn view(&self) -> Node {
        Node::Text(self.clone())
    }
}

impl View for &str {
    fn view(&self) -> Node {
        Node::Text((*self).to_string())
    }
}

impl View for Vec<Node> {
    fn view(&self) -> Node {
        Node::element("div", [("class", "tpt-fragment")], self.clone())
    }
}

macro_rules! view_via_display {
    ($($t:ty),*) => {$(
        impl View for $t {
            fn view(&self) -> Node { Node::Text(self.to_string()) }
        }
    )*};
}
view_via_display!(bool, i32, i64, u32, u64, usize, f32, f64);

fn is_void(tag: &str) -> bool {
    matches!(
        tag,
        "area"
            | "base"
            | "br"
            | "col"
            | "embed"
            | "hr"
            | "img"
            | "input"
            | "link"
            | "meta"
            | "param"
            | "source"
            | "track"
            | "wbr"
    )
}

/// Escape a string for use in text content.
pub fn escape_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(c),
        }
    }
    out
}

/// Escape a string for use inside a double-quoted attribute value.
pub fn escape_attr(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
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

fn write_node(node: &Node, out: &mut String) {
    match node {
        Node::Text(t) => out.push_str(&escape_text(t)),
        Node::Element {
            tag,
            attrs,
            children,
        } => {
            out.push('<');
            out.push_str(tag);
            for (k, v) in attrs {
                out.push(' ');
                out.push_str(k);
                out.push_str("=\"");
                out.push_str(&escape_attr(v));
                out.push('"');
            }
            if is_void(tag) && children.is_empty() {
                out.push_str(" />");
                return;
            }
            out.push('>');
            for c in children {
                write_node(c, out);
            }
            out.push_str("</");
            out.push_str(tag);
            out.push('>');
        }
    }
}

/// Serialise a node tree to an HTML fragment.
pub fn render_to_html(node: &Node) -> String {
    let mut out = String::new();
    write_node(node, &mut out);
    out
}

/// Wrap a view in a complete, self-contained HTML document.
pub fn export_html(app: impl View) -> String {
    export_html_with_title(app, "TPT App")
}

/// Like [`export_html`] but with a custom document title.
pub fn export_html_with_title(app: impl View, title: &str) -> String {
    format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\" />\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n\
         <title>{}</title>\n</head>\n<body>\n{}\n</body>\n</html>\n",
        escape_text(title),
        render_to_html(&app.view())
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_elements_and_escapes() {
        let n = Node::element(
            "p",
            [("class", "a\"b"), ("data-x", "<y>")],
            vec![Node::text("5 < 6 & \"ok\"")],
        );
        assert_eq!(
            render_to_html(&n),
            "<p class=\"a&quot;b\" data-x=\"&lt;y&gt;\">5 &lt; 6 &amp; \"ok\"</p>"
        );
    }

    #[test]
    fn void_elements_self_close() {
        let n = Node::element("input", [("type", "range")], vec![]);
        assert_eq!(render_to_html(&n), "<input type=\"range\" />");
    }

    #[test]
    fn export_wraps_document() {
        let html = export_html(Node::text("hi"));
        assert!(html.starts_with("<!doctype html>"));
        assert!(html.contains("<html"));
        assert!(html.contains("hi"));
    }
}
