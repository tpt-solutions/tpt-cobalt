//! Wasm-rendered, interactive HTML documents (feature `wasm`).
//!
//! `Document::to_interactive_html` serializes the document to HTML (via the
//! existing [`Document::to_html`] renderer) and embeds those bytes inside a
//! tiny generated WebAssembly module. The module exposes the document bytes
//! from its linear memory and a `len()` export; the bundled HTML host
//! instantiates the module, pulls the bytes back out, and injects them into the
//! page — so the document is *rendered by wasm* in the browser rather than
//! shipped as plain markup. A small JS layer adds interactivity: a client-side
//! search box that highlights matches and collapsible sections.

use wasm_encoder::{
    CodeSection, ConstExpr, ExportKind, ExportSection, Function, FunctionSection, MemorySection,
    MemoryType, Module, TypeSection, ValType,
};

use crate::ast::Document;
use crate::error::DocError;

/// Build a wasm module that stores `html` in linear memory at offset 0 and
/// exports `len()` (the byte length) plus the `memory` itself.
fn build_doc_wasm(html: &[u8]) -> Vec<u8> {
    let len = html.len();
    // One page = 64 KiB; allocate enough to hold the document.
    let pages = ((len / 65_536) + 1) as u64;

    let mut f = Function::new([]);
    {
        let mut i = f.instructions();
        i.i32_const(len as i32).end();
    }

    let mut types = TypeSection::new();
    types.ty().function([], [ValType::I32]);

    let mut funcs = FunctionSection::new();
    funcs.function(0);

    let mut mem = MemorySection::new();
    mem.memory(MemoryType {
        minimum: pages,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });

    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    exports.export("len", ExportKind::Func, 0);

    let mut data = wasm_encoder::DataSection::new();
    data.active(0, &ConstExpr::i32_const(0), html.iter().copied());

    let mut code = CodeSection::new();
    code.function(&f);

    let mut module = Module::new();
    module.section(&types);
    module.section(&funcs);
    module.section(&mem);
    module.section(&exports);
    module.section(&code);
    module.section(&data);
    module.finish()
}

/// The static host page. `__TITLE__` and `__WASM_BYTES__` are filled in at
/// runtime; everything else is a literal so it can live in a raw string.
const HOST_TEMPLATE: &str = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  body { font-family: Georgia, serif; max-width: 48rem; margin: 2rem auto; padding: 0 1rem; }
  #toolbar { position: sticky; top: 0; background: #fff; padding: 0.5rem 0; border-bottom: 1px solid #ddd; }
  #search { padding: 0.3rem; width: 16rem; }
  section { border-left: 3px solid #eee; padding-left: 0.75rem; margin: 1rem 0; }
  h1, h2, h3, h4, h5, h6 { cursor: pointer; }
  .collapsed > *:not(h1):not(h2):not(h3):not(h4):not(h5):not(h6) { display: none; }
  mark { background: #ffe08a; }
</style>
</head>
<body>
<div id="toolbar">
  <input id="search" type="search" placeholder="Search this document…" autocomplete="off">
</div>
<div id="doc"></div>
<script type="module">
const wasmBytes = new Uint8Array([__WASM_BYTES__]);
const dec = new TextDecoder("utf-8");
WebAssembly.instantiate(wasmBytes).then(({ module, instance }) => {
  const len = instance.exports.len();
  const bytes = new Uint8Array(instance.exports.memory.buffer, 0, len);
  document.getElementById("doc").innerHTML = dec.decode(bytes);
  wireInteractivity();
});

function wireInteractivity() {
  // Collapsible sections: click a heading to toggle its body.
  document.querySelectorAll("#doc section h1, #doc section h2, #doc section h3").forEach(h => {
    h.addEventListener("click", () => h.parentElement.classList.toggle("collapsed"));
  });
  // Client-side search: highlight matches across the document text.
  const search = document.getElementById("search");
  search.addEventListener("input", () => {
    const q = search.value.trim().toLowerCase();
    document.querySelectorAll("#doc mark").forEach(m => {
      const t = document.createTextNode(m.textContent);
      m.replaceWith(t);
    });
    document.normalize();
    if (!q) return;
    const walker = document.createTreeWalker(document.getElementById("doc"), NodeFilter.SHOW_TEXT);
    const targets = [];
    while (walker.nextNode()) {
      if (walker.currentNode.nodeValue.toLowerCase().includes(q)) targets.push(walker.currentNode);
    }
    for (const node of targets) {
      const idx = node.nodeValue.toLowerCase().indexOf(q);
      if (idx < 0) continue;
      const range = document.createRange();
      range.setStart(node, idx);
      range.setEnd(node, idx + q.length);
      const mark = document.createElement("mark");
      range.surroundContents(mark);
    }
  });
}
</script>
</body>
</html>"##;

/// Render an interactive HTML document: the static HTML is embedded in a wasm
/// module and re-materialized in the browser, with a search box and collapsible
/// sections layered on top via a small inline script.
fn interactive_html(doc: &Document) -> String {
    let base = doc.to_html();
    let wasm = build_doc_wasm(base.as_bytes());
    let wasm_literal = wasm
        .iter()
        .map(|b| b.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let title = doc
        .title
        .clone()
        .unwrap_or_else(|| "TPT Document".to_string());
    HOST_TEMPLATE
        .replace("__TITLE__", &escape_attr(&title))
        .replace("__WASM_BYTES__", &wasm_literal)
}

/// Minimal HTML-attribute escaping for the document title.
fn escape_attr(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

impl Document {
    /// Render the document as a self-contained, wasm-rendered interactive HTML
    /// page (feature `wasm`).
    pub fn to_interactive_html(&self) -> String {
        interactive_html(self)
    }

    /// Write the interactive HTML page to `path`.
    pub fn write_interactive_html(&self, path: impl AsRef<std::path::Path>) -> Result<(), DocError> {
        std::fs::write(path, self.to_interactive_html()).map_err(|e| DocError::Io(e.to_string()))
    }

    /// Write the standalone wasm module (the embedded document bytes) to `path`.
    pub fn to_wasm_doc(&self, path: impl AsRef<std::path::Path>) -> Result<(), DocError> {
        let base = self.to_html();
        std::fs::write(path, build_doc_wasm(base.as_bytes())).map_err(|e| DocError::Io(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interactive_html_embeds_wasm_and_content() {
        let mut doc = Document::new();
        doc.set_title("My Wasm Doc");
        let html = doc.to_interactive_html();
        assert!(html.contains("WebAssembly.instantiate"));
        assert!(html.contains("wireInteractivity"));
        assert!(html.contains("My Wasm Doc"));
        // The wasm byte array literal must be present and non-empty.
        assert!(html.contains("Uint8Array([") && html.contains("])"));
    }

    #[test]
    fn wasm_doc_has_magic_header() {
        let dir = std::env::temp_dir();
        let path = dir.join("tpt_doc_test.wasm");
        let doc = Document::new();
        doc.to_wasm_doc(&path).expect("writes wasm");
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.starts_with(b"\0asm"), "wasm magic header");
    }
}
