//! `cargo-tpt` — the cargo subcommand for the TPT stack.
//!
//! Installed as a binary called `cargo-tpt`, cargo dispatches `cargo tpt <cmd>`
//! to it by re-invoking the binary as `cargo-tpt tpt <cmd>`, so `main` drops a
//! leading `tpt` argument when it sees one. The binary also works when invoked
//! directly as `cargo-tpt <cmd>`.
//!
//! Commands:
//!
//! * `cargo tpt new <name>` — scaffold a workspace member that depends on the
//!   TPT crates by path and prints a real tensor.
//! * `cargo tpt doctor` — environment report plus the capability table below.
//! * `cargo tpt features` — the capability table on its own.
//! * `cargo tpt serve` — build a `tpt-ui`/`tpt-script` Wasm app, bundle a static
//!   host, and serve it over a local HTTP server (no external infra).
//! * `cargo tpt playground` — build the in-browser **Try TPT** playground (a
//!   Wasm-compiled `tpt-script` runtime + editor) and serve it locally.
//!
//! Deliberately dependency-free: `std` only.

use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "\
cargo-tpt — cargo subcommand for the TPT scientific stack

USAGE:
    cargo tpt <COMMAND> [OPTIONS]

COMMANDS:
    new <name>      Scaffold a new TPT project.
                    Inside a tpt-rust6 checkout it is created at
                    `examples/<name>`; elsewhere at `./<name>`.
    doctor          Report the environment plus the stack's optional features.
    features        Report the stack's optional cargo features.
    serve           Build + serve a Wasm app over a local HTTP server.
    playground      Build + serve the in-browser 'Try TPT' playground.
    help            Print this message.

OPTIONS (new):
    --path <DIR>    Create `<DIR>/<name>` instead of the default location.

EXAMPLES:
    cargo tpt new my-analysis
    cargo tpt doctor
    cargo tpt features
";

/// Optional cargo features declared somewhere in the TPT workspace:
/// `(feature, declaring crate, what it turns on)`.
///
/// This table is static knowledge; the `DECLARED` column printed next to it is
/// re-read from the crates' `Cargo.toml` files whenever a checkout is found.
const FEATURES: &[(&str, &str, &str)] = &[
    (
        "wasm",
        "tpt-omni",
        "wasm32 build surface for the data engine",
    ),
    (
        "wasm",
        "tpt-ui",
        "browser backend for #[tpt_app] dashboards",
    ),
    (
        "gpu",
        "tpt-viz",
        "WebGPU (wgpu) instanced point-sprite scatter renderer (Renderer::render_scatter_rgba)",
    ),
    (
        "omni",
        "tpt-viz",
        "Tensor / OmniFrame / Table plotting adapters",
    ),
    (
        "hdf5",
        "tpt-io",
        "HDF5 reader (needs the libhdf5 system library)",
    ),
    ("pdf", "tpt-doc", "text-only PDF backend via printpdf"),
    (
        "viz",
        "tpt-lab",
        "SVG heatmaps and Plot cells inside notebooks",
    ),
];

/// Crates a scaffolded project depends on.
const SCAFFOLD_DEPS: &[&str] = &["tpt-omni", "tpt-io", "tpt-grad"];

/// Placeholder used when no tpt checkout can be located.
const UNKNOWN_CRATES_DIR: &str = "../path/to/tpt-rust6/crates";

fn main() -> ExitCode {
    let mut args: Vec<String> = env::args().skip(1).collect();

    // `cargo tpt <args>` re-invokes this binary as `cargo-tpt tpt <args>`.
    if args.first().is_some_and(|a| a == "tpt") {
        args.remove(0);
    }

    let Some((command, rest)) = args.split_first() else {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    };

    match command.as_str() {
        "new" => match cmd_new(rest) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("error: {message}");
                ExitCode::FAILURE
            }
        },
        "doctor" => {
            cmd_doctor();
            ExitCode::SUCCESS
        }
        "features" => {
            print_feature_table(find_workspace_root().as_deref());
            ExitCode::SUCCESS
        }
        "serve" | "tpt-serve" => match cmd_serve(rest) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("error: {message}");
                ExitCode::FAILURE
            }
        },
        "playground" | "tpt-playground" => match cmd_playground(rest) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("error: {message}");
                ExitCode::FAILURE
            }
        },
        "help" | "--help" | "-h" => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        "version" | "--version" | "-V" => {
            println!("cargo-tpt {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("error: unknown command `{other}`\n");
            eprint!("{USAGE}");
            ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------------------
// cargo tpt new
// ---------------------------------------------------------------------------

fn cmd_new(args: &[String]) -> Result<(), String> {
    let mut name: Option<&str> = None;
    let mut explicit_dir: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--path" => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| "`--path` needs a directory argument".to_string())?;
                explicit_dir = Some(PathBuf::from(value));
            }
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(());
            }
            flag if flag.starts_with('-') => {
                return Err(format!("unknown option `{flag}` for `cargo tpt new`"));
            }
            value => {
                if name.is_some() {
                    return Err(format!("unexpected extra argument `{value}`"));
                }
                name = Some(value);
            }
        }
        i += 1;
    }

    let name = name.ok_or_else(|| {
        "`cargo tpt new` needs a project name, e.g. `cargo tpt new my-analysis`".to_string()
    })?;
    validate_name(name)?;

    let root = find_workspace_root();
    let (dir, crates_dir, in_examples) = match (explicit_dir, root.as_deref()) {
        // Explicit destination: point at the checkout absolutely when we know
        // where it is, otherwise leave an obvious placeholder to edit.
        (Some(base), found) => {
            let crates = match found {
                Some(r) => toml_path(&r.join("crates")),
                None => UNKNOWN_CRATES_DIR.to_string(),
            };
            (base.join(name), crates, false)
        }
        // Inside a checkout: a sibling of the other examples.
        (None, Some(r)) => (
            r.join("examples").join(name),
            "../../crates".to_string(),
            true,
        ),
        // Outside a checkout: right here, with a placeholder path.
        (None, None) => (PathBuf::from(name), UNKNOWN_CRATES_DIR.to_string(), false),
    };

    if dir_is_non_empty(&dir) {
        return Err(format!(
            "`{}` already exists and is not empty",
            dir.display()
        ));
    }

    let src = dir.join("src");
    fs::create_dir_all(&src).map_err(|e| format!("could not create `{}`: {e}", src.display()))?;

    let manifest = dir.join("Cargo.toml");
    let main_rs = src.join("main.rs");
    write_file(&manifest, &manifest_template(name, &crates_dir))?;
    write_file(&main_rs, MAIN_TEMPLATE)?;

    println!("Created TPT project `{name}`");
    println!("  {}", manifest.display());
    println!("  {}", main_rs.display());
    println!();
    println!("Dependencies wired by path: {}", SCAFFOLD_DEPS.join(", "));
    println!();
    println!("Next steps:");
    if in_examples {
        println!("  1. add \"examples/{name}\" to the `members` array in the workspace Cargo.toml");
        println!("  2. cargo run -p {name}");
    } else {
        if crates_dir == UNKNOWN_CRATES_DIR {
            println!(
                "  1. no tpt-rust6 checkout was found, so the path dependencies in\n     \
                 {} are placeholders — point them at your checkout\n     \
                 (TPT is pre-1.0 and not yet published to crates.io)",
                manifest.display()
            );
        } else {
            println!(
                "  1. review the path dependencies in {}",
                manifest.display()
            );
        }
        println!("  2. cargo run --manifest-path {}", manifest.display());
    }
    Ok(())
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("the project name must not be empty".to_string());
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(format!(
            "`{name}` is not a valid package name (use ASCII letters, digits, `-` and `_`)"
        ));
    }
    if name.starts_with(|c: char| c.is_ascii_digit() || c == '-') {
        return Err(format!(
            "`{name}` is not a valid package name (it must not start with a digit or `-`)"
        ));
    }
    Ok(())
}

fn dir_is_non_empty(dir: &Path) -> bool {
    match fs::read_dir(dir) {
        Ok(mut entries) => entries.next().is_some(),
        Err(_) => false,
    }
}

fn write_file(path: &Path, contents: &str) -> Result<(), String> {
    fs::write(path, contents).map_err(|e| format!("could not write `{}`: {e}", path.display()))
}

/// Cargo accepts forward slashes on every platform, and they need no TOML
/// escaping, unlike Windows backslashes.
fn toml_path(path: &Path) -> String {
    path.display().to_string().replace('\\', "/")
}

fn manifest_template(name: &str, crates_dir: &str) -> String {
    let mut deps = String::new();
    for dep in SCAFFOLD_DEPS {
        deps.push_str(&format!("{dep} = {{ path = \"{crates_dir}/{dep}\" }}\n"));
    }
    format!(
        "[package]\n\
         name = \"{name}\"\n\
         version = \"0.1.0\"\n\
         edition = \"2021\"\n\
         publish = false\n\
         \n\
         # TPT is pre-1.0 and is not published to crates.io yet, so its crates are\n\
         # referenced by path. Adjust these if you move this project.\n\
         [dependencies]\n\
         {deps}"
    )
}

const MAIN_TEMPLATE: &str = r#"//! Generated by `cargo tpt new`.

use tpt_omni::ndarray::{ArrayD, IxDyn};
use tpt_omni::Tensor;

fn main() {
    println!("Hello, TPT!");

    // `Tensor<f64>` is the owned N-D view over the same primitive buffers that
    // `OmniFrame` hands out zero-copy from an Arrow column.
    let values: Vec<f64> = (0..12).map(|i| f64::from(i) * 0.5).collect();
    let tensor = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3, 4]), values).expect("3 x 4 = 12 elements"),
    );

    println!("shape = {:?}", tensor.shape());
    println!("mean  = {:.4}", tensor.mean());
    println!("std   = {:.4}", tensor.std());

    // Arithmetic broadcasts automatically and reduces in parallel via Rayon.
    let centred = &tensor - tensor.mean();
    println!("centred mean = {:.4}", centred.mean());
}
"#;

// ---------------------------------------------------------------------------
// cargo tpt doctor / cargo tpt features
// ---------------------------------------------------------------------------

fn cmd_doctor() {
    println!("cargo-tpt {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("environment");
    println!(
        "  os / arch        : {} / {}",
        env::consts::OS,
        env::consts::ARCH
    );
    println!("  pointer width    : {} bits", usize::BITS);
    println!(
        "  cargo-tpt build  : {}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );

    let root = find_workspace_root();
    match &root {
        Some(r) => println!("  tpt workspace    : {}", r.display()),
        None => println!("  tpt workspace    : not found from the current directory"),
    }
    println!();
    print_feature_table(root.as_deref());
}

fn print_feature_table(root: Option<&Path>) {
    println!("optional capabilities of the TPT stack");
    println!(
        "  {:<8} {:<10} {:<9} TURNS ON",
        "FEATURE", "CRATE", "DECLARED"
    );
    for &(feature, krate, what) in FEATURES {
        let declared = match root {
            None => "unknown",
            Some(r) if crate_declares_feature(r, krate, feature) => "yes",
            Some(_) => "no",
        };
        println!("  {feature:<8} {krate:<10} {declared:<9} {what}");
    }

    println!();
    println!("None of these are compiled into cargo-tpt, so this table reports what the");
    println!("stack *declares*, not what your build enabled. They are per-crate cargo");
    println!("features; turn them on where you need them:");
    println!("  cargo build -p tpt-io   --features hdf5");
    println!("  cargo build -p tpt-doc  --features pdf");
    println!("  cargo build -p tpt-viz  --features gpu,omni");
    println!("  cargo build -p tpt-lab  --features viz");
    println!("  cargo build -p tpt-omni --features wasm --target wasm32-unknown-unknown");

    println!();
    match root {
        Some(_) => println!(
            "The DECLARED column was read from each crate's Cargo.toml in the checkout above."
        ),
        None => println!(
            "DECLARED is `unknown`: run this inside a tpt-rust6 checkout to have each\n\
             crate's Cargo.toml consulted."
        ),
    }
}

/// Does `crates/<krate>/Cargo.toml` declare `<feature>` in its `[features]`
/// table? A deliberately small scan, so the CLI needs no TOML dependency.
fn crate_declares_feature(root: &Path, krate: &str, feature: &str) -> bool {
    let manifest = root.join("crates").join(krate).join("Cargo.toml");
    let Ok(text) = fs::read_to_string(manifest) else {
        return false;
    };
    let mut in_features = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_features = trimmed == "[features]";
            continue;
        }
        if !in_features || trimmed.starts_with('#') {
            continue;
        }
        if let Some((key, _)) = trimmed.split_once('=') {
            if key.trim() == feature {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Walk up from the current directory looking for the tpt-rust6 workspace root.
fn find_workspace_root() -> Option<PathBuf> {
    let mut dir = env::current_dir().ok()?;
    loop {
        let manifest = dir.join("Cargo.toml");
        if manifest.is_file() {
            if let Ok(text) = fs::read_to_string(&manifest) {
                if text.contains("[workspace]") && text.contains("crates/tpt-omni") {
                    return Some(dir);
                }
            }
        }
        if !dir.pop() {
            return None;
        }
    }
}

// ---------------------------------------------------------------------------
// cargo tpt serve
// ---------------------------------------------------------------------------

/// `cargo tpt serve [--pkg NAME] [--port N] [--host H] [--dir DIR] [--no-build]
/// [--open]` — build a Wasm app, bundle a static host, serve it locally.
fn cmd_serve(args: &[String]) -> Result<(), String> {
    let mut pkg: Option<String> = None;
    let mut port: u16 = 8080;
    let mut host = "127.0.0.1".to_string();
    let mut dir: Option<PathBuf> = None;
    let mut no_build = false;
    let mut open = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--pkg" => {
                pkg = Some(args.get(i + 1).ok_or("`--pkg` needs a name")?.clone());
                i += 2;
            }
            "--port" => {
                let v = args.get(i + 1).ok_or("`--port` needs a number")?;
                port = v.parse().map_err(|_| format!("invalid --port `{v}`"))?;
                i += 2;
            }
            "--host" => {
                host = args.get(i + 1).ok_or("`--host` needs a value")?.clone();
                i += 2;
            }
            "--dir" => {
                dir = Some(PathBuf::from(
                    args.get(i + 1).ok_or("`--dir` needs a path")?,
                ));
                i += 2;
            }
            "--no-build" => {
                no_build = true;
                i += 1;
            }
            "--open" => {
                open = true;
                i += 1;
            }
            other => return Err(format!("unknown option `{other}` for `cargo tpt serve`")),
        }
    }

    let root = match &dir {
        Some(d) => d.clone(),
        None => {
            let base = find_workspace_root()
                .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
            base.join("target").join("tpt-serve")
        }
    };
    fs::create_dir_all(&root)
        .map_err(|e| format!("cannot create serve directory {}: {e}", root.display()))?;

    if !no_build {
        if let Some(pkg) = &pkg {
            build_wasm(pkg, &root)?;
        }
    }

    let index = root.join("index.html");
    if !index.exists() {
        let body = host_page(pkg.as_deref(), &root);
        fs::write(&index, body)
            .map_err(|e| format!("cannot write {}: {e}", index.display()))?;
    }

    if open {
        let _ = open_browser(&format!("http://{host}:{port}/"));
    }
    serve_dir(&root, &host, port)
}

/// Run `cargo build --release --target wasm32-unknown-unknown -p <pkg>` and copy
/// the resulting `.wasm` into `root`, then write a host page that loads it.
fn build_wasm(pkg: &str, root: &Path) -> Result<(), String> {
    println!("building {pkg} for wasm32-unknown-unknown ...");
    let status = std::process::Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
        .args([
            "build",
            "--release",
            "--target",
            "wasm32-unknown-unknown",
            "-p",
            pkg,
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            let ws = find_workspace_root().unwrap_or_else(|| PathBuf::from("."));
            let release_dir = ws.join("target/wasm32-unknown-unknown/release");
            // A cdylib's output file uses the lib target name, which is the
            // package name with hyphens replaced by underscores (`tpt-playground`
            // -> `tpt_playground.wasm`), so accept either form.
            let src_name = if release_dir.join(format!("{pkg}.wasm")).exists() {
                format!("{pkg}.wasm")
            } else {
                format!("{}.wasm", pkg.replace('-', "_"))
            };
            let wasm = release_dir.join(&src_name);
            if wasm.exists() {
                let dest = root.join(format!("{pkg}.wasm"));
                fs::copy(&wasm, &dest).map_err(|e| format!("copy {}: {e}", wasm.display()))?;
                println!("  copied {}", dest.display());
            } else {
                println!(
                    "  warning: expected {} but it was not found; serving the \
                     directory without a bundled wasm",
                    wasm.display()
                );
            }
            Ok(())
        }
        Ok(_) => Err(format!("`cargo build -p {pkg}` failed (non-zero exit)")),
        Err(e) => {
            println!(
                "  warning: wasm build skipped ({e}); serving the directory as-is.\n  \
                 install the target with `rustup target add wasm32-unknown-unknown` \
                 to bundle a wasm app."
            );
            Ok(())
        }
    }
}

/// A self-contained static host page.
fn host_page(pkg: Option<&str>, root: &Path) -> String {
    let base = root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("tpt-serve");
    match pkg {
        Some(pkg) => format!(
            "<!doctype html>\n<html lang=\"en\">\n<head><meta charset=\"utf-8\">\n\
             <title>TPT · {pkg}</title></head>\n<body>\n\
             <h1>TPT app: {pkg}</h1>\n\
             <p>Served by <code>cargo tpt serve</code> from <code>{base}/</code>.</p>\n\
             <script type=\"module\">\n\
             const wasm = \"{pkg}.wasm\";\n\
             fetch(wasm).then(r => r.arrayBuffer()).then(b => WebAssembly.instantiate(b))\n\
             .then(() => console.log(\"loaded \" + wasm))\n\
             .catch(e => console.error(\"wasm load failed (expected for a plain Rust \
             lib without a wasm-bindgen entry point):\", e));\n\
             </script>\n</body></html>\n"
        ),
        None => format!(
            "<!doctype html>\n<html lang=\"en\">\n<head><meta charset=\"utf-8\">\n\
             <title>TPT · serve</title></head>\n<body>\n\
             <h1>TPT serve root</h1>\n\
             <p>Served by <code>cargo tpt serve</code> from <code>{base}/</code>.</p>\n\
             <p>Re-run with <code>--pkg &lt;name&gt;</code> to build and bundle a \
             Wasm app into this directory.</p>\n</body></html>\n"
        ),
    }
}

/// `cargo tpt playground [--port N] [--host H] [--dir DIR] [--no-build]
/// [--open]` — build the Wasm `tpt-playground` runtime, bundle the editor
/// host page, and serve them over a local HTTP server.
fn cmd_playground(args: &[String]) -> Result<(), String> {
    let mut port: u16 = 8080;
    let mut host = "127.0.0.1".to_string();
    let mut dir: Option<PathBuf> = None;
    let mut no_build = false;
    let mut open = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                let v = args.get(i + 1).ok_or("`--port` needs a number")?;
                port = v.parse().map_err(|_| format!("invalid --port `{v}`"))?;
                i += 2;
            }
            "--host" => {
                host = args.get(i + 1).ok_or("`--host` needs a value")?.clone();
                i += 2;
            }
            "--dir" => {
                dir = Some(PathBuf::from(
                    args.get(i + 1).ok_or("`--dir` needs a path")?,
                ));
                i += 2;
            }
            "--no-build" => {
                no_build = true;
                i += 1;
            }
            "--open" => {
                open = true;
                i += 1;
            }
            other => return Err(format!("unknown option `{other}` for `cargo tpt playground`")),
        }
    }

    let root = match &dir {
        Some(d) => d.clone(),
        None => {
            let base = find_workspace_root()
                .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
            base.join("target").join("tpt-playground")
        }
    };
    fs::create_dir_all(&root)
        .map_err(|e| format!("cannot create playground directory {}: {e}", root.display()))?;

    if !no_build {
        build_wasm("tpt-playground", &root)?;
    }

    let index = root.join("index.html");
    if !index.exists() {
        let body = playground_page();
        fs::write(&index, body)
            .map_err(|e| format!("cannot write {}: {e}", index.display()))?;
    }

    if open {
        let _ = open_browser(&format!("http://{host}:{port}/"));
    }
    serve_dir(&root, &host, port)
}

/// The static editor host for the Try TPT playground. It loads
/// `tpt-playground.wasm` and drives it through the three C-ABI exports
/// (`tpt_alloc` / `tpt_run` / `tpt_output_*`).
fn playground_page() -> String {
    const DEFAULT_SCRIPT: &str = "\
# Try TPT — paste a .tpt script and press Run.
let a = 2 + 3 * 4
let b = a / 2
print a
print b
let m = [1, 2, 3, 4, 5]
print sum(m)
";

    format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Try TPT</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, sans-serif; margin: 0; display: grid;
         grid-template-rows: auto 1fr auto; height: 100vh; }}
  header {{ padding: 0.5rem 1rem; border-bottom: 1px solid #8884; }}
  header h1 {{ font-size: 1rem; margin: 0; }}
  header small {{ opacity: 0.6; }}
  main {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1px;
         background: #8884; min-height: 0; }}
  .pane {{ display: flex; flex-direction: column; min-height: 0; }}
  .pane > label {{ padding: 0.25rem 0.5rem; font-size: 0.8rem; opacity: 0.7; }}
  textarea, pre {{ flex: 1; margin: 0; padding: 0.5rem; font-family: ui-monospace,
                  monospace; font-size: 0.85rem; white-space: pre-wrap;
                  overflow: auto; border: 0; }}
  textarea {{ resize: none; outline: none; }}
  pre {{ background: #0001; }}
  footer {{ display: flex; gap: 0.5rem; align-items: center; padding: 0.5rem 1rem;
           border-top: 1px solid #8884; }}
  button {{ padding: 0.4rem 1rem; font-size: 0.9rem; cursor: pointer; }}
  #status {{ margin-left: auto; font-size: 0.8rem; opacity: 0.7; }}
</style>
</head>
<body>
<header><h1>Try TPT <small>— a Wasm-compiled tpt-script playground</small></h1></header>
<main>
  <div class="pane"><label for="code">script.tpt</label>
    <textarea id="code" spellcheck="false">{default}</textarea></div>
  <div class="pane"><label for="out">output</label><pre id="out"></pre></div>
</main>
<footer>
  <button id="run">Run ▶</button>
  <span id="status">loading wasm…</span>
</footer>
<script type="module">
const WASM = "tpt-playground.wasm";
let api = null;

async function load() {{
  try {{
    const mod = await WebAssembly.instantiateStreaming(fetch(WASM),
      {{}}).catch(async () => WebAssembly.instantiate(await (await fetch(WASM)).arrayBuffer(), {{}}));
    const ex = mod.instance.exports;
    if (!ex.tpt_run || !ex.tpt_alloc || !ex.tpt_output_ptr) {{
      setStatus("wasm is missing the expected exports (build with `cargo tpt playground`)");
      return;
    }}
    api = ex;
    setStatus("ready");
  }} catch (e) {{
    setStatus("failed to load " + WASM + ": " + e.message);
  }}
}}

function run() {{
  if (!api) {{ setStatus("wasm not ready"); return; }}
  const src = document.getElementById("code").value;
  const bytes = new TextEncoder().encode(src);
  const ptr = api.tpt_alloc(bytes.length);
  new Uint8Array(api.memory.buffer, ptr, bytes.length).set(bytes);
  const outLen = api.tpt_run(ptr, bytes.length);
  // The output buffer may live in freshly grown linear memory, so re-read it.
  const outPtr = api.tpt_output_ptr();
  const outBytes = new Uint8Array(api.memory.buffer, outPtr, outLen);
  document.getElementById("out").textContent = new TextDecoder().decode(outBytes);
  setStatus("ran " + bytes.length + " bytes");
}}

function setStatus(s) {{ document.getElementById("status").textContent = s; }}

document.getElementById("run").addEventListener("click", run);
document.getElementById("code").addEventListener("keydown", (e) => {{
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {{ e.preventDefault(); run(); }}
}});
load();
</script>
</body>
</html>
"#,
        default = DEFAULT_SCRIPT.replace('{', "{{").replace('}', "}}")
    )
}

/// Serve `root` over HTTP on `host:port` until interrupted.
fn serve_dir(root: &Path, host: &str, port: u16) -> Result<(), String> {
    let addr = format!("{host}:{port}");
    let listener = TcpListener::bind(&addr)
        .map_err(|e| format!("cannot bind {addr}: {e} (is it already in use?)"))?;
    println!("serving {} at http://{addr}/  (Ctrl-C to stop)", root.display());
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                if let Err(e) = handle_request(s, root) {
                    eprintln!("request error: {e}");
                }
            }
            Err(_) => break,
        }
    }
    Ok(())
}

/// Handle one HTTP request: only GET/HEAD are supported; everything else 405s.
fn handle_request(mut stream: TcpStream, root: &Path) -> std::io::Result<()> {
    let mut buf = [0u8; 1024];
    let n = stream.read(&mut buf)?;
    let request = String::from_utf8_lossy(&buf[..n]);
    let Some(line) = request.lines().next() else {
        return Ok(());
    };
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("/");

    if method != "GET" && method != "HEAD" {
        return write_response(&mut stream, 405, "text/plain", b"Method Not Allowed");
    }

    let rel = path.trim_start_matches('/');
    let rel = if rel.is_empty() { "index.html" } else { rel };
    let mut candidate = root.join(sanitize(rel));
    if candidate.is_dir() {
        candidate = candidate.join("index.html");
    }
    match fs::read(&candidate) {
        Ok(body) => {
            let ct = content_type(&candidate);
            write_response(&mut stream, 200, ct, &body)
        }
        Err(_) => write_response(&mut stream, 404, "text/plain", b"404 Not Found"),
    }
}

/// Reject path traversal: collapse `..` and drop absolute segments.
fn sanitize(rel: &str) -> PathBuf {
    let mut out = PathBuf::new();
    for part in rel.split('/') {
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." {
            out.pop();
            continue;
        }
        out.push(part);
    }
    out
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("wasm") => "application/wasm",
        Some("css") => "text/css; charset=utf-8",
        Some("json") => "application/json; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        _ => "application/octet-stream",
    }
}

fn write_response(
    stream: &mut TcpStream,
    code: u16,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let status = match code {
        200 => "200 OK",
        404 => "404 Not Found",
        405 => "405 Method Not Allowed",
        _ => "500 Internal Server Error",
    };
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

/// Best-effort open of a URL in the default browser (used by `--open`).
fn open_browser(url: &str) -> std::io::Result<()> {
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd").args(["/c", "start", "", url]).spawn()?;
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open").arg(url).spawn()?;
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        std::process::Command::new("xdg-open").arg(url).spawn()?;
    }
    Ok(())
}
