fn main() {
    #[cfg(feature = "lsp")]
    tpt_lab::lsp::run_lsp();

    #[cfg(not(feature = "lsp"))]
    {
        eprintln!("tpt-lab-lsp requires the `lsp` feature: cargo build -p tpt-lab --features lsp");
        std::process::exit(2);
    }
}
