//! The `tpt-lsp` binary: TPT Script language server over stdio.

#[tokio::main]
async fn main() {
    tpt_lsp::server::run_stdio().await;
}
