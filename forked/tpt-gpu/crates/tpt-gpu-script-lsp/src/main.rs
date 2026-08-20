use tpt_gpu_script_lsp::TptLspService;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("tpt_gpu_script_lsp=info")
        .with_writer(std::io::stderr)
        .init();

    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = TptLspService::new();
    tower_lsp::Server::new(stdin, stdout, socket)
        .serve(service)
        .await;
}
