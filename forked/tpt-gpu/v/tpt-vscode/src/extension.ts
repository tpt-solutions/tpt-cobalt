import * as vscode from 'vscode';
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: vscode.ExtensionContext) {
  const serverPath = vscode.workspace.getConfiguration('tpt-gpu-lsp')
    .get<string>('serverPath', 'tpt-gpu-lsp');

  const serverOptions: ServerOptions = {
    run: { command: serverPath },
    debug: { command: serverPath },
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: 'file', language: 'tpt' }],
    synchronize: {
      fileEvents: vscode.workspace.createFileSystemWatcher('**/*.tpts'),
    },
    outputChannelName: 'TPT Script LSP',
    traceOutputChannel: vscode.window.createOutputChannel('TPT Script LSP Trace'),
  };

  client = new LanguageClient(
    'tpt-gpu-lsp',
    'TPT Script Language Server',
    serverOptions,
    clientOptions
  );

  client.start();

  // Register format command
  const formatCommand = vscode.commands.registerCommand('tpt.format', async () => {
    const editor = vscode.window.activeTextEditor;
    if (editor && editor.document.languageId === 'tpt') {
      await vscode.commands.executeCommand('editor.action.formatDocument');
    }
  });

  // Register lint command
  const lintCommand = vscode.commands.registerCommand('tpt.lint', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'tpt') {
      return;
    }
    const uri = editor.document.uri;
    const diagnostics = vscode.languages.getDiagnostics(uri);
    const warningCount = diagnostics.length;
    if (warningCount === 0) {
      vscode.window.showInformationMessage('TPT Script: No issues found!');
    } else {
      vscode.window.showWarningMessage(
        `TPT Script: ${warningCount} issue(s) found. Check the Problems panel.`
      );
    }
  });

  context.subscriptions.push(formatCommand, lintCommand);
}

export function deactivate(): Thenable<void> | undefined {
  if (!client) {
    return undefined;
  }
  return client.stop();
}