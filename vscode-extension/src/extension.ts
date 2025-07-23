import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind
} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: vscode.ExtensionContext) {
    console.log('Augustium extension is now active!');

    // Get configuration
    const config = vscode.workspace.getConfiguration('augustium');
    const lspPath = config.get<string>('lsp.path', 'augustium-lsp');
    const traceLevel = config.get<string>('lsp.trace.server', 'off');

    // Server options
    const serverOptions: ServerOptions = {
        command: lspPath,
        args: ['--stdio'],
        transport: TransportKind.stdio
    };

    // Client options
    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'augustium' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.aug')
        },
        outputChannelName: 'Augustium Language Server',
        traceOutputChannel: vscode.window.createOutputChannel('Augustium LSP Trace')
    };

    // Create and start the language client
    client = new LanguageClient(
        'augustium-lsp',
        'Augustium Language Server',
        serverOptions,
        clientOptions
    );

    // Set trace level
    if (traceLevel !== 'off') {
        client.setTrace(traceLevel as any);
    }

    // Start the client and server
    client.start().then(() => {
        console.log('Augustium LSP client started successfully');
    }).catch((error) => {
        console.error('Failed to start Augustium LSP client:', error);
        vscode.window.showErrorMessage(
            `Failed to start Augustium Language Server. Make sure 'augustium-lsp' is installed and in your PATH. Error: ${error.message}`
        );
    });

    // Register commands
    const restartCommand = vscode.commands.registerCommand('augustium.restart', async () => {
        if (client) {
            await client.stop();
            await client.start();
            vscode.window.showInformationMessage('Augustium Language Server restarted');
        }
    });

    context.subscriptions.push(restartCommand);
}

export async function deactivate(): Promise<void> {
    if (client) {
        await client.stop();
    }
}
