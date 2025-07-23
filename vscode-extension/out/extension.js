"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.deactivate = exports.activate = void 0;
const vscode = require("vscode");
const node_1 = require("vscode-languageclient/node");
let client;
function activate(context) {
    console.log('Augustium extension is now active!');
    // Get configuration
    const config = vscode.workspace.getConfiguration('augustium');
    const lspPath = config.get('lsp.path', 'augustium-lsp');
    const traceLevel = config.get('lsp.trace.server', 'off');
    // Server options
    const serverOptions = {
        command: lspPath,
        args: ['--stdio'],
        transport: node_1.TransportKind.stdio
    };
    // Client options
    const clientOptions = {
        documentSelector: [{ scheme: 'file', language: 'augustium' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.aug')
        },
        outputChannelName: 'Augustium Language Server',
        traceOutputChannel: vscode.window.createOutputChannel('Augustium LSP Trace')
    };
    // Create and start the language client
    client = new node_1.LanguageClient('augustium-lsp', 'Augustium Language Server', serverOptions, clientOptions);
    // Set trace level
    if (traceLevel !== 'off') {
        client.setTrace(traceLevel);
    }
    // Start the client and server
    client.start().then(() => {
        console.log('Augustium LSP client started successfully');
    }).catch((error) => {
        console.error('Failed to start Augustium LSP client:', error);
        vscode.window.showErrorMessage(`Failed to start Augustium Language Server. Make sure 'augustium-lsp' is installed and in your PATH. Error: ${error.message}`);
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
exports.activate = activate;
async function deactivate() {
    if (client) {
        await client.stop();
    }
}
exports.deactivate = deactivate;
//# sourceMappingURL=extension.js.map