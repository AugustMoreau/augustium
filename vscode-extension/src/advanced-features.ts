import * as vscode from 'vscode';
import { LanguageClient } from 'vscode-languageclient/node';

export class AdvancedFeatures {
    private client: LanguageClient;
    private diagnosticCollection: vscode.DiagnosticCollection;

    constructor(client: LanguageClient) {
        this.client = client;
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection('augustium');
    }

    public activate(context: vscode.ExtensionContext) {
        this.registerCommands(context);
        this.setupRealTimeErrorHighlighting(context);
        this.setupMLTypeCompletion(context);
        this.setupRefactoringTools(context);
        this.setupCodeAnalysis(context);
    }

    private registerCommands(context: vscode.ExtensionContext) {
        // Format document command
        context.subscriptions.push(
            vscode.commands.registerCommand('augustium.formatDocument', async () => {
                const editor = vscode.window.activeTextEditor;
                if (!editor || editor.document.languageId !== 'augustium') {
                    return;
                }

                const config = vscode.workspace.getConfiguration('augustium.formatter');
                const indentSize = config.get<number>('indentSize', 4);
                const maxLineLength = config.get<number>('maxLineLength', 100);

                try {
                    const result = await this.client.sendRequest('augustium/format', {
                        textDocument: { uri: editor.document.uri.toString() },
                        options: {
                            indentSize,
                            maxLineLength
                        }
                    });

                    if (result) {
                        const edit = new vscode.WorkspaceEdit();
                        const fullRange = new vscode.Range(
                            editor.document.positionAt(0),
                            editor.document.positionAt(editor.document.getText().length)
                        );
                        edit.replace(editor.document.uri, fullRange, result.formattedText);
                        await vscode.workspace.applyEdit(edit);
                    }
                } catch (error) {
                    vscode.window.showErrorMessage(`Formatting failed: ${error}`);
                }
            })
        );

        // Extract function refactoring
        context.subscriptions.push(
            vscode.commands.registerCommand('augustium.refactor.extractFunction', async () => {
                const editor = vscode.window.activeTextEditor;
                if (!editor || editor.document.languageId !== 'augustium') {
                    return;
                }

                const selection = editor.selection;
                if (selection.isEmpty) {
                    vscode.window.showWarningMessage('Please select code to extract into a function');
                    return;
                }

                const functionName = await vscode.window.showInputBox({
                    prompt: 'Enter function name',
                    validateInput: (value) => {
                        if (!value || !/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(value)) {
                            return 'Please enter a valid function name';
                        }
                        return null;
                    }
                });

                if (!functionName) return;

                try {
                    const result = await this.client.sendRequest('augustium/refactor/extractFunction', {
                        textDocument: { uri: editor.document.uri.toString() },
                        range: {
                            start: { line: selection.start.line, character: selection.start.character },
                            end: { line: selection.end.line, character: selection.end.character }
                        },
                        functionName
                    });

                    if (result && result.edits) {
                        const edit = new vscode.WorkspaceEdit();
                        for (const textEdit of result.edits) {
                            const range = new vscode.Range(
                                textEdit.range.start.line,
                                textEdit.range.start.character,
                                textEdit.range.end.line,
                                textEdit.range.end.character
                            );
                            edit.replace(editor.document.uri, range, textEdit.newText);
                        }
                        await vscode.workspace.applyEdit(edit);
                    }
                } catch (error) {
                    vscode.window.showErrorMessage(`Extract function failed: ${error}`);
                }
            })
        );

        // Gas optimization analysis
        context.subscriptions.push(
            vscode.commands.registerCommand('augustium.analysis.gasOptimization', async () => {
                const editor = vscode.window.activeTextEditor;
                if (!editor || editor.document.languageId !== 'augustium') {
                    return;
                }

                try {
                    const result = await this.client.sendRequest('augustium/analysis/gas', {
                        textDocument: { uri: editor.document.uri.toString() }
                    });

                    if (result && result.suggestions) {
                        const panel = vscode.window.createWebviewPanel(
                            'augustiumGasAnalysis',
                            'Gas Optimization Analysis',
                            vscode.ViewColumn.Two,
                            { enableScripts: true }
                        );

                        panel.webview.html = this.generateGasAnalysisHTML(result.suggestions);
                    }
                } catch (error) {
                    vscode.window.showErrorMessage(`Gas analysis failed: ${error}`);
                }
            })
        );

        // Security audit
        context.subscriptions.push(
            vscode.commands.registerCommand('augustium.analysis.securityAudit', async () => {
                const editor = vscode.window.activeTextEditor;
                if (!editor || editor.document.languageId !== 'augustium') {
                    return;
                }

                try {
                    const result = await this.client.sendRequest('augustium/analysis/security', {
                        textDocument: { uri: editor.document.uri.toString() }
                    });

                    if (result && result.issues) {
                        const panel = vscode.window.createWebviewPanel(
                            'augustiumSecurityAudit',
                            'Security Audit Results',
                            vscode.ViewColumn.Two,
                            { enableScripts: true }
                        );

                        panel.webview.html = this.generateSecurityAuditHTML(result.issues);
                    }
                } catch (error) {
                    vscode.window.showErrorMessage(`Security audit failed: ${error}`);
                }
            })
        );
    }

    private setupRealTimeErrorHighlighting(context: vscode.ExtensionContext) {
        const config = vscode.workspace.getConfiguration('augustium');
        if (!config.get<boolean>('realTimeErrors', true)) {
            return;
        }

        // Listen for document changes and provide real-time diagnostics
        context.subscriptions.push(
            vscode.workspace.onDidChangeTextDocument(async (event) => {
                if (event.document.languageId !== 'augustium') {
                    return;
                }

                // Debounce the diagnostics request
                setTimeout(async () => {
                    try {
                        const result = await this.client.sendRequest('augustium/diagnostics', {
                            textDocument: { uri: event.document.uri.toString() },
                            text: event.document.getText()
                        });

                        if (result && result.diagnostics) {
                            const diagnostics: vscode.Diagnostic[] = result.diagnostics.map((diag: any) => {
                                const range = new vscode.Range(
                                    diag.range.start.line,
                                    diag.range.start.character,
                                    diag.range.end.line,
                                    diag.range.end.character
                                );
                                
                                let severity = vscode.DiagnosticSeverity.Information;
                                switch (diag.severity) {
                                    case 'error': severity = vscode.DiagnosticSeverity.Error; break;
                                    case 'warning': severity = vscode.DiagnosticSeverity.Warning; break;
                                    case 'info': severity = vscode.DiagnosticSeverity.Information; break;
                                }

                                return new vscode.Diagnostic(range, diag.message, severity);
                            });

                            this.diagnosticCollection.set(event.document.uri, diagnostics);
                        }
                    } catch (error) {
                        console.error('Real-time diagnostics failed:', error);
                    }
                }, 500); // 500ms debounce
            })
        );
    }

    private setupMLTypeCompletion(context: vscode.ExtensionContext) {
        const config = vscode.workspace.getConfiguration('augustium.mlTypes');
        if (!config.get<boolean>('completionEnabled', true)) {
            return;
        }

        // Register ML-specific completion provider
        context.subscriptions.push(
            vscode.languages.registerCompletionItemProvider(
                'augustium',
                {
                    provideCompletionItems: async (document, position, token) => {
                        const line = document.lineAt(position).text;
                        const prefix = line.substring(0, position.character);

                        // Check if we're in an ML context
                        if (prefix.includes('ml::') || prefix.includes('MLModel') || prefix.includes('Tensor')) {
                            try {
                                const result = await this.client.sendRequest('augustium/completion/ml', {
                                    textDocument: { uri: document.uri.toString() },
                                    position: { line: position.line, character: position.character },
                                    context: prefix
                                });

                                if (result && result.items) {
                                    return result.items.map((item: any) => {
                                        const completion = new vscode.CompletionItem(item.label, vscode.CompletionItemKind.Method);
                                        completion.detail = item.detail;
                                        completion.documentation = new vscode.MarkdownString(item.documentation);
                                        completion.insertText = item.insertText;
                                        return completion;
                                    });
                                }
                            } catch (error) {
                                console.error('ML completion failed:', error);
                            }
                        }

                        return [];
                    }
                },
                ':', ':' // Trigger on ::
            )
        );
    }

    private setupRefactoringTools(context: vscode.ExtensionContext) {
        // Rename symbol command
        context.subscriptions.push(
            vscode.commands.registerCommand('augustium.refactor.renameSymbol', async () => {
                const editor = vscode.window.activeTextEditor;
                if (!editor || editor.document.languageId !== 'augustium') {
                    return;
                }

                const position = editor.selection.active;
                const wordRange = editor.document.getWordRangeAtPosition(position);
                if (!wordRange) {
                    vscode.window.showWarningMessage('No symbol found at cursor position');
                    return;
                }

                const currentName = editor.document.getText(wordRange);
                const newName = await vscode.window.showInputBox({
                    prompt: `Rename symbol '${currentName}' to:`,
                    value: currentName,
                    validateInput: (value) => {
                        if (!value || !/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(value)) {
                            return 'Please enter a valid identifier';
                        }
                        return null;
                    }
                });

                if (!newName || newName === currentName) return;

                try {
                    const result = await this.client.sendRequest('augustium/refactor/rename', {
                        textDocument: { uri: editor.document.uri.toString() },
                        position: { line: position.line, character: position.character },
                        newName
                    });

                    if (result && result.changes) {
                        const edit = new vscode.WorkspaceEdit();
                        for (const [uri, edits] of Object.entries(result.changes)) {
                            for (const textEdit of edits as any[]) {
                                const range = new vscode.Range(
                                    textEdit.range.start.line,
                                    textEdit.range.start.character,
                                    textEdit.range.end.line,
                                    textEdit.range.end.character
                                );
                                edit.replace(vscode.Uri.parse(uri), range, textEdit.newText);
                            }
                        }
                        await vscode.workspace.applyEdit(edit);
                    }
                } catch (error) {
                    vscode.window.showErrorMessage(`Rename failed: ${error}`);
                }
            })
        );
    }

    private setupCodeAnalysis(context: vscode.ExtensionContext) {
        // Code lens provider for gas costs
        context.subscriptions.push(
            vscode.languages.registerCodeLensProvider(
                'augustium',
                {
                    provideCodeLenses: async (document, token) => {
                        try {
                            const result = await this.client.sendRequest('augustium/codeLens/gasCosts', {
                                textDocument: { uri: document.uri.toString() }
                            });

                            if (result && result.lenses) {
                                return result.lenses.map((lens: any) => {
                                    const range = new vscode.Range(
                                        lens.range.start.line,
                                        lens.range.start.character,
                                        lens.range.end.line,
                                        lens.range.end.character
                                    );
                                    
                                    const codeLens = new vscode.CodeLens(range);
                                    codeLens.command = {
                                        title: `⛽ ${lens.gasCost} gas`,
                                        command: 'augustium.showGasDetails',
                                        arguments: [lens.details]
                                    };
                                    return codeLens;
                                });
                            }
                        } catch (error) {
                            console.error('Code lens failed:', error);
                        }
                        return [];
                    }
                }
            )
        );
    }

    private generateGasAnalysisHTML(suggestions: any[]): string {
        return `
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Gas Optimization Analysis</title>
                <style>
                    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
                    .suggestion { margin: 10px 0; padding: 10px; border-left: 4px solid #007acc; background: #f8f8f8; }
                    .high { border-left-color: #d73a49; }
                    .medium { border-left-color: #f66a0a; }
                    .low { border-left-color: #28a745; }
                </style>
            </head>
            <body>
                <h1>Gas Optimization Suggestions</h1>
                ${suggestions.map(s => `
                    <div class="suggestion ${s.priority}">
                        <h3>${s.title}</h3>
                        <p>${s.description}</p>
                        <p><strong>Potential savings:</strong> ${s.gasSavings} gas</p>
                    </div>
                `).join('')}
            </body>
            </html>
        `;
    }

    private generateSecurityAuditHTML(issues: any[]): string {
        return `
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Security Audit Results</title>
                <style>
                    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
                    .issue { margin: 10px 0; padding: 10px; border-left: 4px solid #007acc; background: #f8f8f8; }
                    .critical { border-left-color: #d73a49; }
                    .high { border-left-color: #f66a0a; }
                    .medium { border-left-color: #ffd33d; }
                    .low { border-left-color: #28a745; }
                </style>
            </head>
            <body>
                <h1>Security Audit Results</h1>
                ${issues.map(issue => `
                    <div class="issue ${issue.severity}">
                        <h3>${issue.title}</h3>
                        <p>${issue.description}</p>
                        <p><strong>Location:</strong> Line ${issue.line}</p>
                        <p><strong>Recommendation:</strong> ${issue.recommendation}</p>
                    </div>
                `).join('')}
            </body>
            </html>
        `;
    }
}
