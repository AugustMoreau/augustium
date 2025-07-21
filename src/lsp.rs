// Language Server Protocol for IDE integration
// Handles syntax highlighting, autocomplete, error checking etc.

use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Write, Read};
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::semantic::SemanticAnalyzer;


/// LSP Server state
pub struct LanguageServer {
    documents: Arc<Mutex<HashMap<String, Document>>>,
    capabilities: ServerCapabilities,
}

/// Document representation
#[derive(Debug, Clone)]
struct Document {
    #[allow(dead_code)]
    uri: String,
    content: String,
    version: i32,
    diagnostics: Vec<Diagnostic>,
}

/// LSP Server capabilities
#[derive(Debug, Serialize)]
struct ServerCapabilities {
    #[serde(rename = "textDocumentSync")]
    text_document_sync: i32,
    #[serde(rename = "completionProvider")]
    completion_provider: CompletionOptions,
    #[serde(rename = "hoverProvider")]
    hover_provider: bool,
    #[serde(rename = "definitionProvider")]
    definition_provider: bool,
    #[serde(rename = "documentFormattingProvider")]
    document_formatting_provider: bool,
    #[serde(rename = "documentSymbolProvider")]
    document_symbol_provider: bool,
}

#[derive(Debug, Serialize)]
struct CompletionOptions {
    #[serde(rename = "triggerCharacters")]
    trigger_characters: Vec<String>,
}

/// LSP Diagnostic
#[derive(Debug, Clone, Serialize)]
struct Diagnostic {
    range: Range,
    severity: i32,
    message: String,
    source: String,
}

/// LSP Range
#[derive(Debug, Clone, Serialize)]
struct Range {
    start: Position,
    end: Position,
}

/// LSP Position
#[derive(Debug, Clone, Serialize)]
struct Position {
    line: u32,
    character: u32,
}

/// LSP Request/Response types
#[derive(Debug, Deserialize)]
struct LSPMessage {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: Option<String>,
    params: Option<Value>,
    #[allow(dead_code)]
    result: Option<Value>,
    #[allow(dead_code)]
    error: Option<Value>,
}

/// Completion item
#[derive(Debug, Serialize)]
struct CompletionItem {
    label: String,
    kind: i32,
    detail: Option<String>,
    documentation: Option<String>,
    #[serde(rename = "insertText")]
    insert_text: Option<String>,
}

impl LanguageServer {
    /// Create a new Language Server instance
    pub fn new() -> Self {
        Self {
            documents: Arc::new(Mutex::new(HashMap::new())),
            capabilities: ServerCapabilities {
                text_document_sync: 1, // Full sync
                completion_provider: CompletionOptions {
                    trigger_characters: vec![".".to_string(), "::".to_string()],
                },
                hover_provider: true,
                definition_provider: true,
                document_formatting_provider: true,
                document_symbol_provider: true,
            },
        }
    }

    /// Start the LSP server
    pub fn start(&self) -> io::Result<()> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        let reader = BufReader::new(stdin);

        eprintln!("Augustium LSP Server started");

        for line in reader.lines() {
            let line = line?;
            
            // Skip empty lines
            if line.trim().is_empty() {
                continue;
            }

            // Parse Content-Length header
            if line.starts_with("Content-Length:") {
                let length: usize = line
                    .split(':')
                    .nth(1)
                    .unwrap()
                    .trim()
                    .parse()
                    .unwrap_or(0);

                // Read the JSON message
                let mut buffer = vec![0; length];
                io::stdin().read_exact(&mut buffer)?;
                let message = String::from_utf8_lossy(&buffer);

                // Process the LSP message
                if let Ok(lsp_msg) = serde_json::from_str::<LSPMessage>(&message) {
                    let response = self.handle_message(lsp_msg);
                    if let Some(response) = response {
                        let response_str = serde_json::to_string(&response).unwrap();
                        let content_length = response_str.len();
                        
                        write!(stdout, "Content-Length: {}\r\n\r\n{}", content_length, response_str)?;
                        stdout.flush()?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle incoming LSP message
    fn handle_message(&self, message: LSPMessage) -> Option<Value> {
        match message.method.as_deref() {
            Some("initialize") => self.handle_initialize(message.id),
            Some("textDocument/didOpen") => {
                self.handle_did_open(message.params);
                None
            }
            Some("textDocument/didChange") => {
                self.handle_did_change(message.params);
                None
            }
            Some("textDocument/completion") => self.handle_completion(message.id, message.params),
            Some("textDocument/hover") => self.handle_hover(message.id, message.params),
            Some("textDocument/definition") => self.handle_definition(message.id, message.params),
            Some("textDocument/formatting") => self.handle_formatting(message.id, message.params),
            Some("shutdown") => {
                json!({
                    "jsonrpc": "2.0",
                    "id": message.id,
                    "result": null
                }).into()
            }
            Some("exit") => {
                std::process::exit(0);
            }
            _ => None,
        }
    }

    /// Handle initialize request
    fn handle_initialize(&self, id: Option<Value>) -> Option<Value> {
        Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "capabilities": self.capabilities
            }
        }))
    }

    /// Handle document open
    fn handle_did_open(&self, params: Option<Value>) {
        if let Some(params) = params {
            if let Ok(text_document) = serde_json::from_value::<Value>(params) {
                if let (Some(uri), Some(text)) = (
                    text_document["textDocument"]["uri"].as_str(),
                    text_document["textDocument"]["text"].as_str(),
                ) {
                    let document = Document {
                        uri: uri.to_string(),
                        content: text.to_string(),
                        version: 1,
                        diagnostics: Vec::new(),
                    };

                    // Analyze document and generate diagnostics
                    let diagnostics = self.analyze_document(&document.content);
                    
                    let mut docs = self.documents.lock().unwrap();
                    docs.insert(uri.to_string(), document);
                    
                    // Send diagnostics
                    self.publish_diagnostics(uri, diagnostics);
                }
            }
        }
    }

    /// Handle document change
    fn handle_did_change(&self, params: Option<Value>) {
        if let Some(params) = params {
            if let Ok(change_event) = serde_json::from_value::<Value>(params) {
                if let (Some(uri), Some(changes)) = (
                    change_event["textDocument"]["uri"].as_str(),
                    change_event["contentChanges"].as_array(),
                ) {
                    if let Some(change) = changes.first() {
                        if let Some(text) = change["text"].as_str() {
                            let mut docs = self.documents.lock().unwrap();
                            if let Some(doc) = docs.get_mut(uri) {
                                doc.content = text.to_string();
                                doc.version += 1;
                                
                                // Re-analyze and update diagnostics
                                let diagnostics = self.analyze_document(&doc.content);
                                doc.diagnostics = diagnostics.clone();
                                
                                // Send updated diagnostics
                                self.publish_diagnostics(uri, diagnostics);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Handle completion request
    fn handle_completion(&self, id: Option<Value>, _params: Option<Value>) -> Option<Value> {
        let completions = vec![
            CompletionItem {
                label: "contract".to_string(),
                kind: 14, // Keyword
                detail: Some("Contract declaration".to_string()),
                documentation: Some("Define a new smart contract".to_string()),
                insert_text: Some("contract ${1:ContractName} {\n\t$0\n}".to_string()),
            },
            CompletionItem {
                label: "function".to_string(),
                kind: 3, // Function
                detail: Some("Function declaration".to_string()),
                documentation: Some("Define a new function".to_string()),
                insert_text: Some("fn ${1:function_name}(${2:params}) -> ${3:ReturnType} {\n\t$0\n}".to_string()),
            },
            CompletionItem {
                label: "pub".to_string(),
                kind: 14, // Keyword
                detail: Some("Public visibility modifier".to_string()),
                documentation: Some("Make function or field public".to_string()),
                insert_text: Some("pub ".to_string()),
            },
            CompletionItem {
                label: "state".to_string(),
                kind: 14, // Keyword
                detail: Some("Contract state declaration".to_string()),
                documentation: Some("Define contract state variables".to_string()),
                insert_text: Some("state {\n\t$0\n}".to_string()),
            },
            CompletionItem {
                label: "address".to_string(),
                kind: 25, // Type
                detail: Some("Address type".to_string()),
                documentation: Some("Ethereum address type".to_string()),
                insert_text: Some("address".to_string()),
            },
            CompletionItem {
                label: "u256".to_string(),
                kind: 25, // Type
                detail: Some("256-bit unsigned integer".to_string()),
                documentation: Some("Large unsigned integer type".to_string()),
                insert_text: Some("u256".to_string()),
            },
        ];

        Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": completions
        }))
    }

    /// Handle hover request
    fn handle_hover(&self, id: Option<Value>, _params: Option<Value>) -> Option<Value> {
        Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "contents": {
                    "kind": "markdown",
                    "value": "**Augustium Language**\n\nHover information for Augustium smart contracts."
                }
            }
        }))
    }

    /// Handle go-to-definition request
    fn handle_definition(&self, id: Option<Value>, _params: Option<Value>) -> Option<Value> {
        Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": []
        }))
    }

    /// Handle formatting request
    fn handle_formatting(&self, id: Option<Value>, _params: Option<Value>) -> Option<Value> {
        Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": []
        }))
    }

    /// Analyze document and generate diagnostics
    fn analyze_document(&self, content: &str) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        // Try to parse the document
        let mut lexer = Lexer::new(content, "<lsp>");
        match lexer.tokenize() {
            Ok(tokens) => {
                let mut parser = Parser::new(tokens);
                match parser.parse() {
                    Ok(ast) => {
                        // Try semantic analysis
                        let mut analyzer = SemanticAnalyzer::new();
                        if let Err(e) = analyzer.analyze(&ast) {
                            diagnostics.push(Diagnostic {
                                range: Range {
                                    start: Position { line: 0, character: 0 },
                                    end: Position { line: 0, character: 0 },
                                },
                                severity: 1, // Error
                                message: format!("Semantic error: {}", e),
                                source: "augustium".to_string(),
                            });
                        }
                    }
                    Err(e) => {
                        diagnostics.push(Diagnostic {
                            range: Range {
                                start: Position { line: 0, character: 0 },
                                end: Position { line: 0, character: 0 },
                            },
                            severity: 1, // Error
                            message: format!("Parse error: {}", e),
                            source: "augustium".to_string(),
                        });
                    }
                }
            }
            Err(e) => {
                diagnostics.push(Diagnostic {
                    range: Range {
                        start: Position { line: 0, character: 0 },
                        end: Position { line: 0, character: 0 },
                    },
                    severity: 1, // Error
                    message: format!("Lexical error: {}", e),
                    source: "augustium".to_string(),
                });
            }
        }

        diagnostics
    }

    /// Publish diagnostics to client
    fn publish_diagnostics(&self, uri: &str, diagnostics: Vec<Diagnostic>) {
        let notification = json!({
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": uri,
                "diagnostics": diagnostics
            }
        });

        let notification_str = serde_json::to_string(&notification).unwrap();
        let content_length = notification_str.len();
        
        print!("Content-Length: {}\r\n\r\n{}", content_length, notification_str);
        io::stdout().flush().unwrap();
    }
}

/// Start the LSP server
pub fn start_lsp_server() -> io::Result<()> {
    let server = LanguageServer::new();
    server.start()
}