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
    fn handle_completion(&self, id: Option<Value>, params: Option<Value>) -> Option<Value> {
        let mut completions = Vec::new();
        
        // Extract position and document URI from params
        let (line, character, uri) = if let Some(ref params) = params {
            let position = params.get("position");
            let text_document = params.get("textDocument");
            
            let line = position
                .and_then(|p| p.get("line"))
                .and_then(|l| l.as_u64())
                .unwrap_or(0) as u32;
            let character = position
                .and_then(|p| p.get("character"))
                .and_then(|c| c.as_u64())
                .unwrap_or(0) as u32;
            let uri = text_document
                .and_then(|td| td.get("uri"))
                .and_then(|u| u.as_str())
                .unwrap_or("");
                
            (line, character, uri)
        } else {
            (0, 0, "")
        };
        
        // Get document content for context-aware completion
        let context = if !uri.is_empty() {
            self.documents.lock().unwrap()
                .get(uri)
                .map(|doc| doc.content.clone())
                .unwrap_or_default()
        } else {
            String::new()
        };
        
        // Analyze context to provide relevant completions
        let current_line = context.lines().nth(line as usize).unwrap_or("");
        let prefix = &current_line[..character.min(current_line.len() as u32) as usize];
        
        // Context-aware completions
        if prefix.trim().is_empty() || prefix.ends_with('{') {
            // Top-level or block-level completions
            completions.extend(self.get_top_level_completions());
        }
        
        if prefix.contains("contract") && !prefix.contains('{') {
            // Inside contract declaration
            completions.extend(self.get_contract_completions());
        }
        
        if prefix.contains("fn") || prefix.contains("function") {
            // Function-related completions
            completions.extend(self.get_function_completions());
        }
        
        // Type completions
        if prefix.ends_with(':') || prefix.contains(":") {
            completions.extend(self.get_type_completions());
        }
        
        // Always include basic completions
        completions.extend(self.get_basic_completions());
        
        // Remove duplicates
        completions.sort_by(|a, b| a.label.cmp(&b.label));
        completions.dedup_by(|a, b| a.label == b.label);

        Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": completions
        }))
    }
    
    fn get_top_level_completions(&self) -> Vec<CompletionItem> {
        vec![
            CompletionItem {
                label: "contract".to_string(),
                kind: 14, // Keyword
                detail: Some("Contract declaration".to_string()),
                documentation: Some("Define a new smart contract\n\n```augustium\ncontract MyContract {\n    // Contract body\n}\n```".to_string()),
                insert_text: Some("contract ${1:ContractName} {\n\t$0\n}".to_string()),
            },
            CompletionItem {
                label: "import".to_string(),
                kind: 14, // Keyword
                detail: Some("Import statement".to_string()),
                documentation: Some("Import external modules or contracts".to_string()),
                insert_text: Some("import \"${1:module_path}\"".to_string()),
            },
            CompletionItem {
                label: "use".to_string(),
                kind: 14, // Keyword
                detail: Some("Use statement".to_string()),
                documentation: Some("Bring items into scope".to_string()),
                insert_text: Some("use ${1:path}".to_string()),
            },
        ]
    }
    
    fn get_contract_completions(&self) -> Vec<CompletionItem> {
        vec![
            CompletionItem {
                label: "state".to_string(),
                kind: 14, // Keyword
                detail: Some("Contract state declaration".to_string()),
                documentation: Some("Define contract state variables\n\n```augustium\nstate {\n    balance: u256,\n    owner: address\n}\n```".to_string()),
                insert_text: Some("state {\n\t$0\n}".to_string()),
            },
            CompletionItem {
                label: "constructor".to_string(),
                kind: 3, // Function
                detail: Some("Contract constructor".to_string()),
                documentation: Some("Define contract initialization function".to_string()),
                insert_text: Some("constructor(${1:params}) {\n\t$0\n}".to_string()),
            },
            CompletionItem {
                label: "event".to_string(),
                kind: 14, // Keyword
                detail: Some("Event declaration".to_string()),
                documentation: Some("Define contract event for logging".to_string()),
                insert_text: Some("event ${1:EventName}(${2:params})".to_string()),
            },
        ]
    }
    
    fn get_function_completions(&self) -> Vec<CompletionItem> {
        vec![
            CompletionItem {
                label: "pub".to_string(),
                kind: 14, // Keyword
                detail: Some("Public visibility modifier".to_string()),
                documentation: Some("Make function publicly accessible".to_string()),
                insert_text: Some("pub ".to_string()),
            },
            CompletionItem {
                label: "payable".to_string(),
                kind: 14, // Keyword
                detail: Some("Payable function modifier".to_string()),
                documentation: Some("Allow function to receive Ether".to_string()),
                insert_text: Some("payable ".to_string()),
            },
            CompletionItem {
                label: "view".to_string(),
                kind: 14, // Keyword
                detail: Some("View function modifier".to_string()),
                documentation: Some("Function that doesn't modify state".to_string()),
                insert_text: Some("view ".to_string()),
            },
        ]
    }
    
    fn get_type_completions(&self) -> Vec<CompletionItem> {
        vec![
            CompletionItem {
                label: "address".to_string(),
                kind: 25, // Type
                detail: Some("Address type".to_string()),
                documentation: Some("Ethereum address type (20 bytes)".to_string()),
                insert_text: Some("address".to_string()),
            },
            CompletionItem {
                label: "u256".to_string(),
                kind: 25, // Type
                detail: Some("256-bit unsigned integer".to_string()),
                documentation: Some("Large unsigned integer type".to_string()),
                insert_text: Some("u256".to_string()),
            },
            CompletionItem {
                label: "u128".to_string(),
                kind: 25, // Type
                detail: Some("128-bit unsigned integer".to_string()),
                documentation: Some("Medium unsigned integer type".to_string()),
                insert_text: Some("u128".to_string()),
            },
            CompletionItem {
                label: "u64".to_string(),
                kind: 25, // Type
                detail: Some("64-bit unsigned integer".to_string()),
                documentation: Some("Standard unsigned integer type".to_string()),
                insert_text: Some("u64".to_string()),
            },
            CompletionItem {
                label: "u32".to_string(),
                kind: 25, // Type
                detail: Some("32-bit unsigned integer".to_string()),
                documentation: Some("Small unsigned integer type".to_string()),
                insert_text: Some("u32".to_string()),
            },
            CompletionItem {
                label: "bool".to_string(),
                kind: 25, // Type
                detail: Some("Boolean type".to_string()),
                documentation: Some("True or false value".to_string()),
                insert_text: Some("bool".to_string()),
            },
            CompletionItem {
                label: "string".to_string(),
                kind: 25, // Type
                detail: Some("String type".to_string()),
                documentation: Some("UTF-8 encoded text".to_string()),
                insert_text: Some("string".to_string()),
            },
            CompletionItem {
                label: "bytes".to_string(),
                kind: 25, // Type
                detail: Some("Dynamic byte array".to_string()),
                documentation: Some("Variable-length byte sequence".to_string()),
                insert_text: Some("bytes".to_string()),
            },
        ]
    }
    
    fn get_basic_completions(&self) -> Vec<CompletionItem> {
        vec![
            CompletionItem {
                label: "fn".to_string(),
                kind: 14, // Keyword
                detail: Some("Function declaration".to_string()),
                documentation: Some("Define a new function\n\n```augustium\nfn function_name(param: Type) -> ReturnType {\n    // Function body\n}\n```".to_string()),
                insert_text: Some("fn ${1:function_name}(${2:params}) -> ${3:ReturnType} {\n\t$0\n}".to_string()),
            },
            CompletionItem {
                label: "let".to_string(),
                kind: 14, // Keyword
                detail: Some("Variable declaration".to_string()),
                documentation: Some("Declare a new variable".to_string()),
                insert_text: Some("let ${1:variable_name}: ${2:Type} = ${3:value}".to_string()),
            },
            CompletionItem {
                label: "if".to_string(),
                kind: 14, // Keyword
                detail: Some("Conditional statement".to_string()),
                documentation: Some("Execute code conditionally".to_string()),
                insert_text: Some("if ${1:condition} {\n\t$0\n}".to_string()),
            },
            CompletionItem {
                label: "for".to_string(),
                kind: 14, // Keyword
                detail: Some("For loop".to_string()),
                documentation: Some("Iterate over a range or collection".to_string()),
                insert_text: Some("for ${1:item} in ${2:iterable} {\n\t$0\n}".to_string()),
            },
            CompletionItem {
                label: "while".to_string(),
                kind: 14, // Keyword
                detail: Some("While loop".to_string()),
                documentation: Some("Loop while condition is true".to_string()),
                insert_text: Some("while ${1:condition} {\n\t$0\n}".to_string()),
            },
            CompletionItem {
                label: "return".to_string(),
                kind: 14, // Keyword
                detail: Some("Return statement".to_string()),
                documentation: Some("Return a value from function".to_string()),
                insert_text: Some("return ${1:value}".to_string()),
            },
        ]
    }

    /// Handle hover request
    fn handle_hover(&self, id: Option<Value>, params: Option<Value>) -> Option<Value> {
        let hover_content = if let Some(params) = params {
            let position = params.get("position");
            let text_document = params.get("textDocument");
            
            let line = position
                .and_then(|p| p.get("line"))
                .and_then(|l| l.as_u64())
                .unwrap_or(0) as u32;
            let character = position
                .and_then(|p| p.get("character"))
                .and_then(|c| c.as_u64())
                .unwrap_or(0) as u32;
            let uri = text_document
                .and_then(|td| td.get("uri"))
                .and_then(|u| u.as_str())
                .unwrap_or("");
            
            // Get document content
            let content = if !uri.is_empty() {
                self.documents.lock().unwrap()
                    .get(uri)
                    .map(|doc| doc.content.clone())
                    .unwrap_or_default()
            } else {
                String::new()
            };
            
            self.get_hover_info(&content, line, character)
        } else {
            "**Augustium Language**\n\nHover information for Augustium smart contracts.".to_string()
        };
        
        Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "contents": {
                    "kind": "markdown",
                    "value": hover_content
                }
            }
        }))
    }
    
    fn get_hover_info(&self, content: &str, line: u32, character: u32) -> String {
        let lines: Vec<&str> = content.lines().collect();
        if line as usize >= lines.len() {
            return "**Augustium Language**\n\nNo information available.".to_string();
        }
        
        let current_line = lines[line as usize];
        let word = self.get_word_at_position(current_line, character);
        
        match word.as_str() {
            "contract" => {
                "**contract** *(keyword)*\n\nDefines a smart contract.\n\n```augustium\ncontract MyContract {\n    // Contract implementation\n}\n```\n\nContracts are the fundamental building blocks of Augustium applications.".to_string()
            }
            "fn" | "function" => {
                "**fn** *(keyword)*\n\nDefines a function.\n\n```augustium\nfn function_name(param: Type) -> ReturnType {\n    // Function body\n}\n```\n\nFunctions encapsulate reusable code logic.".to_string()
            }
            "state" => {
                "**state** *(keyword)*\n\nDefines contract state variables.\n\n```augustium\nstate {\n    balance: u256,\n    owner: address\n}\n```\n\nState variables persist between function calls.".to_string()
            }
            "pub" => {
                "**pub** *(keyword)*\n\nPublic visibility modifier.\n\nMakes functions or state variables accessible from outside the contract.".to_string()
            }
            "payable" => {
                "**payable** *(modifier)*\n\nAllows a function to receive Ether.\n\n```augustium\npub payable fn deposit() {\n    // Function can receive Ether\n}\n```".to_string()
            }
            "view" => {
                "**view** *(modifier)*\n\nIndicates a function that doesn't modify state.\n\n```augustium\npub view fn get_balance() -> u256 {\n    return self.balance;\n}\n```".to_string()
            }
            "address" => {
                "**address** *(type)*\n\nEthereum address type (20 bytes).\n\n```augustium\nlet recipient: address = 0x742d35Cc6634C0532925a3b8D4C0C8b3C2e1e1e1;\n```\n\nUsed to represent Ethereum addresses.".to_string()
            }
            "u256" => {
                "**u256** *(type)*\n\n256-bit unsigned integer.\n\n```augustium\nlet large_number: u256 = 1000000000000000000;\n```\n\nCommonly used for token amounts and large numbers.".to_string()
            }
            "u128" => {
                "**u128** *(type)*\n\n128-bit unsigned integer.\n\n```augustium\nlet medium_number: u128 = 1000000;\n```".to_string()
            }
            "u64" => {
                "**u64** *(type)*\n\n64-bit unsigned integer.\n\n```augustium\nlet number: u64 = 1000;\n```".to_string()
            }
            "u32" => {
                "**u32** *(type)*\n\n32-bit unsigned integer.\n\n```augustium\nlet small_number: u32 = 100;\n```".to_string()
            }
            "bool" => {
                "**bool** *(type)*\n\nBoolean type (true or false).\n\n```augustium\nlet is_active: bool = true;\n```".to_string()
            }
            "string" => {
                "**string** *(type)*\n\nUTF-8 encoded text.\n\n```augustium\nlet message: string = \"Hello, Augustium!\";\n```".to_string()
            }
            "bytes" => {
                "**bytes** *(type)*\n\nDynamic byte array.\n\n```augustium\nlet data: bytes = [0x01, 0x02, 0x03];\n```".to_string()
            }
            "let" => {
                "**let** *(keyword)*\n\nDeclares a variable.\n\n```augustium\nlet variable_name: Type = value;\n```".to_string()
            }
            "if" => {
                "**if** *(keyword)*\n\nConditional statement.\n\n```augustium\nif condition {\n    // Execute if true\n}\n```".to_string()
            }
            "for" => {
                "**for** *(keyword)*\n\nFor loop iteration.\n\n```augustium\nfor item in collection {\n    // Process each item\n}\n```".to_string()
            }
            "while" => {
                "**while** *(keyword)*\n\nWhile loop.\n\n```augustium\nwhile condition {\n    // Loop body\n}\n```".to_string()
            }
            "return" => {
                "**return** *(keyword)*\n\nReturns a value from a function.\n\n```augustium\nreturn value;\n```".to_string()
            }
            "constructor" => {
                "**constructor** *(function)*\n\nContract initialization function.\n\n```augustium\nconstructor(initial_value: u256) {\n    self.value = initial_value;\n}\n```\n\nCalled once when the contract is deployed.".to_string()
            }
            "event" => {
                "**event** *(keyword)*\n\nDefines a contract event for logging.\n\n```augustium\nevent Transfer(from: address, to: address, amount: u256);\n```\n\nEvents are emitted to log important contract activities.".to_string()
            }
            "import" => {
                "**import** *(keyword)*\n\nImports external modules or contracts.\n\n```augustium\nimport \"./other_contract.aug\";\n```".to_string()
            }
            "use" => {
                "**use** *(keyword)*\n\nBrings items into scope.\n\n```augustium\nuse std::collections::HashMap;\n```".to_string()
            }
            _ => {
                if word.is_empty() {
                    "**Augustium Language**\n\nSmart contract programming language for blockchain development.".to_string()
                } else {
                    format!("**{}**\n\nNo documentation available for this symbol.", word)
                }
            }
        }
    }
    
    fn get_word_at_position(&self, line: &str, character: u32) -> String {
        let chars: Vec<char> = line.chars().collect();
        let pos = character as usize;
        
        if pos >= chars.len() {
            return String::new();
        }
        
        // Find word boundaries
        let mut start = pos;
        let mut end = pos;
        
        // Move start backwards to find word start
        while start > 0 && (chars[start - 1].is_alphanumeric() || chars[start - 1] == '_') {
            start -= 1;
        }
        
        // Move end forwards to find word end
        while end < chars.len() && (chars[end].is_alphanumeric() || chars[end] == '_') {
            end += 1;
        }
        
        chars[start..end].iter().collect()
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
    fn handle_formatting(&self, id: Option<Value>, params: Option<Value>) -> Option<Value> {
        let text_edits = if let Some(params) = params {
            let text_document = params.get("textDocument");
            let uri = text_document
                .and_then(|td| td.get("uri"))
                .and_then(|u| u.as_str())
                .unwrap_or("");
            
            // Get document content
            let content = if !uri.is_empty() {
                self.documents.lock().unwrap()
                    .get(uri)
                    .map(|doc| doc.content.clone())
                    .unwrap_or_default()
            } else {
                String::new()
            };
            
            self.format_document(&content)
        } else {
            Vec::new()
        };
        
        Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": text_edits
        }))
    }
    
    fn format_document(&self, content: &str) -> Vec<Value> {
        let lines: Vec<&str> = content.lines().collect();
        let mut formatted_lines = Vec::new();
        let mut indent_level: i32 = 0;
        let indent_size = 4; // 4 spaces per indent level
        
        for line in &lines {
            let trimmed = line.trim();
            
            // Decrease indent for closing braces
            if trimmed.starts_with('}') {
                indent_level = indent_level.saturating_sub(1);
            }
            
            // Format the line with proper indentation
            let formatted_line = if trimmed.is_empty() {
                String::new()
            } else {
                format!("{}{}", " ".repeat((indent_level * indent_size) as usize), trimmed)
            };
            
            formatted_lines.push(formatted_line);
            
            // Increase indent for opening braces
            if trimmed.ends_with('{') {
                indent_level += 1;
            }
        }
        
        let formatted_content = formatted_lines.join("\n");
        
        // Return a single text edit that replaces the entire document
        if formatted_content != content {
            vec![json!({
                "range": {
                    "start": { "line": 0, "character": 0 },
                    "end": { "line": lines.len(), "character": 0 }
                },
                "newText": formatted_content
            })]
        } else {
            Vec::new()
        }
    }

    /// Analyze document and generate diagnostics
    fn analyze_document(&self, content: &str) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        
        // Basic syntax validation
        diagnostics.extend(self.check_basic_syntax(content));
        
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
                        
                        // Additional semantic checks
                        diagnostics.extend(self.check_semantic_issues(content));
                    }
                    Err(e) => {
                        let (line, character) = self.extract_error_position(&e.to_string(), content);
                        diagnostics.push(Diagnostic {
                            range: Range {
                                start: Position { line, character },
                                end: Position { line, character: character + 1 },
                            },
                            severity: 1, // Error
                            message: format!("Parse error: {}", e),
                            source: "augustium".to_string(),
                        });
                    }
                }
            }
            Err(e) => {
                let (line, character) = self.extract_error_position(&e.to_string(), content);
                diagnostics.push(Diagnostic {
                    range: Range {
                        start: Position { line, character },
                        end: Position { line, character: character + 1 },
                    },
                    severity: 1, // Error
                    message: format!("Lexical error: {}", e),
                    source: "augustium".to_string(),
                });
            }
        }

        diagnostics
    }
    
    fn check_basic_syntax(&self, content: &str) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        
        for (line_num, line) in lines.iter().enumerate() {
            let line_num = line_num as u32;
            
            // Check for unmatched braces
            let open_braces = line.matches('{').count();
            let close_braces = line.matches('}').count();
            
            // Check for missing semicolons (simple heuristic)
            if line.trim().ends_with('}') || line.trim().ends_with('{') {
                // These are fine
            } else if line.contains('=') && !line.trim().ends_with(';') && !line.trim().ends_with(',') && !line.trim().is_empty() {
                if let Some(pos) = line.rfind('=') {
                    diagnostics.push(Diagnostic {
                        range: Range {
                            start: Position { line: line_num, character: line.len() as u32 },
                            end: Position { line: line_num, character: line.len() as u32 },
                        },
                        severity: 2, // Warning
                        message: "Missing semicolon".to_string(),
                        source: "augustium".to_string(),
                    });
                }
            }
            
            // Check for invalid characters in identifiers
            if let Some(pos) = line.find(|c: char| !c.is_ascii() && c.is_alphabetic()) {
                diagnostics.push(Diagnostic {
                    range: Range {
                        start: Position { line: line_num, character: pos as u32 },
                        end: Position { line: line_num, character: pos as u32 + 1 },
                    },
                    severity: 2, // Warning
                    message: "Non-ASCII character in identifier".to_string(),
                    source: "augustium".to_string(),
                });
            }
            
            // Check for TODO/FIXME comments
            if line.contains("TODO") || line.contains("FIXME") {
                if let Some(pos) = line.find("TODO").or_else(|| line.find("FIXME")) {
                    diagnostics.push(Diagnostic {
                        range: Range {
                            start: Position { line: line_num, character: pos as u32 },
                            end: Position { line: line_num, character: (pos + 4) as u32 },
                        },
                        severity: 3, // Information
                        message: "TODO/FIXME comment".to_string(),
                        source: "augustium".to_string(),
                    });
                }
            }
        }
        
        diagnostics
    }
    
    fn check_semantic_issues(&self, content: &str) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        
        for (line_num, line) in lines.iter().enumerate() {
            let line_num = line_num as u32;
            
            // Check for unused variables (simple heuristic)
            if line.trim().starts_with("let ") && !content.contains(&line.split_whitespace().nth(1).unwrap_or("")) {
                if let Some(pos) = line.find("let ") {
                    diagnostics.push(Diagnostic {
                        range: Range {
                            start: Position { line: line_num, character: pos as u32 },
                            end: Position { line: line_num, character: line.len() as u32 },
                        },
                        severity: 2, // Warning
                        message: "Potentially unused variable".to_string(),
                        source: "augustium".to_string(),
                    });
                }
            }
            
            // Check for magic numbers
            for word in line.split_whitespace() {
                if let Ok(num) = word.parse::<i64>() {
                    if num > 100 && !line.contains("//") { // Ignore commented lines
                        if let Some(pos) = line.find(word) {
                            diagnostics.push(Diagnostic {
                                range: Range {
                                    start: Position { line: line_num, character: pos as u32 },
                                    end: Position { line: line_num, character: (pos + word.len()) as u32 },
                                },
                                severity: 3, // Information
                                message: "Consider using a named constant for this magic number".to_string(),
                                source: "augustium".to_string(),
                            });
                        }
                    }
                }
            }
        }
        
        diagnostics
    }
    
    fn extract_error_position(&self, error_msg: &str, content: &str) -> (u32, u32) {
        // Try to extract line and column information from error message
        // This is a simple implementation - in a real LSP, you'd want more sophisticated error reporting
        
        // Look for patterns like "line 5" or "at line 3, column 10"
        if let Some(line_match) = error_msg.find("line ") {
            let after_line = &error_msg[line_match + 5..];
            if let Some(space_pos) = after_line.find(' ') {
                let line_str = &after_line[..space_pos];
                if let Ok(line_num) = line_str.parse::<u32>() {
                    return (line_num.saturating_sub(1), 0); // Convert to 0-based
                }
            }
        }
        
        // Default to start of document
        (0, 0)
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