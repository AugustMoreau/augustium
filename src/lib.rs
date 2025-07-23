// Augustium compiler library
// Main modules for lexing, parsing, semantic analysis, and codegen

pub mod ast;
pub mod lexer;
pub mod parser;
pub mod semantic;
pub mod codegen;
pub mod optimization;
pub mod avm;
pub mod error;
pub mod lsp;
pub mod stdlib;
pub mod package_manager;
pub mod evm_compat;
pub mod cross_chain;
pub mod ide_plugins;
pub mod web3_libs;
pub mod deployment_tools;

// Re-export commonly used types for convenience
pub use ast::*;
pub use lexer::Lexer;
pub use parser::Parser;
pub use semantic::SemanticAnalyzer;
pub use codegen::CodeGenerator;
pub use optimization::Optimizer;
pub use avm::AVM;
pub use error::*;