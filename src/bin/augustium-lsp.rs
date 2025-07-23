//! Augustium Language Server Protocol (LSP) Binary
//! 
//! This is a dedicated LSP server for the Augustium programming language.
//! It provides IDE features like autocompletion, hover information, 
//! go-to-definition, diagnostics, and formatting.

use std::env;
use std::process;
use augustc::lsp;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    // Handle command line arguments
    if args.len() > 1 {
        match args[1].as_str() {
            "--help" | "-h" => {
                print_help();
                return;
            }
            "--version" | "-v" => {
                print_version();
                return;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[1]);
                eprintln!("Use --help for usage information.");
                process::exit(1);
            }
        }
    }
    
    // Start the LSP server
    eprintln!("Starting Augustium Language Server Protocol...");
    eprintln!("LSP server listening on stdio");
    
    if let Err(e) = lsp::start_lsp_server() {
        eprintln!("LSP server error: {}", e);
        process::exit(1);
    }
}

fn print_help() {
    println!("Augustium Language Server Protocol (augustium-lsp) v1.0.1");
    println!("Language server for Augustium smart contract development");
    println!();
    println!("USAGE:");
    println!("    augustium-lsp [options]");
    println!();
    println!("OPTIONS:");
    println!("    -h, --help       Show this help message");
    println!("    -v, --version    Show version information");
    println!();
    println!("DESCRIPTION:");
    println!("    The Augustium LSP server provides IDE features including:");
    println!("    - Syntax highlighting and error detection");
    println!("    - Code completion and IntelliSense");
    println!("    - Hover information and documentation");
    println!("    - Go-to-definition and symbol navigation");
    println!("    - Code formatting and refactoring");
    println!();
    println!("    This server communicates via JSON-RPC over stdin/stdout");
    println!("    and is designed to be used with LSP-compatible editors.");
}

fn print_version() {
    println!("augustium-lsp 1.0.1");
    println!("Augustium Language Server Protocol");
    println!("Built with Rust {}", env!("CARGO_PKG_RUST_VERSION"));
}