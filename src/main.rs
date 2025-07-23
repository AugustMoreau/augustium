// Augustium compiler (augustc)
// Compiles .aug files into bytecode for our virtual machine

use std::env;
use std::fs;
use std::path::Path;
use std::process;

mod lexer;
mod parser;
mod semantic;
mod codegen;
mod optimization;
pub mod ast;
mod avm;
mod error;
mod lsp;
mod profiler;
mod package_manager;
mod evm_compat;
mod cross_chain;
mod ide_plugins;
mod web3_libs;
mod deployment_tools;
pub mod consensus;
pub mod transaction;
pub mod gas;
pub mod stdlib;
pub mod syntax_standard;
pub mod dev_tools;

use crate::error::{CompilerError, Result};
use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::semantic::SemanticAnalyzer;
use crate::codegen::CodeGenerator;
use crate::optimization::Optimizer;
use crate::avm::AVM;

// Show help when user runs augustc without args or with --help
fn print_help() {
    println!("Augustium Compiler (augustc) v0.1.0");
    println!("Smart contract compiler for Augustium blockchain");
    println!();
    println!("USAGE:");
    println!("    augustc <command> [options]");
    println!();
    println!("COMMANDS:");
    println!("    compile <file>     Compile Augustium source file");
    println!("    run <file>         Compile and execute Augustium source file");
    println!("    new <name>         Create new Augustium project");
    println!("    build              Build current project");
    println!("    test               Run project tests");
    println!("    clean              Clean build artifacts");
    println!("    fmt                Format source code");
    println!("    lsp                Start Language Server Protocol");
    println!();
    println!("OPTIONS:");
    println!("    --output <file>    Output bytecode file");
    println!("    --run             Compile and run immediately");
    println!("    --debug           Enable debug output");
    println!("    --optimize        Enable optimizations");
    println!("    --check           Only check syntax and types");
    println!("    --target <arch>   Target architecture (avm, evm)");
    println!("    --gas-limit <n>   Set gas limit for execution");
    println!("    --verbose, -v     Verbose output");
    println!("    --run, -r         Compile and immediately execute bytecode");
    println!("    --quiet, -q       Suppress output");
    println!("    --help, -h        Show this help message");
    println!("    --version, -V     Show version information");
    println!();
    println!("EXAMPLES:");
    println!("    augustc new my_contract           # Create new project");
    println!("    augustc compile contract.aug      # Compile single file");
    println!("    augustc run contract.aug          # Compile and execute file");
    println!("    augustc build --optimize          # Build with optimizations");
    println!("    augustc test --verbose            # Run tests with verbose output");
    println!("    augustc lsp                       # Start LSP server");
}

/// Print version information
fn print_version() {
    println!("augustc 0.1.0");
    println!("Augustium Compiler for smart contract development");
    println!("Built with Rust");
}

/// Main compiler driver
fn main() {
    let args: Vec<String> = env::args().collect();
    
    // Handle special flags first
    if args.len() >= 2 {
        match args[1].as_str() {
            "--help" | "-h" => {
                print_help();
                return;
            }
            "--version" | "-V" => {
                print_version();
                return;
            }
            _ => {}
        }
    }
    
    if args.len() < 2 {
        print_help();
        process::exit(1);
    }
    
    let command = &args[1];
    
    match command.as_str() {
        "compile" => {
            if args.len() < 3 {
                eprintln!("Error: compile command requires a source file");
                process::exit(1);
            }
            let source_file = &args[2];
            let options = parse_options(&args[3..]);
            handle_compile(source_file, options);
        }
        "run" => {
            if args.len() < 3 {
                eprintln!("Error: run command requires a source file");
                process::exit(1);
            }
            let source_file = &args[2];
            let options = parse_options(&args[3..]);
            handle_run(source_file, options);
        }
        "new" => {
            if args.len() < 3 {
                eprintln!("Error: new command requires a project name");
                process::exit(1);
            }
            let project_name = &args[2];
            handle_new_project(project_name);
        }
        "build" => {
            let options = parse_options(&args[2..]);
            handle_build(options);
        }
        "test" => {
            let options = parse_options(&args[2..]);
            handle_test(options);
        }
        "clean" => {
            handle_clean();
        }
        "fmt" => {
            handle_format();
        }
        "lsp" => {
            handle_lsp();
        }
        _ => {
            // Legacy mode: treat first argument as source file
            let source_file = &args[1];
            let options = parse_options(&args[2..]);
            handle_compile(source_file, options);
        }
    }
}

/// Parse command line options
fn parse_options(args: &[String]) -> CompilerOptions {
    let mut options = CompilerOptions {
        target: "avm".to_string(),
        ..Default::default()
    };
    
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                if i + 1 < args.len() {
                    options.output_file = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: --output requires a filename");
                    process::exit(1);
                }
            }
            "--run" => {
                options.run_immediately = true;
                i += 1;
            }
            "--debug" => {
                options.debug = true;
                i += 1;
            }
            "--optimize" => {
                options.optimize = true;
                i += 1;
            }
            "--check" => {
                options.check_only = true;
                i += 1;
            }
            "--target" => {
                if i + 1 < args.len() {
                    options.target = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("Error: --target requires an architecture");
                    process::exit(1);
                }
            }
            "--gas-limit" => {
                if i + 1 < args.len() {
                    match args[i + 1].parse::<u64>() {
                        Ok(limit) => options.gas_limit = Some(limit),
                        Err(_) => {
                            eprintln!("Error: --gas-limit requires a valid number");
                            process::exit(1);
                        }
                    }
                    i += 2;
                } else {
                    eprintln!("Error: --gas-limit requires a number");
                    process::exit(1);
                }
            }
            "--verbose" | "-v" => {
                options.verbose = true;
                i += 1;
            }
            "--quiet" | "-q" => {
                options.quiet = true;
                i += 1;
            }
            _ => {
                eprintln!("Error: Unknown option {}", args[i]);
                process::exit(1);
            }
        }
    }
    
    options
}

/// Handle compile command
fn handle_compile(source_file: &str, options: CompilerOptions) {
    let check_only = options.check_only;
    let quiet = options.quiet;
    
    match compile_file(source_file, options) {
        Ok(()) => {
            if !quiet {
                if !check_only {
                    println!("✓ Compilation successful!");
                } else {
                    println!("✓ Syntax and type checking passed!");
                }
            }
        }
        Err(e) => {
            eprintln!("✗ Compilation failed: {}", e);
            process::exit(1);
        }
    }
}

/// Handle run command (compile and execute)
fn handle_run(source_file: &str, mut options: CompilerOptions) {
    let quiet = options.quiet;
    
    // Force immediate execution
    options.run_immediately = true;
    options.debug = true; // Enable debug for run command
    
    match compile_file(source_file, options) {
        Ok(()) => {
            if !quiet {
                println!("✓ Compilation and execution completed!");
            }
        }
        Err(e) => {
            eprintln!("✗ Compilation/execution failed: {}", e);
            process::exit(1);
        }
    }
}

/// Compiler options
#[derive(Debug, Default)]
struct CompilerOptions {
    output_file: Option<String>,
    run_immediately: bool,
    debug: bool,
    optimize: bool,
    check_only: bool,
    target: String,
    gas_limit: Option<u64>,
    verbose: bool,
    quiet: bool,
}

/// Project configuration
#[allow(dead_code)]
#[derive(Debug)]
struct ProjectConfig {
    name: String,
    version: String,
    dependencies: Vec<String>,
    source_dir: String,
    build_dir: String,
}

/// Compile a single Augustium source file
fn compile_file(source_file: &str, options: CompilerOptions) -> Result<()> {
    // Read source file
    let source_code = fs::read_to_string(source_file)
        .map_err(|e| CompilerError::IoError(format!("Failed to read {}: {}", source_file, e)))?;
    
    if options.debug {
        println!("Compiling: {}", source_file);
        println!("Source length: {} characters", source_code.len());
    }
    
    // Phase 1: Lexical Analysis
    if options.debug {
        println!("Phase 1: Lexical Analysis");
    }
    let mut lexer = Lexer::new(&source_code, source_file);
    let tokens = lexer.tokenize()?;
    
    if options.debug {
        println!("Generated {} tokens", tokens.len());
    }
    
    // Phase 2: Syntax Analysis
    if options.debug {
        println!("Phase 2: Syntax Analysis");
    }
    let mut parser = Parser::new(tokens);
    let mut ast = parser.parse()?;
    
    if options.debug {
        println!("Generated AST with {} top-level items", ast.items.len());
    }
    
    // Phase 3: Semantic Analysis
    if options.debug {
        println!("Phase 3: Semantic Analysis");
    }
    let mut analyzer = SemanticAnalyzer::new();
    analyzer.analyze(&ast)?;
    
    if options.debug {
        println!("Semantic analysis completed");
    }
    
    // If check-only mode, stop here
    if options.check_only {
        return Ok(());
    }
    
    // Phase 3.5: Optimization (if enabled)
    if options.optimize {
        if options.debug {
            println!("Phase 3.5: AST Optimization");
        }
        let optimizer = Optimizer::new();
        optimizer.optimize_ast(&mut ast)?;
        
        if options.debug {
            println!("AST optimization completed");
        }
    }
    
    // Phase 4: Code Generation
    if options.debug {
        println!("Phase 4: Code Generation");
    }
    let mut codegen = CodeGenerator::new();
    let mut bytecode = codegen.generate(&ast)?;
    
    // Phase 4.5: Bytecode Optimization (if enabled)
    if options.optimize {
        if options.debug {
            println!("Phase 4.5: Bytecode Optimization");
        }
        let optimizer = Optimizer::new();
        optimizer.optimize_bytecode(&mut bytecode)?;
        
        if options.debug {
            println!("Bytecode optimization completed");
        }
    }
    
    if options.debug {
        println!("Generated {} instructions of bytecode", bytecode.len());
    }
    
    // Determine output file
    let output_file = options.output_file.unwrap_or_else(|| {
        let path = Path::new(source_file);
        let stem = path.file_stem().unwrap().to_str().unwrap();
        format!("{}.avm", stem)
    });
    
    // Write bytecode to file
    fs::write(&output_file, bytecode.to_bytes())
        .map_err(|e| CompilerError::IoError(format!("Failed to write {}: {}", output_file, e)))?;
    
    if options.debug {
        println!("Bytecode written to: {}", output_file);
    }
    
    // Run immediately if requested
    if options.run_immediately {
        if options.debug {
            println!("Phase 5: Execution");
        }
        let mut vm = AVM::new();
        let result = vm.execute(&bytecode)?;
        
        if options.debug {
            println!("Execution result: {:?}", result);
        }
    }
    
    Ok(())
}

/// Handle new project creation
fn handle_new_project(project_name: &str) {
    use std::fs;
    
    println!("Creating new Augustium project: {}", project_name);
    
    // Create project directory structure
    let project_dir = format!("./{}", project_name);
    if let Err(e) = fs::create_dir_all(&format!("{}/src", project_dir)) {
        eprintln!("Error creating project directory: {}", e);
        process::exit(1);
    }
    
    if let Err(e) = fs::create_dir_all(&format!("{}/tests", project_dir)) {
        eprintln!("Error creating tests directory: {}", e);
        process::exit(1);
    }
    
    // Use package manager to initialize the project
    if let Err(e) = package_manager::init_new_package(project_name, &std::path::Path::new(&project_dir)) {
        eprintln!("Error initializing package: {}", e);
        process::exit(1);
    }
    
    // Create main contract file
    let main_content = format!(r#"//! {} - Augustium Smart Contract
//! 
//! This is the main contract for the {} project.

contract {} {{
    // Contract state
    state {{
        owner: address,
        initialized: bool,
    }}
    
    // Constructor
    pub fn new(owner: address) -> Self {{
        Self {{
            owner,
            initialized: true,
        }}
    }}
    
    // Public functions
    pub fn get_owner() -> address {{
        return self.owner;
    }}
    
    pub fn is_initialized() -> bool {{
        return self.initialized;
    }}
}}
"#, project_name, project_name, project_name);
    
    if let Err(e) = fs::write(&format!("{}/src/main.aug", project_dir), main_content) {
        eprintln!("Error creating main.aug: {}", e);
        process::exit(1);
    }
    
    // Create test file
    let test_content = format!(r#"//! Tests for {}

use super::*;

#[test]
fn test_contract_creation() {{
    let owner = address("0x1234567890123456789012345678901234567890");
    let contract = {}::new(owner);
    
    assert_eq!(contract.get_owner(), owner);
    assert!(contract.is_initialized());
}}

#[test]
fn test_owner_functionality() {{
    let owner = address("0x1234567890123456789012345678901234567890");
    let contract = {}::new(owner);
    
    // Test owner retrieval
    assert_eq!(contract.get_owner(), owner);
}}
"#, project_name, project_name, project_name);
    
    if let Err(e) = fs::write(&format!("{}/tests/test_{}.aug", project_dir, project_name.to_lowercase()), test_content) {
        eprintln!("Error creating test file: {}", e);
        process::exit(1);
    }
    
    // Create README
    let readme_content = format!(r#"# {}

Augustium smart contract project.

## Building

```bash
augustc build
```

## Testing

```bash
augustc test
```

## Running

```bash
augustc build --run
```
"#, project_name);
    
    if let Err(e) = fs::write(&format!("{}/README.md", project_dir), readme_content) {
        eprintln!("Error creating README.md: {}", e);
        process::exit(1);
    }
    
    println!("✓ Project '{}' created successfully!", project_name);
    println!("  cd {}", project_name);
    println!("  augustc build");
}

/// Handle build command
fn handle_build(options: CompilerOptions) {
    // Look for Aug.toml in current directory
    if !Path::new("Aug.toml").exists() {
        eprintln!("Error: No Aug.toml found. Run 'augustc new <project>' to create a new project.");
        process::exit(1);
    }
    
    println!("Building Augustium project...");
    
    // Find all .aug files in src directory
    let src_dir = Path::new("src");
    if !src_dir.exists() {
        eprintln!("Error: src directory not found");
        process::exit(1);
    }
    
    let mut compiled_files = 0;
    let mut failed_files = 0;
    
    // Create build directory
    if let Err(e) = fs::create_dir_all("build") {
        eprintln!("Error creating build directory: {}", e);
        process::exit(1);
    }
    
    // Compile all .aug files
    if let Ok(entries) = fs::read_dir(src_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("aug") {
                    if let Some(file_path) = path.to_str() {
                        let mut build_options = options.clone();
                        
                        // Set output file in build directory
                        let file_stem = path.file_stem().unwrap().to_str().unwrap();
                        build_options.output_file = Some(format!("build/{}.avm", file_stem));
                        
                        if !options.quiet {
                            println!("  Compiling {}...", file_path);
                        }
                        
                        match compile_file(file_path, build_options) {
                            Ok(()) => {
                                compiled_files += 1;
                                if options.verbose {
                                    println!("    ✓ {}", file_path);
                                }
                            }
                            Err(e) => {
                                failed_files += 1;
                                eprintln!("    ✗ {}: {}", file_path, e);
                            }
                        }
                    }
                }
            }
        }
    }
    
    if failed_files == 0 {
        println!("✓ Build completed successfully! ({} files compiled)", compiled_files);
    } else {
        eprintln!("✗ Build failed! ({} succeeded, {} failed)", compiled_files, failed_files);
        process::exit(1);
    }
}

/// Handle test command
fn handle_test(options: CompilerOptions) {
    println!("Running Augustium tests...");
    
    let tests_dir = Path::new("tests");
    if !tests_dir.exists() {
        println!("No tests directory found. Skipping tests.");
        return;
    }
    
    let mut test_files = 0;
    let mut passed_tests = 0;
    let mut failed_tests = 0;
    
    // Run all test files
    if let Ok(entries) = fs::read_dir(tests_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("aug") {
                    if let Some(file_path) = path.to_str() {
                        test_files += 1;
                        
                        if !options.quiet {
                            println!("  Running {}...", file_path);
                        }
                        
                        let mut test_options = options.clone();
                        test_options.run_immediately = true;
                        test_options.check_only = false;
                        
                        match compile_file(file_path, test_options) {
                            Ok(()) => {
                                passed_tests += 1;
                                if options.verbose {
                                    println!("    ✓ {}", file_path);
                                }
                            }
                            Err(e) => {
                                failed_tests += 1;
                                eprintln!("    ✗ {}: {}", file_path, e);
                            }
                        }
                    }
                }
            }
        }
    }
    
    if test_files == 0 {
        println!("No test files found.");
    } else if failed_tests == 0 {
        println!("✓ All tests passed! ({} tests)", passed_tests);
    } else {
        eprintln!("✗ Tests failed! ({} passed, {} failed)", passed_tests, failed_tests);
        process::exit(1);
    }
}

/// Handle clean command
fn handle_clean() {
    println!("Cleaning build artifacts...");
    
    if Path::new("build").exists() {
        if let Err(e) = fs::remove_dir_all("build") {
            eprintln!("Error cleaning build directory: {}", e);
            process::exit(1);
        }
        println!("✓ Build directory cleaned");
    } else {
        println!("Build directory already clean");
    }
}

/// Handle format command
fn handle_format() {
    println!("Formatting Augustium source code...");
    println!("✓ Code formatting completed (formatter not yet implemented)");
}

/// Handle LSP server command
fn handle_lsp() {
    println!("Starting Augustium Language Server Protocol...");
    println!("LSP server listening on stdio");
    
    // Start the actual LSP server
    if let Err(e) = lsp::start_lsp_server() {
        eprintln!("LSP server error: {}", e);
        process::exit(1);
    }
}

// Add Clone trait to CompilerOptions
impl Clone for CompilerOptions {
    fn clone(&self) -> Self {
        CompilerOptions {
            output_file: self.output_file.clone(),
            run_immediately: self.run_immediately,
            debug: self.debug,
            optimize: self.optimize,
            check_only: self.check_only,
            target: self.target.clone(),
            gas_limit: self.gas_limit,
            verbose: self.verbose,
            quiet: self.quiet,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_compile_hello_world() {
        let source = r#"
            contract HelloWorld {
                pub fn greet() -> u32 {
                    return 42;
                }
            }
        "#;
        
        let temp_file = NamedTempFile::new().unwrap();
        fs::write(&temp_file, source).unwrap();
        
        let options = CompilerOptions {
            check_only: true,
            debug: false,
            ..Default::default()
        };
        
        let result = compile_file(temp_file.path().to_str().unwrap(), options);
        if let Err(e) = &result {
            eprintln!("Compilation error: {}", e);
        }
        assert!(result.is_ok(), "Hello World compilation should succeed");
    }
    
    #[test]
    fn test_invalid_syntax() {
        let source = r#"
            contract InvalidSyntax {
                fn broken_function( {
                    // Missing closing parenthesis
                }
            }
        "#;
        
        let temp_file = NamedTempFile::new().unwrap();
        fs::write(&temp_file, source).unwrap();
        
        let options = CompilerOptions {
            check_only: true,
            debug: false,
            ..Default::default()
        };
        
        let result = compile_file(temp_file.path().to_str().unwrap(), options);
        assert!(result.is_err(), "Invalid syntax should fail compilation");
    }
}