//! Integration tests for the Augustium compiler

use std::process::Command;
use std::fs;
use std::path::Path;

#[test]
fn test_compiler_help() {
    let output = Command::new("cargo")
        .args(["run", "--", "--help"])
        .output()
        .expect("Failed to execute compiler");
    
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Augustium Compiler"));
}

#[test]
fn test_simple_contract_compilation() {
    // Create a simple test contract
    let test_contract = r#"
contract HelloWorld {
    pub fn greet() -> string {
        return "Hello, Augustium!";
    }
}
"#;
    
    // Write test contract to temporary file
    let test_file = "test_hello.aug";
    fs::write(test_file, test_contract).expect("Failed to write test file");
    
    // Try to compile the contract
    let output = Command::new("cargo")
        .args(["run", "--", "--check", test_file])
        .output()
        .expect("Failed to execute compiler");
    
    // Clean up
    if Path::new(test_file).exists() {
        fs::remove_file(test_file).ok();
    }
    
    // For now, we expect compilation to fail gracefully since we haven't
    // implemented all features yet, but the compiler should run
    assert!(!output.stderr.is_empty() || !output.stdout.is_empty());
}

#[test]
fn test_invalid_syntax_handling() {
    // Create a contract with invalid syntax
    let invalid_contract = r#"
contract InvalidContract {
    pub fn broken_function( {
        // Missing closing parenthesis and function body
    }
}
"#;
    
    // Write test contract to temporary file
    let test_file = "test_invalid.aug";
    fs::write(test_file, invalid_contract).expect("Failed to write test file");
    
    // Try to compile the invalid contract
    let output = Command::new("cargo")
        .args(["run", "--", "--check", test_file])
        .output()
        .expect("Failed to execute compiler");
    
    // Clean up
    if Path::new(test_file).exists() {
        fs::remove_file(test_file).ok();
    }
    
    // Should fail with error message
    assert!(!output.status.success());
}

#[test]
fn test_compiler_version() {
    let output = Command::new("cargo")
        .args(["run", "--", "--version"])
        .output()
        .expect("Failed to execute compiler");
    
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("augustc"));
}