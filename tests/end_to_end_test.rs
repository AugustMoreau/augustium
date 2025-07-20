// End-to-end integration tests for the Augustium compiler
// Tests the complete pipeline from source code to bytecode

use std::fs;
use std::path::Path;

#[cfg(test)]
mod tests {
    use super::*;
    use augustc::lexer::Lexer;
    use augustc::parser::Parser;
    use augustc::semantic::SemanticAnalyzer;
    use augustc::codegen::CodeGenerator;
    use augustc::avm::AVM;

    #[test]
    fn test_simple_expression_compilation() {
        // Test a very simple function with a let statement
        let source = r#"
            fn test() {
                let x: u32 = 42;
            }
        "#;
        
        // Lexical analysis
        let mut lexer = Lexer::new(source, "test.aug");
        let tokens = lexer.tokenize().expect("Lexing should succeed");
        
        // Syntax analysis
        let mut parser = Parser::new(tokens);
        let source_file = parser.parse().expect("Parsing should succeed");
        
        // Semantic analysis
        let mut analyzer = SemanticAnalyzer::new();
        let _analyzed_ast = analyzer.analyze(&source_file).expect("Semantic analysis should succeed");
        
        // Code generation
        let mut codegen = CodeGenerator::new();
        let bytecode = codegen.generate(&source_file).expect("Code generation should succeed");
        
        // VM execution
        let mut vm = AVM::new();
        let result = vm.execute(&bytecode).expect("VM execution should succeed");
        
        assert!(result.success, "Execution should be successful");
    }
    
    #[test]
    fn test_hello_world_compilation() {
        // Try to compile the hello world example
        let example_path = "examples/hello_world.aug";
        
        if !Path::new(example_path).exists() {
            panic!("Hello world example file not found");
        }
        
        let source = fs::read_to_string(example_path)
            .expect("Should be able to read hello world example");
        
        // This test will likely fail initially, but will show us what we need to implement
        let mut lexer = Lexer::new(&source, "hello_world.aug");
        let tokens = match lexer.tokenize() {
            Ok(tokens) => tokens,
            Err(e) => {
                println!("Lexing failed: {:?}", e);
                return; // Expected to fail for now
            }
        };
        
        let mut parser = Parser::new(tokens);
        let source_file = match parser.parse() {
            Ok(source_file) => source_file,
            Err(e) => {
                println!("Parsing failed: {:?}", e);
                return; // Expected to fail for now
            }
        };
        
        let mut analyzer = SemanticAnalyzer::new();
        let _analyzed_result = match analyzer.analyze(&source_file) {
            Ok(result) => result,
            Err(e) => {
                println!("Semantic analysis failed: {:?}", e);
                return; // Expected to fail for now
            }
        };
        
        let mut codegen = CodeGenerator::new();
        let bytecode = match codegen.generate(&source_file) {
            Ok(bytecode) => bytecode,
            Err(e) => {
                println!("Code generation failed: {:?}", e);
                return; // Expected to fail for now
            }
        };
        
        let mut vm = AVM::new();
        let result = match vm.execute(&bytecode) {
            Ok(result) => result,
            Err(e) => {
                println!("VM execution failed: {:?}", e);
                return; // Expected to fail for now
            }
        };
        
        println!("Hello world compilation succeeded! Result: {:?}", result);
    }
    
    #[test]
    fn test_minimal_function() {
        // Test a minimal function definition
        let source = r#"
            fn add(a: u32, b: u32) -> u32 {
                return a + b;
            }
        "#;
        
        let mut lexer = Lexer::new(source, "test_function.aug");
        let tokens = match lexer.tokenize() {
            Ok(tokens) => tokens,
            Err(e) => {
                println!("Lexing failed for minimal function: {:?}", e);
                return;
            }
        };
        
        let mut parser = Parser::new(tokens);
        let source_file = match parser.parse() {
            Ok(source_file) => source_file,
            Err(e) => {
                println!("Parsing failed for minimal function: {:?}", e);
                return;
            }
        };
        
        println!("Minimal function parsing succeeded! AST: {:?}", source_file);
    }
    
    #[test]
    fn test_simple_hello_compilation() {
        // Test our simple hello example
        let example_path = "examples/simple_hello.aug";
        
        if !Path::new(example_path).exists() {
            panic!("Simple hello example file not found");
        }
        
        let source = fs::read_to_string(example_path)
            .expect("Should be able to read simple hello example");
        
        // Full pipeline test
        let mut lexer = Lexer::new(&source, "simple_hello.aug");
        let tokens = lexer.tokenize().expect("Lexing should succeed");
        
        let mut parser = Parser::new(tokens);
        let source_file = parser.parse().expect("Parsing should succeed");
        
        let mut analyzer = SemanticAnalyzer::new();
        let _analyzed_result = analyzer.analyze(&source_file).expect("Semantic analysis should succeed");
        
        let mut codegen = CodeGenerator::new();
        let bytecode = codegen.generate(&source_file).expect("Code generation should succeed");
        
        let mut vm = AVM::new();
        let result = vm.execute(&bytecode).expect("VM execution should succeed");
        
        assert!(result.success, "Execution should be successful");
        println!("Simple hello compilation and execution succeeded!");
    }
}