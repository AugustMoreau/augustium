use std::env;
use std::fs;
use std::io::{self, Write};

use augustc::lexer::Lexer; // assuming crate name is augustc (the root)
use augustc::parser::Parser;
use augustc::semantic::SemanticAnalyzer;
use augustc::codegen::{CodeGenerator, Bytecode, Value};
use augustc::avm::AVM;

fn compile_contract(source: &str) -> anyhow::Result<Bytecode> {
    // Pipeline similar to tests in codegen.rs
    let mut lexer = Lexer::new(source, "<stdin>");
    let tokens = lexer.tokenize()?;

    let mut parser = Parser::new(tokens);
    let ast = parser.parse()?;

    let mut analyzer = SemanticAnalyzer::new();
    analyzer.analyze(&ast)?;

    let mut codegen = CodeGenerator::new();
    let bytecode = codegen.generate(&ast)?;

    Ok(bytecode)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path-to-.aug>", args[0]);
        std::process::exit(1);
    }

    let source_path = &args[1];
    let source = fs::read_to_string(source_path)?;

    println!("Compiling {source_path} ...");
    let bytecode = compile_contract(&source)?;

    if bytecode.contracts.is_empty() {
        eprintln!("No contracts found in source file.");
        std::process::exit(1);
    }

    // Choose first contract
    let (contract_name, contract_bc) = bytecode.contracts.iter().next().unwrap();
    println!("Found contract '{contract_name}'. Deploying...");

    let mut avm = AVM::new();
    let contract_addr = avm.deploy_contract(contract_bc.clone(), Vec::new())?;

    println!("Contract deployed at address {:?}", contract_addr);
    println!("Type 'help' to list functions, 'quit' to exit. Enter calls like: add 3 5\n");

    let mut input = String::new();
    loop {
        input.clear();
        print!("> ");
        io::stdout().flush()?;
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }
        let line = input.trim();
        if line.is_empty() {
            continue;
        }
        if line == "quit" || line == "exit" {
            break;
        }
        if line == "help" {
            println!("Available functions:");
            for fname in contract_bc.functions.keys() {
                println!("  {fname}");
            }
            continue;
        }

        // Parse: first token function name, rest args (i64 for now)
        let mut parts = line.split_whitespace();
        let func = match parts.next() {
            Some(f) => f,
            None => continue,
        };
        let mut args_vec: Vec<Value> = Vec::new();
        for p in parts {
            // Try i64 first, fallback to u64.
            if let Ok(v) = p.parse::<i64>() {
                args_vec.push(Value::I64(v));
            } else if let Ok(vu) = p.parse::<u64>() {
                args_vec.push(Value::U64(vu));
            } else {
                println!("Could not parse argument '{p}'. Only integer args supported.");
                args_vec.clear();
                break;
            }
        }
        if args_vec.is_empty() && line.split_whitespace().count() > 1 {
            continue;
        }

        if !contract_bc.functions.contains_key(func) {
            println!("Unknown function '{func}'. Type 'help' to list.");
            continue;
        }

        // Call
        match avm.call_contract(contract_addr, func, args_vec) {
            Ok(result) => {
                if result.success {
                    if let Some(val) = result.return_value {
                        println!("=> {:?}", val);
                    } else {
                        println!("=> <no return>");
                    }
                } else {
                    println!("Execution failed: {}", result.error.unwrap_or_else(|| "unknown error".to_string()));
                }
            }
            Err(e) => println!("Error: {e}"),
        }
    }

    Ok(())
}
