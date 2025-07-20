# Augustium API Reference

Complete reference for all Augustium compiler and runtime APIs.

## Table of Contents

- [Compiler API](#compiler-api)
- [Standard Library](#standard-library)
- [CLI Tools](#cli-tools)
- [Language Server Protocol](#language-server-protocol)

## Compiler API

### Lexer

```rust
use augustc::lexer::Lexer;

let source = "contract MyContract { }";
let mut lexer = Lexer::new(source);
let tokens = lexer.tokenize()?;
```

### Parser

```rust
use augustc::parser::Parser;

let mut parser = Parser::new(tokens);
let ast = parser.parse()?;
```

### Semantic Analysis

```rust
use augustc::semantic::SemanticAnalyzer;

let mut analyzer = SemanticAnalyzer::new();
let analyzed_ast = analyzer.analyze(ast)?;
```

### Code Generation

```rust
use augustc::codegen::CodeGenerator;

let mut generator = CodeGenerator::new();
let bytecode = generator.generate(analyzed_ast)?;
```

### AVM (Virtual Machine)

```rust
use augustc::avm::AVM;

let mut vm = AVM::new();
let result = vm.execute(bytecode)?;
```

## Standard Library

### Core Types

- `U8`, `U16`, `U32`, `U64`, `U128`, `U256` - Unsigned integers
- `I8`, `I16`, `I32`, `I64`, `I128`, `I256` - Signed integers  
- `Bool` - Boolean type
- `Address` - Blockchain address type
- `AugString` - String type

### Collections

```augustium
use std::collections::Vec;

let mut numbers = Vec::new();
numbers.push(42);
let first = numbers.get(0);
```

### Crypto Functions

```augustium
use std::crypto::Hash;

let data = "hello world".as_bytes();
let hash = Hash::sha256(data)?;
let keccak_hash = Hash::keccak256(data)?;
```

### Math Operations

```augustium
use std::math::SafeMath;

let result = SafeMath::safe_add(a, b)?;  // Overflow-safe addition
let sqrt_result = SafeMath::sqrt(number)?;
```

### DeFi Utilities

```augustium
use std::defi::LiquidityPool;

let pool = LiquidityPool::new(token_a, token_b, fee_rate);
let price = pool.get_price()?;
```

### Time Functions

```augustium
use std::time::Timestamp;

let now = Timestamp::now();
let duration = Duration::from_seconds(3600);
let future = now.add(duration);
```

## CLI Tools

### augustc (Compiler)

```bash
# Compile a contract
augustc compile contract.aug

# Compile and run immediately  
augustc run contract.aug

# Debug a contract
augustc debug contract.aug
```

### august (Project Manager)

```bash
# Create new project
august new my_project

# Build current project
august build --release

# Run the main contract
august run

# Add dependency
august install augustium-defi

# Run tests
august test
```

## Language Server Protocol

The Augustium LSP server provides IDE integration:

- **Syntax highlighting**
- **Error diagnostics**
- **Auto-completion** 
- **Go-to-definition**
- **Hover information**
- **Code formatting**

### VS Code Integration

Install the Augustium extension from the marketplace or use the generated plugin files.

### Other Editors

Generated plugins are available for:
- **Vim/Neovim**
- **Emacs** 
- **Sublime Text**
- **Atom**

## Error Handling

All API functions return `Result<T, CompilerError>` types:

```rust
match result {
    Ok(value) => println!("Success: {:?}", value),
    Err(error) => println!("Error: {}", error),
}
```

Common error types:
- `LexError` - Tokenization errors
- `ParseError` - Syntax errors
- `SemanticError` - Type checking errors
- `VmError` - Runtime errors

## Configuration

### Compiler Options

```toml
# Aug.toml
[compiler]
optimization_level = 2
target = "avm"
debug_info = true

[features]
cross_chain = true
defi = true
governance = false
```

### Runtime Configuration

```rust
let mut vm = AVM::new();
vm.set_gas_limit(1_000_000);
vm.set_debug_mode(true);
```

For detailed examples and usage patterns, see the [Examples](../examples/) directory.
