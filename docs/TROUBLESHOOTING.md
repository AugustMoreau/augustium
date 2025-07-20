# Troubleshooting Guide

Common issues and solutions when working with Augustium.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Compilation Errors](#compilation-errors)
- [Runtime Errors](#runtime-errors)
- [IDE Integration](#ide-integration)
- [Performance Issues](#performance-issues)
- [Debugging Tips](#debugging-tips)

## Installation Issues

### Rust Installation Problems

**Error**: `rustc: command not found`

**Solution**:
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Restart shell or run:
source $HOME/.cargo/env

# Verify installation
rustc --version
```

### Build Failures

**Error**: `error: could not compile augustc`

**Solutions**:
```bash
# Update Rust to latest version
rustup update

# Clean and rebuild
cargo clean
cargo build --release

# Check Rust version (need 1.70+)
rustc --version
```

### Missing Dependencies

**Error**: `error: linker 'cc' not found`

**Solutions**:

**On Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install build-essential
```

**On macOS:**
```bash
xcode-select --install
```

**On Windows:**
```bash
# Install Visual Studio Build Tools or Visual Studio Community
```

## Compilation Errors

### Syntax Errors

**Error**: `ParseError: unexpected token`

**Common causes and fixes**:

```augustium
// ❌ Missing semicolon
let x = 42

// ✅ Correct
let x = 42;

// ❌ Wrong function syntax
function add(a: u32, b: u32) -> u32 {
    return a + b;
}

// ✅ Correct
pub fn add(a: u32, b: u32) -> u32 {
    return a + b;
}
```

### Type Errors

**Error**: `SemanticError: type mismatch`

**Common fixes**:
```augustium
// ❌ Type mismatch
let number: u32 = "hello";

// ✅ Correct types
let number: u32 = 42;
let text: string = "hello";

// ❌ Function return type mismatch
pub fn get_number() -> u32 {
    return "42";  // String instead of u32
}

// ✅ Correct return type
pub fn get_number() -> u32 {
    return 42;
}
```

### Import Errors

**Error**: `module not found`

**Solutions**:
```augustium
// ❌ Wrong import path
use crypto::Hash;

// ✅ Correct import path
use std::crypto::Hash;

// Make sure module exists in Aug.toml dependencies
```

## Runtime Errors

### Gas Limit Exceeded

**Error**: `VmError: gas limit exceeded`

**Solutions**:
```rust
// Increase gas limit
let mut vm = AVM::new();
vm.set_gas_limit(2_000_000);  // Increase from default

// Or optimize your contract:
// - Avoid loops where possible
// - Use more efficient algorithms
// - Break large operations into smaller ones
```

### Stack Overflow

**Error**: `VmError: stack overflow`

**Causes and solutions**:
```augustium
// ❌ Infinite recursion
pub fn factorial(n: u32) -> u32 {
    return n * factorial(n - 1);  // No base case!
}

// ✅ Proper recursion with base case
pub fn factorial(n: u32) -> u32 {
    if n <= 1 {
        return 1;
    }
    return n * factorial(n - 1);
}
```

### Arithmetic Overflow

**Error**: `overflow in arithmetic operation`

**Solutions**:
```augustium
// ❌ Can overflow
let result = a + b;

// ✅ Safe arithmetic
use std::math::SafeMath;
let result = SafeMath::safe_add(a, b)?;

// Or use explicit overflow checking
let result = a.checked_add(b).require("overflow")?;
```

## IDE Integration

### VS Code Issues

**Problem**: No syntax highlighting

**Solutions**:
1. Install the Augustium extension
2. Reload VS Code window (`Ctrl+Shift+P` → "Developer: Reload Window")
3. Check file extension is `.aug`

**Problem**: LSP server not starting

**Solutions**:
```bash
# Check if augustc is in PATH
which augustc

# If not, add to PATH or configure extension settings:
{
    "augustium.compilerPath": "/path/to/augustium/target/release/augustc"
}
```

### Vim/Neovim Issues

**Problem**: No syntax highlighting

**Solution**:
```vim
" Add to .vimrc
set runtimepath+=/path/to/augustium/ide-plugins/vim

" For Neovim with LSP
lua require'lspconfig'.augustium.setup{}
```

## Performance Issues

### Slow Compilation

**Problem**: Compilation takes too long

**Solutions**:
```bash
# Use parallel compilation
RUSTFLAGS="-C target-cpu=native" cargo build --release -j$(nproc)

# Enable compiler caching
export RUSTC_WRAPPER=sccache

# For development, use debug builds
cargo build  # Instead of --release
```

### Slow Runtime

**Problem**: Contract execution is slow

**Solutions**:
```augustium
// Optimize loops
// ❌ Inefficient
for i in 0..1000000 {
    // Heavy computation each iteration
    let result = expensive_function(i);
    storage[i] = result;
}

// ✅ More efficient
let batch_size = 1000;
for batch in 0..(1000000/batch_size) {
    let results = batch_compute(batch * batch_size, batch_size);
    for (i, result) in results.enumerate() {
        storage[batch * batch_size + i] = result;
    }
}
```

## Debugging Tips

### Enable Debug Mode

```bash
# Compile with debug info
augustc compile --debug contract.aug

# Run with debugger
augustc debug contract.aug
```

### Add Debug Prints

```augustium
// Use require for debugging (gets removed in production)
require(balance >= amount, "Debug: balance check failed");

// Log values
debug_print("Balance:", balance);
debug_print("Amount:", amount);
```

### Step-by-Step Debugging

```bash
# Start interactive debugger
augustc debug contract.aug

# Set breakpoints
(augustc-debug) break line 25
(augustc-debug) break function transfer

# Run until breakpoint
(augustc-debug) run

# Inspect variables
(augustc-debug) print balance
(augustc-debug) print msg.sender

# Step through code
(augustc-debug) step
(augustc-debug) next
(augustc-debug) continue
```

### Memory Issues

**Problem**: High memory usage

**Solutions**:
```rust
// Check VM memory limits
let mut vm = AVM::new();
vm.set_memory_limit(512 * 1024 * 1024);  // 512MB limit

// Profile memory usage
augustc profile --memory contract.aug
```

## Common Patterns and Fixes

### Working with Mappings

```augustium
// ❌ Wrong mapping syntax
mapping<address => u256> balances;

// ✅ Correct mapping syntax
balances: mapping<address, u256>;
```

### Event Emission

```augustium
// ❌ Wrong event syntax
emit Transfer(from, to, amount);

// ✅ Correct event syntax  
event Transfer(from: address, to: address, amount: u256);

// In function:
emit Transfer(from_addr, to_addr, transfer_amount);
```

### Error Handling

```augustium
// ❌ Not handling errors
let result = risky_operation();

// ✅ Proper error handling
match risky_operation() {
    Ok(value) => {
        // Handle success
    }
    Err(error) => {
        // Handle error
        return Err(error);
    }
}

// Or use ? operator
let result = risky_operation()?;
```

## Getting Help

### Before Asking for Help

1. **Check this troubleshooting guide**
2. **Read error messages carefully** - they often contain the solution
3. **Try minimal examples** - isolate the problem
4. **Check the examples** in the `examples/` directory
5. **Update to latest version** - your issue might be fixed

### Where to Get Help

1. **GitHub Issues** - For bugs and feature requests
2. **Documentation** - Check all docs in `docs/` directory
3. **Examples** - Look at working code in `examples/`
4. **Community** - Join discussions and forums

### Reporting Bugs

When reporting issues, include:
- **Augustium version** (`augustc --version`)
- **Rust version** (`rustc --version`)
- **Operating system**
- **Complete error message**
- **Minimal code example** that reproduces the issue
- **Expected vs actual behavior**

### Performance Issues

```bash
# Generate performance report
augustc profile --detailed contract.aug

# Check gas usage
augustc analyze-gas contract.aug

# Profile compilation time
RUSTC_LOG=info cargo build --release
```

---

**Still having issues?** Don't hesitate to open an issue on GitHub with a detailed description of your problem!
