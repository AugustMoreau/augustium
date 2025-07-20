# Contributing to Augustium

Thanks for your interest in contributing to Augustium! This guide will help you get started.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/augustium
   cd augustium
   ```

2. **Install Rust** (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

3. **Build the Augustium toolchain:**
   ```bash
   # Build the Rust codebase (compiler, CLI tools, etc.)
   cargo build --release
   ```

4. **Run compiler tests:**
   ```bash
   # Test the Rust implementation
   cargo test
   ```

5. **Add to PATH (for testing):**
   ```bash
   # Add the built binaries to your PATH
   export PATH=$PATH:$(pwd)/target/release
   ```

6. **Test the tools:**
   ```bash
   # Test the compiler directly
   augustc --version
   
   # Test the project manager
   august --version
   
   # Create a test project
   august new test_project
   cd test_project
   august build
   ```

## Important Distinction

**When contributing to Augustium itself**, you use `cargo` commands because you're developing the Rust codebase:
- `cargo build` - Build the compiler
- `cargo test` - Test the compiler
- `cargo run --bin augustc` - Run the compiler directly

**When using Augustium for smart contracts**, end users use `august` commands:
- `august new my_contract` - Create new project
- `august build` - Build Augustium contracts
- `august run` - Run contracts

## Project Structure

- `src/` - Core compiler implementation
- `src/stdlib/` - Standard library modules  
- `examples/` - Example Augustium programs
- `tests/` - Test suite
- `docs/` - Documentation files

## Making Changes

1. **Create a branch** for your changes
2. **Add tests** for new functionality
3. **Update documentation** if needed
4. **Run tests** to make sure everything works
5. **Submit a pull request**

## Code Style

- Follow standard Rust conventions
- Keep comments natural and helpful
- Add examples for new features
- Keep functions focused and simple

## Getting Help

- Check the issues page for known problems
- Look at the language specification in `language-specification.md`
- Study the examples in `examples/`

I appreciate all contributions, from bug fixes to new features!
