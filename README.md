# Augustium

> A secure, blockchain-native programming language for smart contract development

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/built%20with-Rust-000000.svg?logo=rust)](https://www.rust-lang.org/)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](CHANGELOG.md)

Augustium is a modern programming language specifically designed for blockchain development, combining the safety of Rust with the familiarity of C-like syntax. It provides built-in security features, automatic optimizations, and seamless multi-chain deployment capabilities.

## Key Features

- **Built-in Security**: Automatic overflow protection, reentrancy guards, memory safety
- **Blockchain Native**: Native address types, gas optimization, event system
- **Multi-Chain**: Deploy to Ethereum, BSC, Polygon, Avalanche, Arbitrum, Optimism
- **Developer Experience**: Full IDE support, debugging tools, comprehensive CLI
- **Performance**: Optimized bytecode generation, efficient execution
- **Rich Standard Library**: Crypto, DeFi, governance utilities built-in

##  Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AugustMoreau/augustium.git
cd augustium

# Build from source
cargo build --release

# Add to PATH
export PATH=$PATH:$(pwd)/target/release

# Verify installation
august --version
```

### VS Code Extension

Get the best development experience with our official VS Code extension:

#### Installation Options:

**Option 1: VS Code Marketplace**
1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "Augustium"
4. Click Install

**Option 2: Open VSX Registry**
1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "Augustium" by AugustMoreau
4. Click Install

**Option 3: Manual Installation**
```bash
# Download the latest .vsix file from releases
code --install-extension augustium-1.0.4.vsix
```

#### Features:
- 🎨 **Syntax Highlighting**: Full language support with rich colors
- 📝 **Code Snippets**: Pre-built templates for contracts, functions, and patterns
- 🔍 **IntelliSense**: Auto-completion and error detection
- 🛠️ **Build Integration**: Compile and run directly from VS Code
- 📚 **Documentation**: Hover hints and inline documentation

### Create Your First Contract

```bash
# Create a new project
august new my_token
cd my_token

# Build and run
august build
august run
```

## 💡 Example Contract

```augustium
contract MyToken {
    balances: mapping<address, u256>;
    total_supply: u256;
    owner: address;
    
    constructor(initial_supply: u256) {
        total_supply = initial_supply;
        owner = msg.sender;
        balances[msg.sender] = initial_supply;
        
        emit Transfer(address(0), msg.sender, initial_supply);
    }
    
    pub fn transfer(to: address, amount: u256) -> bool {
        require(to != address(0), "Transfer to zero address");
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        balances[msg.sender] -= amount;  // Automatic overflow protection
        balances[to] += amount;
        
        emit Transfer(msg.sender, to, amount);
        return true;
    }
    
    pub fn balance_of(account: address) -> u256 {
        return balances[account];
    }
    
    event Transfer(from: address, to: address, value: u256);
}
```

## 📖 Documentation

| Resource | Description |
|----------|-------------|
| [Quick Start Guide](docs/QUICK_START.md) | Step-by-step tutorial for beginners |
| [Language Specification](language-specification.md) | Complete language reference |
| [API Reference](docs/API_REFERENCE.md) | Comprehensive API documentation |
| [Security Guide](docs/SECURITY.md) | Security best practices |
| [Examples](docs/EXAMPLES.md) | Working contract examples |
| [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) | Multi-chain deployment |
| [FAQ](docs/FAQ.md) | Frequently asked questions |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and solutions |

## 🛠️ CLI Commands

```bash
# Project Management
august new <name>        # Create new project
august init              # Initialize in current directory
august build             # Build the project
august run               # Build and execute
august test              # Run tests
august clean             # Clean build artifacts

# Package Management
august install <package> # Add dependency
august update            # Update dependencies
august publish           # Publish to registry

# Development Tools
august fmt               # Format code
august check             # Check for errors
august doc               # Generate documentation
```

## 🔧 Development

### Prerequisites

- Rust 1.70 or later
- Git

### Building from Source

```bash
# Clone the repository
git clone https://github.com/AugustMoreau/augustium.git
cd augustium

# Install dependencies and build
cargo build --release

# Run tests
cargo test

# Test the CLI tools
./target/release/august --version
./target/release/augustc --version
```

### Project Structure

```
augustium/
├── src/                 # Compiler source code
│   ├── bin/            # CLI binaries (august, augustc)
│   ├── lexer.rs        # Lexical analyzer
│   ├── parser.rs       # Parser
│   ├── semantic.rs     # Semantic analyzer
│   ├── codegen.rs      # Code generation
│   ├── avm.rs          # Virtual machine
│   └── stdlib/         # Standard library
├── examples/           # Example contracts
├── docs/              # Documentation
├── tests/             # Test suite
└── benches/           # Benchmarks
```

## 🌟 What Makes Augustium Different?

| Feature | Augustium | Solidity | Rust |
|---------|-----------|----------|------|
| **Memory Safety** | ✅ Automatic | ❌ Manual | ✅ Ownership |
| **Overflow Protection** | ✅ Built-in | ❌ Manual | ❌ Manual |
| **Reentrancy Protection** | ✅ Built-in | ❌ Manual | ❌ N/A |
| **Blockchain Types** | ✅ Native | ✅ Basic | ❌ None |
| **Multi-Chain** | ✅ Built-in | ❌ Manual | ❌ N/A |
| **Learning Curve** | 📈 Moderate | 📈 Steep | 📈 Steep |

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup instructions
- Code style guidelines
- Testing requirements
- Pull request process

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

Augustium Programming Language and Compiler are created and maintained by [August Moreau](https://github.com/AugustMoreau).

## 🔗 Links

- **GitHub**: [AugustMoreau/augustium](https://github.com/AugustMoreau/augustium)
- **Issues**: [Report bugs or request features](https://github.com/AugustMoreau/augustium/issues)
- **Releases**: [View all releases](https://github.com/AugustMoreau/augustium/releases)

---

*Built by August Moreau*
