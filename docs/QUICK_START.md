# Quick Start Guide

Get up and running with Augustium in 5 minutes.

## Installation

### Prerequisites

- Rust 1.70+ (install from [rustup.rs](https://rustup.rs))
- Git

### Install Augustium

```bash
# Clone the repository
git clone https://github.com/yourusername/augustium
cd augustium

# Build the compiler and tools
cargo build --release

# Add to PATH (optional)
export PATH=$PATH:$(pwd)/target/release
```

## Your First Contract

### 1. Create a New Project

```bash
# Using the august project manager
./target/release/august new hello_defi
cd hello_defi
```

This creates:
```
hello_defi/
├── Aug.toml          # Project configuration
├── src/
│   └── main.aug      # Main contract file
└── tests/            # Test directory
```

### 2. Write Your Contract

Edit `src/main.aug`:

```augustium
// Simple token contract
contract MyToken {
    // State variables
    balances: mapping<address, u256>;
    total_supply: u256;
    owner: address;
    
    // Constructor
    constructor(supply: u256) {
        owner = msg.sender;
        total_supply = supply;
        balances[owner] = supply;
    }
    
    // Transfer tokens
    pub fn transfer(to: address, amount: u256) -> bool {
        require(balances[msg.sender] >= amount);
        
        balances[msg.sender] -= amount;
        balances[to] += amount;
        
        emit Transfer(msg.sender, to, amount);
        return true;
    }
    
    // Get balance
    pub fn balance_of(account: address) -> u256 {
        return balances[account];
    }
}

// Events
event Transfer(from: address, to: address, value: u256);
```

### 3. Build and Run

```bash
# Build the contract
august build

# Run the contract
august run
```

## Core Features Demo

### Safe Arithmetic

```augustium
contract SafeMath {
    pub fn safe_add(a: u256, b: u256) -> u256 {
        let result = a + b;
        require(result >= a);  // Overflow check
        return result;
    }
}
```

### Blockchain Integration

```augustium
contract BlockchainInfo {
    pub fn get_info() -> (address, u256, u256) {
        return (
            msg.sender,      // Who called this
            block.timestamp, // Current time
            block.number     // Block height
        );
    }
}
```

### DeFi Features

```augustium
use std::defi::LiquidityPool;

contract SimpleAMM {
    pool: LiquidityPool;
    
    constructor() {
        pool = LiquidityPool::new(
            token_a_info,
            token_b_info,
            30  // 0.3% fee
        );
    }
    
    pub fn swap(amount_in: u256) -> u256 {
        return pool.swap_exact_input(amount_in)?;
    }
}
```

## Development Workflow

### 1. Development Loop

```bash
# Make changes to your contract
vim src/main.aug

# Check for errors
august check

# Build if no errors
august build

# Run and test
august run
```

### 2. Testing

Create `tests/token_test.aug`:

```augustium
use std::test::TestFramework;

test "token transfer works" {
    let token = MyToken::new(1000);
    let success = token.transfer(alice_addr, 100);
    
    assert(success == true);
    assert(token.balance_of(alice_addr) == 100);
}
```

Run tests:
```bash
august test
```

### 3. Adding Dependencies

```bash
# Add DeFi utilities
august install augustium-defi

# Add oracle support  
august install augustium-oracle

# Update all dependencies
august update
```

## IDE Integration

### VS Code

1. Install the Augustium extension
2. Open your project folder
3. Get syntax highlighting, error checking, and autocomplete

### Other Editors

Generated plugins available for Vim, Emacs, Sublime Text, and more in the `ide-plugins/` directory.

## Next Steps

Now that you have the basics working:

1. **Learn the Language** - Read the [Language Specification](../language-specification.md)
2. **Explore Examples** - Check out [examples/](../examples/) for more contracts
3. **DeFi Development** - See [DeFi Templates](../defi-templates.md)
4. **Safety Features** - Learn about [Security](../safety-features.md)
5. **Advanced Topics** - Dive into [Compiler Architecture](../compiler-architecture.md)

## Getting Help

- **Examples** - See [examples/](../examples/) directory
- **Documentation** - Browse [docs/](../docs/) for detailed guides
- **Issues** - Report problems on GitHub
- **Community** - Join discussions and get help

Happy coding with Augustium! 🚀
