# Augustium Contract Examples

This directory contains practical examples of smart contracts written in Augustium, demonstrating various patterns, features, and best practices for blockchain development.

## Overview

These examples showcase Augustium's key features:
- **Memory Safety**: Automatic bounds checking and memory management
- **Arithmetic Safety**: Built-in overflow/underflow protection
- **Reentrancy Protection**: Automatic guards against reentrancy attacks
- **Type Safety**: Strong static typing with compile-time guarantees
- **Gas Optimization**: Efficient bytecode generation
- **Developer Experience**: Clear syntax and comprehensive error messages

## Examples

### 1. Simple Token (`simple-token.aug`)

A comprehensive ERC-20 token implementation demonstrating:
- Basic token functionality (transfer, approve, allowance)
- Minting and burning capabilities
- Access control with owner-only functions
- Built-in safety features (overflow protection, zero address checks)
- Comprehensive test suite

**Key Features:**
- Automatic overflow/underflow protection
- Reentrancy guards on state-changing functions
- Role-based access control
- Event emission for transparency
- Gas-optimized operations

**Usage:**
```augustium
// Deploy a new token
let token = SimpleToken::new(
    "My Token",
    "MTK", 
    18,
    1000000 * 10u256.pow(18)
);

// Transfer tokens
token.transfer(recipient, 100 * 10u256.pow(18));

// Approve spending
token.approve(spender, 200 * 10u256.pow(18));
```

### 2. Voting Contract (`voting-contract.aug`)

A decentralized governance system featuring:
- Proposal creation and voting
- Time-based voting periods
- Quorum and approval thresholds
- Execution delays for security
- Voter registration and weight management

**Key Features:**
- Time-locked proposal execution
- Configurable voting parameters
- Voter weight management
- Proposal state tracking
- Emergency pause functionality

**Usage:**
```augustium
// Deploy voting contract
let voting = VotingContract::new(
    24,  // 24 hours voting duration
    1,   // minimum voting weight
    20,  // 20% quorum threshold
    60,  // 60% approval threshold
    48   // 48 hours execution delay
);

// Register voters
voting.register_voter(voter_address, 10);

// Create proposal
let proposal_id = voting.create_proposal(
    "Increase protocol fee to 0.1%",
    Some(target_contract),
    Some(call_data),
    0
);

// Vote on proposal
voting.cast_vote(proposal_id, true, "I support this change");
```

### 3. Liquidity Pool (`liquidity-pool.aug`)

An Automated Market Maker (AMM) implementation featuring:
- Constant product formula (x * y = k)
- Liquidity provision and removal
- Token swapping with fees
- Flash loans
- Price oracle functionality
- Slippage protection

**Key Features:**
- Mathematical precision with overflow protection
- Fee collection and distribution
- Price impact calculation
- Flash loan functionality
- Emergency pause mechanisms
- Comprehensive testing

**Usage:**
```augustium
// Deploy liquidity pool
let pool = LiquidityPool::new(
    token_a_address,
    token_b_address,
    30,   // 0.3% trading fee
    5,    // 0.05% protocol fee
    fee_collector,
    500   // 5% max slippage
);

// Add liquidity
let (amount_a, amount_b, liquidity) = pool.add_liquidity(
    1000 * 1e18,  // desired amount A
    1000 * 1e18,  // desired amount B
    950 * 1e18,   // minimum amount A
    950 * 1e18,   // minimum amount B
    deadline
);

// Swap tokens
let amount_out = pool.swap(
    token_a_address,
    100 * 1e18,   // amount in
    95 * 1e18,    // minimum amount out
    deadline
);
```

## Common Patterns

### Safety Patterns

1. **Automatic Overflow Protection**
```augustium
// Augustium automatically prevents overflows
let result = a + b; // Safe addition
let product = x * y; // Safe multiplication
```

2. **Reentrancy Protection**
```augustium
#[non_reentrant]
fn withdraw(&mut self, amount: u256) {
    // Function is automatically protected against reentrancy
    self.balance -= amount;
    self.transfer_to_user(amount);
}
```

3. **Access Control**
```augustium
#[only_owner]
fn admin_function(&mut self) {
    // Only contract owner can call this function
}
```

### Error Handling

```augustium
// Use require! for input validation
require!(amount > 0, "Amount must be positive");
require!(recipient != address::zero(), "Invalid recipient");

// Use Result types for fallible operations
fn transfer(&mut self, to: address, amount: u256) -> Result<bool, string> {
    if self.balance < amount {
        return Err("Insufficient balance".to_string());
    }
    // ... transfer logic
    Ok(true)
}
```

### Event Emission

```augustium
#[derive(Event)]
struct Transfer {
    from: address,
    to: address,
    value: u256
}

// Emit events for transparency
emit Transfer {
    from: msg.sender,
    to: recipient,
    value: amount
};
```

## Testing

All examples include comprehensive test suites demonstrating:
- Unit testing of individual functions
- Integration testing of contract interactions
- Edge case testing
- Security testing (overflow, reentrancy, access control)

### Running Tests

```bash
# Compile and test a specific contract
augustium test examples/simple-token.aug

# Run all example tests
augustium test examples/

# Run tests with coverage
augustium test --coverage examples/
```

### Test Structure

```augustium
#[cfg(test)]
mod tests {
    use super::*;
    use std::testing::*;
    
    #[test]
    fn test_basic_functionality() {
        // Test setup
        let mut contract = Contract::new(...);
        
        // Test execution
        let result = contract.some_function(...);
        
        // Assertions
        assert_eq!(result, expected_value);
        assert!(condition);
    }
    
    #[test]
    fn test_error_conditions() {
        let mut contract = Contract::new(...);
        
        // Test that errors are properly handled
        let result = test::try_call(|| {
            contract.failing_function();
        });
        
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Expected error message"));
    }
}
```

## Security Considerations

### Built-in Protections

1. **Arithmetic Safety**: All arithmetic operations are checked for overflow/underflow
2. **Memory Safety**: Automatic bounds checking prevents buffer overflows
3. **Reentrancy Protection**: Built-in guards prevent reentrancy attacks
4. **Type Safety**: Strong typing prevents many classes of bugs

### Best Practices

1. **Input Validation**
   - Always validate function parameters
   - Check for zero addresses and zero amounts
   - Validate array bounds and string lengths

2. **State Management**
   - Update state before external calls
   - Use checks-effects-interactions pattern
   - Implement proper access controls

3. **Error Handling**
   - Use descriptive error messages
   - Handle all possible error conditions
   - Provide graceful failure modes

4. **Gas Optimization**
   - Minimize storage operations
   - Use appropriate data types
   - Batch operations when possible

## Deployment

### Local Development

```bash
# Compile contracts
augustium build examples/

# Deploy to local testnet
augustium deploy --network local examples/simple-token.aug

# Interact with deployed contract
augustium call --network local <contract_address> "transfer(address,uint256)" <recipient> <amount>
```

### Testnet Deployment

```bash
# Deploy to testnet
augustium deploy --network testnet examples/simple-token.aug

# Verify contract
augustium verify --network testnet <contract_address> examples/simple-token.aug
```

### Mainnet Deployment

```bash
# Deploy to mainnet (use with caution)
augustium deploy --network mainnet examples/simple-token.aug

# Verify contract
augustium verify --network mainnet <contract_address> examples/simple-token.aug
```

## Gas Optimization Tips

1. **Use Appropriate Types**
   ```augustium
   // Use smaller types when possible
   let small_number: u8 = 255;
   let medium_number: u64 = 1000000;
   ```

2. **Minimize Storage Operations**
   ```augustium
   // Bad: Multiple storage writes
   self.balance = self.balance - amount;
   self.total_supply = self.total_supply - amount;
   
   // Good: Batch operations
   let new_balance = self.balance - amount;
   let new_total = self.total_supply - amount;
   self.balance = new_balance;
   self.total_supply = new_total;
   ```

3. **Use Events Instead of Storage**
   ```augustium
   // For data that doesn't need to be queried on-chain
   emit DataLogged { data: important_info };
   ```

## Contributing

To contribute new examples:

1. Follow the existing code style and patterns
2. Include comprehensive tests
3. Add documentation and comments
4. Ensure security best practices
5. Test on multiple networks

### Example Template

```augustium
// Contract Description
// Explains what the contract does and key features

use std::collections::HashMap;
use std::events::Event;
use std::access::Ownable;

// Events
#[derive(Event)]
struct SomeEvent {
    // Event fields
}

// Main contract
contract ExampleContract extends Ownable {
    // State variables
    
    // Constructor
    constructor(/* parameters */) {
        // Initialization logic
    }
    
    // Public functions
    
    // View functions
    
    // Internal functions
    
    // Tests
    #[cfg(test)]
    mod tests {
        // Test cases
    }
}
```

## Resources

- [Augustium Language Specification](../language-specification.md)
- [Syntax Reference](../syntax-reference.md)
- [Type System](../type-system.md)
- [Safety Features](../safety-features.md)
- [Standard Library](../stdlib-reference.md)
- [DeFi Templates](../defi-templates.md)

## License

These examples are provided under the MIT License. See the main project LICENSE file for details.