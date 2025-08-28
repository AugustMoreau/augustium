# Augustium Migration Guide

## Table of Contents
1. [Overview](#overview)
2. [Breaking Changes](#breaking-changes)
3. [New Features Migration](#new-features-migration)
4. [Code Examples](#code-examples)
5. [Tooling Updates](#tooling-updates)
6. [Performance Improvements](#performance-improvements)
7. [Troubleshooting](#troubleshooting)

## Overview

This guide helps you migrate existing Augustium contracts and applications to take advantage of the new language features including generics, async/await, macros, enhanced pattern matching, operator overloading, and improved tooling.

## Breaking Changes

### AST Structure Changes

**Before:**
```augustium
struct Function {
    name: Identifier,
    parameters: Vec<Parameter>,
    return_type: Option<Type>,
    body: Block,
}
```

**After:**
```augustium
struct Function {
    name: Identifier,
    type_parameters: Vec<TypeParameter>,  // NEW
    parameters: Vec<Parameter>,
    return_type: Option<Type>,
    body: Block,
    is_async: bool,                      // NEW
    where_clause: Option<WhereClause>,   // NEW
}
```

### Pattern Matching Syntax

**Before:**
```augustium
match value {
    Some(x) => x,
    None => 0,
}
```

**After (Enhanced):**
```augustium
match value {
    Some(x) if x > 100 => x * 2,        // Guards now supported
    Some(x @ 50..=100) => x,            // Binding patterns
    Some(x) | None => 0,                // Or patterns
}
```

## New Features Migration

### 1. Adding Generics to Existing Contracts

**Before:**
```augustium
contract TokenContract {
    let mut balances: Map<Address, u64>;
    
    pub fn transfer(&mut self, to: Address, amount: u64) -> bool {
        // Implementation
    }
}
```

**After:**
```augustium
contract TokenContract<T: Numeric + Copy + Default> {
    let mut balances: Map<Address, T>;
    
    pub fn transfer(&mut self, to: Address, amount: T) -> bool 
    where 
        T: PartialOrd + Sub<Output = T> + Add<Output = T>
    {
        // Implementation with generic amounts
    }
}

// Usage
type StandardToken = TokenContract<u64>;
type PrecisionToken = TokenContract<u256>;
```

### 2. Converting to Async Operations

**Before:**
```augustium
contract Oracle {
    pub fn update_price(&mut self, symbol: String) {
        let price = self.fetch_external_price(symbol);
        self.prices.insert(symbol, price);
    }
}
```

**After:**
```augustium
contract Oracle {
    pub async fn update_price(&mut self, symbol: String) {
        match self.fetch_external_price(symbol).await {
            Ok(price) => {
                self.prices.insert(symbol.clone(), price);
                emit PriceUpdated { symbol, price };
            },
            Err(e) => {
                emit PriceUpdateFailed { symbol, error: e.to_string() };
            }
        }
    }
    
    async fn fetch_external_price(&self, symbol: String) -> Result<u64, Error> {
        let response = http::get(&format!("https://api.price.com/{}", symbol)).await?;
        let data: PriceData = response.json().await?;
        Ok(data.price)
    }
}
```

### 3. Using Macros for Code Generation

**Before:**
```augustium
contract Storage {
    let mut value1: u64;
    let mut value2: String;
    let mut value3: bool;
    
    pub fn get_value1(&self) -> u64 { self.value1 }
    pub fn set_value1(&mut self, v: u64) { self.value1 = v; }
    
    pub fn get_value2(&self) -> String { self.value2.clone() }
    pub fn set_value2(&mut self, v: String) { self.value2 = v; }
    
    pub fn get_value3(&self) -> bool { self.value3 }
    pub fn set_value3(&mut self, v: bool) { self.value3 = v; }
}
```

**After:**
```augustium
macro_rules! property {
    ($name:ident, $type:ty) => {
        paste! {
            pub fn [<get_ $name>](&self) -> $type {
                self.$name.clone()
            }
            
            pub fn [<set_ $name>](&mut self, value: $type) {
                self.$name = value;
            }
        }
    };
}

contract Storage {
    let mut value1: u64;
    let mut value2: String;
    let mut value3: bool;
    
    property!(value1, u64);
    property!(value2, String);
    property!(value3, bool);
}
```

### 4. Enhanced Pattern Matching

**Before:**
```augustium
fn process_transaction(tx: Transaction) -> String {
    match tx.transaction_type {
        0 => format!("Transfer: {} to {}", tx.from, tx.to),
        1 => format!("Mint: {} tokens", tx.amount),
        2 => format!("Burn: {} tokens", tx.amount),
        _ => "Unknown".to_string(),
    }
}
```

**After:**
```augustium
enum Transaction {
    Transfer { from: Address, to: Address, amount: u64 },
    Mint { to: Address, amount: u64 },
    Burn { from: Address, amount: u64 },
}

fn process_transaction(tx: Transaction) -> String {
    match tx {
        Transaction::Transfer { from, to, amount } if amount > 1000 => {
            format!("Large transfer: {} from {} to {}", amount, from, to)
        },
        Transaction::Transfer { from, to, amount } => {
            format!("Transfer: {} from {} to {}", amount, from, to)
        },
        Transaction::Mint { to, amount } | Transaction::Burn { from: to, amount } => {
            format!("Token operation: {} tokens for {}", amount, to)
        },
    }
}
```

### 5. Operator Overloading

**Before:**
```augustium
struct Money {
    amount: u64,
    currency: String,
}

impl Money {
    pub fn add_money(&self, other: &Money) -> Result<Money, Error> {
        if self.currency != other.currency {
            return Err("Currency mismatch");
        }
        Ok(Money {
            amount: self.amount + other.amount,
            currency: self.currency.clone(),
        })
    }
}

// Usage
let total = money1.add_money(&money2)?;
```

**After:**
```augustium
struct Money {
    amount: u64,
    currency: String,
}

impl Add for Money {
    type Output = Result<Money, Error>;
    
    fn add(self, other: Money) -> Self::Output {
        if self.currency != other.currency {
            return Err("Currency mismatch");
        }
        Ok(Money {
            amount: self.amount + other.amount,
            currency: self.currency,
        })
    }
}

// Usage
let total = (money1 + money2)?;
```

## Code Examples

### Complete Migration Example

**Before (Legacy Contract):**
```augustium
contract LegacyToken {
    let mut total_supply: u64;
    let mut balances: Map<Address, u64>;
    let mut allowances: Map<Address, Map<Address, u64>>;
    
    pub fn transfer(&mut self, to: Address, amount: u64) -> bool {
        let sender_balance = self.balances.get(&msg.sender).unwrap_or(&0);
        if *sender_balance < amount {
            return false;
        }
        
        self.balances.entry(msg.sender).and_modify(|e| *e -= amount);
        self.balances.entry(to).and_modify(|e| *e += amount).or_insert(amount);
        
        true
    }
    
    pub fn get_price_from_oracle(&mut self, symbol: String) -> u64 {
        // Blocking call
        self.call_external_api(symbol)
    }
}
```

**After (Modern Contract):**
```augustium
use stdlib::ml::LinearRegression;

contract ModernToken<T: TokenAmount> 
where 
    T: Add<Output = T> + Sub<Output = T> + PartialOrd + Copy + Default
{
    let mut total_supply: T;
    let mut balances: Map<Address, T>;
    let mut allowances: Map<Address, Map<Address, T>>;
    let mut price_predictor: LinearRegression;
    
    pub fn transfer(&mut self, to: Address, amount: T) -> Result<(), TokenError> {
        let sender_balance = self.balances.get(&msg.sender).copied()
            .unwrap_or_default();
        
        match sender_balance >= amount {
            true => {
                self.balances.entry(msg.sender).and_modify(|e| *e = *e - amount);
                self.balances.entry(to)
                    .and_modify(|e| *e = *e + amount)
                    .or_insert(amount);
                
                emit Transfer { from: msg.sender, to, amount };
                Ok(())
            },
            false => Err(TokenError::InsufficientBalance),
        }
    }
    
    pub async fn get_price_with_prediction(&mut self, symbol: String) -> Result<T, Error> {
        // Async oracle call
        let current_price = self.fetch_oracle_price(symbol.clone()).await?;
        
        // ML prediction for price trend
        let historical_data = self.get_price_history(&symbol).await?;
        let prediction = self.price_predictor.predict(&historical_data);
        
        emit PricePrediction { 
            symbol, 
            current_price, 
            predicted_change: prediction 
        };
        
        Ok(current_price)
    }
    
    async fn fetch_oracle_price(&self, symbol: String) -> Result<T, Error> {
        let response = http::get(&format!("https://api.oracle.com/price/{}", symbol)).await?;
        let price_data: PriceResponse = response.json().await?;
        Ok(price_data.price.into())
    }
}

// Specialized implementations
type StandardToken = ModernToken<u64>;
type HighPrecisionToken = ModernToken<u256>;
```

## Tooling Updates

### VS Code Extension

The VS Code extension now provides:

1. **Real-time Error Highlighting**: Errors are shown as you type
2. **Advanced Refactoring**: Extract functions, rename symbols, inline variables
3. **ML-aware Completions**: Type-aware suggestions for ML operations
4. **Gas Optimization**: Automated suggestions for gas improvements
5. **Security Auditing**: Real-time security vulnerability detection

### Configuration

Update your `.vscode/settings.json`:

```json
{
    "augustium.formatter.indentSize": 4,
    "augustium.formatter.maxLineLength": 100,
    "augustium.linter.enableRealTimeChecking": true,
    "augustium.analysis.enableGasOptimization": true,
    "augustium.analysis.enableSecurityAudit": true,
    "augustium.ml.enableTypeAwareCompletion": true
}
```

### Build Configuration

Update your `Augustium.toml`:

```toml
[package]
name = "my-contract"
version = "2.0.0"
edition = "2024"

[features]
default = ["async", "ml", "generics"]
async = []
ml = []
generics = []
macros = []

[dependencies]
augustium-std = "2.0"
augustium-ml = "2.0"
augustium-async = "2.0"

[build]
target = "wasm32"
optimization = "release"
```

## Performance Improvements

### Gas Optimization

The new compiler includes automatic optimizations:

1. **Dead Code Elimination**: Unused code is removed
2. **Constant Folding**: Compile-time constant evaluation
3. **Loop Unrolling**: Small loops are unrolled for efficiency
4. **Storage Packing**: Structs are automatically packed
5. **Batch Operations**: Multiple operations are batched when possible

### Memory Management

```augustium
// Old approach - inefficient
fn process_large_data(&self, data: Vec<u8>) -> Vec<u8> {
    let mut result = Vec::new();
    for item in data {
        result.push(process_item(item));
    }
    result
}

// New approach - memory efficient
fn process_large_data(&self, data: &[u8]) -> impl Iterator<Item = u8> + '_ {
    data.iter().map(|&item| self.process_item(item))
}
```

## Troubleshooting

### Common Migration Issues

1. **Generic Type Inference**:
   ```augustium
   // Problem: Type cannot be inferred
   let token = TokenContract::new();
   
   // Solution: Specify type explicitly
   let token: TokenContract<u64> = TokenContract::new();
   ```

2. **Async Function Calls**:
   ```augustium
   // Problem: Forgot to await
   let result = async_function();
   
   // Solution: Add await
   let result = async_function().await?;
   ```

3. **Pattern Matching Exhaustiveness**:
   ```augustium
   // Problem: Non-exhaustive patterns
   match status {
       Status::Active => "active",
       Status::Inactive => "inactive",
       // Missing Status::Pending case
   }
   
   // Solution: Add all cases or wildcard
   match status {
       Status::Active => "active",
       Status::Inactive => "inactive",
       Status::Pending => "pending",
   }
   ```

### Migration Checklist

- [ ] Update `Augustium.toml` with new features
- [ ] Add type parameters to generic contracts
- [ ] Convert blocking calls to async where appropriate
- [ ] Replace repetitive code with macros
- [ ] Enhance pattern matching with new syntax
- [ ] Implement operator overloading for custom types
- [ ] Update tests to use new testing framework
- [ ] Configure VS Code extension settings
- [ ] Run security audit on migrated contracts
- [ ] Test gas optimization improvements

### Getting Help

- **Documentation**: Check the updated API reference
- **Community**: Join the Augustium Discord server
- **Issues**: Report bugs on GitHub
- **Examples**: See the examples directory for migration patterns

This migration guide provides a comprehensive path to upgrading existing Augustium projects to take advantage of all the new language features and improvements.
