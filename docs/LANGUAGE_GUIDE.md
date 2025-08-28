# Augustium Language Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Basic Syntax](#basic-syntax)
3. [Type System](#type-system)
4. [Generics](#generics)
5. [Pattern Matching](#pattern-matching)
6. [Async Programming](#async-programming)
7. [Macros](#macros)
8. [Operator Overloading](#operator-overloading)
9. [Machine Learning](#machine-learning)
10. [Contract Development](#contract-development)
11. [Testing](#testing)
12. [Security](#security)

## Introduction

Augustium is a modern, high-performance programming language designed for blockchain smart contracts and decentralized applications. It combines the safety of Rust with blockchain-specific features and native machine learning capabilities.

## Basic Syntax

### Variables and Constants

```augustium
// Immutable variable (default)
let x = 42;
let name = "Alice";

// Mutable variable
let mut counter = 0;
counter += 1;

// Constants
const MAX_SUPPLY: u64 = 1_000_000;
```

### Functions

```augustium
// Basic function
fn add(a: u32, b: u32) -> u32 {
    a + b
}

// Public function (visible outside module)
pub fn get_balance() -> u64 {
    self.balance
}

// Function with multiple return values
fn divide_with_remainder(a: u32, b: u32) -> (u32, u32) {
    (a / b, a % b)
}
```

### Control Flow

```augustium
// If expressions
let result = if condition {
    "true branch"
} else {
    "false branch"
};

// Loops
for i in 0..10 {
    println!("Number: {}", i);
}

while condition {
    // loop body
}

loop {
    if should_break {
        break;
    }
}
```

## Type System

### Primitive Types

```augustium
// Integers
let small: u8 = 255;
let medium: u32 = 4_294_967_295;
let large: u64 = 18_446_744_073_709_551_615;
let signed: i32 = -2_147_483_648;

// Blockchain-specific integers
let wei: u256 = 1_000_000_000_000_000_000; // 1 ETH in wei
let gas: u64 = 21_000;

// Floating point
let price: f64 = 3.14159;

// Boolean
let is_valid: bool = true;

// String
let message: String = "Hello, Augustium!";

// Address (blockchain address)
let owner: Address = "0x742d35Cc6634C0532925a3b8D4C2c4c4c4c4c4c4";
```

### Compound Types

```augustium
// Arrays
let numbers: [u32; 5] = [1, 2, 3, 4, 5];
let dynamic_array: Vec<u32> = vec![1, 2, 3];

// Tuples
let point: (f64, f64) = (3.0, 4.0);
let (x, y) = point; // Destructuring

// Structs
struct User {
    name: String,
    age: u32,
    active: bool,
}

let user = User {
    name: "Alice".to_string(),
    age: 30,
    active: true,
};

// Enums
enum Status {
    Active,
    Inactive,
    Pending(String),
}

let current_status = Status::Pending("Verification required".to_string());
```

## Generics

Augustium supports powerful generic programming with type parameters, bounds, and associated types.

### Generic Functions

```augustium
// Generic function
fn swap<T>(a: &mut T, b: &mut T) {
    let temp = std::mem::replace(a, std::mem::replace(b, temp));
}

// Generic function with bounds
fn compare<T: Ord>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

// Multiple type parameters
fn zip<T, U>(a: Vec<T>, b: Vec<U>) -> Vec<(T, U)> {
    a.into_iter().zip(b.into_iter()).collect()
}
```

### Generic Structs

```augustium
// Generic struct
struct Container<T> {
    value: T,
}

impl<T> Container<T> {
    fn new(value: T) -> Self {
        Container { value }
    }
    
    fn get(&self) -> &T {
        &self.value
    }
}

// Generic struct with bounds
struct Wallet<T: Token> {
    balance: u64,
    token: T,
}
```

### Generic Traits

```augustium
// Generic trait
trait Converter<T> {
    fn convert(&self) -> T;
}

// Trait with associated types
trait Iterator {
    type Item;
    
    fn next(&mut self) -> Option<Self::Item>;
}

// Implementation with associated types
impl Iterator for Counter {
    type Item = u32;
    
    fn next(&mut self) -> Option<Self::Item> {
        // implementation
    }
}
```

## Pattern Matching

Augustium provides powerful pattern matching with advanced destructuring capabilities.

### Basic Patterns

```augustium
match value {
    0 => "zero",
    1 => "one",
    2..=10 => "small number",
    _ => "other",
}
```

### Destructuring Patterns

```augustium
// Tuple destructuring
let point = (3, 4);
match point {
    (0, 0) => "origin",
    (x, 0) => format!("on x-axis at {}", x),
    (0, y) => format!("on y-axis at {}", y),
    (x, y) => format!("point at ({}, {})", x, y),
}

// Struct destructuring
struct Person { name: String, age: u32 }
let person = Person { name: "Alice".to_string(), age: 30 };

match person {
    Person { name, age: 0..=17 } => format!("{} is a minor", name),
    Person { name, age: 18..=64 } => format!("{} is an adult", name),
    Person { name, age } => format!("{} is {} years old", name, age),
}

// Array destructuring
match array {
    [] => "empty",
    [x] => format!("single element: {}", x),
    [x, y] => format!("two elements: {}, {}", x, y),
    [first, .., last] => format!("first: {}, last: {}", first, last),
    [head, tail @ ..] => format!("head: {}, tail: {:?}", head, tail),
}
```

### Guards and Bindings

```augustium
match value {
    x if x > 100 => "large",
    x @ 50..=100 => format!("medium: {}", x),
    x => format!("small: {}", x),
}

// Or patterns
match status {
    Status::Active | Status::Pending(_) => "operational",
    Status::Inactive => "down",
}
```

## Async Programming

Augustium supports async/await for concurrent programming and non-blocking operations.

### Async Functions

```augustium
// Async function
async fn fetch_data(url: String) -> Result<String, Error> {
    let response = http_client.get(&url).await?;
    let body = response.text().await?;
    Ok(body)
}

// Calling async functions
async fn process_data() {
    let data = fetch_data("https://api.example.com/data".to_string()).await;
    match data {
        Ok(content) => println!("Received: {}", content),
        Err(e) => println!("Error: {}", e),
    }
}
```

### Futures and Streams

```augustium
use std::future::Future;
use std::stream::Stream;

// Working with futures
let future1 = fetch_data("url1".to_string());
let future2 = fetch_data("url2".to_string());

// Concurrent execution
let (result1, result2) = futures::join!(future1, future2);

// Stream processing
async fn process_stream<S>(mut stream: S) 
where 
    S: Stream<Item = String> + Unpin,
{
    while let Some(item) = stream.next().await {
        println!("Processing: {}", item);
    }
}
```

### Async in Contracts

```augustium
contract AsyncExample {
    async fn fetch_oracle_price(&self, symbol: String) -> Result<u64, Error> {
        let oracle_response = self.call_oracle(symbol).await?;
        Ok(oracle_response.price)
    }
    
    pub async fn update_price(&mut self, symbol: String) {
        match self.fetch_oracle_price(symbol).await {
            Ok(price) => self.prices.insert(symbol, price),
            Err(e) => emit PriceUpdateFailed { error: e.to_string() },
        }
    }
}
```

## Macros

Augustium provides a powerful macro system for metaprogramming.

### Declarative Macros

```augustium
// Simple macro
macro_rules! say_hello {
    () => {
        println!("Hello, Augustium!");
    };
}

// Macro with parameters
macro_rules! create_function {
    ($func_name:ident, $return_type:ty, $value:expr) => {
        fn $func_name() -> $return_type {
            $value
        }
    };
}

create_function!(get_answer, u32, 42);

// Variadic macro
macro_rules! vec_of_strings {
    ($($x:expr),*) => {
        vec![$(String::from($x)),*]
    };
}

let strings = vec_of_strings!["hello", "world", "augustium"];
```

### Built-in Macros

```augustium
// Assertion macros
assert!(condition, "Custom error message");
assert_eq!(left, right);
assert_ne!(left, right);

// Debugging macros
debug_assert!(expensive_check());
dbg!(variable);

// Formatting macros
println!("Value: {}", value);
format!("Hello, {}!", name);

// Contract-specific macros
require!(msg.sender == owner, "Only owner can call this function");
emit!(Transfer { from: sender, to: recipient, amount: value });
```

## Operator Overloading

Augustium allows custom types to implement operators for natural syntax.

### Arithmetic Operators

```augustium
struct Vector2D {
    x: f64,
    y: f64,
}

// Addition operator
impl Add for Vector2D {
    type Output = Vector2D;
    
    fn add(self, other: Vector2D) -> Vector2D {
        Vector2D {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

// Usage
let v1 = Vector2D { x: 1.0, y: 2.0 };
let v2 = Vector2D { x: 3.0, y: 4.0 };
let v3 = v1 + v2; // Calls the add method
```

### Comparison Operators

```augustium
#[derive(PartialEq, Eq)]
struct Money {
    amount: u64,
    currency: String,
}

impl PartialOrd for Money {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.currency == other.currency {
            self.amount.partial_cmp(&other.amount)
        } else {
            None // Can't compare different currencies
        }
    }
}
```

### Index Operators

```augustium
struct Matrix {
    data: Vec<Vec<f64>>,
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;
    
    fn index(&self, (row, col): (usize, usize)) -> &f64 {
        &self.data[row][col]
    }
}

// Usage
let matrix = Matrix { /* ... */ };
let value = matrix[(0, 1)]; // Calls index method
```

## Machine Learning

Augustium provides native machine learning capabilities for smart contracts.

### Creating Models

```augustium
use stdlib::ml::{NeuralNetwork, LinearRegression, DecisionTree};

contract MLContract {
    let mut price_model: LinearRegression;
    let mut classifier: NeuralNetwork;
    
    fn constructor() {
        // Linear regression for price prediction
        self.price_model = LinearRegression::new();
        
        // Neural network for classification
        self.classifier = NeuralNetwork::new(vec![10, 5, 2]);
    }
}
```

### Training Models

```augustium
pub fn train_price_model(&mut self, training_data: Vec<(Vec<f64>, f64)>) {
    let metrics = self.price_model.train(&training_data, TrainingConfig {
        learning_rate: 0.01,
        epochs: 100,
        batch_size: 32,
    });
    
    emit ModelTrained {
        model_type: "LinearRegression".to_string(),
        accuracy: metrics.accuracy,
        loss: metrics.final_loss,
    };
}
```

### Making Predictions

```augustium
pub fn predict_price(&self, features: Vec<f64>) -> f64 {
    require!(features.len() == self.price_model.input_size(), "Invalid feature count");
    
    let prediction = self.price_model.predict(&features);
    
    emit PredictionMade {
        features: features,
        prediction: prediction,
        timestamp: block.timestamp,
    };
    
    prediction
}
```

### Advanced ML Features

```augustium
// Tensor operations
let tensor_a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let tensor_b = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let result = tensor_a.matmul(&tensor_b);

// GPU acceleration (when available)
let gpu_model = NeuralNetwork::new_gpu(vec![1000, 500, 100, 10]);
gpu_model.train_parallel(&large_dataset);

// Model serialization
let model_bytes = self.classifier.serialize();
storage.store("model_v1", model_bytes);
```

## Contract Development

### Basic Contract Structure

```augustium
contract Token {
    // State variables
    let mut total_supply: u64;
    let mut balances: Map<Address, u64>;
    let mut allowances: Map<Address, Map<Address, u64>>;
    let owner: Address;
    
    // Events
    event Transfer {
        from: Address,
        to: Address,
        amount: u64,
    }
    
    event Approval {
        owner: Address,
        spender: Address,
        amount: u64,
    }
    
    // Constructor
    fn constructor(initial_supply: u64) {
        self.total_supply = initial_supply;
        self.owner = msg.sender;
        self.balances.insert(msg.sender, initial_supply);
    }
    
    // Public functions
    pub fn transfer(&mut self, to: Address, amount: u64) -> bool {
        require!(self.balances.get(&msg.sender).unwrap_or(&0) >= &amount, "Insufficient balance");
        
        self.balances.entry(msg.sender).and_modify(|e| *e -= amount);
        self.balances.entry(to).and_modify(|e| *e += amount).or_insert(amount);
        
        emit Transfer {
            from: msg.sender,
            to: to,
            amount: amount,
        };
        
        true
    }
    
    pub fn balance_of(&self, account: Address) -> u64 {
        self.balances.get(&account).copied().unwrap_or(0)
    }
}
```

### Access Control

```augustium
contract AccessControlled {
    let owner: Address;
    let mut admins: Set<Address>;
    
    modifier only_owner() {
        require!(msg.sender == self.owner, "Only owner");
        _;
    }
    
    modifier only_admin() {
        require!(self.admins.contains(&msg.sender) || msg.sender == self.owner, "Only admin");
        _;
    }
    
    pub fn add_admin(&mut self, admin: Address) only_owner {
        self.admins.insert(admin);
    }
    
    pub fn critical_function(&mut self) only_admin {
        // Critical functionality here
    }
}
```

## Testing

### Unit Tests

```augustium
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_token_transfer() {
        let mut token = Token::new(1000);
        let alice = Address::from("0x1111...");
        let bob = Address::from("0x2222...");
        
        // Setup
        token.balances.insert(alice, 100);
        
        // Test transfer
        assert!(token.transfer_from(alice, bob, 50));
        assert_eq!(token.balance_of(alice), 50);
        assert_eq!(token.balance_of(bob), 50);
    }
    
    #[test]
    #[should_panic(expected = "Insufficient balance")]
    fn test_transfer_insufficient_balance() {
        let mut token = Token::new(1000);
        let alice = Address::from("0x1111...");
        let bob = Address::from("0x2222...");
        
        token.balances.insert(alice, 10);
        token.transfer_from(alice, bob, 50); // Should panic
    }
}
```

### Property-Based Testing

```augustium
#[cfg(test)]
mod property_tests {
    use super::*;
    use augustium::testing::PropertyTest;
    
    #[property_test]
    fn test_transfer_preserves_total_supply(
        initial_supply: u64,
        from_balance: u64,
        to_balance: u64,
        transfer_amount: u64
    ) {
        let mut token = Token::new(initial_supply);
        
        // Assume valid initial state
        assume!(from_balance + to_balance <= initial_supply);
        assume!(transfer_amount <= from_balance);
        
        let alice = Address::from("0x1111...");
        let bob = Address::from("0x2222...");
        
        token.balances.insert(alice, from_balance);
        token.balances.insert(bob, to_balance);
        
        let total_before = token.total_supply();
        token.transfer_from(alice, bob, transfer_amount);
        let total_after = token.total_supply();
        
        assert_eq!(total_before, total_after);
    }
}
```

### Integration Tests

```augustium
#[cfg(test)]
mod integration_tests {
    use super::*;
    use augustium::testing::{TestEnvironment, deploy_contract};
    
    #[test]
    async fn test_contract_interaction() {
        let env = TestEnvironment::new();
        
        // Deploy contracts
        let token = deploy_contract::<Token>(&env, 1000000).await;
        let exchange = deploy_contract::<Exchange>(&env, token.address()).await;
        
        // Test interaction
        let alice = env.create_account("alice", 1000);
        
        token.transfer(alice.address(), 1000).send_from(env.owner()).await?;
        token.approve(exchange.address(), 500).send_from(alice).await?;
        
        let result = exchange.swap_tokens(500).send_from(alice).await?;
        assert!(result.success);
    }
}
```

## Security

### Common Vulnerabilities

```augustium
// ❌ Vulnerable to reentrancy
contract VulnerableContract {
    let mut balances: Map<Address, u64>;
    
    pub fn withdraw(&mut self, amount: u64) {
        let balance = self.balances.get(&msg.sender).unwrap_or(&0);
        require!(*balance >= amount, "Insufficient balance");
        
        // External call before state change - VULNERABLE!
        msg.sender.transfer(amount);
        
        self.balances.entry(msg.sender).and_modify(|e| *e -= amount);
    }
}

// ✅ Protected against reentrancy
contract SecureContract {
    let mut balances: Map<Address, u64>;
    let mut locked: bool;
    
    modifier non_reentrant() {
        require!(!self.locked, "Reentrant call");
        self.locked = true;
        _;
        self.locked = false;
    }
    
    pub fn withdraw(&mut self, amount: u64) non_reentrant {
        let balance = self.balances.get(&msg.sender).unwrap_or(&0);
        require!(*balance >= amount, "Insufficient balance");
        
        // State change before external call
        self.balances.entry(msg.sender).and_modify(|e| *e -= amount);
        
        msg.sender.transfer(amount);
    }
}
```

### Safe Math Operations

```augustium
use stdlib::math::SafeMath;

contract SafeContract {
    let mut total_supply: u64;
    
    pub fn mint(&mut self, amount: u64) {
        // Safe addition prevents overflow
        self.total_supply = self.total_supply.safe_add(amount)
            .expect("Total supply overflow");
    }
    
    pub fn burn(&mut self, amount: u64) {
        // Safe subtraction prevents underflow
        self.total_supply = self.total_supply.safe_sub(amount)
            .expect("Total supply underflow");
    }
}
```

### Input Validation

```augustium
contract ValidatedContract {
    pub fn transfer(&mut self, to: Address, amount: u64) {
        // Validate inputs
        require!(to != Address::zero(), "Cannot transfer to zero address");
        require!(to != msg.sender, "Cannot transfer to self");
        require!(amount > 0, "Amount must be positive");
        require!(amount <= MAX_TRANSFER, "Amount exceeds maximum");
        
        // Additional business logic validation
        require!(self.is_transfer_allowed(msg.sender, to), "Transfer not allowed");
        
        // Proceed with transfer...
    }
}
```

This comprehensive language guide covers all the major features and improvements we've implemented for Augustium. The language now has modern capabilities including generics, async/await, macros, operator overloading, advanced pattern matching, and comprehensive tooling support.
