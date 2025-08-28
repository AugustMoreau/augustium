//! Property-based testing framework for Augustium
//! Generates random test cases to verify contract properties

use crate::ast::*;
use crate::avm::AugustiumVM;
use crate::codegen::Value;
use crate::error::{Result, VmError};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::fmt;

/// Property-based test configuration
#[derive(Debug, Clone)]
pub struct PropertyTestConfig {
    pub num_tests: usize,
    pub max_size: usize,
    pub seed: Option<u64>,
    pub shrink_attempts: usize,
}

impl Default for PropertyTestConfig {
    fn default() -> Self {
        Self {
            num_tests: 100,
            max_size: 100,
            seed: None,
            shrink_attempts: 100,
        }
    }
}

/// Property test result
#[derive(Debug)]
pub enum PropertyTestResult {
    Success {
        tests_run: usize,
    },
    Failure {
        failing_input: TestInput,
        error: String,
        shrunk_input: Option<TestInput>,
    },
}

/// Test input generator
pub trait Generator<T> {
    fn generate(&self, rng: &mut StdRng, size: usize) -> T;
    fn shrink(&self, value: &T) -> Vec<T>;
}

/// Test input wrapper
#[derive(Debug, Clone)]
pub struct TestInput {
    pub values: HashMap<String, Value>,
}

/// Property test runner
pub struct PropertyTester {
    config: PropertyTestConfig,
    rng: StdRng,
}

impl PropertyTester {
    pub fn new(config: PropertyTestConfig) -> Self {
        let seed = config.seed.unwrap_or_else(|| rand::random());
        let rng = StdRng::seed_from_u64(seed);
        
        Self { config, rng }
    }

    /// Run a property test
    pub fn test_property<F>(&mut self, property: F) -> PropertyTestResult 
    where
        F: Fn(&TestInput) -> Result<bool>,
    {
        for i in 0..self.config.num_tests {
            let size = (i * self.config.max_size) / self.config.num_tests;
            let input = self.generate_input(size);
            
            match property(&input) {
                Ok(true) => continue,
                Ok(false) => {
                    // Property failed, try to shrink
                    let shrunk = self.shrink_input(&input, &property);
                    return PropertyTestResult::Failure {
                        failing_input: input,
                        error: "Property assertion failed".to_string(),
                        shrunk_input: shrunk,
                    };
                }
                Err(e) => {
                    let shrunk = self.shrink_input(&input, &property);
                    return PropertyTestResult::Failure {
                        failing_input: input,
                        error: format!("Property test error: {}", e),
                        shrunk_input: shrunk,
                    };
                }
            }
        }
        
        PropertyTestResult::Success {
            tests_run: self.config.num_tests,
        }
    }

    fn generate_input(&mut self, size: usize) -> TestInput {
        let mut values = HashMap::new();
        
        // Generate various types of test values
        values.insert("uint".to_string(), Value::U64(self.rng.gen_range(0..=size as u64)));
        values.insert("int".to_string(), Value::I64(self.rng.gen_range(-(size as i64)..=size as i64)));
        values.insert("bool".to_string(), Value::Bool(self.rng.gen()));
        values.insert("address".to_string(), Value::Address(format!("0x{:040x}", self.rng.gen::<u64>())));
        
        // Generate string of random length
        let str_len = self.rng.gen_range(0..=size.min(50));
        let random_string: String = (0..str_len)
            .map(|_| self.rng.gen_range(b'a'..=b'z') as char)
            .collect();
        values.insert("string".to_string(), Value::String(random_string));
        
        // Generate array of random values
        let array_len = self.rng.gen_range(0..=size.min(20));
        let array_values: Vec<Value> = (0..array_len)
            .map(|_| Value::U64(self.rng.gen_range(0..100)))
            .collect();
        values.insert("array".to_string(), Value::Array(array_values));
        
        TestInput { values }
    }

    fn shrink_input<F>(&mut self, input: &TestInput, property: &F) -> Option<TestInput>
    where
        F: Fn(&TestInput) -> Result<bool>,
    {
        let mut current = input.clone();
        
        for _ in 0..self.config.shrink_attempts {
            let mut shrunk = current.clone();
            let mut changed = false;
            
            // Try to shrink each value
            for (key, value) in shrunk.values.iter_mut() {
                match value {
                    Value::U64(n) if *n > 0 => {
                        *n = *n / 2;
                        changed = true;
                    }
                    Value::I64(n) if n.abs() > 0 => {
                        *n = *n / 2;
                        changed = true;
                    }
                    Value::String(s) if !s.is_empty() => {
                        s.truncate(s.len() / 2);
                        changed = true;
                    }
                    Value::Array(arr) if !arr.is_empty() => {
                        arr.truncate(arr.len() / 2);
                        changed = true;
                    }
                    _ => {}
                }
            }
            
            if !changed {
                break;
            }
            
            // Test if the shrunk input still fails
            match property(&shrunk) {
                Ok(false) | Err(_) => {
                    current = shrunk;
                }
                Ok(true) => {
                    // Shrunk input passes, keep the previous one
                    break;
                }
            }
        }
        
        if current.values != input.values {
            Some(current)
        } else {
            None
        }
    }
}

/// Built-in generators
pub struct IntGenerator {
    pub min: i64,
    pub max: i64,
}

impl Generator<i64> for IntGenerator {
    fn generate(&self, rng: &mut StdRng, _size: usize) -> i64 {
        rng.gen_range(self.min..=self.max)
    }

    fn shrink(&self, value: &i64) -> Vec<i64> {
        let mut shrunk = Vec::new();
        
        if *value != 0 {
            shrunk.push(0);
        }
        
        if value.abs() > 1 {
            shrunk.push(value / 2);
            shrunk.push(-value / 2);
        }
        
        if *value > self.min {
            shrunk.push(value - 1);
        }
        
        if *value < self.max {
            shrunk.push(value + 1);
        }
        
        shrunk
    }
}

pub struct StringGenerator {
    pub max_length: usize,
}

impl Generator<String> for StringGenerator {
    fn generate(&self, rng: &mut StdRng, size: usize) -> String {
        let len = rng.gen_range(0..=self.max_length.min(size));
        (0..len)
            .map(|_| rng.gen_range(b'a'..=b'z') as char)
            .collect()
    }

    fn shrink(&self, value: &String) -> Vec<String> {
        let mut shrunk = Vec::new();
        
        if !value.is_empty() {
            shrunk.push(String::new());
        }
        
        if value.len() > 1 {
            shrunk.push(value[..value.len()/2].to_string());
            shrunk.push(value[1..].to_string());
            shrunk.push(value[..value.len()-1].to_string());
        }
        
        shrunk
    }
}

/// Contract property testing utilities
pub struct ContractPropertyTester {
    vm: AugustiumVM,
    tester: PropertyTester,
}

impl ContractPropertyTester {
    pub fn new(config: PropertyTestConfig) -> Self {
        Self {
            vm: AugustiumVM::new(),
            tester: PropertyTester::new(config),
        }
    }

    /// Test that a contract function never reverts for valid inputs
    pub fn test_no_revert<F>(&mut self, function_name: &str, input_generator: F) -> PropertyTestResult
    where
        F: Fn(&mut StdRng, usize) -> Vec<Value>,
    {
        self.tester.test_property(|input| {
            // Execute the function with the generated input
            let args = input_generator(&mut StdRng::from_entropy(), 10);
            match self.vm.call_function(function_name, args) {
                Ok(_) => Ok(true),
                Err(VmError { kind: crate::error::VmErrorKind::Revert, .. }) => Ok(false),
                Err(e) => Err(e.into()),
            }
        })
    }

    /// Test that a contract maintains an invariant
    pub fn test_invariant<F>(&mut self, invariant: F) -> PropertyTestResult
    where
        F: Fn(&AugustiumVM) -> Result<bool>,
    {
        self.tester.test_property(|_input| {
            invariant(&self.vm)
        })
    }

    /// Test that two implementations are equivalent
    pub fn test_equivalence<F1, F2>(&mut self, impl1: F1, impl2: F2) -> PropertyTestResult
    where
        F1: Fn(&TestInput) -> Result<Value>,
        F2: Fn(&TestInput) -> Result<Value>,
    {
        self.tester.test_property(|input| {
            let result1 = impl1(input)?;
            let result2 = impl2(input)?;
            Ok(result1 == result2)
        })
    }
}

/// Fuzzing utilities
pub struct ContractFuzzer {
    vm: AugustiumVM,
    rng: StdRng,
    crash_inputs: Vec<TestInput>,
}

impl ContractFuzzer {
    pub fn new(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| rand::random());
        Self {
            vm: AugustiumVM::new(),
            rng: StdRng::seed_from_u64(seed),
            crash_inputs: Vec::new(),
        }
    }

    /// Fuzz a contract function with random inputs
    pub fn fuzz_function(&mut self, function_name: &str, iterations: usize) -> FuzzResult {
        let mut crashes = 0;
        let mut reverts = 0;
        let mut successes = 0;

        for _ in 0..iterations {
            let input = self.generate_random_input();
            let args = self.input_to_args(&input);

            match self.vm.call_function(function_name, args) {
                Ok(_) => successes += 1,
                Err(VmError { kind: crate::error::VmErrorKind::Revert, .. }) => reverts += 1,
                Err(_) => {
                    crashes += 1;
                    self.crash_inputs.push(input);
                }
            }
        }

        FuzzResult {
            iterations,
            successes,
            reverts,
            crashes,
            crash_inputs: self.crash_inputs.clone(),
        }
    }

    fn generate_random_input(&mut self) -> TestInput {
        let mut values = HashMap::new();
        
        // Generate extreme values that might cause issues
        let extreme_values = vec![
            Value::U64(0),
            Value::U64(u64::MAX),
            Value::U64(u64::MAX / 2),
            Value::I64(i64::MIN),
            Value::I64(i64::MAX),
            Value::I64(0),
            Value::String(String::new()),
            Value::String("a".repeat(1000)),
            Value::Array(vec![]),
            Value::Array(vec![Value::U64(0); 1000]),
        ];

        for (i, extreme_val) in extreme_values.iter().enumerate() {
            values.insert(format!("extreme_{}", i), extreme_val.clone());
        }

        // Add some random values
        values.insert("random_uint".to_string(), Value::U64(self.rng.gen()));
        values.insert("random_int".to_string(), Value::I64(self.rng.gen()));
        values.insert("random_bool".to_string(), Value::Bool(self.rng.gen()));

        TestInput { values }
    }

    fn input_to_args(&self, input: &TestInput) -> Vec<Value> {
        input.values.values().cloned().collect()
    }
}

#[derive(Debug)]
pub struct FuzzResult {
    pub iterations: usize,
    pub successes: usize,
    pub reverts: usize,
    pub crashes: usize,
    pub crash_inputs: Vec<TestInput>,
}

impl fmt::Display for FuzzResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, 
            "Fuzz Results: {} iterations, {} successes, {} reverts, {} crashes",
            self.iterations, self.successes, self.reverts, self.crashes
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_property_tester() {
        let config = PropertyTestConfig::default();
        let mut tester = PropertyTester::new(config);

        // Test a simple property: all generated uints are non-negative
        let result = tester.test_property(|input| {
            if let Some(Value::U64(n)) = input.values.get("uint") {
                Ok(*n >= 0)
            } else {
                Ok(false)
            }
        });

        match result {
            PropertyTestResult::Success { tests_run } => {
                assert_eq!(tests_run, 100);
            }
            PropertyTestResult::Failure { .. } => {
                panic!("Property test should not fail");
            }
        }
    }

    #[test]
    fn test_int_generator() {
        let gen = IntGenerator { min: -10, max: 10 };
        let mut rng = StdRng::seed_from_u64(42);
        
        let value = gen.generate(&mut rng, 5);
        assert!(value >= -10 && value <= 10);
        
        let shrunk = gen.shrink(&value);
        assert!(!shrunk.is_empty());
    }

    #[test]
    fn test_string_generator() {
        let gen = StringGenerator { max_length: 20 };
        let mut rng = StdRng::seed_from_u64(42);
        
        let value = gen.generate(&mut rng, 10);
        assert!(value.len() <= 20);
        
        let shrunk = gen.shrink(&value);
        if !value.is_empty() {
            assert!(!shrunk.is_empty());
        }
    }
}
