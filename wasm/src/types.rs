//! Type definitions and conversions for WASM bindings

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use std::collections::HashMap;

/// Re-export types for JavaScript bindings
pub use crate::compiler::{CompileOptions, CompileResult, CompileMetadata, MethodInfo, ParameterInfo};
pub use crate::runtime::{ContractState, MethodArguments, MethodResult, ContractEvent};

/// JavaScript-compatible error type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmError {
    pub message: String,
    pub code: String,
    pub details: Option<HashMap<String, serde_json::Value>>,
}

impl WasmError {
    pub fn new(message: &str, code: &str) -> Self {
        Self {
            message: message.to_string(),
            code: code.to_string(),
            details: None,
        }
    }
    
    pub fn with_details(mut self, details: HashMap<String, serde_json::Value>) -> Self {
        self.details = Some(details);
        self
    }
}

/// Contract deployment options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentOptions {
    pub initial_balance: Option<u64>,
    pub owner: Option<String>,
    pub gas_limit: Option<u64>,
    pub metadata: Option<HashMap<String, String>>,
    pub address: Option<String>,
    pub constructor_args: Option<Vec<serde_json::Value>>,
    pub enable_debug: Option<bool>,
    pub max_memory: Option<u32>,
    pub timeout_ms: Option<u64>,
}

impl Default for DeploymentOptions {
    fn default() -> Self {
        Self {
            initial_balance: Some(0),
            owner: None,
            gas_limit: Some(1_000_000),
            metadata: None,
            address: None,
            constructor_args: None,
            enable_debug: Some(false),
            max_memory: Some(64 * 1024 * 1024), // 64MB
            timeout_ms: Some(30000), // 30 seconds
        }
    }
}

/// Transaction information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionInfo {
    pub hash: String,
    pub from: String,
    pub to: Option<String>,
    pub value: u64,
    pub gas_limit: u64,
    pub gas_used: u64,
    pub status: TransactionStatus,
    pub block_number: Option<u64>,
    pub timestamp: f64,
}

/// Transaction status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionStatus {
    Pending,
    Success,
    Failed,
    Reverted,
}

/// Blockchain context for contract execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockchainContext {
    pub block_number: u64,
    pub block_timestamp: u64,
    pub block_hash: String,
    pub chain_id: u64,
    pub gas_price: u64,
    pub difficulty: u64,
    pub coinbase: String,
    pub timestamp: f64,
}

impl Default for BlockchainContext {
    fn default() -> Self {
        Self {
            block_number: 1,
            block_timestamp: (js_sys::Date::now() / 1000.0) as u64,
            block_hash: "0x0000000000000000000000000000000000000000000000000000000000000000".to_string(),
            chain_id: 1,
            gas_price: 20_000_000_000, // 20 gwei
            difficulty: 1000,
            coinbase: "0x0000000000000000000000000000000000000000".to_string(),
            timestamp: js_sys::Date::now(),
        }
    }
}

/// AVM state representation for WASM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvmState {
    pub stack: Vec<serde_json::Value>,
    pub memory: Vec<u8>,
    pub storage: HashMap<String, serde_json::Value>,
    pub contracts: HashMap<String, ContractInfo>,
    pub gas_used: u64,
    pub gas_limit: u64,
    pub call_stack_depth: u32,
    pub events: Vec<ContractEvent>,
    pub debug_mode: bool,
    pub halted: bool,
}

impl Default for AvmState {
    fn default() -> Self {
        Self {
            stack: Vec::new(),
            memory: vec![0; 64 * 1024], // 64KB initial memory
            storage: HashMap::new(),
            contracts: HashMap::new(),
            gas_used: 0,
            gas_limit: 1_000_000,
            call_stack_depth: 0,
            events: Vec::new(),
            debug_mode: false,
            halted: false,
        }
    }
}

/// Contract information for state management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractInfo {
    pub address: String,
    pub bytecode: Vec<u8>,
    pub storage: HashMap<String, serde_json::Value>,
    pub balance: u64,
    pub nonce: u64,
    pub code_hash: String,
}

/// Cross-chain operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossChainOperation {
    pub operation_type: CrossChainOpType,
    pub source_chain: u64,
    pub target_chain: u64,
    pub data: serde_json::Value,
    pub gas_limit: u64,
    pub timeout: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossChainOpType {
    Transfer,
    ContractCall,
    StateSync,
    MessagePassing,
}

/// Advanced ML operation types
#[cfg(any(feature = "ml-basic", feature = "ml-deep"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLOperation {
    pub operation_type: MLOpType,
    pub model_id: u32,
    pub parameters: HashMap<String, serde_json::Value>,
    pub gas_cost: u64,
}

#[cfg(any(feature = "ml-basic", feature = "ml-deep"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MLOpType {
    CreateModel,
    TrainModel,
    Predict,
    UpdateWeights,
    SaveModel,
    LoadModel,
    EvaluateModel,
    CrossValidate,
    FeatureSelection,
    HyperparameterTuning,
    ModelEnsemble,
    TransferLearning,
}

#[cfg(any(feature = "ml-basic", feature = "ml-deep"))]
/// ML model configuration for WASM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLModelConfig {
    pub model_type: String,
    pub architecture: Vec<u32>,
    pub activation_functions: Vec<String>,
    pub learning_rate: f64,
    pub batch_size: u32,
    pub epochs: u32,
    pub optimizer: String,
}

#[cfg(any(feature = "ml-basic", feature = "ml-deep"))]
impl Default for MLModelConfig {
    fn default() -> Self {
        Self {
            model_type: "neural_network".to_string(),
            architecture: vec![784, 128, 64, 10],
            activation_functions: vec!["relu".to_string(), "relu".to_string(), "softmax".to_string()],
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            optimizer: "adam".to_string(),
        }
    }
}

#[cfg(any(feature = "ml-basic", feature = "ml-deep"))]
/// ML training data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLTrainingData {
    pub features: Vec<Vec<f64>>,
    pub labels: Vec<f64>,
    pub validation_split: f64,
    pub shuffle: bool,
}

#[cfg(any(feature = "ml-basic", feature = "ml-deep"))]
/// ML prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPredictionResult {
    pub predictions: Vec<f64>,
    pub confidence: Vec<f64>,
    pub model_accuracy: f64,
    pub inference_time_ms: f64,
}

/// Debug information for contract execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugInfo {
    pub execution_trace: Vec<ExecutionStep>,
    pub gas_usage: Vec<GasUsage>,
    pub memory_usage: MemoryUsage,
    pub stack_trace: Vec<StackFrame>,
}

/// Execution step in debug trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStep {
    pub instruction: String,
    pub pc: u32,
    pub gas_cost: u64,
    pub stack_before: Vec<String>,
    pub stack_after: Vec<String>,
}

/// Gas usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasUsage {
    pub operation: String,
    pub gas_cost: u64,
    pub cumulative_gas: u64,
}

/// Memory usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub total_allocated: u32,
    pub peak_usage: u32,
    pub current_usage: u32,
    pub allocations: Vec<MemoryAllocation>,
}

/// Memory allocation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAllocation {
    pub address: u32,
    pub size: u32,
    pub allocation_type: String,
}

/// Stack frame for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackFrame {
    pub function_name: String,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub column: Option<u32>,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub compilation_time_ms: f64,
    pub execution_time_ms: f64,
    pub memory_peak_kb: u32,
    pub gas_efficiency: f64,
    pub bytecode_size_bytes: u32,
}

/// Contract ABI (Application Binary Interface)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractABI {
    pub contract_name: String,
    pub version: String,
    pub functions: Vec<ABIFunction>,
    pub events: Vec<ABIEvent>,
    pub constructor: Option<ABIFunction>,
}

/// ABI function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABIFunction {
    pub name: String,
    pub inputs: Vec<ABIParameter>,
    pub outputs: Vec<ABIParameter>,
    pub state_mutability: String,
    pub visibility: String,
}

/// ABI event definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABIEvent {
    pub name: String,
    pub inputs: Vec<ABIParameter>,
    pub anonymous: bool,
}

/// ABI parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABIParameter {
    pub name: String,
    pub param_type: String,
    pub indexed: bool,
}

/// Utility functions for type conversions
impl From<augustc::error::CompilerError> for WasmError {
    fn from(error: augustc::error::CompilerError) -> Self {
        WasmError::new(&error.to_string(), "COMPILER_ERROR")
    }
}

impl From<WasmError> for JsValue {
    fn from(error: WasmError) -> Self {
        serde_wasm_bindgen::to_value(&error).unwrap_or_else(|_| {
            JsValue::from_str(&error.message)
        })
    }
}

/// Type validation utilities
pub fn validate_compile_options(options: &CompileOptions) -> Result<(), WasmError> {
    if let Some(memory_limit) = options.memory_limit {
        if memory_limit < 64 * 1024 {
            return Err(WasmError::new(
                "Memory limit must be at least 64KB",
                "INVALID_MEMORY_LIMIT",
            ));
        }
        if u64::from(memory_limit) > 4u64 * 1024 * 1024 * 1024 {
            return Err(WasmError::new(
                "Memory limit cannot exceed 4GB",
                "INVALID_MEMORY_LIMIT",
            ));
        }
    }
    
    if let Some(stack_size) = options.stack_size {
        if stack_size < 4 * 1024 {
            return Err(WasmError::new(
                "Stack size must be at least 4KB",
                "INVALID_STACK_SIZE",
            ));
        }
    }
    
    Ok(())
}

/// Convert Augustium type to WASM type
pub fn augustium_type_to_wasm(aug_type: &str) -> String {
    match aug_type {
        "u8" | "u16" | "u32" | "bool" => "i32".to_string(),
        "u64" | "i64" => "i64".to_string(),
        "f32" => "f32".to_string(),
        "f64" => "f64".to_string(),
        "string" | "address" => "i32".to_string(), // Pointer to memory
        _ => "i32".to_string(), // Default to i32 for complex types
    }
}

/// Convert WASM type to JavaScript type
pub fn wasm_type_to_js(wasm_type: &str) -> String {
    match wasm_type {
        "i32" => "number".to_string(),
        "i64" => "bigint".to_string(),
        "f32" | "f64" => "number".to_string(),
        _ => "any".to_string(),
    }
}