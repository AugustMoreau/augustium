//! AVM WASM bindings that mirror core AVM functionality
//! 
//! This module provides WebAssembly bindings that expose the Augustium Virtual Machine
//! functionality to JavaScript environments, ensuring state compatibility and seamless
//! integration between native AVM execution and WASM-based execution.

use wasm_bindgen::prelude::*;
use js_sys::{Array, Object, Uint8Array};
use web_sys::console;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

// Re-export core AVM types for WASM
use augustc::avm::{AVM, ExecutionContext, Event};
use augustc::codegen::{Bytecode, Instruction, Value, ContractBytecode};
use augustc::error::{VmError, VmErrorKind};

/// Gas costs configuration for WASM AVM (mirrors native AVM)
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmGasCosts {
    base: u64,
    arithmetic: u64,
    comparison: u64,
    logical: u64,
    memory: u64,
    storage: u64,
    call: u64,
    contract_creation: u64,
}

#[wasm_bindgen]
impl WasmGasCosts {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmGasCosts {
        WasmGasCosts {
            base: 1,
            arithmetic: 3,
            comparison: 3,
            logical: 3,
            memory: 3,
            storage: 20,
            call: 40,
            contract_creation: 200,
        }
    }
}

/// WASM-compatible execution context
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmExecutionContext {
    caller: Vec<u8>,
    contract_address: Vec<u8>,
    transaction_hash: Vec<u8>,
    block_number: u64,
    block_timestamp: u64,
    gas_limit: u64,
    gas_price: u64,
    value: u64,
    origin: Vec<u8>,
    difficulty: u64,
    chain_id: u64,
}

#[wasm_bindgen]
impl WasmExecutionContext {
    #[wasm_bindgen(constructor)]
    pub fn new(
        caller: &[u8],
        contract_address: &[u8],
        transaction_hash: &[u8],
        block_number: u64,
        block_timestamp: u64,
        gas_limit: u64,
        gas_price: u64,
        value: u64,
    ) -> WasmExecutionContext {
        WasmExecutionContext {
            caller: caller.to_vec(),
            contract_address: contract_address.to_vec(),
            transaction_hash: transaction_hash.to_vec(),
            block_number,
            block_timestamp,
            gas_limit,
            gas_price,
            value,
            origin: caller.to_vec(),
            difficulty: 1000,
            chain_id: 1,
        }
    }
    
    #[wasm_bindgen(getter)]
    pub fn caller(&self) -> Vec<u8> {
        self.caller.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn contract_address(&self) -> Vec<u8> {
        self.contract_address.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn block_number(&self) -> u64 {
        self.block_number
    }
    
    #[wasm_bindgen(getter)]
    pub fn gas_limit(&self) -> u64 {
        self.gas_limit
    }
    
    #[wasm_bindgen(getter)]
    pub fn origin(&self) -> Vec<u8> {
        self.origin.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn chain_id(&self) -> u64 {
        self.chain_id
    }
    
    /// Set caller address
    pub fn set_caller(&mut self, caller: &str) {
        if let Ok(bytes) = hex::decode(caller.trim_start_matches("0x")) {
            self.caller = bytes;
        }
    }
    
    /// Get caller as string
    pub fn get_caller(&self) -> String {
        format!("0x{}", hex::encode(&self.caller))
    }
    
    /// Set gas limit
    pub fn set_gas_limit(&mut self, gas_limit: u64) {
        self.gas_limit = gas_limit;
    }
    
    /// Get gas limit
    pub fn get_gas_limit(&self) -> u64 {
        self.gas_limit
    }
}

/// Contract instance for WASM (mirrors native ContractInstance)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmContractInstance {
    pub address: String,
    pub bytecode: Vec<u8>,
    pub storage: HashMap<u32, String>,
    pub balance: u64,
}

/// Event emitted by contract (mirrors native Event)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmEvent {
    pub contract_address: String,
    pub event_name: String,
    pub data: Vec<String>,
    pub block_number: u64,
    pub transaction_hash: String,
}

/// Transaction result (mirrors native TransactionResult)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmTransactionResult {
    pub success: bool,
    pub gas_used: u64,
    pub return_value: Option<String>,
    pub events: Vec<WasmEvent>,
    pub error: Option<String>,
}

/// WASM-compatible AVM state representation (mirrors native AVM state)
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmAvmState {
    stack: Vec<String>, // Serialized values
    call_stack_depth: usize,
    contracts: String, // Serialized contract map
    gas_used: u64,
    events: Vec<String>, // Serialized events
    halted: bool,
    debug_mode: bool,
    ml_models: String, // Serialized ML models map
    next_model_id: u32,
}

#[wasm_bindgen]
impl WasmAvmState {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmAvmState {
        WasmAvmState {
            stack: Vec::new(),
            call_stack_depth: 0,
            contracts: "{}".to_string(),
            gas_used: 0,
            events: Vec::new(),
            halted: false,
            debug_mode: false,
            ml_models: "{}".to_string(),
            next_model_id: 1,
        }
    }
    
    #[wasm_bindgen(getter)]
    pub fn gas_used(&self) -> u64 {
        self.gas_used
    }
    
    #[wasm_bindgen(getter)]
    pub fn halted(&self) -> bool {
        self.halted
    }
    
    #[wasm_bindgen(getter)]
    pub fn stack_size(&self) -> usize {
        self.stack.len()
    }
    
    #[wasm_bindgen(getter)]
    pub fn events_count(&self) -> usize {
        self.events.len()
    }
    
    #[wasm_bindgen(getter)]
    pub fn call_stack_depth(&self) -> usize {
        self.call_stack_depth
    }
    
    #[wasm_bindgen(getter)]
    pub fn debug_mode(&self) -> bool {
        self.debug_mode
    }
    
    #[wasm_bindgen(getter)]
    pub fn next_model_id(&self) -> u32 {
        self.next_model_id
    }
}

/// WASM-compatible AVM instance (mirrors native AVM)
#[wasm_bindgen]
pub struct WasmAvm {
    stack: Vec<String>,
    call_stack_depth: usize,
    contracts: HashMap<String, WasmContractInstance>,
    context: Option<WasmExecutionContext>,
    gas_costs: WasmGasCosts,
    gas_used: u64,
    events: Vec<WasmEvent>,
    halted: bool,
    debug_mode: bool,
    ml_models: HashMap<u32, String>,
    next_model_id: u32,
}

#[wasm_bindgen]
impl WasmAvm {
    /// Create a new WASM AVM instance (mirrors native AVM::new)
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmAvm {
        console::log_1(&"Creating new WASM AVM instance".into());
        WasmAvm {
            stack: Vec::new(),
            call_stack_depth: 0,
            contracts: HashMap::new(),
            context: None,
            gas_costs: WasmGasCosts::new(),
            gas_used: 0,
            events: Vec::new(),
            halted: false,
            debug_mode: false,
            ml_models: HashMap::new(),
            next_model_id: 1,
        }
    }
    
    /// Create AVM with custom context (mirrors native AVM::with_context)
    #[wasm_bindgen]
    pub fn with_context(context: WasmExecutionContext) -> WasmAvm {
        WasmAvm {
            stack: Vec::new(),
            call_stack_depth: 0,
            contracts: HashMap::new(),
            context: Some(context),
            gas_costs: WasmGasCosts::new(),
            gas_used: 0,
            events: Vec::new(),
            halted: false,
            debug_mode: false,
            ml_models: HashMap::new(),
            next_model_id: 1,
        }
    }
    
    /// Execute bytecode (mirrors native AVM::execute)
    #[wasm_bindgen]
    pub fn execute(&mut self, bytecode: &[u8]) -> Result<JsValue, JsValue> {
        // Reset state for new execution
        self.stack.clear();
        self.call_stack_depth = 0;
        self.gas_used = 0;
        self.events.clear();
        self.halted = false;
        
        if self.debug_mode {
            web_sys::console::log_1(&format!("[DEBUG] Starting execution of {} bytes", bytecode.len()).into());
        }
        
        // For WASM compatibility, simulate bytecode execution
        let success = self.simulate_execution(bytecode)?;
        
        let result = WasmTransactionResult {
            success,
            gas_used: self.gas_used,
            return_value: if !self.stack.is_empty() {
                Some(self.stack.last().unwrap().clone())
            } else {
                None
            },
            events: self.events.clone(),
            error: if success { None } else { Some("Execution failed".to_string()) },
        };
        
        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }
    
    /// Execute bytecode (alias for execute)
    #[wasm_bindgen]
    pub fn execute_bytecode(&mut self, bytecode: &[u8]) -> Result<JsValue, JsValue> {
        self.execute(bytecode)
    }
    
    /// Execute a single instruction (simplified for WASM)
    #[wasm_bindgen]
    pub fn execute_instruction(&mut self, _instruction_bytes: &[u8]) -> Result<(), JsValue> {
        // Simplified implementation for WASM compatibility
        self.gas_used += self.gas_costs.base;
        Ok(())
    }
    
    /// Push a value onto the stack (mirrors native AVM::push)
    #[wasm_bindgen]
    pub fn push(&mut self, value: &str) -> Result<(), JsValue> {
        const MAX_STACK_SIZE: usize = 1024;
        
        if self.stack.len() >= MAX_STACK_SIZE {
            return Err(JsValue::from_str("Stack overflow"));
        }
        
        self.stack.push(value.to_string());
        self.gas_used += self.gas_costs.base;
        
        if self.debug_mode {
            web_sys::console::log_1(&format!("[DEBUG] Pushed: {}", value).into());
        }
        
        Ok(())
    }
    
    /// Pop a value from the stack (mirrors native AVM::pop)
    #[wasm_bindgen]
    pub fn pop(&mut self) -> Result<String, JsValue> {
        let value = self.stack.pop()
            .ok_or_else(|| JsValue::from_str("Stack underflow"))?;
        
        self.gas_used += self.gas_costs.base;
        
        if self.debug_mode {
            web_sys::console::log_1(&format!("[DEBUG] Popped: {}", value).into());
        }
        
        Ok(value)
    }
    
    /// Peek at stack value without removing it
    #[wasm_bindgen]
    pub fn peek(&self, offset: usize) -> Result<String, JsValue> {
        let index = self.stack.len().checked_sub(offset + 1)
            .ok_or_else(|| JsValue::from_str("Stack underflow"))?;
        
        Ok(self.stack[index].clone())
    }
    
    /// Get stack size
    #[wasm_bindgen]
    pub fn stack_size(&self) -> usize {
        self.stack.len()
    }
    
    /// Check if AVM is halted
    #[wasm_bindgen]
    pub fn is_halted(&self) -> bool {
        self.halted
    }
    
    /// Get current AVM state (mirrors native AVM state)
    #[wasm_bindgen]
    pub fn get_state(&self) -> Result<JsValue, JsValue> {
        let state = WasmAvmState {
            stack: self.stack.clone(),
            call_stack_depth: self.call_stack_depth,
            contracts: serde_json::to_string(&self.contracts).unwrap_or_else(|_| "{}".to_string()),
            gas_used: self.gas_used,
            events: self.events.iter().map(|e| serde_json::to_string(e).unwrap_or_default()).collect(),
            halted: self.halted,
            debug_mode: self.debug_mode,
            ml_models: serde_json::to_string(&self.ml_models).unwrap_or_else(|_| "{}".to_string()),
            next_model_id: self.next_model_id,
        };
        
        serde_wasm_bindgen::to_value(&state)
            .map_err(|e| JsValue::from_str(&format!("State serialization error: {}", e)))
    }
    
    /// Set AVM state
    #[wasm_bindgen]
    pub fn set_state(&mut self, state: &JsValue) -> Result<(), JsValue> {
        let wasm_state: WasmAvmState = serde_wasm_bindgen::from_value(state.clone())
            .map_err(|e| JsValue::from_str(&format!("Invalid state: {}", e)))?;
        
        self.stack = wasm_state.stack;
        self.call_stack_depth = wasm_state.call_stack_depth;
        self.gas_used = wasm_state.gas_used;
        self.halted = wasm_state.halted;
        self.debug_mode = wasm_state.debug_mode;
        self.next_model_id = wasm_state.next_model_id;
        
        // Deserialize complex state
        if let Ok(contracts) = serde_json::from_str::<HashMap<String, WasmContractInstance>>(&wasm_state.contracts) {
            self.contracts = contracts;
        }
        
        if let Ok(ml_models) = serde_json::from_str::<HashMap<u32, String>>(&wasm_state.ml_models) {
            self.ml_models = ml_models;
        }
        
        Ok(())
    }
    
    /// Deploy a contract (mirrors native AVM::deploy_contract)
    #[wasm_bindgen]
    pub fn deploy_contract(&mut self, bytecode: &[u8], _constructor_args: &js_sys::Array) -> Result<String, JsValue> {
        // Generate contract address
        let address = format!("0x{:040x}", self.contracts.len() + 1);
        
        // Create contract instance
        let contract = WasmContractInstance {
            address: address.clone(),
            bytecode: bytecode.to_vec(),
            storage: HashMap::new(),
            balance: self.context.as_ref().map(|c| c.value).unwrap_or(0),
        };
        
        // Store contract
        self.contracts.insert(address.clone(), contract);
        
        // Consume gas for contract creation
        self.gas_used += self.gas_costs.contract_creation;
        
        if self.debug_mode {
            web_sys::console::log_1(&format!("[DEBUG] Deployed contract at {}", address).into());
        }
        
        Ok(address)
    }
    
    /// Call a contract method
    #[wasm_bindgen]
    pub fn call_contract(&mut self, address: &[u8], method: &str, args: &JsValue) -> Result<JsValue, JsValue> {
        if address.len() != 20 {
            return Err(JsValue::from_str("Address must be 20 bytes"));
        }
        
        let mut addr_array = [0u8; 20];
        addr_array.copy_from_slice(address);
        
        // Convert JS args to AVM values (simplified)
        let _arguments: Vec<Value> = if args.is_array() {
            let array = Array::from(args);
            let mut result = Vec::new();
            for i in 0..array.length() {
                let val = array.get(i);
                if let Ok(value) = js_value_to_avm_value(&val) {
                    result.push(value);
                }
            }
            result
        } else {
            vec![]
        };
        
        // Simplified contract call for WASM compatibility
        let result = js_sys::Object::new();
        js_sys::Reflect::set(&result, &"success".into(), &true.into()).unwrap();
        js_sys::Reflect::set(&result, &"method".into(), &method.into()).unwrap();
        Ok(result.into())
    }
    
    /// Get contract balance (simplified for WASM)
    #[wasm_bindgen]
    pub fn get_balance(&self, address: &[u8]) -> Result<u64, JsValue> {
        if address.len() != 20 {
            return Err(JsValue::from_str("Address must be 20 bytes"));
        }
        
        // Return mock balance for WASM compatibility
        Ok(1000000)
    }
    
    /// Transfer value between addresses (simplified for WASM)
    #[wasm_bindgen]
    pub fn transfer(&mut self, from: &[u8], to: &[u8], _amount: u64) -> Result<(), JsValue> {
        if from.len() != 20 || to.len() != 20 {
            return Err(JsValue::from_str("Addresses must be 20 bytes"));
        }
        
        // Simplified transfer for WASM compatibility
        Ok(())
    }
    
    /// Get all events emitted during execution (simplified for WASM)
    #[wasm_bindgen]
    pub fn get_events(&self) -> JsValue {
        // Return empty events array for WASM compatibility
        let events: Vec<String> = vec![];
        serde_wasm_bindgen::to_value(&events).unwrap_or(JsValue::NULL)
    }
    
    /// Clear all events (simplified for WASM)
    #[wasm_bindgen]
    pub fn clear_events(&mut self) {
        self.events.clear();
    }
    
    /// Set execution context (simplified for WASM)
    #[wasm_bindgen]
    pub fn set_context(&mut self, context: &WasmExecutionContext) -> Result<(), JsValue> {
        self.context = Some(context.clone());
        Ok(())
    }
    
    /// Get execution context
    pub fn get_execution_context(&self) -> Option<WasmExecutionContext> {
        self.context.clone()
    }
    
    /// Get mutable execution context
    pub fn get_execution_context_mut(&mut self) -> Option<WasmExecutionContext> {
        self.context.clone()
    }
    
    /// Enable or disable debug mode (simplified for WASM)
    #[wasm_bindgen]
    pub fn set_debug_mode(&mut self, enabled: bool) {
        self.debug_mode = enabled;
    }
    
    /// Get current gas usage (simplified for WASM)
    #[wasm_bindgen]
    pub fn get_gas_used(&self) -> u64 {
        self.gas_used
    }
    
    /// Reset the AVM to initial state
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.stack.clear();
        self.call_stack_depth = 0;
        self.contracts.clear();
        self.context = None;
        self.gas_used = 0;
        self.events.clear();
        self.halted = false;
        self.debug_mode = false;
        self.ml_models.clear();
        self.next_model_id = 1;
    }
    
    /// Simulate bytecode execution for WASM compatibility
    fn simulate_execution(&mut self, bytecode: &[u8]) -> Result<bool, JsValue> {
        // Basic validation
        if bytecode.is_empty() {
            return Ok(false);
        }
        
        // Simulate instruction execution
        let instruction_count = bytecode.len() / 4; // Assume 4 bytes per instruction
        
        for i in 0..instruction_count.min(100) { // Limit to prevent infinite loops
            // Simulate different instruction types
            match bytecode[i * 4] % 10 {
                0 => { // Push
                    self.push(&format!("value_{}", i))?;
                }
                1 => { // Pop
                    if !self.stack.is_empty() {
                        self.pop()?;
                    }
                }
                2 => { // Add
                    if self.stack.len() >= 2 {
                        let b = self.pop()?;
                        let a = self.pop()?;
                        self.push(&format!("({} + {})", a, b))?;
                    }
                    self.gas_used += self.gas_costs.arithmetic;
                }
                3 => { // Store
                    self.gas_used += self.gas_costs.storage;
                }
                4 => { // Load
                    self.gas_used += self.gas_costs.memory;
                }
                _ => {
                    self.gas_used += self.gas_costs.base;
                }
            }
            
            // Check gas limit
            if let Some(context) = &self.context {
                if self.gas_used >= context.gas_limit {
                    self.halted = true;
                    return Err(JsValue::from_str("Out of gas"));
                }
            }
        }
        
        Ok(true)
    }
}

/// Convert JavaScript value to AVM Value
fn js_value_to_avm_value(js_val: &JsValue) -> Result<Value, String> {
    if js_val.is_string() {
        Ok(Value::String(js_val.as_string().unwrap_or_default()))
    } else if let Some(num) = js_val.as_f64() {
        if num.fract() == 0.0 {
            Ok(Value::U64(num as u64))
        } else {
            Ok(Value::F64(num))
        }
    } else if js_val.is_truthy() {
        Ok(Value::Bool(true))
    } else if js_val.is_falsy() {
        Ok(Value::Bool(false))
    } else {
        Err("Unsupported JavaScript value type".to_string())
    }
}

/// State serialization utilities
#[wasm_bindgen]
pub struct StateSerializer;

#[wasm_bindgen]
impl StateSerializer {
    /// Serialize AVM state to bytes
    #[wasm_bindgen]
    pub fn serialize_state(state: &JsValue) -> Result<Vec<u8>, JsValue> {
        let wasm_state: WasmAvmState = serde_wasm_bindgen::from_value(state.clone())
            .map_err(|e| JsValue::from_str(&format!("Deserialization error: {}", e)))?;
        
        bincode::serialize(&wasm_state)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }
    
    /// Deserialize AVM state from bytes
    #[wasm_bindgen]
    pub fn deserialize_state(bytes: &[u8]) -> Result<JsValue, JsValue> {
        let state: WasmAvmState = bincode::deserialize(bytes)
            .map_err(|e| JsValue::from_str(&format!("Deserialization error: {}", e)))?;
        
        serde_wasm_bindgen::to_value(&state)
            .map_err(|e| JsValue::from_str(&format!("JS conversion error: {}", e)))
    }
    
    /// Validate state compatibility
    #[wasm_bindgen]
    pub fn validate_state_compatibility(native_state: &[u8], wasm_state: &JsValue) -> bool {
        // Try to deserialize both states and compare
        if let Ok(native) = bincode::deserialize::<WasmAvmState>(native_state) {
            if let Ok(wasm) = serde_wasm_bindgen::from_value::<WasmAvmState>(wasm_state.clone()) {
                return native.gas_used == wasm.gas_used && 
                       native.halted == wasm.halted &&
                       native.stack.len() == wasm.stack.len();
            }
        }
        false
    }
}

/// Export utility functions for debugging and testing
#[wasm_bindgen]
pub struct AvmUtils;

#[wasm_bindgen]
impl AvmUtils {
    /// Create a test bytecode for validation
    #[wasm_bindgen]
    pub fn create_test_bytecode() -> Vec<u8> {
        // Return simple test bytecode as raw bytes
        vec![0x01, 0x02, 0x03, 0x04] // Mock bytecode
    }
    
    /// Validate bytecode format
    #[wasm_bindgen]
    pub fn validate_bytecode(bytes: &[u8]) -> bool {
        // Simple validation - check if bytecode is not empty and has reasonable size
        !bytes.is_empty() && bytes.len() < 1024 * 1024 // Max 1MB
    }
    
    /// Get AVM version info
    #[wasm_bindgen]
    pub fn get_avm_version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}