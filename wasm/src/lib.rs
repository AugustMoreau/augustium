//! WebAssembly bindings for the Augustium programming language compiler
//! 
//! This module provides JavaScript-accessible functions for compiling Augustium
//! smart contracts to WebAssembly and executing them in browser environments.

mod utils;
mod compiler;
mod runtime;
mod types;
mod avm_bindings;
mod js_api;

use wasm_bindgen::prelude::*;
use js_sys::Promise;
use web_sys::console;

// Re-export the main API components
pub use avm_bindings::*;
pub use js_api::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn main() {
    utils::set_panic_hook();
    console::log_1(&"Augustium WASM compiler initialized".into());
}

/// Compile Augustium source code to WebAssembly bytecode
#[wasm_bindgen]
pub fn compile_to_wasm(source: &str, options: &JsValue) -> Result<JsValue, JsValue> {
    let compile_options = serde_wasm_bindgen::from_value(options.clone())
        .map_err(|e| JsValue::from_str(&format!("Invalid options: {}", e)))?;
    
    match compiler::compile_augustium_to_wasm(source, compile_options) {
        Ok(result) => {
            serde_wasm_bindgen::to_value(&result)
                .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
        }
        Err(e) => Err(JsValue::from_str(&format!("Compilation error: {}", e)))
    }
}

/// Compile Augustium source code asynchronously
#[wasm_bindgen]
pub fn compile_to_wasm_async(source: String, options: JsValue) -> Promise {
    wasm_bindgen_futures::future_to_promise(async move {
        let compile_options = serde_wasm_bindgen::from_value(options)
            .map_err(|e| JsValue::from_str(&format!("Invalid options: {}", e)))?;
        
        match compiler::compile_augustium_to_wasm(&source, compile_options) {
            Ok(result) => {
                serde_wasm_bindgen::to_value(&result)
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
            }
            Err(e) => Err(JsValue::from_str(&format!("Compilation error: {}", e)))
        }
    })
}

/// Create a new Augustium contract instance
#[wasm_bindgen]
pub struct AugustiumContract {
    inner: runtime::ContractInstance,
}

#[wasm_bindgen]
impl AugustiumContract {
    /// Create a new contract instance from compiled bytecode
    #[wasm_bindgen(constructor)]
    pub fn new(bytecode: &[u8], initial_state: &JsValue) -> Result<AugustiumContract, JsValue> {
        let state = serde_wasm_bindgen::from_value(initial_state.clone())
            .map_err(|e| JsValue::from_str(&format!("Invalid initial state: {}", e)))?;
        
        match runtime::ContractInstance::new(bytecode, state) {
            Ok(instance) => Ok(AugustiumContract { inner: instance }),
            Err(e) => Err(JsValue::from_str(&format!("Contract creation error: {}", e)))
        }
    }
    
    /// Call a contract method
    #[wasm_bindgen]
    pub fn call_method(&mut self, method_name: &str, args: &JsValue) -> Result<JsValue, JsValue> {
        let arguments = serde_wasm_bindgen::from_value(args.clone())
            .map_err(|e| JsValue::from_str(&format!("Invalid arguments: {}", e)))?;
        
        match self.inner.call_method(method_name, arguments) {
            Ok(result) => {
                serde_wasm_bindgen::to_value(&result)
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
            }
            Err(e) => Err(JsValue::from_str(&format!("Method call error: {}", e)))
        }
    }
    
    /// Get the current contract state
    #[wasm_bindgen]
    pub fn get_state(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.inner.get_state())
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }
    
    /// Set contract state
    #[wasm_bindgen]
    pub fn set_state(&mut self, state: &JsValue) -> Result<(), JsValue> {
        let new_state = serde_wasm_bindgen::from_value(state.clone())
            .map_err(|e| JsValue::from_str(&format!("Invalid state: {}", e)))?;
        
        self.inner.set_state(new_state)
            .map_err(|e| JsValue::from_str(&format!("State update error: {}", e)))
    }
}

/// Utility functions for working with Augustium types
#[wasm_bindgen]
pub struct AugustiumUtils;

#[wasm_bindgen]
impl AugustiumUtils {
    /// Parse Augustium source code and return AST
    #[wasm_bindgen]
    pub fn parse_source(source: &str) -> Result<JsValue, JsValue> {
        match compiler::parse_augustium_source(source) {
            Ok(ast) => {
                serde_wasm_bindgen::to_value(&ast)
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
            }
            Err(e) => Err(JsValue::from_str(&format!("Parse error: {}", e)))
        }
    }
    
    /// Validate Augustium source code
    #[wasm_bindgen]
    pub fn validate_source(source: &str) -> Result<bool, JsValue> {
        compiler::validate_augustium_source(source)
            .map_err(|e| JsValue::from_str(&format!("Validation error: {}", e)))
    }
    
    /// Get compiler version information
    #[wasm_bindgen]
    pub fn get_version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
    
    /// Get supported language features
    #[wasm_bindgen]
    pub fn get_features() -> JsValue {
        let features = compiler::get_supported_features();
        serde_wasm_bindgen::to_value(&features).unwrap_or(JsValue::NULL)
    }
}

/// Export types for TypeScript definitions
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "CompileOptions")]
    pub type CompileOptions;
    
    #[wasm_bindgen(typescript_type = "CompileResult")]
    pub type CompileResult;
    
    #[wasm_bindgen(typescript_type = "ContractState")]
    pub type ContractState;
    
    #[wasm_bindgen(typescript_type = "MethodArguments")]
    pub type MethodArguments;
}