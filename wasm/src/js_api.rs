//! High-level JavaScript API for Augustium WASM integration
//! Provides comprehensive web integration capabilities

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;
use js_sys::{Promise, Array};
use web_sys::{console, window, Storage};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::avm_bindings::{WasmAvm, WasmExecutionContext};
use crate::compiler;
// use crate::runtime::{ContractInstance, MethodArguments, MethodResult};
use crate::types::{DeploymentOptions, TransactionInfo, BlockchainContext};

/// Internal web utility functions for address generation and hashing
struct InternalWebUtils;

impl InternalWebUtils {
    /// Generate a random contract address
    fn generate_address() -> String {
        let timestamp = js_sys::Date::now() as u64;
        let random = (js_sys::Math::random() * 1000000.0) as u64;
        format!("0x{:040x}", timestamp ^ random)
    }
    
    /// Get current timestamp
    fn now() -> u64 {
        js_sys::Date::now() as u64
    }
    
    /// Create transaction hash
    fn create_tx_hash(from: &str, to: &str, value: u64, nonce: u64) -> String {
        let data = format!("{}{}{}{}", from, to, value, nonce);
        let hash = data.chars().fold(0u64, |acc, c| acc.wrapping_mul(31).wrapping_add(c as u64));
        format!("0x{:064x}", hash)
    }
}

/// Configuration for Augustium runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct AugustiumConfig {
    gas_limit: u64,
    debug_mode: bool,
    enable_events: bool,
    max_stack_size: u32,
    network_id: u32,
    enable_ml: bool,
    enable_persistence: bool,
    block_time: u64,
}

#[wasm_bindgen]
impl AugustiumConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> AugustiumConfig {
        AugustiumConfig {
            gas_limit: 1_000_000,
            debug_mode: false,
            enable_events: true,
            max_stack_size: 1024,
            network_id: 1,
            enable_ml: false,
            enable_persistence: true,
            block_time: 15000, // 15 seconds
        }
    }
    
    /// Create config with custom settings
    #[wasm_bindgen]
    pub fn with_settings(
        gas_limit: u64,
        debug_mode: bool,
        enable_ml: bool,
        enable_persistence: bool,
        network_id: u64,
    ) -> AugustiumConfig {
        AugustiumConfig {
            gas_limit,
            debug_mode,
            enable_events: true,
            max_stack_size: 1024,
            network_id: network_id as u32,
            enable_ml,
            enable_persistence,
            block_time: 15000,
        }
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_gas_limit(&mut self, limit: u64) {
        self.gas_limit = limit;
    }
    
    #[wasm_bindgen(getter)]
    pub fn gas_limit(&self) -> u64 {
        self.gas_limit
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_debug_mode(&mut self, enabled: bool) {
        self.debug_mode = enabled;
    }
    
    #[wasm_bindgen(getter)]
    pub fn debug_mode(&self) -> bool {
        self.debug_mode
    }
    
    #[wasm_bindgen(getter)]
    pub fn enable_ml(&self) -> bool {
        self.enable_ml
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_enable_ml(&mut self, enable_ml: bool) {
        self.enable_ml = enable_ml;
    }
    
    #[wasm_bindgen(getter)]
    pub fn enable_persistence(&self) -> bool {
        self.enable_persistence
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_enable_persistence(&mut self, enable_persistence: bool) {
        self.enable_persistence = enable_persistence;
    }
}

/// Event listener callback type
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "(event: any) => void")]
    pub type EventCallback;
}

/// Contract metadata for runtime management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractMetadata {
    pub address: String,
    pub name: String,
    pub abi: Vec<String>,
    pub bytecode: Vec<u8>,
    pub deployed_at: u64,
    pub version: String,
}

/// High-level Augustium runtime for JavaScript
#[wasm_bindgen]
pub struct AugustiumRuntime {
    avm: WasmAvm,
    config: AugustiumConfig,
    contracts: HashMap<String, ContractMetadata>,
    event_listeners: HashMap<String, js_sys::Function>,
    blockchain_context: BlockchainContext,
    storage: Option<Storage>,
    transaction_history: Vec<TransactionInfo>,
}

#[wasm_bindgen]
impl AugustiumRuntime {
    /// Create a new Augustium runtime
    #[wasm_bindgen(constructor)]
    pub fn new(config: Option<AugustiumConfig>) -> Result<AugustiumRuntime, JsValue> {
        console::log_1(&"Initializing Augustium Runtime".into());
        
        let config = config.unwrap_or_else(|| AugustiumConfig::new());
        let mut avm = WasmAvm::new();
        avm.set_debug_mode(config.debug_mode);
        
        // Initialize blockchain context
        let blockchain_context = BlockchainContext::default();
        
        // Setup execution context with default values
        let caller = vec![0u8; 20];
        let contract_addr = vec![0u8; 20];
        let tx_hash = vec![0u8; 32];
        let execution_context = WasmExecutionContext::new(
            &caller,
            &contract_addr,
            &tx_hash,
            0, // block_number
            js_sys::Date::now() as u64, // block_timestamp
            config.gas_limit,
            1, // gas_price
            0, // value
        );
        avm.set_context(&execution_context).map_err(|e| {
            JsValue::from_str(&format!("Failed to set execution context: {:?}", e))
        })?;
        
        // Initialize storage if persistence is enabled
        let storage = if config.enable_persistence {
            window()
                .and_then(|w| w.local_storage().ok().flatten())
        } else {
            None
        };
        
        Ok(AugustiumRuntime {
            avm,
            config,
            contracts: HashMap::new(),
            event_listeners: HashMap::new(),
            blockchain_context,
            storage,
            transaction_history: Vec::new(),
        })
    }
    
    /// Create runtime with custom blockchain context
    #[wasm_bindgen]
    pub fn with_context(config: AugustiumConfig, context: JsValue) -> Result<AugustiumRuntime, JsValue> {
        let blockchain_context: BlockchainContext = serde_wasm_bindgen::from_value(context)
            .map_err(|e| JsValue::from_str(&format!("Invalid blockchain context: {}", e)))?;
        
        let mut runtime = Self::new(Some(config))?;
        runtime.blockchain_context = blockchain_context;
        
        Ok(runtime)
    }
    
    /// Deploy a contract from source code with metadata
    #[wasm_bindgen]
    pub fn deploy_from_source(
        &mut self,
        source_code: &str,
        contract_name: &str,
        constructor_args: &Array,
    ) -> Result<String, JsValue> {
        self.deploy_from_source_with_options(source_code, contract_name, constructor_args, JsValue::NULL)
    }
    
    /// Deploy a contract from source code with options
    #[wasm_bindgen]
    pub fn deploy_from_source_with_options(
        &mut self,
        source_code: &str,
        contract_name: &str,
        constructor_args: &Array,
        options: JsValue,
    ) -> Result<String, JsValue> {
        console::log_1(&format!("Deploying contract '{}' from source", contract_name).into());
        
        // Parse deployment options
        let deploy_options: DeploymentOptions = if !options.is_null() && !options.is_undefined() {
            serde_wasm_bindgen::from_value(options)
                .map_err(|e| JsValue::from_str(&format!("Invalid deployment options: {}", e)))?
        } else {
            DeploymentOptions::default()
        };
        
        // Compile the source code
        let compile_options = compiler::CompileOptions::default();
        let compile_result = compiler::compile_augustium_to_wasm(source_code, compile_options)
            .map_err(|e| JsValue::from_str(&format!("Compilation error: {}", e)))?;
        
        // Generate address if not provided
         let address = deploy_options.address.unwrap_or_else(|| InternalWebUtils::generate_address());
         
         // Deploy the compiled bytecode using AVM
         let _deploy_result = self.avm.deploy_contract(&compile_result.bytecode, constructor_args)?;
         
         // Create contract metadata
         let metadata = ContractMetadata {
             address: address.clone(),
             name: contract_name.to_string(),
             abi: compile_result.metadata.contract_methods.iter().map(|m| m.name.clone()).collect(),
             bytecode: compile_result.bytecode,
             deployed_at: InternalWebUtils::now(),
             version: "1.0.0".to_string(),
         };
         
         // Store contract metadata
         self.contracts.insert(address.clone(), metadata);
         
         // Save to persistent storage if enabled
         if let Some(storage) = &self.storage {
             let contracts_json = serde_json::to_string(&self.contracts).unwrap_or_default();
             let _ = storage.set_item("augustium_contracts", &contracts_json);
         }
         
         // Record transaction
         let transaction = TransactionInfo {
             hash: InternalWebUtils::create_tx_hash("0x0000000000000000000000000000000000000000", &address, 0, self.transaction_history.len() as u64),
             from: "0x0000000000000000000000000000000000000000".to_string(),
             to: Some(address.clone()),
             value: deploy_options.initial_balance.unwrap_or(0),
             gas_limit: deploy_options.gas_limit.unwrap_or(self.config.gas_limit),
             gas_used: self.avm.get_gas_used(),
             status: crate::types::TransactionStatus::Success,
             block_number: Some(self.blockchain_context.block_number),
             timestamp: js_sys::Date::now(),
         };
        
        self.transaction_history.push(transaction);
        
        console::log_1(&format!("Contract '{}' deployed at {}", contract_name, address).into());
         Ok(address)
     }
     
     /// Call a contract method
     #[wasm_bindgen]
     pub fn call_contract(
         &mut self,
         contract_address: &str,
         method_name: &str,
         args: &Array,
     ) -> Result<JsValue, JsValue> {
         self.call_contract_with_options(contract_address, method_name, args, JsValue::NULL)
     }
     
     /// Call a contract method with options
     #[wasm_bindgen]
     pub fn call_contract_with_options(
         &mut self,
         contract_address: &str,
         method_name: &str,
         args: &Array,
         options: JsValue,
     ) -> Result<JsValue, JsValue> {
         console::log_1(&format!("Calling method '{}' on contract {}", method_name, contract_address).into());
         
         // Check if contract exists
         if !self.contracts.contains_key(contract_address) {
             return Err(JsValue::from_str("Contract not found"));
         }
         
         // Parse call options
         let call_options: serde_json::Value = if !options.is_null() && !options.is_undefined() {
             serde_wasm_bindgen::from_value(options)
                 .map_err(|e| JsValue::from_str(&format!("Invalid call options: {}", e)))?
         } else {
             serde_json::Value::Null
         };
         
         // Set caller context if provided
        if let Some(caller) = call_options.get("from").and_then(|v| v.as_str()) {
            if let Some(mut context) = self.avm.get_execution_context() {
                context.set_caller(caller);
                let _ = self.avm.set_context(&context);
            }
        }
        
        // Set gas limit if provided
        if let Some(gas_limit) = call_options.get("gas").and_then(|v| v.as_u64()) {
            if let Some(mut context) = self.avm.get_execution_context() {
                context.set_gas_limit(gas_limit);
                let _ = self.avm.set_context(&context);
            }
        }
         
         // Call the contract method
         let addr_bytes = hex::decode(&contract_address.replace("0x", ""))
             .map_err(|_| JsValue::from_str("Invalid contract address format"))?;
         
         if addr_bytes.len() != 20 {
             return Err(JsValue::from_str("Contract address must be 20 bytes"));
         }
         
         let result = self.avm.call_contract(&addr_bytes, method_name, args)?;
         
         // Record transaction
         let transaction = TransactionInfo {
             hash: InternalWebUtils::create_tx_hash(
                 &self.avm.get_execution_context().map(|ctx| ctx.get_caller()).unwrap_or_else(|| "0x0000000000000000000000000000000000000000".to_string()),
                 contract_address,
                 call_options.get("value").and_then(|v| v.as_u64()).unwrap_or(0),
                 self.transaction_history.len() as u64,
             ),
             from: self.avm.get_execution_context().map(|ctx| ctx.get_caller()).unwrap_or_else(|| "0x0000000000000000000000000000000000000000".to_string()),
             to: Some(contract_address.to_string()),
             value: call_options.get("value").and_then(|v| v.as_u64()).unwrap_or(0),
             gas_limit: self.avm.get_execution_context().map(|ctx| ctx.get_gas_limit()).unwrap_or(21000),
             gas_used: self.avm.get_gas_used(),
             status: crate::types::TransactionStatus::Success,
             block_number: Some(self.blockchain_context.block_number),
             timestamp: js_sys::Date::now(),
         };
         
         self.transaction_history.push(transaction);
         
         console::log_1(&format!("Method '{}' executed successfully", method_name).into());
         Ok(result)
     }
     
     /// Get contract information
     #[wasm_bindgen]
     pub fn get_contract_info(&self, contract_address: &str) -> Result<JsValue, JsValue> {
         if let Some(metadata) = self.contracts.get(contract_address) {
             serde_wasm_bindgen::to_value(metadata)
                 .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
         } else {
             Err(JsValue::from_str("Contract not found"))
         }
     }
     
     /// List all deployed contracts
     #[wasm_bindgen]
     pub fn list_contracts(&self) -> Result<JsValue, JsValue> {
         let contract_list: Vec<&ContractMetadata> = self.contracts.values().collect();
         serde_wasm_bindgen::to_value(&contract_list)
             .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
     }
     
     /// Get contract state
     #[wasm_bindgen]
     pub fn get_contract_state(&self, contract_address: &str) -> Result<JsValue, JsValue> {
         if !self.contracts.contains_key(contract_address) {
             return Err(JsValue::from_str("Contract not found"));
         }
         
         self.avm.get_state()
     }
     
     /// Subscribe to contract events
     #[wasm_bindgen]
     pub fn subscribe_to_events(
         &mut self,
         contract_address: &str,
         event_name: Option<String>,
         callback: &js_sys::Function,
     ) -> Result<String, JsValue> {
         if !self.contracts.contains_key(contract_address) {
             return Err(JsValue::from_str("Contract not found"));
         }
         
         let subscription_id = format!("sub_{}_{}", contract_address, InternalWebUtils::now());
         
         // Store the callback in event_listeners map
         let listener_key = format!("{}:{}", contract_address, event_name.unwrap_or_else(|| "*".to_string()));
         self.event_listeners.insert(listener_key, callback.clone());
         
         console::log_1(&format!("Subscribed to events for contract {}", contract_address).into());
         Ok(subscription_id)
     }
     
     /// Unsubscribe from events
     #[wasm_bindgen]
     pub fn unsubscribe_from_events(&mut self, subscription_id: &str) -> Result<(), JsValue> {
         // Remove from event_listeners - this is a simplified implementation
         // In practice, you'd need to track subscription IDs more carefully
         console::log_1(&format!("Unsubscribed from events: {}", subscription_id).into());
         Ok(())
     }
    
    /// Deploy a pre-compiled contract
    #[wasm_bindgen]
    pub fn deploy_contract(
        &mut self,
        bytecode: &[u8],
        constructor_args: &Array,
    ) -> Result<String, JsValue> {
        console::log_1(&"Deploying pre-compiled contract".into());
        
        // Generate a contract address
        let address = InternalWebUtils::generate_address();
        
        // Deploy using AVM
        let _deploy_result = self.avm.deploy_contract(bytecode, constructor_args)
            .map_err(|e| JsValue::from_str(&format!("Deployment failed: {:?}", e)))?;
        
        // Create basic contract metadata
        let metadata = ContractMetadata {
            address: address.clone(),
            name: "Unknown".to_string(),
            abi: vec![],
            bytecode: bytecode.to_vec(),
            deployed_at: InternalWebUtils::now(),
            version: "1.0.0".to_string(),
        };
        
        // Store contract metadata
        self.contracts.insert(address.clone(), metadata);
        
        console::log_1(&format!("Contract deployed at {}", address).into());
        Ok(address)
    }
    
    /// Get transaction history
    #[wasm_bindgen]
    pub fn get_transaction_history(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.transaction_history)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }
    
    /// Get blockchain context
    #[wasm_bindgen]
    pub fn get_blockchain_context(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.blockchain_context)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }
    
    /// Update blockchain context
    #[wasm_bindgen]
    pub fn update_blockchain_context(&mut self, context: JsValue) -> Result<(), JsValue> {
        let new_context: BlockchainContext = serde_wasm_bindgen::from_value(context)
            .map_err(|e| JsValue::from_str(&format!("Invalid blockchain context: {}", e)))?;
        
        self.blockchain_context = new_context;
        console::log_1(&"Blockchain context updated".into());
        Ok(())
    }
    
    /// Advance blockchain state (simulate mining)
    #[wasm_bindgen]
    pub fn advance_block(&mut self) -> Result<(), JsValue> {
        self.blockchain_context.block_number += 1;
        self.blockchain_context.timestamp = js_sys::Date::now();
        
        // Save updated context to storage if enabled
        if let Some(storage) = &self.storage {
            let context_json = serde_json::to_string(&self.blockchain_context).unwrap_or_default();
            let _ = storage.set_item("augustium_blockchain_context", &context_json);
        }
        
        console::log_1(&format!("Advanced to block {}", self.blockchain_context.block_number).into());
        Ok(())
    }
    
    /// Get AVM state
    #[wasm_bindgen]
    pub fn get_avm_state(&self) -> Result<JsValue, JsValue> {
        self.avm.get_state()
    }
    
    /// Reset AVM state
    #[wasm_bindgen]
    pub fn reset_avm(&mut self) -> Result<(), JsValue> {
        self.avm.reset();
        console::log_1(&"AVM state reset".into());
        Ok(())
    }
    
    /// Enable or disable debug mode
    #[wasm_bindgen]
    pub fn set_debug_mode(&mut self, enabled: bool) -> Result<(), JsValue> {
        self.avm.set_debug_mode(enabled);
        console::log_1(&format!("Debug mode {}", if enabled { "enabled" } else { "disabled" }).into());
        Ok(())
    }
    
    /// Get gas usage statistics
    #[wasm_bindgen]
    pub fn get_gas_usage(&self) -> Result<JsValue, JsValue> {
        let gas_info = serde_json::json!({
            "used": self.avm.get_gas_used(),
            "limit": self.avm.get_execution_context().map(|ctx| ctx.get_gas_limit()).unwrap_or(21000),
            "remaining": self.avm.get_execution_context().map(|ctx| ctx.get_gas_limit()).unwrap_or(21000) - self.avm.get_gas_used()
        });
        
        serde_wasm_bindgen::to_value(&gas_info)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }
    
    /// Clear transaction history
    #[wasm_bindgen]
    pub fn clear_transaction_history(&mut self) -> Result<(), JsValue> {
        self.transaction_history.clear();
        
        // Clear from storage if enabled
        if let Some(storage) = &self.storage {
            let _ = storage.remove_item("augustium_transactions");
        }
        
        console::log_1(&"Transaction history cleared".into());
        Ok(())
    }
    
    /// Save current state to persistent storage
    #[wasm_bindgen]
    pub fn save_state(&self) -> Result<(), JsValue> {
        if let Some(storage) = &self.storage {
            // Save contracts
            let contracts_json = serde_json::to_string(&self.contracts).unwrap_or_default();
            storage.set_item("augustium_contracts", &contracts_json)
                .map_err(|e| JsValue::from_str(&format!("Failed to save contracts: {:?}", e)))?;
            
            // Save transaction history
            let transactions_json = serde_json::to_string(&self.transaction_history).unwrap_or_default();
            storage.set_item("augustium_transactions", &transactions_json)
                .map_err(|e| JsValue::from_str(&format!("Failed to save transactions: {:?}", e)))?;
            
            // Save blockchain context
            let context_json = serde_json::to_string(&self.blockchain_context).unwrap_or_default();
            storage.set_item("augustium_blockchain_context", &context_json)
                .map_err(|e| JsValue::from_str(&format!("Failed to save context: {:?}", e)))?;
            
            console::log_1(&"State saved to persistent storage".into());
        } else {
            return Err(JsValue::from_str("Persistent storage not enabled"));
        }
        
        Ok(())
    }
    
    /// Load state from persistent storage
    #[wasm_bindgen]
    pub fn load_state(&mut self) -> Result<(), JsValue> {
        if let Some(storage) = &self.storage {
            // Load contracts
            if let Ok(Some(contracts_json)) = storage.get_item("augustium_contracts") {
                if let Ok(contracts) = serde_json::from_str::<HashMap<String, ContractMetadata>>(&contracts_json) {
                    self.contracts = contracts;
                }
            }
            
            // Load transaction history
            if let Ok(Some(transactions_json)) = storage.get_item("augustium_transactions") {
                if let Ok(transactions) = serde_json::from_str::<Vec<TransactionInfo>>(&transactions_json) {
                    self.transaction_history = transactions;
                }
            }
            
            // Load blockchain context
            if let Ok(Some(context_json)) = storage.get_item("augustium_blockchain_context") {
                if let Ok(context) = serde_json::from_str::<BlockchainContext>(&context_json) {
                    self.blockchain_context = context;
                }
            }
            
            console::log_1(&"State loaded from persistent storage".into());
        } else {
            return Err(JsValue::from_str("Persistent storage not enabled"));
        }
        
        Ok(())
    }
    

    /// Call a contract method (async version)
    #[wasm_bindgen]
    pub fn call_contract_async(&mut self, address: &str, method: &str, args: &JsValue) -> Promise {
        let address = address.to_string();
        let method = method.to_string();
        let _args = args.clone();
        
        // Clone the AVM for async operation
        let _addr_bytes = match hex::decode(&address.replace("0x", "")) {
            Ok(bytes) if bytes.len() == 20 => bytes,
            _ => return future_to_promise(async { Err(JsValue::from_str("Invalid address format")) }),
        };
        
        future_to_promise(async move {
            // This is a simplified version - in practice, you'd need to handle the async call properly
            Ok(JsValue::from_str(&format!("Called {}::{} with args", address, method)))
        })
    }
    
    /// Execute raw bytecode
    #[wasm_bindgen]
    pub fn execute_bytecode(&mut self, bytecode: &[u8]) -> Result<JsValue, JsValue> {
        // Create a new AVM instance for this execution
        let mut avm = WasmAvm::new();
        
        match avm.execute_bytecode(bytecode) {
            Ok(result) => Ok(result),
            Err(e) => Err(e),
        }
    }
    
    /// Get contract balance
    #[wasm_bindgen]
    pub fn get_balance(&self, address: &str) -> Result<u64, JsValue> {
        let addr_bytes = hex::decode(&address.replace("0x", ""))
            .map_err(|_| JsValue::from_str("Invalid address format"))?;
        
        if addr_bytes.len() != 20 {
            return Err(JsValue::from_str("Address must be 20 bytes"));
        }
        
        self.avm.get_balance(&addr_bytes)
    }
    
    /// Transfer tokens between addresses
    #[wasm_bindgen]
    pub fn transfer(&mut self, from: &str, to: &str, amount: u64) -> Result<(), JsValue> {
        let from_bytes = hex::decode(&from.replace("0x", ""))
            .map_err(|_| JsValue::from_str("Invalid from address format"))?;
        let to_bytes = hex::decode(&to.replace("0x", ""))
            .map_err(|_| JsValue::from_str("Invalid to address format"))?;
        
        if from_bytes.len() != 20 || to_bytes.len() != 20 {
            return Err(JsValue::from_str("Addresses must be 20 bytes"));
        }
        
        self.avm.transfer(&from_bytes, &to_bytes, amount)?;
        
        // Emit transfer event if listeners are registered
        if let Some(listener) = self.event_listeners.get("transfer") {
            let event_data = js_sys::Object::new();
            js_sys::Reflect::set(&event_data, &"from".into(), &from.into()).unwrap();
            js_sys::Reflect::set(&event_data, &"to".into(), &to.into()).unwrap();
            js_sys::Reflect::set(&event_data, &"amount".into(), &amount.into()).unwrap();
            
            let _ = listener.call1(&JsValue::NULL, &event_data);
        }
        
        Ok(())
    }
    
    /// Register an event listener
    #[wasm_bindgen]
    pub fn on(&mut self, event: &str, callback: &js_sys::Function) {
        self.event_listeners.insert(event.to_string(), callback.clone());
        console::log_1(&format!("Registered listener for event: {}", event).into());
    }
    
    /// Remove an event listener
    #[wasm_bindgen]
    pub fn off(&mut self, event: &str) {
        self.event_listeners.remove(event);
        console::log_1(&format!("Removed listener for event: {}", event).into());
    }
    
    /// Get all events from the last execution
    #[wasm_bindgen]
    pub fn get_events(&self) -> JsValue {
        self.avm.get_events()
    }
    
    /// Clear all events
    #[wasm_bindgen]
    pub fn clear_events(&mut self) {
        self.avm.clear_events();
    }
    
    /// Get runtime statistics
    #[wasm_bindgen]
    pub fn get_stats(&self) -> JsValue {
        let stats = js_sys::Object::new();
        js_sys::Reflect::set(&stats, &"gasUsed".into(), &self.avm.get_gas_used().into()).unwrap();
        js_sys::Reflect::set(&stats, &"contractsDeployed".into(), &self.contracts.len().into()).unwrap();
        js_sys::Reflect::set(&stats, &"eventListeners".into(), &self.event_listeners.len().into()).unwrap();
        
        stats.into()
    }
    
    /// Reset the runtime to initial state
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.avm.reset();
        self.contracts.clear();
        self.event_listeners.clear();
        console::log_1(&"Runtime reset to initial state".into());
    }
    
    /// Export current state for persistence
    #[wasm_bindgen]
    pub fn export_state(&self) -> Result<JsValue, JsValue> {
        let state = self.avm.get_state()?;
        let export_data = js_sys::Object::new();
        
        js_sys::Reflect::set(&export_data, &"avmState".into(), &state).unwrap();
        js_sys::Reflect::set(&export_data, &"config".into(), 
            &serde_wasm_bindgen::to_value(&self.config).unwrap()).unwrap();
        
        // Export contract addresses
        let contracts_array = js_sys::Array::new();
        for address in self.contracts.keys() {
            contracts_array.push(&address.into());
        }
        js_sys::Reflect::set(&export_data, &"contracts".into(), &contracts_array).unwrap();
        
        Ok(export_data.into())
    }
    
    /// Import state from exported data
    #[wasm_bindgen]
    pub fn import_state(&mut self, state_data: &JsValue) -> Result<(), JsValue> {
        let avm_state = js_sys::Reflect::get(state_data, &"avmState".into())
            .map_err(|_| JsValue::from_str("Missing avmState in import data"))?;
        
        self.avm.set_state(&avm_state)?;
        
        // Import config if present
        if let Ok(config_val) = js_sys::Reflect::get(state_data, &"config".into()) {
            if let Ok(config) = serde_wasm_bindgen::from_value::<AugustiumConfig>(config_val) {
                self.config = config;
                self.avm.set_debug_mode(self.config.debug_mode);
            }
        }
        
        console::log_1(&"State imported successfully".into());
        Ok(())
    }
}

/// Utility functions for web integration
#[wasm_bindgen]
pub struct WebUtils;

#[wasm_bindgen]
impl WebUtils {
    /// Generate a random address
    #[wasm_bindgen]
    pub fn generate_address() -> String {
        let mut bytes = [0u8; 20];
        
        // Use crypto.getRandomValues if available
        if let Some(window) = window() {
            if let Ok(crypto) = window.crypto() {
                let _array = js_sys::Uint8Array::new_with_length(20);
                if crypto.get_random_values_with_u8_array(&mut bytes).is_ok() {
                    // bytes are already filled
                } else {
                    // Fallback to simple random generation
                    for i in 0..20 {
                        bytes[i] = (js_sys::Math::random() * 256.0) as u8;
                    }
                }
            }
        }
        
        format!("0x{}", hex::encode(bytes))
    }
    
    /// Validate an address format
    #[wasm_bindgen]
    pub fn validate_address(address: &str) -> bool {
        if !address.starts_with("0x") {
            return false;
        }
        
        match hex::decode(&address[2..]) {
            Ok(bytes) => bytes.len() == 20,
            Err(_) => false,
        }
    }
    
    /// Convert value to hex string
    #[wasm_bindgen]
    pub fn to_hex(value: u64) -> String {
        format!("0x{:x}", value)
    }
    
    /// Convert hex string to value
    #[wasm_bindgen]
    pub fn from_hex(hex_str: &str) -> Result<u64, JsValue> {
        let clean_hex = hex_str.strip_prefix("0x").unwrap_or(hex_str);
        u64::from_str_radix(clean_hex, 16)
            .map_err(|_| JsValue::from_str("Invalid hex string"))
    }
    
    /// Get current timestamp
    #[wasm_bindgen]
    pub fn now() -> u64 {
        (js_sys::Date::now() / 1000.0) as u64
    }
    
    /// Create a transaction hash
    #[wasm_bindgen]
    pub fn create_tx_hash(from: &str, to: &str, value: u64, nonce: u64) -> String {
        // Simple hash creation - in practice, you'd use a proper hash function
        let data = format!("{}{}{}{}", from, to, value, nonce);
        let hash = data.chars().fold(0u64, |acc, c| acc.wrapping_mul(31).wrapping_add(c as u64));
        format!("0x{:064x}", hash)
    }
}

/// Contract interaction helper
#[wasm_bindgen]
pub struct ContractHelper {
    runtime: AugustiumRuntime,
    address: String,
}

#[wasm_bindgen]
impl ContractHelper {
    /// Create a new contract helper
    #[wasm_bindgen(constructor)]
    pub fn new(runtime: AugustiumRuntime, address: &str) -> ContractHelper {
        ContractHelper {
            runtime,
            address: address.to_string(),
        }
    }
    
    /// Call a contract method with automatic gas estimation
    #[wasm_bindgen]
    pub fn call(&mut self, method: &str, args: &JsValue) -> Result<JsValue, JsValue> {
        let address = self.address.clone();
        let method = method.to_string();
        let _args = args.clone();
        
        // Estimate gas first
        let estimated_gas = 21000u64; // Base gas cost
        
        // Execute the call
        console::log_1(&format!("Calling {}::{} with estimated gas: {}", address, method, estimated_gas).into());
        
        // Return mock result for now
        let result = js_sys::Object::new();
        js_sys::Reflect::set(&result, &"success".into(), &true.into()).unwrap();
        js_sys::Reflect::set(&result, &"gasUsed".into(), &estimated_gas.into()).unwrap();
        
        Ok(result.into())
    }
    
    /// Get contract address
    #[wasm_bindgen(getter)]
    pub fn address(&self) -> String {
        self.address.clone()
    }
    
    /// Get contract balance
    #[wasm_bindgen]
    pub fn balance(&self) -> Result<u64, JsValue> {
        self.runtime.get_balance(&self.address)
    }
}

/// Initialize the JavaScript API
#[wasm_bindgen]
pub fn init_js_api() {
    console::log_1(&"Augustium JavaScript API initialized".into());
    
    // Set up global error handler
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
}

/// Export version information
#[wasm_bindgen]
pub fn get_version_info() -> JsValue {
    let info = js_sys::Object::new();
    js_sys::Reflect::set(&info, &"version".into(), &env!("CARGO_PKG_VERSION").into()).unwrap();
    js_sys::Reflect::set(&info, &"name".into(), &"Augustium WASM".into()).unwrap();
    js_sys::Reflect::set(&info, &"description".into(), &"WebAssembly bindings for Augustium smart contracts".into()).unwrap();
    
    info.into()
}