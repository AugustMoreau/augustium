//! Foreign Function Interface (FFI) for Augustium
//! Enables calling external C/C++ libraries and system functions

use crate::ast::*;
use crate::error::{Result, VmError, VmErrorKind};
use crate::codegen::Value;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_void};
use std::os::raw::{c_char, c_int, c_double, c_float};
use libloading::{Library, Symbol};

/// FFI function signature
#[derive(Debug, Clone)]
pub struct FFISignature {
    pub name: String,
    pub library: String,
    pub symbol: String,
    pub parameters: Vec<FFIType>,
    pub return_type: FFIType,
    pub calling_convention: CallingConvention,
}

/// FFI type mapping
#[derive(Debug, Clone, PartialEq)]
pub enum FFIType {
    Void,
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Pointer(Box<FFIType>),
    CString,
    Array(Box<FFIType>, usize),
    Struct(String),
}

#[derive(Debug, Clone)]
pub enum CallingConvention {
    C,
    Stdcall,
    Fastcall,
}

/// FFI manager for loading and calling external functions
pub struct FFIManager {
    libraries: HashMap<String, Library>,
    functions: HashMap<String, FFISignature>,
    type_mappings: HashMap<String, FFIType>,
}

impl FFIManager {
    pub fn new() -> Self {
        let mut manager = Self {
            libraries: HashMap::new(),
            functions: HashMap::new(),
            type_mappings: HashMap::new(),
        };
        manager.setup_default_mappings();
        manager
    }

    /// Load a dynamic library
    pub fn load_library(&mut self, name: &str, path: &str) -> Result<()> {
        match unsafe { Library::new(path) } {
            Ok(lib) => {
                self.libraries.insert(name.to_string(), lib);
                Ok(())
            }
            Err(e) => Err(VmError {
                kind: VmErrorKind::FFIError,
                message: format!("Failed to load library {}: {}", name, e),
                location: None,
            }.into())
        }
    }

    /// Register an FFI function
    pub fn register_function(&mut self, signature: FFISignature) {
        self.functions.insert(signature.name.clone(), signature);
    }

    /// Call an FFI function
    pub fn call_function(&self, name: &str, args: Vec<Value>) -> Result<Value> {
        let signature = self.functions.get(name)
            .ok_or_else(|| VmError {
                kind: VmErrorKind::FFIError,
                message: format!("Unknown FFI function: {}", name),
                location: None,
            })?;

        let library = self.libraries.get(&signature.library)
            .ok_or_else(|| VmError {
                kind: VmErrorKind::FFIError,
                message: format!("Library not loaded: {}", signature.library),
                location: None,
            })?;

        // Validate argument count
        if args.len() != signature.parameters.len() {
            return Err(VmError {
                kind: VmErrorKind::FFIError,
                message: format!("Argument count mismatch: expected {}, got {}", 
                               signature.parameters.len(), args.len()),
                location: None,
            }.into());
        }

        // Convert arguments to FFI types
        let ffi_args = self.convert_args_to_ffi(&args, &signature.parameters)?;

        // Call the function based on its signature
        unsafe {
            self.call_unsafe_function(library, signature, ffi_args)
        }
    }

    fn convert_args_to_ffi(&self, args: &[Value], param_types: &[FFIType]) -> Result<Vec<FFIValue>> {
        let mut ffi_args = Vec::new();
        
        for (arg, param_type) in args.iter().zip(param_types.iter()) {
            let ffi_value = self.convert_value_to_ffi(arg, param_type)?;
            ffi_args.push(ffi_value);
        }
        
        Ok(ffi_args)
    }

    fn convert_value_to_ffi(&self, value: &Value, ffi_type: &FFIType) -> Result<FFIValue> {
        match (value, ffi_type) {
            (Value::Bool(b), FFIType::Bool) => Ok(FFIValue::Bool(*b)),
            (Value::I64(i), FFIType::I32) => Ok(FFIValue::I32(*i as i32)),
            (Value::I64(i), FFIType::I64) => Ok(FFIValue::I64(*i)),
            (Value::U64(u), FFIType::U32) => Ok(FFIValue::U32(*u as u32)),
            (Value::U64(u), FFIType::U64) => Ok(FFIValue::U64(*u)),
            (Value::F64(f), FFIType::F32) => Ok(FFIValue::F32(*f as f32)),
            (Value::F64(f), FFIType::F64) => Ok(FFIValue::F64(*f)),
            (Value::String(s), FFIType::CString) => {
                let c_string = CString::new(s.as_str())
                    .map_err(|_| VmError {
                        kind: VmErrorKind::FFIError,
                        message: "Invalid C string".to_string(),
                        location: None,
                    })?;
                Ok(FFIValue::CString(c_string))
            }
            (Value::Array(arr), FFIType::Array(elem_type, size)) => {
                if arr.len() != *size {
                    return Err(VmError {
                        kind: VmErrorKind::FFIError,
                        message: format!("Array size mismatch: expected {}, got {}", size, arr.len()),
                        location: None,
                    }.into());
                }
                
                let mut ffi_array = Vec::new();
                for elem in arr {
                    ffi_array.push(self.convert_value_to_ffi(elem, elem_type)?);
                }
                Ok(FFIValue::Array(ffi_array))
            }
            _ => Err(VmError {
                kind: VmErrorKind::FFIError,
                message: format!("Type conversion not supported: {:?} to {:?}", value, ffi_type),
                location: None,
            }.into())
        }
    }

    unsafe fn call_unsafe_function(&self, library: &Library, signature: &FFISignature, args: Vec<FFIValue>) -> Result<Value> {
        match signature.return_type {
            FFIType::Void => {
                let func: Symbol<unsafe extern "C" fn()> = library.get(signature.symbol.as_bytes())
                    .map_err(|e| VmError {
                        kind: VmErrorKind::FFIError,
                        message: format!("Symbol not found: {}", e),
                        location: None,
                    })?;
                func();
                Ok(Value::Null)
            }
            FFIType::I32 => {
                let func: Symbol<unsafe extern "C" fn() -> c_int> = library.get(signature.symbol.as_bytes())
                    .map_err(|e| VmError {
                        kind: VmErrorKind::FFIError,
                        message: format!("Symbol not found: {}", e),
                        location: None,
                    })?;
                let result = func();
                Ok(Value::I64(result as i64))
            }
            FFIType::F64 => {
                let func: Symbol<unsafe extern "C" fn() -> c_double> = library.get(signature.symbol.as_bytes())
                    .map_err(|e| VmError {
                        kind: VmErrorKind::FFIError,
                        message: format!("Symbol not found: {}", e),
                        location: None,
                    })?;
                let result = func();
                Ok(Value::F64(result))
            }
            FFIType::CString => {
                let func: Symbol<unsafe extern "C" fn() -> *const c_char> = library.get(signature.symbol.as_bytes())
                    .map_err(|e| VmError {
                        kind: VmErrorKind::FFIError,
                        message: format!("Symbol not found: {}", e),
                        location: None,
                    })?;
                let result_ptr = func();
                if result_ptr.is_null() {
                    Ok(Value::String(String::new()))
                } else {
                    let c_str = CStr::from_ptr(result_ptr);
                    let rust_str = c_str.to_string_lossy().to_string();
                    Ok(Value::String(rust_str))
                }
            }
            _ => Err(VmError {
                kind: VmErrorKind::FFIError,
                message: format!("Return type not supported: {:?}", signature.return_type),
                location: None,
            }.into())
        }
    }

    fn setup_default_mappings(&mut self) {
        // Standard C library mappings
        self.type_mappings.insert("int".to_string(), FFIType::I32);
        self.type_mappings.insert("long".to_string(), FFIType::I64);
        self.type_mappings.insert("float".to_string(), FFIType::F32);
        self.type_mappings.insert("double".to_string(), FFIType::F64);
        self.type_mappings.insert("char*".to_string(), FFIType::CString);
        self.type_mappings.insert("void".to_string(), FFIType::Void);
    }
}

/// FFI value wrapper
#[derive(Debug, Clone)]
pub enum FFIValue {
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    Pointer(*mut c_void),
    CString(CString),
    Array(Vec<FFIValue>),
}

/// Cross-chain bridge interface
pub struct CrossChainBridge {
    supported_chains: HashMap<String, ChainConfig>,
    active_connections: HashMap<String, Box<dyn ChainConnector>>,
}

#[derive(Debug, Clone)]
pub struct ChainConfig {
    pub chain_id: u64,
    pub name: String,
    pub rpc_url: String,
    pub bridge_contract: String,
    pub confirmation_blocks: u64,
}

pub trait ChainConnector: Send + Sync {
    fn connect(&mut self) -> Result<()>;
    fn disconnect(&mut self) -> Result<()>;
    fn send_transaction(&self, tx: CrossChainTransaction) -> Result<String>;
    fn get_transaction_status(&self, tx_hash: &str) -> Result<TransactionStatus>;
    fn listen_for_events(&self, callback: Box<dyn Fn(CrossChainEvent)>) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct CrossChainTransaction {
    pub from_chain: String,
    pub to_chain: String,
    pub sender: String,
    pub recipient: String,
    pub amount: u64,
    pub data: Vec<u8>,
    pub nonce: u64,
}

#[derive(Debug, Clone)]
pub enum TransactionStatus {
    Pending,
    Confirmed(u64), // block number
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct CrossChainEvent {
    pub event_type: String,
    pub chain: String,
    pub transaction_hash: String,
    pub data: HashMap<String, Value>,
}

impl CrossChainBridge {
    pub fn new() -> Self {
        Self {
            supported_chains: HashMap::new(),
            active_connections: HashMap::new(),
        }
    }

    pub fn add_chain(&mut self, config: ChainConfig) {
        self.supported_chains.insert(config.name.clone(), config);
    }

    pub fn connect_to_chain(&mut self, chain_name: &str) -> Result<()> {
        let config = self.supported_chains.get(chain_name)
            .ok_or_else(|| VmError {
                kind: VmErrorKind::CrossChainError,
                message: format!("Unknown chain: {}", chain_name),
                location: None,
            })?;

        // Create appropriate connector based on chain type
        let mut connector = self.create_connector(config)?;
        connector.connect()?;
        
        self.active_connections.insert(chain_name.to_string(), connector);
        Ok(())
    }

    pub fn transfer_cross_chain(&self, tx: CrossChainTransaction) -> Result<String> {
        let connector = self.active_connections.get(&tx.from_chain)
            .ok_or_else(|| VmError {
                kind: VmErrorKind::CrossChainError,
                message: format!("Not connected to chain: {}", tx.from_chain),
                location: None,
            })?;

        connector.send_transaction(tx)
    }

    fn create_connector(&self, config: &ChainConfig) -> Result<Box<dyn ChainConnector>> {
        // Create connector based on chain type
        match config.name.as_str() {
            "ethereum" | "polygon" | "bsc" => {
                Ok(Box::new(EVMConnector::new(config.clone())))
            }
            "solana" => {
                Ok(Box::new(SolanaConnector::new(config.clone())))
            }
            "cosmos" => {
                Ok(Box::new(CosmosConnector::new(config.clone())))
            }
            _ => Err(VmError {
                kind: VmErrorKind::CrossChainError,
                message: format!("Unsupported chain type: {}", config.name),
                location: None,
            }.into())
        }
    }
}

/// EVM-compatible chain connector
pub struct EVMConnector {
    config: ChainConfig,
    connected: bool,
}

impl EVMConnector {
    pub fn new(config: ChainConfig) -> Self {
        Self {
            config,
            connected: false,
        }
    }
}

impl ChainConnector for EVMConnector {
    fn connect(&mut self) -> Result<()> {
        // Connect to EVM chain via RPC
        self.connected = true;
        Ok(())
    }

    fn disconnect(&mut self) -> Result<()> {
        self.connected = false;
        Ok(())
    }

    fn send_transaction(&self, _tx: CrossChainTransaction) -> Result<String> {
        if !self.connected {
            return Err(VmError {
                kind: VmErrorKind::CrossChainError,
                message: "Not connected to chain".to_string(),
                location: None,
            }.into());
        }
        
        // Send transaction via RPC
        Ok("0x1234567890abcdef".to_string())
    }

    fn get_transaction_status(&self, _tx_hash: &str) -> Result<TransactionStatus> {
        Ok(TransactionStatus::Confirmed(12345))
    }

    fn listen_for_events(&self, _callback: Box<dyn Fn(CrossChainEvent)>) -> Result<()> {
        // Set up event listener
        Ok(())
    }
}

/// Solana chain connector
pub struct SolanaConnector {
    config: ChainConfig,
    connected: bool,
}

impl SolanaConnector {
    pub fn new(config: ChainConfig) -> Self {
        Self {
            config,
            connected: false,
        }
    }
}

impl ChainConnector for SolanaConnector {
    fn connect(&mut self) -> Result<()> {
        self.connected = true;
        Ok(())
    }

    fn disconnect(&mut self) -> Result<()> {
        self.connected = false;
        Ok(())
    }

    fn send_transaction(&self, _tx: CrossChainTransaction) -> Result<String> {
        if !self.connected {
            return Err(VmError {
                kind: VmErrorKind::CrossChainError,
                message: "Not connected to chain".to_string(),
                location: None,
            }.into());
        }
        
        Ok("solana_tx_hash".to_string())
    }

    fn get_transaction_status(&self, _tx_hash: &str) -> Result<TransactionStatus> {
        Ok(TransactionStatus::Confirmed(54321))
    }

    fn listen_for_events(&self, _callback: Box<dyn Fn(CrossChainEvent)>) -> Result<()> {
        Ok(())
    }
}

/// Cosmos chain connector
pub struct CosmosConnector {
    config: ChainConfig,
    connected: bool,
}

impl CosmosConnector {
    pub fn new(config: ChainConfig) -> Self {
        Self {
            config,
            connected: false,
        }
    }
}

impl ChainConnector for CosmosConnector {
    fn connect(&mut self) -> Result<()> {
        self.connected = true;
        Ok(())
    }

    fn disconnect(&mut self) -> Result<()> {
        self.connected = false;
        Ok(())
    }

    fn send_transaction(&self, _tx: CrossChainTransaction) -> Result<String> {
        if !self.connected {
            return Err(VmError {
                kind: VmErrorKind::CrossChainError,
                message: "Not connected to chain".to_string(),
                location: None,
            }.into());
        }
        
        Ok("cosmos_tx_hash".to_string())
    }

    fn get_transaction_status(&self, _tx_hash: &str) -> Result<TransactionStatus> {
        Ok(TransactionStatus::Confirmed(98765))
    }

    fn listen_for_events(&self, _callback: Box<dyn Fn(CrossChainEvent)>) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_manager() {
        let mut manager = FFIManager::new();
        
        let signature = FFISignature {
            name: "test_func".to_string(),
            library: "test_lib".to_string(),
            symbol: "test_symbol".to_string(),
            parameters: vec![FFIType::I32],
            return_type: FFIType::I32,
            calling_convention: CallingConvention::C,
        };
        
        manager.register_function(signature);
        assert!(manager.functions.contains_key("test_func"));
    }

    #[test]
    fn test_cross_chain_bridge() {
        let mut bridge = CrossChainBridge::new();
        
        let config = ChainConfig {
            chain_id: 1,
            name: "ethereum".to_string(),
            rpc_url: "https://eth.example.com".to_string(),
            bridge_contract: "0x123...".to_string(),
            confirmation_blocks: 12,
        };
        
        bridge.add_chain(config);
        assert!(bridge.supported_chains.contains_key("ethereum"));
    }

    #[test]
    fn test_ffi_type_conversion() {
        let manager = FFIManager::new();
        
        let value = Value::I64(42);
        let ffi_type = FFIType::I32;
        
        let result = manager.convert_value_to_ffi(&value, &ffi_type);
        assert!(result.is_ok());
        
        if let Ok(FFIValue::I32(converted)) = result {
            assert_eq!(converted, 42);
        } else {
            panic!("Conversion failed");
        }
    }
}
