// Augustium Virtual Machine - executes our compiled bytecode
// Stack-based VM that handles contract execution, state, and security

use crate::codegen::{Bytecode, Instruction, Value, ContractBytecode};
use crate::error::{Result, VmError, VmErrorKind};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

// Stack size limit to prevent overflow attacks
const MAX_STACK_SIZE: usize = 1024;

// Call depth limit to prevent infinite recursion
const MAX_CALL_DEPTH: usize = 256;

// Gas limit for execution (prevents infinite loops)
const MAX_GAS_LIMIT: u64 = 1_000_000;

// How much gas different operations cost
#[derive(Debug, Clone)]
pub struct GasCosts {
    pub base: u64,
    pub arithmetic: u64,
    pub comparison: u64,
    pub logical: u64,
    pub memory: u64,
    pub storage: u64,
    pub call: u64,
    pub contract_creation: u64,
    // ML operation costs - Enhanced Phase 1
    pub ml_create_model: u64,
    pub ml_train: u64,
    pub ml_predict: u64,
    pub ml_forward: u64,
    pub ml_backward: u64,
    pub ml_normalize: u64,
    
    // Enhanced ML Operations
    pub ml_create_tensor: u64,
    
    // Async operation costs
    pub async_spawn: u64,
    pub async_await: u64,
    pub async_yield: u64,
    pub ml_tensor_op: u64,
    pub ml_reshape: u64,
    pub ml_slice: u64,
    pub ml_concat: u64,
    pub ml_split: u64,
    pub ml_reduce: u64,
    pub ml_broadcast: u64,
    
    // Advanced Neural Network Operations
    pub ml_conv2d: u64,
    pub ml_maxpool2d: u64,
    pub ml_dropout: u64,
    pub ml_batch_norm: u64,
    pub ml_layer_norm: u64,
    pub ml_attention: u64,
    pub ml_embedding: u64,
    
    // Model Management
    pub ml_clone_model: u64,
    pub ml_merge_models: u64,
    pub ml_quantize_model: u64,
    pub ml_prune_model: u64,
    pub ml_distill_model: u64,
    
    // Training Operations
    pub ml_set_learning_rate: u64,
    pub ml_schedule_lr: u64,
    pub ml_gradient_clip: u64,
    pub ml_early_stopping: u64,
    pub ml_checkpoint: u64,
    pub ml_restore_checkpoint: u64,
    
    // Data Operations
    pub ml_load_dataset: u64,
    pub ml_save_dataset: u64,
    pub ml_split_dataset: u64,
    pub ml_shuffle_dataset: u64,
    pub ml_augment_data: u64,
    pub ml_preprocess_data: u64,
    
    // Evaluation and Metrics
    pub ml_evaluate: u64,
    pub ml_confusion_matrix: u64,
    pub ml_roc_curve: u64,
    pub ml_feature_importance: u64,
    pub ml_explain_prediction: u64,
    
    // Cross-Chain ML Operations
    pub ml_export_model: u64,
    pub ml_import_model: u64,
    pub ml_verify_model: u64,
    pub ml_sync_model: u64,
}

impl Default for GasCosts {
    fn default() -> Self {
        Self {
            base: 1,
            arithmetic: 3,
            comparison: 3,
            logical: 3,
            memory: 3,
            storage: 20,
            call: 40,
            contract_creation: 200,
            // ML operations are more expensive due to computation
            ml_create_model: 100,
            ml_train: 1000,
            ml_predict: 50,
            ml_forward: 30,
            ml_backward: 100,
            ml_normalize: 20,
            
            // Enhanced ML Operations
            ml_create_tensor: 25,
            ml_tensor_op: 15,
            ml_reshape: 10,
            ml_slice: 8,
            ml_concat: 20,
            ml_split: 15,
            ml_reduce: 30,
            ml_broadcast: 12,
            
            // Advanced Neural Network Operations
            ml_conv2d: 200,
            ml_maxpool2d: 50,
            ml_dropout: 5,
            ml_batch_norm: 40,
            ml_layer_norm: 35,
            ml_attention: 150,
            ml_embedding: 25,
            
            // Model Management
            ml_clone_model: 80,
            ml_merge_models: 120,
            ml_quantize_model: 100,
            ml_prune_model: 90,
            ml_distill_model: 500,
            
            // Training Operations
            ml_set_learning_rate: 5,
            ml_schedule_lr: 10,
            ml_gradient_clip: 15,
            ml_early_stopping: 8,
            ml_checkpoint: 200,
            ml_restore_checkpoint: 150,
            
            // Data Operations
            ml_load_dataset: 100,
            ml_save_dataset: 80,
            ml_split_dataset: 30,
            ml_shuffle_dataset: 25,
            ml_augment_data: 60,
            ml_preprocess_data: 40,
            
            // Evaluation and Metrics
            ml_evaluate: 200,
            ml_confusion_matrix: 50,
            ml_roc_curve: 80,
            ml_feature_importance: 100,
            ml_explain_prediction: 150,
            
            // Cross-Chain ML Operations
            ml_export_model: 300,
            ml_import_model: 250,
            ml_verify_model: 100,
            ml_sync_model: 400,
        }
    }
}

/// Execution context for the AVM
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub caller: [u8; 20],
    pub contract_address: [u8; 20],
    pub transaction_hash: [u8; 32],
    #[allow(dead_code)]
    pub origin: [u8; 20],
    #[allow(dead_code)]
    pub gas_price: u64,
    pub gas_limit: u64,
    pub value: u64,
    pub block_number: u64,
    pub timestamp: u64,
    #[allow(dead_code)]
    pub difficulty: u64,
    #[allow(dead_code)]
    pub chain_id: u64,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            caller: [0; 20],
            contract_address: [0; 20],
            transaction_hash: [0; 32],
            origin: [0; 20],
            gas_price: 1,
            gas_limit: MAX_GAS_LIMIT,
            value: 0,
            block_number: 1,
            timestamp: 1640995200, // 2022-01-01
            difficulty: 1000,
            chain_id: 1,
        }
    }
}

/// Contract instance in the AVM
#[derive(Debug, Clone)]
pub struct ContractInstance {
    #[allow(dead_code)]
    pub address: [u8; 20],
    #[allow(dead_code)]
    pub bytecode: ContractBytecode,
    #[allow(dead_code)]
    pub storage: HashMap<u32, Value>,
    pub balance: u64,
}

/// Call frame for function calls
#[derive(Debug, Clone)]
struct CallFrame {
    instructions: Vec<Instruction>,
    pc: usize,
    locals: Vec<Value>,
    #[allow(dead_code)]
    return_address: Option<usize>,
}

/// Event emitted by a contract
#[derive(Debug, Clone)]
pub struct Event {
    #[allow(dead_code)]
    pub contract_address: [u8; 20],
    #[allow(dead_code)]
    pub event_name: String,
    #[allow(dead_code)]
    pub data: Vec<Value>,
    #[allow(dead_code)]
    pub block_number: u64,
    #[allow(dead_code)]
    pub transaction_hash: [u8; 32],
}

/// Transaction result
#[derive(Debug, Clone)]
pub struct TransactionResult {
    #[allow(dead_code)]
    pub success: bool,
    #[allow(dead_code)]
    pub gas_used: u64,
    #[allow(dead_code)]
    pub return_value: Option<Value>,
    #[allow(dead_code)]
    pub events: Vec<Event>,
    #[allow(dead_code)]
    pub error: Option<String>,
}

/// Augustium Virtual Machine
pub struct AVM {
    stack: Vec<Value>,
    call_stack: Vec<CallFrame>,
    contracts: HashMap<[u8; 20], ContractInstance>,
    context: ExecutionContext,
    gas_costs: GasCosts,
    gas_used: u64,
    events: Vec<Event>,
    halted: bool,
    debug_mode: bool,
    // ML-specific state
    ml_models: HashMap<u32, crate::codegen::Value>,
    next_model_id: u32,
}

impl AVM {
    /// Create a new AVM instance
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            call_stack: Vec::new(),
            contracts: HashMap::new(),
            context: ExecutionContext::default(),
            gas_costs: GasCosts::default(),
            gas_used: 0,
            events: Vec::new(),
            halted: false,
            debug_mode: false,
            ml_models: HashMap::new(),
            next_model_id: 1,
        }
    }
    
    /// Create a new AVM instance with custom context
    #[allow(dead_code)]
    pub fn with_context(context: ExecutionContext) -> Self {
        Self {
            stack: Vec::new(),
            call_stack: Vec::new(),
            contracts: HashMap::new(),
            context,
            gas_costs: GasCosts::default(),
            gas_used: 0,
            events: Vec::new(),
            halted: false,
            debug_mode: false,
            ml_models: HashMap::new(),
            next_model_id: 1,
        }
    }
    
    /// Enable debug mode
    #[allow(dead_code)]
    pub fn set_debug_mode(&mut self, debug: bool) {
        self.debug_mode = debug;
    }
    
    /// Deploy a contract
    #[allow(dead_code)]
    pub fn deploy_contract(
        &mut self,
        bytecode: ContractBytecode,
        constructor_args: Vec<Value>,
    ) -> Result<[u8; 20]> {
        // Generate contract address (simplified)
        let mut address = [0u8; 20];
        address[0] = (self.contracts.len() + 1) as u8;
        
        // Create contract instance
        let contract = ContractInstance {
            address,
            bytecode: bytecode.clone(),
            storage: HashMap::new(),
            balance: 0,
        };
        
        // Execute constructor if present
        if !bytecode.constructor.is_empty() {
            // Push constructor arguments onto stack
            for arg in constructor_args {
                self.push(arg)?;
            }
            
            // Set contract context
            let old_contract_address = self.context.contract_address;
            self.context.contract_address = address;
            
            // Execute constructor
            let call_frame = CallFrame {
                instructions: bytecode.constructor.clone(),
                pc: 0,
                locals: Vec::new(),
                return_address: None,
            };
            
            self.call_stack.push(call_frame);
            
            // Execute until constructor completes
            while !self.call_stack.is_empty() && !self.halted {
                // self.step()?; // TODO: Implement step method
            }
            
            // Restore context
            self.context.contract_address = old_contract_address;
        }
        
        // Store contract
        self.contracts.insert(address, contract);
        
        Ok(address)
    }
    
    /// Deploy a contract from stack (used by Deploy instruction)
    fn deploy_contract_from_stack(&mut self) -> Result<()> {
        // Pop bytecode and constructor arguments from stack
        // In a real implementation, the bytecode would be passed differently
        // For now, we'll create a minimal contract
        
        // Generate contract address
        let mut address = [0u8; 20];
        address[0] = (self.contracts.len() + 1) as u8;
        
        // Create minimal contract bytecode
        let bytecode = ContractBytecode {
            constructor: Vec::new(),
            functions: std::collections::HashMap::new(),
            fields: std::collections::HashMap::new(),
            events: std::collections::HashMap::new(),
        };
        
        // Create contract instance
        let contract = ContractInstance {
            address,
            bytecode,
            storage: std::collections::HashMap::new(),
            balance: self.context.value, // Initial balance from transaction value
        };
        
        // Store contract
        self.contracts.insert(address, contract);
        
        // Push contract address onto stack as result
        self.push(Value::Address(address))?;
        
        Ok(())
    }
    
    /// Call a contract function
    #[allow(dead_code)]
    pub fn call_contract(
        &mut self,
        contract_address: [u8; 20],
        function_name: &str,
        args: Vec<Value>,
    ) -> Result<TransactionResult> {
        // Reset execution state
        self.gas_used = 0;
        self.events.clear();
        self.halted = false;
        
        // Get function bytecode (clone to avoid borrowing issues)
        let function_bytecode = {
            let contract = self.contracts.get(&contract_address)
                .ok_or_else(|| VmError::new(
                    VmErrorKind::ContractNotFound,
                    format!("Contract at address {:?} not found", contract_address),
                ))?;
            
            contract.bytecode.functions.get(function_name)
                .ok_or_else(|| VmError::new(
                    VmErrorKind::FunctionNotFound,
                    format!("Function '{}' not found in contract", function_name),
                ))?
                .clone()
        };
        
        // Push arguments onto stack
        for arg in args {
            self.push(arg)?;
        }
        
        // Create call frame
        let call_frame = CallFrame {
            instructions: function_bytecode.clone(),
            pc: 0,
            locals: Vec::new(),
            return_address: None,
        };
        
        self.call_stack.push(call_frame);
        
        // Execute function
        let result = self.execute_until_return();
        
        // Get return value
        let return_value = if !self.stack.is_empty() {
            Some(self.pop()?)
        } else {
            None
        };
        
        // Create transaction result
        let transaction_result = TransactionResult {
            success: result.is_ok() && !self.halted,
            gas_used: self.gas_used,
            return_value,
            events: self.events.clone(),
            error: result.err().map(|e| e.to_string()),
        };
        
        Ok(transaction_result)
    }
    
    /// Execute bytecode
    pub fn execute(&mut self, bytecode: &Bytecode) -> Result<TransactionResult> {
        // Reset state for new execution
        self.stack.clear();
        self.call_stack.clear();
        self.gas_used = 0;
        self.events.clear();
        self.halted = false;
        
        if self.debug_mode {
            println!("[DEBUG] Starting execution of {} instructions", bytecode.instructions.len());
        }
        
        // Setup call frame
        let call_frame = CallFrame {
            instructions: bytecode.instructions.clone(),
            pc: 0,
            locals: Vec::new(),
            return_address: None,
        };
        
        self.call_stack.push(call_frame);
        
        // Execute until halt or error
        match self.execute_until_return() {
            Ok(()) => {
                let return_value = if !self.stack.is_empty() {
                    Some(self.stack.pop().unwrap())
                } else {
                    None
                };
                
                if self.debug_mode {
                    println!("[DEBUG] Execution completed successfully, gas used: {}", self.gas_used);
                }
                
                Ok(TransactionResult {
                    success: true,
                    gas_used: self.gas_used,
                    return_value,
                    events: self.events.clone(),
                    error: None,
                })
            },
            Err(e) => {
                if self.debug_mode {
                    println!("[DEBUG] Execution failed: {}", e);
                }
                
                Ok(TransactionResult {
                    success: false,
                    gas_used: self.gas_used,
                    return_value: None,
                    events: self.events.clone(),
                    error: Some(e.to_string()),
                })
            },
        }
    }
    
    /// Execute until return or error
    fn execute_until_return(&mut self) -> Result<()> {
        let initial_call_depth = self.call_stack.len();
        
        while !self.halted && self.call_stack.len() >= initial_call_depth {
            self.execute_instruction()?;
        }
        
        Ok(())
    }
    
    /// Execute a single instruction
    fn execute_instruction(&mut self) -> Result<()> {
        // Check gas limit
        if self.gas_used >= self.context.gas_limit {
            return Err(VmError::new(
                VmErrorKind::OutOfGas,
                "Out of gas".to_string(),
            ).into());
        }
        
        // Check call stack depth
        if self.call_stack.len() > MAX_CALL_DEPTH {
            return Err(VmError::new(
                VmErrorKind::StackOverflow,
                "Call stack overflow".to_string(),
            ).into());
        }
        
        // Get current call frame
        let call_frame = self.call_stack.last_mut()
            .ok_or_else(|| VmError::new(
                VmErrorKind::EmptyCallStack,
                "Empty call stack".to_string(),
            ))?;
        
        // Check program counter bounds
        if call_frame.pc >= call_frame.instructions.len() {
            // End of instructions, return from function
            self.call_stack.pop();
            return Ok(());
        }
        
        // Get current instruction
        let instruction = call_frame.instructions[call_frame.pc].clone();
        
        if self.debug_mode {
            println!("PC: {}, Instruction: {:?}, Stack: {:?}", 
                call_frame.pc, instruction, self.stack);
        }
        
        // Advance program counter
        call_frame.pc += 1;
        
        // Execute instruction
        self.execute_single_instruction(instruction)?;
        
        Ok(())
    }
    
    /// Execute a single instruction
    fn execute_single_instruction(&mut self, instruction: Instruction) -> Result<()> {
        match instruction {
            // Stack operations
            Instruction::Push(value) => {
                self.consume_gas(self.gas_costs.base)?;
                self.push(value)?;
            }
            Instruction::Pop => {
                self.consume_gas(self.gas_costs.base)?;
                self.pop()?;
            }
            Instruction::Dup => {
                self.consume_gas(self.gas_costs.base)?;
                let value = self.peek(0)?.clone();
                self.push(value)?;
            }
            Instruction::Swap => {
                self.consume_gas(self.gas_costs.base)?;
                let a = self.pop()?;
                let b = self.pop()?;
                self.push(a)?;
                self.push(b)?;
            }
            
            // Arithmetic operations
            Instruction::Add => {
                self.consume_gas(self.gas_costs.arithmetic)?;
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.add_values(a, b)?;
                self.push(result)?;
            }
            Instruction::Sub => {
                self.consume_gas(self.gas_costs.arithmetic)?;
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.sub_values(a, b)?;
                self.push(result)?;
            }
            Instruction::Mul => {
                self.consume_gas(self.gas_costs.arithmetic)?;
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.mul_values(a, b)?;
                self.push(result)?;
            }
            Instruction::Div => {
                self.consume_gas(self.gas_costs.arithmetic)?;
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.div_values(a, b)?;
                self.push(result)?;
            }
            Instruction::Mod => {
                self.consume_gas(self.gas_costs.arithmetic)?;
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.mod_values(a, b)?;
                self.push(result)?;
            }
            
            // Comparison operations
            Instruction::Eq => {
                self.consume_gas(self.gas_costs.comparison)?;
                let b = self.pop()?;
                let a = self.pop()?;
                let result = Value::Bool(self.values_equal(&a, &b));
                self.push(result)?;
            }
            Instruction::Ne => {
                self.consume_gas(self.gas_costs.comparison)?;
                let b = self.pop()?;
                let a = self.pop()?;
                let result = Value::Bool(!self.values_equal(&a, &b));
                self.push(result)?;
            }
            Instruction::Lt => {
                self.consume_gas(self.gas_costs.comparison)?;
                let b = self.pop()?;
                let a = self.pop()?;
                let result = Value::Bool(self.compare_values(&a, &b)? < 0);
                self.push(result)?;
            }
            Instruction::Le => {
                self.consume_gas(self.gas_costs.comparison)?;
                let b = self.pop()?;
                let a = self.pop()?;
                let result = Value::Bool(self.compare_values(&a, &b)? <= 0);
                self.push(result)?;
            }
            Instruction::Gt => {
                self.consume_gas(self.gas_costs.comparison)?;
                let b = self.pop()?;
                let a = self.pop()?;
                let result = Value::Bool(self.compare_values(&a, &b)? > 0);
                self.push(result)?;
            }
            Instruction::Ge => {
                self.consume_gas(self.gas_costs.comparison)?;
                let b = self.pop()?;
                let a = self.pop()?;
                let result = Value::Bool(self.compare_values(&a, &b)? >= 0);
                self.push(result)?;
            }
            
            // Logical operations
            Instruction::And => {
                self.consume_gas(self.gas_costs.logical)?;
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.and_values(a, b)?;
                self.push(result)?;
            }
            Instruction::Or => {
                self.consume_gas(self.gas_costs.logical)?;
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.or_values(a, b)?;
                self.push(result)?;
            }
            Instruction::Not => {
                self.consume_gas(self.gas_costs.logical)?;
                let a = self.pop()?;
                let result = self.not_value(a)?;
                self.push(result)?;
            }
            
            // Bitwise operations
            Instruction::BitAnd => {
                self.consume_gas(self.gas_costs.arithmetic)?;
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.bitand_values(a, b)?;
                self.push(result)?;
            }
            Instruction::BitOr => {
                self.consume_gas(self.gas_costs.arithmetic)?;
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.bitor_values(a, b)?;
                self.push(result)?;
            }
            Instruction::BitXor => {
                self.consume_gas(self.gas_costs.arithmetic)?;
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.bitxor_values(a, b)?;
                self.push(result)?;
            }
            Instruction::BitNot => {
                self.consume_gas(self.gas_costs.arithmetic)?;
                let a = self.pop()?;
                let result = self.bitnot_value(a)?;
                self.push(result)?;
            }
            Instruction::Shl => {
                self.consume_gas(self.gas_costs.arithmetic)?;
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.shl_values(a, b)?;
                self.push(result)?;
            }
            Instruction::Shr => {
                self.consume_gas(self.gas_costs.arithmetic)?;
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.shr_values(a, b)?;
                self.push(result)?;
            }
            
            // Memory operations
            Instruction::Load(index) => {
                self.consume_gas(self.gas_costs.memory)?;
                let value = self.load_local(index)?;
                self.push(value)?;
            }
            Instruction::Store(index) => {
                self.consume_gas(self.gas_costs.memory)?;
                let value = self.pop()?;
                self.store_local(index, value)?;
            }
            Instruction::LoadField(index) => {
                self.consume_gas(self.gas_costs.storage)?;
                let value = self.load_field(index)?;
                self.push(value)?;
            }
            Instruction::StoreField(index) => {
                self.consume_gas(self.gas_costs.storage)?;
                let value = self.pop()?;
                self.store_field(index, value)?;
            }
            
            // Control flow
            Instruction::Jump(offset) => {
                self.consume_gas(self.gas_costs.base)?;
                self.jump(offset)?;
            }
            Instruction::JumpIf(offset) => {
                self.consume_gas(self.gas_costs.base)?;
                let condition = self.pop()?;
                if self.is_truthy(&condition) {
                    self.jump(offset)?;
                }
            }
            Instruction::JumpIfNot(offset) => {
                self.consume_gas(self.gas_costs.base)?;
                let condition = self.pop()?;
                if !self.is_truthy(&condition) {
                    self.jump(offset)?;
                }
            }
            Instruction::Call(offset) => {
                self.consume_gas(self.gas_costs.call)?;
                self.call_function(offset)?;
            }
            Instruction::Return => {
                self.consume_gas(self.gas_costs.base)?;
                self.return_from_function()?;
            }
            
            // Contract operations
            Instruction::Deploy => {
                self.consume_gas(self.gas_costs.contract_creation)?;
                self.deploy_contract_from_stack()?;
            }
            Instruction::Invoke(function_name) => {
                self.consume_gas(self.gas_costs.call)?;
                self.invoke_function(&function_name)?;
            }
            Instruction::Emit(event_name) => {
                self.consume_gas(self.gas_costs.base)?;
                self.emit_event(&event_name)?;
            }
            
            // Blockchain operations
            Instruction::GetBalance => {
                self.consume_gas(self.gas_costs.base)?;
                let address_value = self.pop()?;
                let balance = self.get_balance(&address_value)?;
                self.push(Value::U64(balance))?;
            }
            Instruction::Transfer => {
                self.consume_gas(self.gas_costs.call)?;
                let amount = self.pop()?;
                let to_address = self.pop()?;
                self.transfer(&to_address, &amount)?;
            }
            Instruction::GetCaller => {
                self.consume_gas(self.gas_costs.base)?;
                self.push(Value::Address(self.context.caller))?;
            }
            Instruction::GetValue => {
                self.consume_gas(self.gas_costs.base)?;
                self.push(Value::U64(self.context.value))?;
            }
            Instruction::GetBlockNumber => {
                self.consume_gas(self.gas_costs.base)?;
                self.push(Value::U64(self.context.block_number))?;
            }
            Instruction::GetTimestamp => {
                self.consume_gas(self.gas_costs.base)?;
                self.push(Value::U64(self.context.timestamp))?;
            }
            
            // Safety operations
            Instruction::Require => {
                self.consume_gas(self.gas_costs.base)?;
                let message = self.pop()?;
                let condition = self.pop()?;
                if !self.is_truthy(&condition) {
                    let error_msg = match message {
                        Value::String(s) => s,
                        _ => "Requirement failed".to_string(),
                    };
                    return Err(VmError::new(
                        VmErrorKind::RequirementFailed,
                        error_msg,
                    ).into());
                }
            }
            Instruction::Assert => {
                self.consume_gas(self.gas_costs.base)?;
                let message = self.pop()?;
                let condition = self.pop()?;
                if !self.is_truthy(&condition) {
                    let error_msg = match message {
                        Value::String(s) => s,
                        _ => "Assertion failed".to_string(),
                    };
                    return Err(VmError::new(
                        VmErrorKind::AssertionFailed,
                        error_msg,
                    ).into());
                }
            }
            Instruction::Revert => {
                self.consume_gas(self.gas_costs.base)?;
                let message = self.pop()?;
                let error_msg = match message {
                    Value::String(s) => s,
                    _ => "Transaction reverted".to_string(),
                };
                return Err(VmError::new(
                    VmErrorKind::TransactionReverted,
                    error_msg,
                ).into());
            }
            
            // Debugging
            Instruction::Debug(message) => {
                self.consume_gas(self.gas_costs.base)?;
                if self.debug_mode {
                    println!("DEBUG: {}", message);
                }
            }
            
            // Machine Learning operations
            Instruction::MLCreateModel(model_type) => {
                self.consume_gas(self.gas_costs.ml_create_model)?;
                self.ml_create_model(&model_type)?;
            }
            Instruction::MLLoadModel(model_id) => {
                self.consume_gas(self.gas_costs.base)?;
                self.ml_load_model(model_id)?;
            }
            Instruction::MLSaveModel(model_id) => {
                self.consume_gas(self.gas_costs.base)?;
                self.ml_save_model(model_id)?;
            }
            Instruction::MLTrain(model_id) => {
                self.consume_gas(self.gas_costs.ml_train)?;
                self.ml_train_model(model_id)?;
            }
            Instruction::MLPredict(model_id) => {
                self.consume_gas(self.gas_costs.ml_predict)?;
                self.ml_predict(model_id)?;
            }
            Instruction::MLSetHyperparams(model_id) => {
                self.consume_gas(self.gas_costs.base)?;
                self.ml_set_hyperparams(model_id)?;
            }
            Instruction::MLGetMetrics(model_id) => {
                self.consume_gas(self.gas_costs.base)?;
                self.ml_get_metrics(model_id)?;
            }
            Instruction::MLForward(model_id) => {
                self.consume_gas(self.gas_costs.ml_forward)?;
                self.ml_forward_pass(model_id)?;
            }
            Instruction::MLBackward(model_id) => {
                self.consume_gas(self.gas_costs.ml_backward)?;
                self.ml_backward_pass(model_id)?;
            }
            Instruction::MLUpdateWeights(model_id) => {
                self.consume_gas(self.gas_costs.base)?;
                self.ml_update_weights(model_id)?;
            }
            Instruction::MLNormalize => {
                self.consume_gas(self.gas_costs.ml_normalize)?;
                self.ml_normalize_data()?;
            }
            Instruction::MLDenormalize => {
                self.consume_gas(self.gas_costs.ml_normalize)?;
                self.ml_denormalize_data()?;
            }
            Instruction::MLActivation(activation_type) => {
                self.consume_gas(self.gas_costs.base)?;
                self.ml_apply_activation(&activation_type)?;
            }
            Instruction::MLLoss(loss_type) => {
                self.consume_gas(self.gas_costs.base)?;
                self.ml_calculate_loss(&loss_type)?;
            }
            Instruction::MLOptimizer(optimizer_type) => {
                self.consume_gas(self.gas_costs.base)?;
                self.ml_apply_optimizer(&optimizer_type)?;
            }
            
            // Enhanced ML Instructions - Phase 1
            Instruction::MLCreateTensor(_dimensions) => {
                self.consume_gas(self.gas_costs.ml_create_tensor)?;
                self.ml_create_tensor()?;
            }
            Instruction::MLTensorOp(op_type) => {
                self.consume_gas(self.gas_costs.ml_tensor_op)?;
                self.ml_tensor_op(&op_type)?;
            }
            Instruction::MLReshape(new_shape) => {
                self.consume_gas(self.gas_costs.ml_reshape)?;
                self.ml_reshape(&new_shape)?;
            }
            Instruction::MLSlice(ranges) => {
                self.consume_gas(self.gas_costs.ml_slice)?;
                self.ml_slice(&ranges)?;
            }
            Instruction::MLConcat(axis) => {
                self.consume_gas(self.gas_costs.ml_concat)?;
                self.ml_concat(axis)?;
            }
            Instruction::MLSplit(axis, sections) => {
                self.consume_gas(self.gas_costs.ml_split)?;
                self.ml_split(axis, sections)?;
            }
            Instruction::MLReduce(op_type, axis) => {
                self.consume_gas(self.gas_costs.ml_reduce)?;
                self.ml_reduce(&op_type, axis)?;
            }
            Instruction::MLBroadcast(target_shape) => {
                self.consume_gas(self.gas_costs.ml_broadcast)?;
                self.ml_broadcast(&target_shape)?;
            }
            
            // Advanced Neural Network Operations
            Instruction::MLConv2D(_filters, _kernel_size, _stride, _padding) => {
                self.consume_gas(self.gas_costs.ml_conv2d)?;
                self.ml_conv2d()?;
            }
            Instruction::MLMaxPool2D(_pool_size, _stride) => {
                self.consume_gas(self.gas_costs.ml_maxpool2d)?;
                self.ml_maxpool2d()?;
            }
            Instruction::MLDropout(_probability) => {
                self.consume_gas(self.gas_costs.ml_dropout)?;
                self.ml_dropout()?;
            }
            Instruction::MLBatchNorm => {
                self.consume_gas(self.gas_costs.ml_batch_norm)?;
                self.ml_batch_norm()?;
            }
            Instruction::MLLayerNorm => {
                self.consume_gas(self.gas_costs.ml_layer_norm)?;
                self.ml_layer_norm()?;
            }
            Instruction::MLAttention => {
                self.consume_gas(self.gas_costs.ml_attention)?;
                self.ml_attention()?;
            }
            Instruction::MLEmbedding(_vocab_size, _embed_dim) => {
                self.consume_gas(self.gas_costs.ml_embedding)?;
                self.ml_embedding()?;
            }
            
            // Model Management
            Instruction::MLCloneModel(model_id) => {
                self.consume_gas(self.gas_costs.ml_clone_model)?;
                self.ml_clone_model(model_id)?;
            }
            Instruction::MLMergeModels(model_ids) => {
                self.consume_gas(self.gas_costs.ml_merge_models)?;
                self.ml_merge_models(&model_ids)?;
            }
            Instruction::MLQuantizeModel(model_id, quantization_type) => {
                self.consume_gas(self.gas_costs.ml_quantize_model)?;
                self.ml_quantize_model(model_id)?;
            }
            Instruction::MLPruneModel(model_id, threshold) => {
                self.consume_gas(self.gas_costs.ml_prune_model)?;
                self.ml_prune_model(model_id, threshold)?;
            }
            Instruction::MLDistillModel(teacher_id, student_id) => {
                self.consume_gas(self.gas_costs.ml_distill_model)?;
                self.ml_distill_model(teacher_id, student_id)?;
            }
            
            // Training Operations
            Instruction::MLSetLearningRate(learning_rate) => {
                self.consume_gas(self.gas_costs.ml_set_learning_rate)?;
                self.ml_set_learning_rate(learning_rate)?;
            }
            Instruction::MLScheduleLR(scheduler_type) => {
                self.consume_gas(self.gas_costs.ml_schedule_lr)?;
                self.ml_schedule_learning_rate(&scheduler_type)?;
            }
            Instruction::MLGradientClip(max_norm) => {
                self.consume_gas(self.gas_costs.ml_gradient_clip)?;
                self.ml_gradient_clip(max_norm)?;
            }
            Instruction::MLEarlyStopping(patience, min_delta) => {
                self.consume_gas(self.gas_costs.ml_early_stopping)?;
                self.ml_early_stopping()?;
            }
            Instruction::MLCheckpoint(checkpoint_name) => {
                self.consume_gas(self.gas_costs.ml_checkpoint)?;
                self.ml_save_checkpoint(&checkpoint_name)?;
            }
            Instruction::MLRestoreCheckpoint(checkpoint_name) => {
                self.consume_gas(self.gas_costs.ml_restore_checkpoint)?;
                self.ml_restore_checkpoint(&checkpoint_name)?;
            }
            
            // Data Operations
            Instruction::MLLoadDataset(dataset_name) => {
                self.consume_gas(self.gas_costs.ml_load_dataset)?;
                self.ml_load_dataset(&dataset_name)?;
            }
            Instruction::MLSaveDataset(dataset_name) => {
                self.consume_gas(self.gas_costs.ml_save_dataset)?;
                self.ml_save_dataset(&dataset_name)?;
            }
            Instruction::MLSplitDataset(train_ratio, val_ratio) => {
                self.consume_gas(self.gas_costs.ml_split_dataset)?;
                self.ml_split_dataset(train_ratio, val_ratio)?;
            }
            Instruction::MLShuffleDataset => {
                self.consume_gas(self.gas_costs.ml_shuffle_dataset)?;
                self.ml_shuffle_dataset()?;
            }
            Instruction::MLAugmentData(augmentation_type) => {
                self.consume_gas(self.gas_costs.ml_augment_data)?;
                self.ml_augment_data()?;
            }
            Instruction::MLPreprocessData(preprocessing_type) => {
                self.consume_gas(self.gas_costs.ml_preprocess_data)?;
                self.ml_preprocess_data(&preprocessing_type)?;
            }
            
            // Evaluation and Metrics
            Instruction::MLEvaluate(model_id) => {
                self.consume_gas(self.gas_costs.ml_evaluate)?;
                self.ml_evaluate(model_id)?;
            }
            Instruction::MLConfusionMatrix => {
                self.consume_gas(self.gas_costs.ml_confusion_matrix)?;
                self.ml_confusion_matrix()?;
            }
            Instruction::MLROCCurve => {
                self.consume_gas(self.gas_costs.ml_roc_curve)?;
                self.ml_roc_curve()?;
            }
            Instruction::MLFeatureImportance => {
                self.consume_gas(self.gas_costs.ml_feature_importance)?;
                self.ml_feature_importance()?;
            }
            Instruction::MLExplainPrediction(model_id) => {
                self.consume_gas(self.gas_costs.ml_explain_prediction)?;
                self.ml_explain_prediction(model_id)?;
            }
            
            // Cross-Chain ML Operations
            Instruction::MLExportModel(export_format) => {
                self.consume_gas(self.gas_costs.ml_export_model)?;
                self.ml_export_model(0)?; // Use dummy model ID
            }
            Instruction::MLImportModel(import_format) => {
                self.consume_gas(self.gas_costs.ml_import_model)?;
                self.ml_import_model(&import_format)?;
            }
            Instruction::MLVerifyModel(verification_type) => {
                self.consume_gas(self.gas_costs.ml_verify_model)?;
                self.ml_verify_model(&verification_type)?;
            }
            Instruction::MLSyncModel(model_id, target_chain) => {
                self.consume_gas(self.gas_costs.ml_sync_model)?;
                self.ml_sync_model(model_id)?;
            }
            
            // Halt execution
            Instruction::Halt => {
                self.consume_gas(self.gas_costs.base)?;
                self.halted = true;
            }
        }
        
        Ok(())
    }
    
    /// Push a value onto the stack
    pub fn push(&mut self, value: Value) -> Result<()> {
        if self.stack.len() >= MAX_STACK_SIZE {
            return Err(VmError::new(
                VmErrorKind::StackOverflow,
                "Stack overflow".to_string(),
            ).into());
        }
        
        self.stack.push(value);
        Ok(())
    }
    
    /// Pop a value from the stack
    fn pop(&mut self) -> Result<Value> {
        self.stack.pop().ok_or_else(|| VmError::new(
            VmErrorKind::StackUnderflow,
            "Stack underflow".to_string(),
        ).into())
    }
    
    /// Peek at a value on the stack without removing it
    fn peek(&self, offset: usize) -> Result<&Value> {
        let index = self.stack.len().checked_sub(offset + 1)
            .ok_or_else(|| VmError::new(
                VmErrorKind::StackUnderflow,
                "Stack underflow".to_string(),
            ))?;
        
        Ok(&self.stack[index])
    }
    
    /// Consume gas
    fn consume_gas(&mut self, amount: u64) -> Result<()> {
        self.gas_used += amount;
        if self.gas_used > self.context.gas_limit {
            return Err(VmError::new(
                VmErrorKind::OutOfGas,
                "Out of gas".to_string(),
            ).into());
        }
        Ok(())
    }
    
    /// Load a local variable
    fn load_local(&self, index: u32) -> Result<Value> {
        let call_frame = self.call_stack.last()
            .ok_or_else(|| VmError::new(
                VmErrorKind::EmptyCallStack,
                "Empty call stack".to_string(),
            ))?;
        
        call_frame.locals.get(index as usize)
            .cloned()
            .ok_or_else(|| VmError::new(
                VmErrorKind::InvalidLocalAccess,
                format!("Invalid local variable access: {}", index),
            ).into())
    }
    
    /// Store a local variable
    fn store_local(&mut self, index: u32, value: Value) -> Result<()> {
        let call_frame = self.call_stack.last_mut()
            .ok_or_else(|| VmError::new(
                VmErrorKind::EmptyCallStack,
                "Empty call stack".to_string(),
            ))?;
        
        // Extend locals vector if necessary
        while call_frame.locals.len() <= index as usize {
            call_frame.locals.push(Value::Null);
        }
        
        call_frame.locals[index as usize] = value;
        Ok(())
    }
    
    /// Load a contract field
    fn load_field(&mut self, index: u32) -> Result<Value> {
        // Get current contract address from context
        let contract_address = self.context.contract_address;
        
        if let Some(contract) = self.contracts.get(&contract_address) {
            // Load field from contract storage
            if let Some(value) = contract.storage.get(&index) {
                Ok(value.clone())
            } else {
                // Field not initialized, return default value
                Ok(Value::Null)
            }
        } else {
            Err(VmError::new(
                VmErrorKind::ContractNotFound,
                format!("Contract at address {:?} not found", contract_address),
            ).into())
        }
    }
    
    /// Store a contract field
    fn store_field(&mut self, index: u32, value: Value) -> Result<()> {
        // Get current contract address from context
        let contract_address = self.context.contract_address;
        
        if let Some(contract) = self.contracts.get_mut(&contract_address) {
            // Store field in contract storage
            contract.storage.insert(index, value);
            Ok(())
        } else {
            Err(VmError::new(
                VmErrorKind::ContractNotFound,
                format!("Contract at address {:?} not found", contract_address),
            ).into())
        }
    }
    
    /// Jump to a specific instruction offset
    fn jump(&mut self, offset: u32) -> Result<()> {
        let call_frame = self.call_stack.last_mut()
            .ok_or_else(|| VmError::new(
                VmErrorKind::EmptyCallStack,
                "Empty call stack".to_string(),
            ))?;
        
        call_frame.pc = offset as usize;
        Ok(())
    }
    
    /// Call a function
    fn call_function(&mut self, offset: u32) -> Result<()> {
        // Check call depth limit
        if self.call_stack.len() >= MAX_CALL_DEPTH {
            return Err(VmError::new(
                VmErrorKind::CallDepthExceeded,
                "Maximum call depth exceeded".to_string(),
            ).into());
        }
        
        // Get current call frame to access instructions
        let current_frame = self.call_stack.last()
            .ok_or_else(|| VmError::new(
                VmErrorKind::EmptyCallStack,
                "No current call frame".to_string(),
            ))?;
        
        // Create new call frame for the function
        let new_frame = CallFrame {
            instructions: current_frame.instructions.clone(), // In a real implementation, this would be function-specific bytecode
            pc: offset as usize,
            locals: Vec::new(),
            return_address: Some(current_frame.pc + 1),
        };
        
        self.call_stack.push(new_frame);
        Ok(())
    }
    
    /// Return from a function
    fn return_from_function(&mut self) -> Result<()> {
        self.call_stack.pop();
        Ok(())
    }
    
    /// Invoke a function by name
    fn invoke_function(&mut self, function_name: &str) -> Result<()> {
        // Get current contract address
        let contract_address = self.context.contract_address;
        
        if let Some(contract) = self.contracts.get(&contract_address) {
            // Look up function in contract bytecode
            if let Some(function_instructions) = contract.bytecode.functions.get(function_name) {
                // Create new call frame for the function
                let new_frame = CallFrame {
                    instructions: function_instructions.clone(),
                    pc: 0,
                    locals: Vec::new(),
                    return_address: self.call_stack.last().map(|f| f.pc + 1),
                };
                
                self.call_stack.push(new_frame);
            } else {
                return Err(VmError::new(
                    VmErrorKind::FunctionNotFound,
                    format!("Function '{}' not found in contract", function_name),
                ).into());
            }
        } else {
            return Err(VmError::new(
                VmErrorKind::ContractNotFound,
                format!("Contract at address {:?} not found", contract_address),
            ).into());
        }
        
        Ok(())
    }
    
    /// Emit an event
    fn emit_event(&mut self, event_name: &str) -> Result<()> {
        // Collect event data from stack (number of data items should be specified or known)
        // For now, we'll collect all available stack items as event data
        let mut event_data = Vec::new();
        
        // In a real implementation, the number of event parameters would be known
        // For now, we'll assume the top stack value indicates the number of parameters
        if !self.stack.is_empty() {
            if let Value::U32(param_count) = self.stack.last().cloned().unwrap_or(Value::U32(0)) {
                self.pop()?; // Remove the parameter count
                
                // Collect the specified number of parameters
                for _ in 0..param_count.min(self.stack.len() as u32) {
                    if let Ok(value) = self.pop() {
                        event_data.push(value);
                    }
                }
            }
        }
        
        let event = Event {
            contract_address: self.context.contract_address,
            event_name: event_name.to_string(),
            data: event_data,
            block_number: self.context.block_number,
            transaction_hash: self.context.transaction_hash,
        };
        
        self.events.push(event);
        Ok(())
    }
    
    /// Get balance of an address
    fn get_balance(&self, address_value: &Value) -> Result<u64> {
        match address_value {
            Value::Address(address) => {
                if let Some(contract) = self.contracts.get(address) {
                    Ok(contract.balance)
                } else {
                    Ok(0) // Address not found, balance is 0
                }
            }
            _ => Err(VmError::new(
                VmErrorKind::InvalidAddress,
                "Invalid address for balance query".to_string(),
            ).into()),
        }
    }
    
    /// Transfer value between addresses
    fn transfer(&mut self, to_address: &Value, amount: &Value) -> Result<()> {
        let from_address = self.context.caller;
        
        let to_addr = match to_address {
            Value::Address(addr) => *addr,
            _ => return Err(VmError::new(
                VmErrorKind::InvalidAddress,
                "Invalid recipient address".to_string(),
            ).into()),
        };
        
        let transfer_amount = match amount {
            Value::U64(amt) => *amt,
            Value::U32(amt) => *amt as u64,
            _ => return Err(VmError::new(
                VmErrorKind::InvalidAmount,
                "Invalid transfer amount".to_string(),
            ).into()),
        };
        
        // Check sender balance
        let sender_balance = if let Some(sender_contract) = self.contracts.get(&from_address) {
            sender_contract.balance
        } else {
            0
        };
        
        if sender_balance < transfer_amount {
            return Err(VmError::new(
                VmErrorKind::InsufficientBalance,
                "Insufficient balance for transfer".to_string(),
            ).into());
        }
        
        // Perform transfer
        if let Some(sender_contract) = self.contracts.get_mut(&from_address) {
            sender_contract.balance -= transfer_amount;
        }
        
        // Add to recipient (create contract entry if doesn't exist)
        self.contracts.entry(to_addr)
            .or_insert_with(|| ContractInstance {
                address: to_addr,
                bytecode: ContractBytecode {
                    constructor: Vec::new(),
                    functions: std::collections::HashMap::new(),
                    fields: std::collections::HashMap::new(),
                    events: std::collections::HashMap::new(),
                },
                storage: std::collections::HashMap::new(),
                balance: 0,
            })
            .balance += transfer_amount;
        
        Ok(())
    }
    
    /// Check if a value is truthy
    fn is_truthy(&self, value: &Value) -> bool {
        match value {
            Value::Bool(b) => *b,
            Value::U8(n) => *n != 0,
            Value::U16(n) => *n != 0,
            Value::U32(n) => *n != 0,
            Value::U64(n) => *n != 0,
            Value::U128(n) => *n != 0,
            Value::U256(bytes) => bytes.iter().any(|&b| b != 0),
            Value::I8(n) => *n != 0,
            Value::I16(n) => *n != 0,
            Value::I32(n) => *n != 0,
            Value::I64(n) => *n != 0,
            Value::I128(n) => *n != 0,
            Value::I256(bytes) => bytes.iter().any(|&b| b != 0),
            Value::F32(f) => *f != 0.0,
            Value::F64(f) => *f != 0.0,
            Value::String(s) => !s.is_empty(),
            Value::Array(arr) => !arr.is_empty(),
            Value::Tuple(tuple) => !tuple.is_empty(),
            Value::Address(_) => true,
            Value::MLModel { .. } => true,
            Value::MLDataset { .. } => true,
            Value::MLMetrics { .. } => true,
            Value::Tensor { data, .. } => !data.is_empty(),
            Value::Matrix(data) => !data.is_empty(),
            Value::Vector(data) => !data.is_empty(),
            Value::MLOptimizer { .. } => true,
            Value::MLScheduler { .. } => true,
            Value::MLCheckpoint { .. } => true,
            Value::MLExplanation { .. } => true,
            Value::Null => false,
        }
    }
    
    /// Check if two values are equal
    fn values_equal(&self, a: &Value, b: &Value) -> bool {
        std::mem::discriminant(a) == std::mem::discriminant(b) && a == b
    }
    
    /// Compare two values (-1, 0, 1)
    fn compare_values(&self, a: &Value, b: &Value) -> Result<i32> {
        match (a, b) {
            (Value::U32(a), Value::U32(b)) => Ok(a.cmp(b) as i32),
            (Value::U64(a), Value::U64(b)) => Ok(a.cmp(b) as i32),
            (Value::I32(a), Value::I32(b)) => Ok(a.cmp(b) as i32),
            (Value::I64(a), Value::I64(b)) => Ok(a.cmp(b) as i32),
            (Value::String(a), Value::String(b)) => Ok(a.cmp(b) as i32),
            _ => Err(VmError::new(
                VmErrorKind::TypeMismatch,
                "Cannot compare values of different types".to_string(),
            ).into()),
        }
    }
    
    // Arithmetic operations
    fn add_values(&self, a: Value, b: Value) -> Result<Value> {
        match (a, b) {
            (Value::U32(a), Value::U32(b)) => Ok(Value::U32(a.wrapping_add(b))),
            (Value::U64(a), Value::U64(b)) => Ok(Value::U64(a.wrapping_add(b))),
            (Value::I32(a), Value::I32(b)) => Ok(Value::I32(a.wrapping_add(b))),
            (Value::I64(a), Value::I64(b)) => Ok(Value::I64(a.wrapping_add(b))),
            _ => Err(VmError::new(
                VmErrorKind::TypeMismatch,
                "Cannot add values of incompatible types".to_string(),
            ).into()),
        }
    }
    
    fn sub_values(&self, a: Value, b: Value) -> Result<Value> {
        match (a, b) {
            (Value::U32(a), Value::U32(b)) => Ok(Value::U32(a.wrapping_sub(b))),
            (Value::U64(a), Value::U64(b)) => Ok(Value::U64(a.wrapping_sub(b))),
            (Value::I32(a), Value::I32(b)) => Ok(Value::I32(a.wrapping_sub(b))),
            (Value::I64(a), Value::I64(b)) => Ok(Value::I64(a.wrapping_sub(b))),
            _ => Err(VmError::new(
                VmErrorKind::TypeMismatch,
                "Cannot subtract values of incompatible types".to_string(),
            ).into()),
        }
    }
    
    fn mul_values(&self, a: Value, b: Value) -> Result<Value> {
        match (a, b) {
            (Value::U32(a), Value::U32(b)) => Ok(Value::U32(a.wrapping_mul(b))),
            (Value::U64(a), Value::U64(b)) => Ok(Value::U64(a.wrapping_mul(b))),
            (Value::I32(a), Value::I32(b)) => Ok(Value::I32(a.wrapping_mul(b))),
            (Value::I64(a), Value::I64(b)) => Ok(Value::I64(a.wrapping_mul(b))),
            _ => Err(VmError::new(
                VmErrorKind::TypeMismatch,
                "Cannot multiply values of incompatible types".to_string(),
            ).into()),
        }
    }
    
    fn div_values(&self, a: Value, b: Value) -> Result<Value> {
        match (a, b) {
            (Value::U32(a), Value::U32(b)) => {
                if b == 0 {
                    return Err(VmError::new(
                        VmErrorKind::DivisionByZero,
                        "Division by zero".to_string(),
                    ).into());
                }
                Ok(Value::U32(a / b))
            }
            (Value::U64(a), Value::U64(b)) => {
                if b == 0 {
                    return Err(VmError::new(
                        VmErrorKind::DivisionByZero,
                        "Division by zero".to_string(),
                    ).into());
                }
                Ok(Value::U64(a / b))
            }
            (Value::I32(a), Value::I32(b)) => {
                if b == 0 {
                    return Err(VmError::new(
                        VmErrorKind::DivisionByZero,
                        "Division by zero".to_string(),
                    ).into());
                }
                Ok(Value::I32(a / b))
            }
            (Value::I64(a), Value::I64(b)) => {
                if b == 0 {
                    return Err(VmError::new(
                        VmErrorKind::DivisionByZero,
                        "Division by zero".to_string(),
                    ).into());
                }
                Ok(Value::I64(a / b))
            }
            _ => Err(VmError::new(
                VmErrorKind::TypeMismatch,
                "Cannot divide values of incompatible types".to_string(),
            ).into()),
        }
    }
    
    fn mod_values(&self, a: Value, b: Value) -> Result<Value> {
        match (a, b) {
            (Value::U32(a), Value::U32(b)) => {
                if b == 0 {
                    return Err(VmError::new(
                        VmErrorKind::DivisionByZero,
                        "Modulo by zero".to_string(),
                    ).into());
                }
                Ok(Value::U32(a % b))
            }
            (Value::U64(a), Value::U64(b)) => {
                if b == 0 {
                    return Err(VmError::new(
                        VmErrorKind::DivisionByZero,
                        "Modulo by zero".to_string(),
                    ).into());
                }
                Ok(Value::U64(a % b))
            }
            _ => Err(VmError::new(
                VmErrorKind::TypeMismatch,
                "Cannot perform modulo on incompatible types".to_string(),
            ).into()),
        }
    }
    
    // Logical operations
    fn and_values(&self, a: Value, b: Value) -> Result<Value> {
        match (a, b) {
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a && b)),
            _ => Err(VmError::new(
                VmErrorKind::TypeMismatch,
                "Logical AND requires boolean values".to_string(),
            ).into()),
        }
    }
    
    fn or_values(&self, a: Value, b: Value) -> Result<Value> {
        match (a, b) {
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a || b)),
            _ => Err(VmError::new(
                VmErrorKind::TypeMismatch,
                "Logical OR requires boolean values".to_string(),
            ).into()),
        }
    }
    
    fn not_value(&self, a: Value) -> Result<Value> {
        match a {
            Value::Bool(a) => Ok(Value::Bool(!a)),
            _ => Err(VmError::new(
                VmErrorKind::TypeMismatch,
                "Logical NOT requires boolean value".to_string(),
            ).into()),
        }
    }
    
    // Bitwise operations
    fn bitand_values(&self, a: Value, b: Value) -> Result<Value> {
        match (a, b) {
            (Value::U32(a), Value::U32(b)) => Ok(Value::U32(a & b)),
            (Value::U64(a), Value::U64(b)) => Ok(Value::U64(a & b)),
            _ => Err(VmError::new(
                VmErrorKind::TypeMismatch,
                "Bitwise AND requires integer values".to_string(),
            ).into()),
        }
    }
    
    fn bitor_values(&self, a: Value, b: Value) -> Result<Value> {
        match (a, b) {
            (Value::U32(a), Value::U32(b)) => Ok(Value::U32(a | b)),
            (Value::U64(a), Value::U64(b)) => Ok(Value::U64(a | b)),
            _ => Err(VmError::new(
                VmErrorKind::TypeMismatch,
                "Bitwise OR requires integer values".to_string(),
            ).into()),
        }
    }
    
    fn bitxor_values(&self, a: Value, b: Value) -> Result<Value> {
        match (a, b) {
            (Value::U32(a), Value::U32(b)) => Ok(Value::U32(a ^ b)),
            (Value::U64(a), Value::U64(b)) => Ok(Value::U64(a ^ b)),
            _ => Err(VmError::new(
                VmErrorKind::TypeMismatch,
                "Bitwise XOR requires integer values".to_string(),
            ).into()),
        }
    }
    
    fn bitnot_value(&self, a: Value) -> Result<Value> {
        match a {
            Value::U32(a) => Ok(Value::U32(!a)),
            Value::U64(a) => Ok(Value::U64(!a)),
            _ => Err(VmError::new(
                VmErrorKind::TypeMismatch,
                "Bitwise NOT requires integer value".to_string(),
            ).into()),
        }
    }
    
    fn shl_values(&self, a: Value, b: Value) -> Result<Value> {
        match (a, b) {
            (Value::U32(a), Value::U32(b)) => Ok(Value::U32(a << b)),
            (Value::U64(a), Value::U64(b)) => Ok(Value::U64(a << b)),
            _ => Err(VmError::new(
                VmErrorKind::TypeMismatch,
                "Left shift requires integer values".to_string(),
            ).into()),
        }
    }
    
    fn shr_values(&self, a: Value, b: Value) -> Result<Value> {
        match (a, b) {
            (Value::U32(a), Value::U32(b)) => Ok(Value::U32(a >> b)),
            (Value::U64(a), Value::U64(b)) => Ok(Value::U64(a >> b)),
            _ => Err(VmError::new(
                VmErrorKind::TypeMismatch,
                "Right shift requires integer values".to_string(),
            ).into()),
        }
    }
    
    // Machine Learning helper methods
    
    /// Create a new ML model
    pub fn ml_create_model(&mut self, model_type: &str) -> Result<()> {
        let model_id = self.next_model_id;
        self.next_model_id += 1;
        
        let model = match model_type {
            "neural_network" => {
                use crate::codegen::Value;
                Value::MLModel {
                    model_id,
                    model_type: model_type.to_string(),
                    weights: vec![0.1, 0.2, 0.3], // Default weights
                    biases: vec![0.0, 0.0],
                    hyperparams: std::collections::HashMap::new(),
                    architecture: vec![3, 2, 1], // Default architecture
                    version: 1,
                    checksum: "default_checksum".to_string(),
                }
            }
            "linear_regression" => {
                use crate::codegen::Value;
                Value::MLModel {
                    model_id,
                    model_type: model_type.to_string(),
                    weights: vec![1.0],
                    biases: vec![0.0],
                    hyperparams: std::collections::HashMap::new(),
                    architecture: vec![1, 1], // Default architecture
                    version: 1,
                    checksum: "linear_checksum".to_string(),
                }
            }
            _ => {
                return Err(VmError::new(
                    VmErrorKind::InvalidOperation,
                    format!("Unsupported model type: {}", model_type),
                ).into());
            }
        };
        
        self.ml_models.insert(model_id, model);
        self.push(crate::codegen::Value::U32(model_id))?;
        Ok(())
    }
    
    /// Load an ML model from storage
    fn ml_load_model(&mut self, model_id: u32) -> Result<()> {
        if let Some(model) = self.ml_models.get(&model_id) {
            self.push(model.clone())?;
        } else {
            return Err(VmError::new(
                VmErrorKind::InvalidOperation,
                format!("Model {} not found", model_id),
            ).into());
        }
        Ok(())
    }
    
    /// Save an ML model to storage
    fn ml_save_model(&mut self, model_id: u32) -> Result<()> {
        let model = self.pop()?;
        self.ml_models.insert(model_id, model);
        Ok(())
    }
    
    /// Train an ML model
    fn ml_train_model(&mut self, model_id: u32) -> Result<()> {
        let dataset = self.pop()?; // Training dataset
        
        if let Some(model) = self.ml_models.get_mut(&model_id) {
            match (model, &dataset) {
                (crate::codegen::Value::MLModel { weights, .. }, 
                 crate::codegen::Value::MLDataset { features, targets, .. }) => {
                    // Simple training simulation - update weights based on data
                    for (i, weight) in weights.iter_mut().enumerate() {
                        if i < features.len() && !features[i].is_empty() {
                            *weight += 0.01 * (targets.get(i).unwrap_or(&0.0) - features[i][0]);
                        }
                    }
                }
                _ => {
                    return Err(VmError::new(
                        VmErrorKind::TypeMismatch,
                        "Invalid model or dataset type for training".to_string(),
                    ).into());
                }
            }
        } else {
            return Err(VmError::new(
                VmErrorKind::InvalidOperation,
                format!("Model {} not found", model_id),
            ).into());
        }
        
        Ok(())
    }
    
    /// Make a prediction with an ML model
    fn ml_predict(&mut self, model_id: u32) -> Result<()> {
        let input = self.pop()?; // Input data
        
        if let Some(model) = self.ml_models.get(&model_id) {
            match (model, &input) {
                (crate::codegen::Value::MLModel { weights, biases, .. }, 
                 crate::codegen::Value::Vector(input_vec)) => {
                    // Simple linear prediction
                    let mut prediction = *biases.get(0).unwrap_or(&0.0);
                    for (i, &input_val) in input_vec.iter().enumerate() {
                        if let Some(&weight) = weights.get(i) {
                            prediction += weight * input_val;
                        }
                    }
                    self.push(crate::codegen::Value::Vector(vec![prediction]))?;
                }
                _ => {
                    return Err(VmError::new(
                        VmErrorKind::TypeMismatch,
                        "Invalid model or input type for prediction".to_string(),
                    ).into());
                }
            }
        } else {
            return Err(VmError::new(
                VmErrorKind::InvalidOperation,
                format!("Model {} not found", model_id),
            ).into());
        }
        
        Ok(())
    }
    
    /// Set hyperparameters for an ML model
    fn ml_set_hyperparams(&mut self, model_id: u32) -> Result<()> {
        let hyperparams = self.pop()?; // Hyperparameters
        
        if let Some(model) = self.ml_models.get_mut(&model_id) {
            if let crate::codegen::Value::MLModel { hyperparams: model_hyperparams, .. } = model {
                // Update hyperparameters (simplified)
                model_hyperparams.insert("learning_rate".to_string(), 0.01);
            }
        } else {
            return Err(VmError::new(
                VmErrorKind::InvalidOperation,
                format!("Model {} not found", model_id),
            ).into());
        }
        
        Ok(())
    }
    
    /// Get metrics for an ML model
    fn ml_get_metrics(&mut self, model_id: u32) -> Result<()> {
        if self.ml_models.contains_key(&model_id) {
            let metrics = crate::codegen::Value::MLMetrics {
                accuracy: 0.85,
                loss: 0.15,
                precision: 0.82,
                recall: 0.88,
                f1_score: 0.85,
                auc_roc: 0.90,
                confusion_matrix: vec![vec![10, 2], vec![1, 15]],
            };
            self.push(metrics)?;
        } else {
            return Err(VmError::new(
                VmErrorKind::InvalidOperation,
                format!("Model {} not found", model_id),
            ).into());
        }
        
        Ok(())
    }
    
    /// Perform forward pass for neural network
    fn ml_forward_pass(&mut self, model_id: u32) -> Result<()> {
        let input = self.pop()?;
        
        if let Some(model) = self.ml_models.get(&model_id) {
            match (model, &input) {
                (crate::codegen::Value::MLModel { weights, biases, .. }, 
                 crate::codegen::Value::Vector(input_vec)) => {
                    // Simple forward pass simulation
                    let mut output = Vec::new();
                    for (i, &bias) in biases.iter().enumerate() {
                        let mut neuron_output = bias;
                        for (j, &input_val) in input_vec.iter().enumerate() {
                            if let Some(&weight) = weights.get(i * input_vec.len() + j) {
                                neuron_output += weight * input_val;
                            }
                        }
                        // Apply ReLU activation
                        output.push(neuron_output.max(0.0));
                    }
                    self.push(crate::codegen::Value::Vector(output))?;
                }
                _ => {
                    return Err(VmError::new(
                        VmErrorKind::TypeMismatch,
                        "Invalid model or input type for forward pass".to_string(),
                    ).into());
                }
            }
        } else {
            return Err(VmError::new(
                VmErrorKind::InvalidOperation,
                format!("Model {} not found", model_id),
            ).into());
        }
        
        Ok(())
    }
    
    /// Perform backward pass for neural network
    fn ml_backward_pass(&mut self, model_id: u32) -> Result<()> {
        let gradients = self.pop()?; // Gradients from loss
        
        if let Some(model) = self.ml_models.get_mut(&model_id) {
            if let crate::codegen::Value::MLModel { weights, .. } = model {
                // Simple gradient update simulation
                for weight in weights.iter_mut() {
                    *weight -= 0.01 * 0.1; // learning_rate * gradient (simplified)
                }
            }
        } else {
            return Err(VmError::new(
                VmErrorKind::InvalidOperation,
                format!("Model {} not found", model_id),
            ).into());
        }
        
        Ok(())
    }
    
    /// Update model weights
    fn ml_update_weights(&mut self, model_id: u32) -> Result<()> {
        let new_weights = self.pop()?;
        
        if let Some(model) = self.ml_models.get_mut(&model_id) {
            if let (crate::codegen::Value::MLModel { weights, .. }, 
                    crate::codegen::Value::Vector(new_weight_vec)) = (model, &new_weights) {
                *weights = new_weight_vec.clone();
            }
        } else {
            return Err(VmError::new(
                VmErrorKind::InvalidOperation,
                format!("Model {} not found", model_id),
            ).into());
        }
        
        Ok(())
    }
    
    /// Normalize data
    fn ml_normalize_data(&mut self) -> Result<()> {
        let data = self.pop()?;
        
        match data {
            crate::codegen::Value::Vector(mut vec) => {
                // Min-max normalization
                if !vec.is_empty() {
                    let min_val = vec.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let max_val = vec.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let range = max_val - min_val;
                    
                    if range > 0.0 {
                        for val in vec.iter_mut() {
                            *val = (*val - min_val) / range;
                        }
                    }
                }
                self.push(crate::codegen::Value::Vector(vec))?;
            }
            _ => {
                return Err(VmError::new(
                    VmErrorKind::TypeMismatch,
                    "Can only normalize vector data".to_string(),
                ).into());
            }
        }
        
        Ok(())
    }
    
    /// Denormalize data
    fn ml_denormalize_data(&mut self) -> Result<()> {
        let normalized_data = self.pop()?;
        let original_stats = self.pop()?; // Min and max values
        
        // Implementation would reverse the normalization process
        self.push(normalized_data)?; // Simplified for now
        Ok(())
    }
    
    /// Apply activation function
    fn ml_apply_activation(&mut self, activation_type: &str) -> Result<()> {
        let input = self.pop()?;
        
        match input {
            crate::codegen::Value::Vector(vec) => {
                let output: Vec<f64> = vec.iter().map(|&x| {
                    match activation_type {
                        "relu" => x.max(0.0),
                        "sigmoid" => 1.0 / (1.0 + (-x).exp()),
                        "tanh" => x.tanh(),
                        _ => x, // Linear activation
                    }
                }).collect();
                
                self.push(crate::codegen::Value::Vector(output))?;
            }
            _ => {
                return Err(VmError::new(
                    VmErrorKind::TypeMismatch,
                    "Activation function requires vector input".to_string(),
                ).into());
            }
        }
        
        Ok(())
    }
    
    /// Calculate loss
    fn ml_calculate_loss(&mut self, loss_type: &str) -> Result<()> {
        let predictions = self.pop()?;
        let targets = self.pop()?;
        
        match (predictions, targets) {
            (crate::codegen::Value::Vector(pred), crate::codegen::Value::Vector(targ)) => {
                let loss = match loss_type {
                    "mse" => {
                        // Mean Squared Error
                        let sum: f64 = pred.iter().zip(targ.iter())
                            .map(|(p, t)| (p - t).powi(2))
                            .sum();
                        sum / pred.len() as f64
                    }
                    "cross_entropy" => {
                        // Cross Entropy (simplified)
                        let sum: f64 = pred.iter().zip(targ.iter())
                            .map(|(p, t)| -t * p.ln())
                            .sum();
                        sum / pred.len() as f64
                    }
                    _ => 0.0, // Default
                };
                
                self.push(crate::codegen::Value::Vector(vec![loss]))?;
            }
            _ => {
                return Err(VmError::new(
                    VmErrorKind::TypeMismatch,
                    "Loss calculation requires vector inputs".to_string(),
                ).into());
            }
        }
        
        Ok(())
    }
    
    /// Apply optimizer
    fn ml_apply_optimizer(&mut self, optimizer_type: &str) -> Result<()> {
        let gradients = self.pop()?;
        let learning_rate = 0.01; // Default learning rate
        
        match gradients {
            crate::codegen::Value::Vector(grad) => {
                let updates: Vec<f64> = grad.iter().map(|&g| {
                    match optimizer_type {
                        "sgd" => -learning_rate * g,
                        "adam" => -learning_rate * g, // Simplified Adam
                        "rmsprop" => -learning_rate * g, // Simplified RMSprop
                        _ => -learning_rate * g, // Default SGD
                    }
                }).collect();
                
                self.push(crate::codegen::Value::Vector(updates))?;
            }
            _ => {
                return Err(VmError::new(
                    VmErrorKind::TypeMismatch,
                    "Optimizer requires vector gradients".to_string(),
                ).into());
            }
        }
        
        Ok(())
    }

    // Additional ML method implementations for new instructions
    
    /// Create tensor from vector data
    pub fn ml_create_tensor(&mut self) -> Result<()> {
        let shape = self.pop()?;
        let data = self.pop()?;
        
        match (data, shape) {
            (crate::codegen::Value::Vector(vec), crate::codegen::Value::Vector(shape_vec)) => {
                let tensor = crate::codegen::Value::Tensor {
                    data: vec,
                    shape: shape_vec.iter().map(|&x| x as usize).collect(),
                    dtype: "f64".to_string(),
                };
                self.push(tensor)?;
            }
            _ => {
                return Err(VmError::new(
                    VmErrorKind::TypeMismatch,
                    "Invalid data or shape for tensor creation".to_string(),
                ).into());
            }
        }
        Ok(())
    }
    
    /// Perform tensor operations
    pub fn ml_tensor_op(&mut self, op: &str) -> Result<()> {
        let b = self.pop()?;
        let a = self.pop()?;
        
        match (a, b) {
            (crate::codegen::Value::Tensor { data: data_a, shape: shape_a, .. },
             crate::codegen::Value::Tensor { data: data_b, shape: shape_b, .. }) => {
                
                let result_data = match op {
                    "add" => {
                        if data_a.len() == data_b.len() {
                            data_a.iter().zip(data_b.iter()).map(|(a, b)| a + b).collect()
                        } else {
                            return Err(VmError::new(
                                VmErrorKind::InvalidOperation,
                                "Tensor dimensions mismatch for addition".to_string(),
                            ).into());
                        }
                    }
                    "mul" => {
                        if data_a.len() == data_b.len() {
                            data_a.iter().zip(data_b.iter()).map(|(a, b)| a * b).collect()
                        } else {
                            return Err(VmError::new(
                                VmErrorKind::InvalidOperation,
                                "Tensor dimensions mismatch for multiplication".to_string(),
                            ).into());
                        }
                    }
                    "matmul" => {
                        // Simplified matrix multiplication for 2D tensors
                        if shape_a.len() == 2 && shape_b.len() == 2 && shape_a[1] == shape_b[0] {
                            let mut result = vec![0.0; shape_a[0] * shape_b[1]];
                            for i in 0..shape_a[0] {
                                for j in 0..shape_b[1] {
                                    for k in 0..shape_a[1] {
                                        result[i * shape_b[1] + j] += 
                                            data_a[i * shape_a[1] + k] * data_b[k * shape_b[1] + j];
                                    }
                                }
                            }
                            result
                        } else {
                            return Err(VmError::new(
                                VmErrorKind::InvalidOperation,
                                "Invalid tensor shapes for matrix multiplication".to_string(),
                            ).into());
                        }
                    }
                    _ => data_a, // Default to first tensor
                };
                
                let result_shape = match op {
                    "matmul" if shape_a.len() == 2 && shape_b.len() == 2 => 
                        vec![shape_a[0], shape_b[1]],
                    _ => shape_a,
                };
                
                self.push(crate::codegen::Value::Tensor {
                    data: result_data,
                    shape: result_shape,
                    dtype: "f64".to_string(),
                })?;
            }
            _ => {
                return Err(VmError::new(
                    VmErrorKind::TypeMismatch,
                    "Tensor operation requires tensor inputs".to_string(),
                ).into());
            }
        }
        Ok(())
    }
    
    /// Perform convolution operation
    fn ml_conv2d(&mut self) -> Result<()> {
        let kernel = self.pop()?;
        let input = self.pop()?;
        
        // Simplified 2D convolution implementation
        match (input, kernel) {
            (crate::codegen::Value::Tensor { data: input_data, shape: input_shape, .. },
             crate::codegen::Value::Tensor { data: kernel_data, shape: kernel_shape, .. }) => {
                
                if input_shape.len() >= 2 && kernel_shape.len() >= 2 {
                    // Simplified convolution - just return input for now
                    self.push(crate::codegen::Value::Tensor {
                        data: input_data,
                        shape: input_shape,
                        dtype: "f64".to_string(),
                    })?;
                } else {
                    return Err(VmError::new(
                        VmErrorKind::InvalidOperation,
                        "Conv2D requires 2D+ tensors".to_string(),
                    ).into());
                }
            }
            _ => {
                return Err(VmError::new(
                    VmErrorKind::TypeMismatch,
                    "Conv2D requires tensor inputs".to_string(),
                ).into());
            }
        }
        Ok(())
    }
    
    /// Perform attention mechanism
    fn ml_attention(&mut self) -> Result<()> {
        let values = self.pop()?;
        let keys = self.pop()?;
        let queries = self.pop()?;
        
        // Simplified attention mechanism
        match (queries, keys, values) {
            (crate::codegen::Value::Tensor { data: q_data, shape: q_shape, .. },
             crate::codegen::Value::Tensor { data: k_data, shape: k_shape, .. },
             crate::codegen::Value::Tensor { data: v_data, shape: v_shape, .. }) => {
                
                // Simplified attention - return values tensor
                self.push(crate::codegen::Value::Tensor {
                    data: v_data,
                    shape: v_shape,
                    dtype: "f64".to_string(),
                })?;
            }
            _ => {
                return Err(VmError::new(
                    VmErrorKind::TypeMismatch,
                    "Attention requires tensor inputs".to_string(),
                ).into());
            }
        }
        Ok(())
    }
    
    /// Clone a model
    fn ml_clone_model(&mut self, model_id: u32) -> Result<()> {
        if let Some(model) = self.ml_models.get(&model_id).cloned() {
            let new_id = self.next_model_id;
            self.next_model_id += 1;
            self.ml_models.insert(new_id, model);
            self.push(crate::codegen::Value::U32(new_id))?;
        } else {
            return Err(VmError::new(
                VmErrorKind::InvalidOperation,
                format!("Model {} not found", model_id),
            ).into());
        }
        Ok(())
    }
    
    /// Quantize a model
    fn ml_quantize_model(&mut self, model_id: u32) -> Result<()> {
        if let Some(model) = self.ml_models.get_mut(&model_id) {
            if let crate::codegen::Value::MLModel { weights, .. } = model {
                // Simple quantization - round to nearest 0.1
                for weight in weights.iter_mut() {
                    *weight = (*weight * 10.0).round() / 10.0;
                }
            }
        } else {
            return Err(VmError::new(
                VmErrorKind::InvalidOperation,
                format!("Model {} not found", model_id),
            ).into());
        }
        Ok(())
    }
    
    /// Set learning rate
    fn ml_set_learning_rate(&mut self, rate: f64) -> Result<()> {
        // Store learning rate in context or model state
        self.push(crate::codegen::Value::Vector(vec![rate]))?;
        Ok(())
    }
    
    /// Implement early stopping
    fn ml_early_stopping(&mut self) -> Result<()> {
        let current_loss = self.pop()?;
        let best_loss = self.pop()?;
        
        match (current_loss, best_loss) {
            (crate::codegen::Value::Vector(curr), crate::codegen::Value::Vector(best)) => {
                let should_stop = if !curr.is_empty() && !best.is_empty() {
                    curr[0] > best[0] // Stop if current loss is worse
                } else {
                    false
                };
                self.push(crate::codegen::Value::Bool(should_stop))?;
            }
            _ => {
                self.push(crate::codegen::Value::Bool(false))?;
            }
        }
        Ok(())
    }
    
    /// Load dataset
    fn ml_load_dataset(&mut self, dataset_id: &str) -> Result<()> {
        // Simulate loading a dataset
        let dataset = crate::codegen::Value::MLDataset {
            features: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            targets: vec![0.0, 1.0],
            normalized: false,
            metadata: std::collections::HashMap::new(),
            split_info: None,
        };
        self.push(dataset)?;
        Ok(())
    }
    
    /// Augment data
    fn ml_augment_data(&mut self) -> Result<()> {
        let data = self.pop()?;
        
        match data {
            crate::codegen::Value::MLDataset { mut features, targets, .. } => {
                // Simple data augmentation - add noise
                for sample in features.iter_mut() {
                    for value in sample.iter_mut() {
                        #[cfg(feature = "crypto")]
                        {
                            *value += (rand::random::<f64>() - 0.5) * 0.1; // Add small noise
                        }
                        #[cfg(not(feature = "crypto"))]
                        {
                            *value += 0.05; // Fixed small noise for WASM
                        }
                    }
                }
                
                self.push(crate::codegen::Value::MLDataset {
                    features,
                    targets,
                    normalized: false,
                    metadata: std::collections::HashMap::new(),
                    split_info: None,
                })?;
            }
            _ => {
                return Err(VmError::new(
                    VmErrorKind::TypeMismatch,
                    "Data augmentation requires dataset input".to_string(),
                ).into());
            }
        }
        Ok(())
    }
    
    /// Evaluate model
    fn ml_evaluate(&mut self, model_id: u32) -> Result<()> {
        let test_data = self.pop()?;
        
        if let Some(_model) = self.ml_models.get(&model_id) {
            // Simulate model evaluation
            let metrics = crate::codegen::Value::MLMetrics {
                accuracy: 0.85,
                loss: 0.15,
                precision: 0.82,
                recall: 0.88,
                f1_score: 0.85,
                auc_roc: 0.90,
                confusion_matrix: vec![vec![50, 5], vec![10, 35]],
            };
            self.push(metrics)?;
        } else {
            return Err(VmError::new(
                VmErrorKind::InvalidOperation,
                format!("Model {} not found", model_id),
            ).into());
        }
        Ok(())
    }
    
    /// Calculate confusion matrix
    fn ml_confusion_matrix(&mut self) -> Result<()> {
        let actual = self.pop()?;
        let predicted = self.pop()?;
        
        match (predicted, actual) {
            (crate::codegen::Value::Vector(pred), crate::codegen::Value::Vector(act)) => {
                // Simplified confusion matrix for binary classification
                let mut matrix = vec![vec![0, 0], vec![0, 0]];
                
                for (p, a) in pred.iter().zip(act.iter()) {
                    let pred_class = if *p > 0.5 { 1 } else { 0 };
                    let actual_class = if *a > 0.5 { 1 } else { 0 };
                    matrix[actual_class][pred_class] += 1;
                }
                
                self.push(crate::codegen::Value::Matrix(matrix.into_iter().map(|row| {
                    row.into_iter().map(|x| x as f64).collect()
                }).collect()))?;
            }
            _ => {
                return Err(VmError::new(
                    VmErrorKind::TypeMismatch,
                    "Confusion matrix requires vector inputs".to_string(),
                ).into());
            }
        }
        Ok(())
    }
    
    /// Export model
    fn ml_export_model(&mut self, model_id: u32) -> Result<()> {
        if let Some(model) = self.ml_models.get(&model_id) {
            // Simulate model export
            self.push(crate::codegen::Value::String(format!("exported_model_{}", model_id)))?;
        } else {
            return Err(VmError::new(
                VmErrorKind::InvalidOperation,
                format!("Model {} not found", model_id),
            ).into());
        }
        Ok(())
    }
    
    /// Sync model across chains
    fn ml_sync_model(&mut self, model_id: u32) -> Result<()> {
        if let Some(_model) = self.ml_models.get(&model_id) {
            // Simulate cross-chain model synchronization
            self.push(crate::codegen::Value::Bool(true))?; // Success
        } else {
            return Err(VmError::new(
                VmErrorKind::InvalidOperation,
                format!("Model {} not found", model_id),
            ).into());
        }
        Ok(())
    }

    // Missing method implementations
    #[cfg(any(feature = "ml-basic", feature = "ml-deep"))]
    fn ml_reshape(&mut self, new_shape: &Vec<usize>) -> Result<()> {
        let tensor_value = self.pop()?;
        
        match tensor_value {
            crate::codegen::Value::Tensor { data, shape: _, dtype } => {
                // Create tensor from data and reshape
                let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                #[cfg(any(feature = "ml-basic", feature = "ml-deep"))]
                let tensor = crate::stdlib::ml::tensor::Tensor::from_data(data_f32, new_shape.clone())
                    .map_err(|e| VmError::new(VmErrorKind::InvalidOperation, e.to_string()))?;
                
                #[cfg(not(any(feature = "ml-basic", feature = "ml-deep")))]
                return Err(VmError::new(
                    VmErrorKind::InvalidOperation,
                    "ML operations not available - compile with ml-basic or ml-deep feature".to_string(),
                ).into());
                
                #[cfg(any(feature = "ml-basic", feature = "ml-deep"))]
                let reshaped_data: Vec<f64> = tensor.to_vec().iter().map(|&x| x as f64).collect();
                self.push(crate::codegen::Value::Tensor {
                    data: reshaped_data,
                    shape: new_shape.clone(),
                    dtype,
                })?;
            }
            _ => {
                return Err(VmError::new(
                    VmErrorKind::TypeMismatch,
                    "Reshape requires tensor input".to_string(),
                ).into());
            }
        }
        Ok(())
    }

    #[cfg(not(any(feature = "ml-basic", feature = "ml-deep")))]
    fn ml_reshape(&mut self, _new_shape: &Vec<usize>) -> Result<()> {
        Err(VmError::new(
            VmErrorKind::InvalidOperation,
            "ML operations not available - compile with ml-basic or ml-deep feature".to_string(),
        ).into())
    }

    #[cfg(any(feature = "ml-basic", feature = "ml-deep"))]
    fn ml_slice(&mut self, ranges: &Vec<(usize, usize)>) -> Result<()> {
        let tensor_value = self.pop()?;
        
        match tensor_value {
            crate::codegen::Value::Tensor { data, shape, dtype } => {
                // For simplicity, implement basic 1D slicing
                if ranges.len() != 1 || shape.len() != 1 {
                    return Err(VmError::new(
                        VmErrorKind::InvalidOperation,
                        "Currently only 1D tensor slicing is supported".to_string(),
                    ).into());
                }
                
                let (start, end) = ranges[0];
                if start >= data.len() || end > data.len() || start >= end {
                    return Err(VmError::new(
                        VmErrorKind::InvalidOperation,
                        "Invalid slice range".to_string(),
                    ).into());
                }
                
                let sliced_data = data[start..end].to_vec();
                let new_shape = vec![end - start];
                
                self.push(crate::codegen::Value::Tensor {
                    data: sliced_data,
                    shape: new_shape,
                    dtype,
                })?;
            }
            _ => {
                return Err(VmError::new(
                    VmErrorKind::TypeMismatch,
                    "Slice requires tensor input".to_string(),
                ).into());
            }
        }
        Ok(())
    }

    #[cfg(not(any(feature = "ml-basic", feature = "ml-deep")))]
    fn ml_slice(&mut self, _ranges: &Vec<(usize, usize)>) -> Result<()> {
        Err(VmError::new(
            VmErrorKind::InvalidOperation,
            "ML operations not available - compile with ml-basic or ml-deep feature".to_string(),
        ).into())
    }

    #[cfg(any(feature = "ml-basic", feature = "ml-deep"))]
    fn ml_concat(&mut self, axis: usize) -> Result<()> {
        let tensor2_value = self.pop()?;
        let tensor1_value = self.pop()?;
        
        match (tensor1_value, tensor2_value) {
            (crate::codegen::Value::Tensor { data: data1, shape: shape1, dtype: dtype1 },
             crate::codegen::Value::Tensor { data: data2, shape: shape2, dtype: dtype2 }) => {
                
                if dtype1 != dtype2 {
                    return Err(VmError::new(
                        VmErrorKind::TypeMismatch,
                        "Cannot concatenate tensors with different data types".to_string(),
                    ).into());
                }
                
                // For simplicity, only support 1D concatenation along axis 0
                if axis != 0 || shape1.len() != 1 || shape2.len() != 1 {
                    return Err(VmError::new(
                        VmErrorKind::InvalidOperation,
                        "Currently only 1D concatenation along axis 0 is supported".to_string(),
                    ).into());
                }
                
                let mut concatenated_data = data1;
                concatenated_data.extend(data2);
                let new_shape = vec![shape1[0] + shape2[0]];
                
                self.push(crate::codegen::Value::Tensor {
                    data: concatenated_data,
                    shape: new_shape,
                    dtype: dtype1,
                })?;
            }
            _ => {
                return Err(VmError::new(
                    VmErrorKind::TypeMismatch,
                    "Concatenation requires tensor inputs".to_string(),
                ).into());
            }
        }
        Ok(())
    }

    #[cfg(not(any(feature = "ml-basic", feature = "ml-deep")))]
    fn ml_concat(&mut self, _axis: usize) -> Result<()> {
        Err(VmError::new(
            VmErrorKind::InvalidOperation,
            "ML operations not available - compile with ml-basic or ml-deep feature".to_string(),
        ).into())
    }

    fn ml_split(&mut self, _axis: usize, _sections: usize) -> Result<()> {
        let tensor = self.pop()?;
        // Simplified split - just push back the tensor
        self.push(tensor.clone())?;
        self.push(tensor)?;
        Ok(())
    }

    fn ml_reduce(&mut self, op_type: &str, axis: Option<usize>) -> Result<()> {
        let tensor_value = self.pop()?;
        
        match tensor_value {
            crate::codegen::Value::Tensor { data, shape, dtype } => {
                #[cfg(not(any(feature = "ml-basic", feature = "ml-deep")))]
                return Err(VmError::new(
                    VmErrorKind::InvalidOperation,
                    "ML operations not available - compile with ml-basic or ml-deep feature".to_string(),
                ).into());
                
                #[cfg(any(feature = "ml-basic", feature = "ml-deep"))]
                {
                    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                    let tensor = crate::stdlib::ml::tensor::Tensor::from_data(data_f32, shape)
                        .map_err(|e| VmError::new(VmErrorKind::InvalidOperation, e.to_string()))?;
                
                    let result = match op_type {
                    "sum" => tensor.sum(axis, false)
                        .map_err(|e| VmError::new(VmErrorKind::InvalidOperation, e.to_string()))?,
                    "mean" => tensor.mean(axis, false)
                        .map_err(|e| VmError::new(VmErrorKind::InvalidOperation, e.to_string()))?,
                    _ => {
                        return Err(VmError::new(
                            VmErrorKind::InvalidOperation,
                            format!("Unsupported reduction operation: {}", op_type),
                        ).into());
                    }
                };
                
                    let result_data: Vec<f64> = result.to_vec().iter().map(|&x| x as f64).collect();
                    let result_shape = result.shape().dims.clone();
                    
                    self.push(crate::codegen::Value::Tensor {
                        data: result_data,
                        shape: result_shape,
                        dtype,
                    })?;
                }
            }
            _ => {
                return Err(VmError::new(
                    VmErrorKind::TypeMismatch,
                    "Reduce requires tensor input".to_string(),
                ).into());
            }
        }
        Ok(())
    }

    fn ml_broadcast(&mut self, _target_shape: &Vec<usize>) -> Result<()> {
        let tensor = self.pop()?;
        // Simplified broadcast - just push back the tensor
        self.push(tensor)?;
        Ok(())
    }

    #[cfg(any(feature = "ml-basic", feature = "ml-deep"))]
    fn ml_dropout(&mut self) -> Result<()> {
        let prob_value = self.pop()?;
        let tensor_value = self.pop()?;
        
        let dropout_prob = match prob_value {
            crate::codegen::Value::Vector(ref v) if v.len() == 1 => v[0] as f32,
            crate::codegen::Value::F64(p) => p as f32,
            _ => 0.5, // Default dropout probability
        };
        
        match tensor_value {
            crate::codegen::Value::Tensor { data, shape, dtype } => {
                let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                let tensor = crate::stdlib::ml::tensor::Tensor::from_data(data_f32, shape)
                    .map_err(|e| VmError::new(VmErrorKind::InvalidOperation, e.to_string()))?;
                
                let dropout = crate::stdlib::ml::deep_learning::Dropout::new(dropout_prob);
                let result = dropout.forward(&tensor)
                    .map_err(|e| VmError::new(VmErrorKind::InvalidOperation, e.to_string()))?;
                
                let result_data: Vec<f64> = result.to_vec().iter().map(|&x| x as f64).collect();
                let result_shape = result.shape().dims.clone();
                
                self.push(crate::codegen::Value::Tensor {
                    data: result_data,
                    shape: result_shape,
                    dtype,
                })?;
            }
            _ => {
                return Err(VmError::new(
                    VmErrorKind::TypeMismatch,
                    "Dropout requires tensor input".to_string(),
                ).into());
            }
        }
        Ok(())
    }

    #[cfg(not(any(feature = "ml-basic", feature = "ml-deep")))]
    fn ml_dropout(&mut self) -> Result<()> {
        Err(VmError::new(
            VmErrorKind::InvalidOperation,
            "ML operations not available - compile with ml-basic or ml-deep feature".to_string(),
        ).into())
    }

    fn ml_embedding(&mut self) -> Result<()> {
        let input = self.pop()?;
        // Simplified embedding - return a tensor
        self.push(crate::codegen::Value::Tensor {
            data: vec![1.0, 2.0, 3.0],
            shape: vec![1, 3],
            dtype: "f64".to_string(),
        })?;
        Ok(())
    }

    fn ml_maxpool2d(&mut self) -> Result<()> {
        let tensor = self.pop()?;
        // Simplified maxpool - just push back the tensor
        self.push(tensor)?;
        Ok(())
    }

    #[cfg(any(feature = "ml-basic", feature = "ml-deep"))]
    fn ml_batch_norm(&mut self) -> Result<()> {
        let tensor_value = self.pop()?;
        
        match tensor_value {
            crate::codegen::Value::Tensor { data, shape, dtype } => {
                let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                let tensor = crate::stdlib::ml::tensor::Tensor::from_data(data_f32, shape.clone())
                    .map_err(|e| VmError::new(VmErrorKind::InvalidOperation, e.to_string()))?;
                
                // Create batch normalization layer
                let num_features = if shape.len() > 1 { shape[1] } else { shape[0] };
                let mut batch_norm = crate::stdlib::ml::deep_learning::BatchNorm::new(num_features, 1e-5, 0.1)
                    .map_err(|e| VmError::new(VmErrorKind::InvalidOperation, e.to_string()))?;
                
                let result = batch_norm.forward(&tensor)
                    .map_err(|e| VmError::new(VmErrorKind::InvalidOperation, e.to_string()))?;
                
                let result_data: Vec<f64> = result.to_vec().iter().map(|&x| x as f64).collect();
                let result_shape = result.shape().dims.clone();
                
                self.push(crate::codegen::Value::Tensor {
                    data: result_data,
                    shape: result_shape,
                    dtype,
                })?;
            }
            _ => {
                return Err(VmError::new(
                    VmErrorKind::TypeMismatch,
                    "Batch normalization requires tensor input".to_string(),
                ).into());
            }
        }
        Ok(())
    }

    #[cfg(not(any(feature = "ml-basic", feature = "ml-deep")))]
    fn ml_batch_norm(&mut self) -> Result<()> {
        Err(VmError::new(
            VmErrorKind::InvalidOperation,
            "ML operations not available - compile with ml-basic or ml-deep feature".to_string(),
        ).into())
    }

    #[cfg(any(feature = "ml-basic", feature = "ml-deep"))]
    fn ml_layer_norm(&mut self) -> Result<()> {
        let tensor_value = self.pop()?;
        
        match tensor_value {
            crate::codegen::Value::Tensor { data, shape, dtype } => {
                let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                let tensor = crate::stdlib::ml::tensor::Tensor::from_data(data_f32, shape.clone())
                    .map_err(|e| VmError::new(VmErrorKind::InvalidOperation, e.to_string()))?;
                
                // Create layer normalization
                let layer_norm = crate::stdlib::ml::deep_learning::LayerNorm::new(shape.clone(), 1e-5)
                    .map_err(|e| VmError::new(VmErrorKind::InvalidOperation, e.to_string()))?;
                
                let result = layer_norm.forward(&tensor)
                    .map_err(|e| VmError::new(VmErrorKind::InvalidOperation, e.to_string()))?;
                
                let result_data: Vec<f64> = result.to_vec().iter().map(|&x| x as f64).collect();
                let result_shape = result.shape().dims.clone();
                
                self.push(crate::codegen::Value::Tensor {
                    data: result_data,
                    shape: result_shape,
                    dtype,
                })?;
            }
            _ => {
                return Err(VmError::new(
                    VmErrorKind::TypeMismatch,
                    "Layer normalization requires tensor input".to_string(),
                ).into());
            }
        }
        Ok(())
    }

    #[cfg(not(any(feature = "ml-basic", feature = "ml-deep")))]
    fn ml_layer_norm(&mut self) -> Result<()> {
        Err(VmError::new(
            VmErrorKind::InvalidOperation,
            "ML operations not available - compile with ml-basic or ml-deep feature".to_string(),
        ).into())
    }

    fn ml_merge_models(&mut self, _model_ids: &Vec<u32>) -> Result<()> {
        // Simplified merge - create a new model
        let new_id = self.next_model_id;
        self.next_model_id += 1;
        let model = crate::codegen::Value::MLModel {
            model_id: new_id,
            model_type: "merged".to_string(),
            weights: vec![1.0, 2.0, 3.0],
            biases: vec![0.0, 0.0],
            hyperparams: std::collections::HashMap::new(),
            architecture: vec![3, 2, 1],
            version: 1,
            checksum: "merged_checksum".to_string(),
        };
        self.ml_models.insert(new_id, model);
        self.push(crate::codegen::Value::U32(new_id))?;
        Ok(())
    }

    fn ml_prune_model(&mut self, model_id: u32, _threshold: f64) -> Result<()> {
        if let Some(_model) = self.ml_models.get_mut(&model_id) {
            // Simplified pruning - just mark as successful
            self.push(crate::codegen::Value::Bool(true))?;
        } else {
            return Err(VmError::new(
                VmErrorKind::InvalidOperation,
                format!("Model {} not found", model_id),
            ).into());
        }
        Ok(())
    }

    fn ml_distill_model(&mut self, _teacher_id: u32, _student_id: u32) -> Result<()> {
        // Simplified distillation - just mark as successful
        self.push(crate::codegen::Value::Bool(true))?;
        Ok(())
    }

    fn ml_schedule_learning_rate(&mut self, _scheduler_type: &str) -> Result<()> {
        // Simplified scheduler - just push a learning rate
        self.push(crate::codegen::Value::Vector(vec![0.001]))?;
        Ok(())
    }

    fn ml_gradient_clip(&mut self, _max_norm: f64) -> Result<()> {
        // Simplified gradient clipping - just mark as successful
        self.push(crate::codegen::Value::Bool(true))?;
        Ok(())
    }

    fn ml_save_checkpoint(&mut self, _checkpoint_name: &str) -> Result<()> {
        // Simplified checkpoint saving - just mark as successful
        self.push(crate::codegen::Value::Bool(true))?;
        Ok(())
    }

    fn ml_restore_checkpoint(&mut self, _checkpoint_name: &str) -> Result<()> {
        // Simplified checkpoint restoration - just mark as successful
        self.push(crate::codegen::Value::Bool(true))?;
        Ok(())
    }

    fn ml_save_dataset(&mut self, _dataset_name: &str) -> Result<()> {
        // Simplified dataset saving - just mark as successful
        self.push(crate::codegen::Value::Bool(true))?;
        Ok(())
    }

    fn ml_split_dataset(&mut self, _train_ratio: f64, _val_ratio: f64) -> Result<()> {
        let dataset = self.pop()?;
        // Simplified split - just push back the dataset twice
        self.push(dataset.clone())?; // validation set
        self.push(dataset)?; // training set
        Ok(())
    }

    fn ml_shuffle_dataset(&mut self) -> Result<()> {
        let dataset = self.pop()?;
        // Simplified shuffle - just push back the dataset
        self.push(dataset)?;
        Ok(())
    }

    fn ml_preprocess_data(&mut self, _preprocessing_type: &str) -> Result<()> {
        let data = self.pop()?;
        // Simplified preprocessing - just push back the data
        self.push(data)?;
        Ok(())
    }

    fn ml_roc_curve(&mut self) -> Result<()> {
        // Simplified ROC curve - return dummy data
        self.push(crate::codegen::Value::Matrix(vec![
            vec![0.0, 0.0], vec![0.5, 0.5], vec![1.0, 1.0]
        ]))?;
        Ok(())
    }

    fn ml_feature_importance(&mut self) -> Result<()> {
        // Simplified feature importance - return dummy data
        self.push(crate::codegen::Value::Vector(vec![0.3, 0.5, 0.2]))?;
        Ok(())
    }

    fn ml_explain_prediction(&mut self, _model_id: u32) -> Result<()> {
        // Simplified explanation - return dummy data
        self.push(crate::codegen::Value::Vector(vec![0.1, 0.3, 0.6]))?;
        Ok(())
    }

    fn ml_import_model(&mut self, _import_format: &str) -> Result<()> {
        // Simplified import - create a new model
        let new_id = self.next_model_id;
        self.next_model_id += 1;
        let model = crate::codegen::Value::MLModel {
            model_id: new_id,
            model_type: "imported".to_string(),
            weights: vec![1.0, 2.0],
            biases: vec![0.0],
            hyperparams: std::collections::HashMap::new(),
            architecture: vec![2, 1],
            version: 1,
            checksum: "imported_checksum".to_string(),
        };
        self.ml_models.insert(new_id, model);
        self.push(crate::codegen::Value::U32(new_id))?;
        Ok(())
    }

    fn ml_verify_model(&mut self, _verification_type: &str) -> Result<()> {
        // Simplified verification - just mark as successful
        self.push(crate::codegen::Value::Bool(true))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::Instruction;
    
    #[test]
    fn test_stack_operations() {
        let mut avm = AVM::new();
        
        // Test push and pop
        avm.push(Value::U32(42)).unwrap();
        avm.push(Value::U32(24)).unwrap();
        
        assert_eq!(avm.pop().unwrap(), Value::U32(24));
        assert_eq!(avm.pop().unwrap(), Value::U32(42));
    }
    
    #[test]
    fn test_arithmetic_operations() {
        let mut avm = AVM::new();
        
        // Test addition
        avm.push(Value::U32(10)).unwrap();
        avm.push(Value::U32(20)).unwrap();
        
        let b = avm.pop().unwrap();
        let a = avm.pop().unwrap();
        let result = avm.add_values(a, b).unwrap();
        
        assert_eq!(result, Value::U32(30));
    }
    
    #[test]
    fn test_comparison_operations() {
        let avm = AVM::new();
        
        assert!(avm.values_equal(&Value::U32(42), &Value::U32(42)));
        assert!(!avm.values_equal(&Value::U32(42), &Value::U32(24)));
        
        assert_eq!(avm.compare_values(&Value::U32(10), &Value::U32(20)).unwrap(), -1);
        assert_eq!(avm.compare_values(&Value::U32(20), &Value::U32(10)).unwrap(), 1);
        assert_eq!(avm.compare_values(&Value::U32(15), &Value::U32(15)).unwrap(), 0);
    }
    
    #[test]
    fn test_simple_bytecode_execution() {
        let mut avm = AVM::new();
        
        let bytecode = Bytecode {
            instructions: vec![
                Instruction::Push(Value::U32(10)),
                Instruction::Push(Value::U32(20)),
                Instruction::Add,
                Instruction::Halt,
            ],
            constants: Vec::new(),
            functions: HashMap::new(),
            contracts: HashMap::new(),
        };
        
        let result = avm.execute(&bytecode).unwrap();
        
        assert!(result.success);
        assert_eq!(result.return_value, Some(Value::U32(30)));
    }
    
    #[test]
    fn test_gas_consumption() {
        let mut avm = AVM::new();
        avm.context.gas_limit = 100;
        
        let bytecode = Bytecode {
            instructions: vec![
                Instruction::Push(Value::U32(1)),
                Instruction::Push(Value::U32(2)),
                Instruction::Add,
                Instruction::Halt,
            ],
            constants: Vec::new(),
            functions: HashMap::new(),
            contracts: HashMap::new(),
        };
        
        let result = avm.execute(&bytecode).unwrap();
        
        assert!(result.success);
        assert!(result.gas_used > 0);
        assert!(result.gas_used <= 100);
    }
}