// Augustium Virtual Machine - executes our compiled bytecode
// Stack-based VM that handles contract execution, state, and security

use crate::codegen::{Bytecode, Instruction, Value, ContractBytecode};
use crate::error::{Result, VmError, VmErrorKind, SourceLocation};
use std::collections::HashMap;

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
        }
    }
}

/// Execution context for the AVM
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub caller: [u8; 20],
    pub origin: [u8; 20],
    pub gas_price: u64,
    pub gas_limit: u64,
    pub value: u64,
    pub block_number: u64,
    pub timestamp: u64,
    pub difficulty: u64,
    pub chain_id: u64,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            caller: [0; 20],
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
    pub address: [u8; 20],
    pub bytecode: ContractBytecode,
    pub storage: HashMap<u32, Value>,
    pub balance: u64,
}

/// Call frame for function calls
#[derive(Debug, Clone)]
struct CallFrame {
    instructions: Vec<Instruction>,
    pc: usize,
    locals: Vec<Value>,
    return_address: Option<usize>,
}

/// Event emitted by a contract
#[derive(Debug, Clone)]
pub struct Event {
    pub contract_address: [u8; 20],
    pub event_name: String,
    pub data: Vec<Value>,
    pub block_number: u64,
    pub transaction_hash: [u8; 32],
}

/// Transaction result
#[derive(Debug, Clone)]
pub struct TransactionResult {
    pub success: bool,
    pub gas_used: u64,
    pub return_value: Option<Value>,
    pub events: Vec<Event>,
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
        }
    }
    
    /// Create a new AVM instance with custom context
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
        }
    }
    
    /// Enable debug mode
    pub fn set_debug_mode(&mut self, debug: bool) {
        self.debug_mode = debug;
    }
    
    /// Deploy a contract
    pub fn deploy_contract(
        &mut self,
        bytecode: ContractBytecode,
        constructor_args: Vec<Value>,
    ) -> Result<[u8; 20]> {
        // Generate contract address (simplified)
        let mut address = [0u8; 20];
        address[0] = (self.contracts.len() + 1) as u8;
        
        // Create contract instance
        let mut contract = ContractInstance {
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
            
            // Execute constructor
            let call_frame = CallFrame {
                instructions: bytecode.constructor.clone(),
                pc: 0,
                locals: Vec::new(),
                return_address: None,
            };
            
            self.call_stack.push(call_frame);
            self.execute_until_return()?;
        }
        
        // Store contract
        self.contracts.insert(address, contract);
        
        Ok(address)
    }
    
    /// Call a contract function
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
                // TODO: Implement contract deployment
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
            
            // Halt execution
            Instruction::Halt => {
                self.consume_gas(self.gas_costs.base)?;
                self.halted = true;
            }
        }
        
        Ok(())
    }
    
    /// Push a value onto the stack
    fn push(&mut self, value: Value) -> Result<()> {
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
    fn load_field(&self, index: u32) -> Result<Value> {
        // TODO: Implement contract field loading
        // For now, return a placeholder value
        Ok(Value::U32(0))
    }
    
    /// Store a contract field
    fn store_field(&mut self, index: u32, value: Value) -> Result<()> {
        // TODO: Implement contract field storage
        Ok(())
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
        // TODO: Implement function calls
        Ok(())
    }
    
    /// Return from a function
    fn return_from_function(&mut self) -> Result<()> {
        self.call_stack.pop();
        Ok(())
    }
    
    /// Invoke a function by name
    fn invoke_function(&mut self, function_name: &str) -> Result<()> {
        // TODO: Implement function invocation by name
        Ok(())
    }
    
    /// Emit an event
    fn emit_event(&mut self, event_name: &str) -> Result<()> {
        // TODO: Collect event data from stack
        let event = Event {
            contract_address: [0; 20], // TODO: Get current contract address
            event_name: event_name.to_string(),
            data: Vec::new(),
            block_number: self.context.block_number,
            transaction_hash: [0; 32], // TODO: Get transaction hash
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
        // TODO: Implement value transfer
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
            Value::String(s) => !s.is_empty(),
            Value::Array(arr) => !arr.is_empty(),
            Value::Tuple(tuple) => !tuple.is_empty(),
            Value::Address(_) => true,
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