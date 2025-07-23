//! EVM Compatibility Layer for Augustium
//!
//! This module provides compatibility with the Ethereum Virtual Machine (EVM),
//! allowing Augustium contracts to interact with Ethereum and other EVM-compatible blockchains.

use crate::ast::*;
use crate::error::{CompilerError, CodegenError, CodegenErrorKind, SourceLocation};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// EVM compatibility configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvmConfig {
    /// Target EVM version (e.g., "london", "berlin", "istanbul")
    pub evm_version: String,
    /// Gas limit for transactions
    pub gas_limit: u64,
    /// Enable EVM precompiles
    pub enable_precompiles: bool,
    /// Chain ID for the target network
    pub chain_id: u64,
}

impl Default for EvmConfig {
    fn default() -> Self {
        Self {
            evm_version: "london".to_string(),
            gas_limit: 30_000_000,
            enable_precompiles: true,
            chain_id: 1, // Ethereum mainnet
        }
    }
}

/// EVM bytecode instruction
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum EvmInstruction {
    // Stack operations
    Push(Vec<u8>),
    Pop,
    Dup(u8),
    Swap(u8),
    
    // Arithmetic operations
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    AddMod,
    MulMod,
    Exp,
    SignExtend,
    
    // Comparison operations
    Lt,
    Gt,
    Slt,
    Sgt,
    Eq,
    IsZero,
    And,
    Or,
    Xor,
    Not,
    Byte,
    Shl,
    Shr,
    Sar,
    
    // Memory operations
    MLoad,
    MStore,
    MStore8,
    SLoad,
    SStore,
    MSize,
    
    // Control flow
    Jump,
    JumpI,
    Pc,
    JumpDest,
    
    // Block information
    BlockHash,
    Coinbase,
    Timestamp,
    Number,
    Difficulty,
    GasLimit,
    ChainId,
    SelfBalance,
    BaseFee,
    
    // Transaction information
    Origin,
    GasPrice,
    CallDataLoad,
    CallDataSize,
    CallDataCopy,
    CodeSize,
    CodeCopy,
    GasLeft,
    
    // Call operations
    Call,
    CallCode,
    DelegateCall,
    StaticCall,
    Create,
    Create2,
    
    // Return operations
    Return,
    Revert,
    Stop,
    Invalid,
    SelfDestruct,
    
    // Logging
    Log0,
    Log1,
    Log2,
    Log3,
    Log4,
}

/// EVM bytecode representation
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct EvmBytecode {
    pub instructions: Vec<EvmInstruction>,
    #[allow(dead_code)]
    pub metadata: EvmMetadata,
}

/// EVM contract metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvmMetadata {
    pub contract_name: String,
    pub compiler_version: String,
    pub source_hash: String,
    pub abi: Vec<EvmAbiEntry>,
    pub storage_layout: HashMap<String, EvmStorageSlot>,
}

/// EVM ABI entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvmAbiEntry {
    pub name: String,
    pub entry_type: EvmAbiType,
    pub inputs: Vec<EvmAbiParam>,
    pub outputs: Vec<EvmAbiParam>,
    pub state_mutability: EvmStateMutability,
}

/// EVM ABI types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvmAbiType {
    Function,
    Constructor,
    Event,
    Error,
    Fallback,
    Receive,
}

/// EVM ABI parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvmAbiParam {
    pub name: String,
    pub param_type: String,
    pub indexed: bool, // For events
}

/// EVM state mutability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvmStateMutability {
    Pure,
    View,
    NonPayable,
    Payable,
}

/// EVM storage slot information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvmStorageSlot {
    pub slot: u64,
    pub offset: u8,
    pub type_name: String,
}

/// EVM compatibility translator
pub struct EvmTranslator {
    function_selectors: HashMap<String, [u8; 4]>,
}

impl EvmTranslator {
    /// Create a new EVM translator
    #[allow(dead_code)]
    #[allow(dead_code)]
    pub fn new(_config: EvmConfig) -> Self {
        Self {
            function_selectors: HashMap::new(),
        }
    }
    
    /// Translate Augustium AST to EVM bytecode
    #[allow(dead_code)]
    pub fn translate_contract(&mut self, contract: &Contract) -> Result<EvmBytecode, CompilerError> {
        let mut instructions = Vec::new();
        let mut abi = Vec::new();
        let mut storage_layout = HashMap::new();
        
        // Note: Constructor handling will be implemented when constructor syntax is added to Contract struct
        
        // Generate function bytecode
        for function in &contract.functions {
            self.translate_function(function, &mut instructions, &mut abi, &mut storage_layout)?;
        }
        
        // Generate field storage layout
        for (index, field) in contract.fields.iter().enumerate() {
            storage_layout.insert(
                field.name.name.clone(),
                EvmStorageSlot {
                    slot: index as u64,
                    offset: 0,
                    type_name: self.augustium_type_to_solidity(&field.type_annotation)?,
                },
            );
        }
        
        let metadata = EvmMetadata {
            contract_name: contract.name.name.clone(),
            compiler_version: "augustium-0.1.0".to_string(),
            source_hash: self.calculate_source_hash(contract)?,
            abi,
            storage_layout,
        };
        
        Ok(EvmBytecode {
            instructions,
            metadata,
        })
    }
    
    /// Translate constructor to EVM bytecode
    #[allow(dead_code)]
    fn translate_constructor(
        &mut self,
        constructor: &Function,
        instructions: &mut Vec<EvmInstruction>,
        abi: &mut Vec<EvmAbiEntry>,
    ) -> Result<(), CompilerError> {
        // Add constructor ABI entry
        abi.push(EvmAbiEntry {
            name: "constructor".to_string(),
            entry_type: EvmAbiType::Constructor,
            inputs: constructor.parameters.iter().map(|p| EvmAbiParam {
                name: p.name.name.clone(),
                param_type: self.augustium_type_to_solidity(&p.type_annotation).unwrap_or_default(),
                indexed: false,
            }).collect(),
            outputs: vec![],
            state_mutability: EvmStateMutability::NonPayable,
        });
        
        // Generate constructor bytecode
        self.translate_function_body(&constructor.body, instructions)?;
        
        Ok(())
    }
    
    /// Translate function to EVM bytecode
    #[allow(dead_code)]
    fn translate_function(
        &mut self,
        function: &Function,
        instructions: &mut Vec<EvmInstruction>,
        abi: &mut Vec<EvmAbiEntry>,
        _storage_layout: &mut HashMap<String, EvmStorageSlot>,
    ) -> Result<(), CompilerError> {
        // Generate function selector
        let selector = self.generate_function_selector(&function.name.name, &function.parameters)?;
        self.function_selectors.insert(function.name.name.clone(), selector);
        
        // Add function ABI entry
        abi.push(EvmAbiEntry {
            name: function.name.name.clone(),
            entry_type: EvmAbiType::Function,
            inputs: function.parameters.iter().map(|p| EvmAbiParam {
                name: p.name.name.clone(),
                param_type: self.augustium_type_to_solidity(&p.type_annotation).unwrap_or_default(),
                indexed: false,
            }).collect(),
            outputs: if let Some(return_type) = &function.return_type {
                vec![EvmAbiParam {
                    name: "result".to_string(),
                    param_type: self.augustium_type_to_solidity(return_type).unwrap_or_default(),
                    indexed: false,
                }]
            } else {
                vec![]
            },
            state_mutability: self.determine_state_mutability(function)?,
        });
        
        // Generate function dispatcher
        instructions.push(EvmInstruction::Push(selector.to_vec()));
        instructions.push(EvmInstruction::Push(vec![0x00, 0x00, 0x00, 0x00])); // calldata selector
        instructions.push(EvmInstruction::CallDataLoad);
        instructions.push(EvmInstruction::Eq);
        instructions.push(EvmInstruction::Push(vec![0x00, 0x00])); // jump destination
        instructions.push(EvmInstruction::JumpI);
        
        // Function body
        instructions.push(EvmInstruction::JumpDest);
        self.translate_function_body(&function.body, instructions)?;
        
        Ok(())
    }
    
    /// Translate function body to EVM instructions
    #[allow(dead_code)]
    fn translate_function_body(
        &mut self,
        body: &Block,
        instructions: &mut Vec<EvmInstruction>,
    ) -> Result<(), CompilerError> {
        for statement in &body.statements {
            self.translate_statement(statement, instructions)?;
        }
        Ok(())
    }
    
    /// Translate statement to EVM instructions
    #[allow(dead_code)]
    fn translate_statement(
        &mut self,
        statement: &Statement,
        instructions: &mut Vec<EvmInstruction>,
    ) -> Result<(), CompilerError> {
        match statement {
            Statement::Expression(expr) => {
                self.translate_expression(expr, instructions)?;
                instructions.push(EvmInstruction::Pop); // Discard result
            }

            Statement::Return(ret) => {
                if let Some(value) = &ret.value {
                    self.translate_expression(value, instructions)?;
                }
                instructions.push(EvmInstruction::Return);
            }
            Statement::If(if_stmt) => {
                self.translate_expression(&if_stmt.condition, instructions)?;
                instructions.push(EvmInstruction::IsZero);
                instructions.push(EvmInstruction::Push(vec![0x00, 0x00])); // else jump
                instructions.push(EvmInstruction::JumpI);
                
                self.translate_function_body(&if_stmt.then_block, instructions)?;
                
                if let Some(else_block) = &if_stmt.else_block {
                    instructions.push(EvmInstruction::JumpDest);
                    self.translate_statement(else_block, instructions)?;
                }
            }
            Statement::While(while_stmt) => {
                instructions.push(EvmInstruction::JumpDest); // Loop start
                self.translate_expression(&while_stmt.condition, instructions)?;
                instructions.push(EvmInstruction::IsZero);
                instructions.push(EvmInstruction::Push(vec![0x00, 0x00])); // exit jump
                instructions.push(EvmInstruction::JumpI);

                self.translate_function_body(&while_stmt.body, instructions)?;
                instructions.push(EvmInstruction::Push(vec![0x00, 0x00])); // loop start
                instructions.push(EvmInstruction::Jump);
                instructions.push(EvmInstruction::JumpDest); // Loop exit
            }

            Statement::Let(var_decl) => {
                if let Some(initializer) = &var_decl.value {
                    self.translate_expression(initializer, instructions)?;
                    // Store in memory/storage based on variable type
                    instructions.push(EvmInstruction::SStore);
                }
            }
            _ => {
                // Other statement types not yet supported
            }
        }
        Ok(())
    }
    
    /// Translate expression to EVM instructions
    #[allow(dead_code)]
    fn translate_expression(
        &mut self,
        expression: &Expression,
        instructions: &mut Vec<EvmInstruction>,
    ) -> Result<(), CompilerError> {
        match expression {
            Expression::Literal(lit) => {
                match lit {
                    Literal::Integer(value) => {
                        instructions.push(EvmInstruction::Push(value.to_be_bytes().to_vec()));
                    }
                    Literal::Float(f) => {
                        let int_value = *f as i64;
                        instructions.push(EvmInstruction::Push(int_value.to_be_bytes().to_vec()));
                    }
                    Literal::String(s) => {
                        instructions.push(EvmInstruction::Push(s.as_bytes().to_vec()));
                    }
                    Literal::Boolean(b) => {
                        let v = if *b { 1u8 } else { 0u8 };
                        instructions.push(EvmInstruction::Push(vec![v]));
                    }
                    Literal::Address(_addr) => {
                        // For now just push zero-padded 20-byte address placeholder
                        instructions.push(EvmInstruction::Push(vec![0u8; 20]));
                    }
                }
            }
            Expression::Identifier(_ident) => {
                // TODO: proper variable resolution – placeholder load from slot 0
                instructions.push(EvmInstruction::Push(vec![0x00]));
                instructions.push(EvmInstruction::SLoad);
            }
            Expression::Binary(binop) => {
                self.translate_expression(&binop.left, instructions)?;
                self.translate_expression(&binop.right, instructions)?;
                match binop.operator {
                    BinaryOperator::Add => instructions.push(EvmInstruction::Add),
                    BinaryOperator::Subtract => instructions.push(EvmInstruction::Sub),
                    BinaryOperator::Multiply => instructions.push(EvmInstruction::Mul),
                    BinaryOperator::Divide => instructions.push(EvmInstruction::Div),
                    BinaryOperator::Modulo => instructions.push(EvmInstruction::Mod),
                    BinaryOperator::Equal => instructions.push(EvmInstruction::Eq),
                    BinaryOperator::NotEqual => {
                        instructions.push(EvmInstruction::Eq);
                        instructions.push(EvmInstruction::IsZero);
                    }
                    BinaryOperator::Less => instructions.push(EvmInstruction::Lt),
                    BinaryOperator::Greater => instructions.push(EvmInstruction::Gt),
                    BinaryOperator::LessEqual => {
                        instructions.push(EvmInstruction::Gt);
                        instructions.push(EvmInstruction::IsZero);
                    }
                    BinaryOperator::GreaterEqual => {
                        instructions.push(EvmInstruction::Lt);
                        instructions.push(EvmInstruction::IsZero);
                    }
                    BinaryOperator::And => instructions.push(EvmInstruction::And),
                    BinaryOperator::Or => instructions.push(EvmInstruction::Or),
                    _ => {}
                }
            }
            Expression::Unary(unop) => {
                self.translate_expression(&unop.operand, instructions)?;
                match unop.operator {
                    UnaryOperator::Not => instructions.push(EvmInstruction::IsZero),
                    UnaryOperator::Minus => {
                        instructions.push(EvmInstruction::Push(vec![0x00]));
                        instructions.push(EvmInstruction::Sub);
                    }
                    UnaryOperator::BitNot => instructions.push(EvmInstruction::Not),
                }
            }
            Expression::Call(call) => {
                for arg in &call.arguments {
                    self.translate_expression(arg, instructions)?;
                }
                if let Expression::Identifier(id) = call.function.as_ref() {
                    if let Some(selector) = self.function_selectors.get(&id.name) {
                        instructions.push(EvmInstruction::Push(selector.to_vec()));
                        instructions.push(EvmInstruction::Call);
                    }
                }
            }
            Expression::FieldAccess(_field) => {
                instructions.push(EvmInstruction::Push(vec![0x00]));
                instructions.push(EvmInstruction::SLoad);
            }
            Expression::Index(index) => {
                self.translate_expression(&index.object, instructions)?;
                self.translate_expression(&index.index, instructions)?;
                instructions.push(EvmInstruction::Add);
                instructions.push(EvmInstruction::SLoad);
            }
            _ => {}
        }
        Ok(())
    }

    /// Translate assignment target
    #[allow(dead_code)]
    fn translate_assignment_target(
        &mut self,
        target: &Expression,
        instructions: &mut Vec<EvmInstruction>,
    ) -> Result<(), CompilerError> {
        match target {
            Expression::Identifier(_ident) => {
                instructions.push(EvmInstruction::Push(vec![0x00])); // storage slot
                instructions.push(EvmInstruction::SStore);
            }
            Expression::FieldAccess(_field) => {
                instructions.push(EvmInstruction::Push(vec![0x00])); // field slot
                instructions.push(EvmInstruction::SStore);
            }
            Expression::Index(index) => {
                self.translate_expression(&index.object, instructions)?;
                self.translate_expression(&index.index, instructions)?;
                instructions.push(EvmInstruction::Add);
                instructions.push(EvmInstruction::SStore);
            }
            _ => {
                return Err(CompilerError::CodegenError(CodegenError {
                    kind: CodegenErrorKind::InvalidAssignmentTarget,
                    location: SourceLocation::default(),
                    message: "Invalid assignment target".to_string(),
                }));
            }
        }
        Ok(())
    }
    
    /// Generate function selector (first 4 bytes of keccak256 hash)
    #[allow(dead_code)]
    fn generate_function_selector(
        &self,
        name: &str,
        parameters: &[Parameter],
    ) -> Result<[u8; 4], CompilerError> {
        let mut signature = name.to_string();
        signature.push('(');
        
        for (i, param) in parameters.iter().enumerate() {
            if i > 0 {
                signature.push(',');
            }
            signature.push_str(&self.augustium_type_to_solidity(&param.type_annotation)?);
        }
        signature.push(')');
        
        // Simple hash for demonstration (in real implementation, use keccak256)
        let hash = self.simple_hash(signature.as_bytes());
        Ok([hash[0], hash[1], hash[2], hash[3]])
    }
    
    /// Convert Augustium type to Solidity type string
    #[allow(dead_code)]
    fn augustium_type_to_solidity(&self, aug_type: &Type) -> Result<String, CompilerError> {
        match aug_type {
            Type::U8 => Ok("uint8".to_string()),
            Type::U256 => Ok("uint256".to_string()),
            Type::Bool => Ok("bool".to_string()),
            Type::Address => Ok("address".to_string()),
            Type::String => Ok("string".to_string()),
            Type::Array { element_type, size } => {
                 let inner_type = self.augustium_type_to_solidity(element_type)?;
                 if let Some(size) = size {
                     Ok(format!("{}[{}]", inner_type, size))
                 } else {
                     Ok(format!("{}[]", inner_type))
                 }
             }
            Type::Named(name) => Ok(name.name.clone()),
            _ => Ok("bytes".to_string()),
        }
    }
    
    /// Determine function state mutability
    #[allow(dead_code)]
    fn determine_state_mutability(
        &self,
        _function: &Function,
    ) -> Result<EvmStateMutability, CompilerError> {
        // Simplified - in real implementation, analyze function body
        Ok(EvmStateMutability::NonPayable)
    }
    
    /// Calculate source hash
    #[allow(dead_code)]
    fn calculate_source_hash(&self, _contract: &Contract) -> Result<String, CompilerError> {
        // Simplified hash calculation
        Ok("0x1234567890abcdef".to_string())
    }
    
    /// Simple hash function (replace with keccak256 in real implementation)
    #[allow(dead_code)]
    fn simple_hash(&self, data: &[u8]) -> Vec<u8> {
        let mut hash = vec![0u8; 32];
        for (i, &byte) in data.iter().enumerate() {
            hash[i % 32] ^= byte;
        }
        hash
    }
    
    /// Convert EVM bytecode to hex string
    #[allow(dead_code)]
    pub fn bytecode_to_hex(&self, bytecode: &EvmBytecode) -> String {
        let mut hex = String::new();
        for instruction in &bytecode.instructions {
            hex.push_str(&self.instruction_to_hex(instruction));
        }
        hex
    }
    
    /// Convert single instruction to hex
    #[allow(dead_code)]
    fn instruction_to_hex(&self, instruction: &EvmInstruction) -> String {
        match instruction {
            EvmInstruction::Push(data) => {
                let opcode = 0x60 + (data.len() - 1) as u8; // PUSH1-PUSH32
                let mut hex = format!("{:02x}", opcode);
                for byte in data {
                    hex.push_str(&format!("{:02x}", byte));
                }
                hex
            }
            EvmInstruction::Add => "01".to_string(),
            EvmInstruction::Sub => "03".to_string(),
            EvmInstruction::Mul => "02".to_string(),
            EvmInstruction::Div => "04".to_string(),
            EvmInstruction::Mod => "06".to_string(),
            EvmInstruction::Eq => "14".to_string(),
            EvmInstruction::Lt => "10".to_string(),
            EvmInstruction::Gt => "11".to_string(),
            EvmInstruction::IsZero => "15".to_string(),
            EvmInstruction::And => "16".to_string(),
            EvmInstruction::Or => "17".to_string(),
            EvmInstruction::SLoad => "54".to_string(),
            EvmInstruction::SStore => "55".to_string(),
            EvmInstruction::Jump => "56".to_string(),
            EvmInstruction::JumpI => "57".to_string(),
            EvmInstruction::JumpDest => "5b".to_string(),
            EvmInstruction::Return => "f3".to_string(),
            EvmInstruction::Revert => "fd".to_string(),
            EvmInstruction::Stop => "00".to_string(),
            _ => "00".to_string(), // Placeholder for other instructions
        }
    }
}

/// EVM deployment helper
#[allow(dead_code)]
pub struct EvmDeployer {
    config: EvmConfig,
}

impl EvmDeployer {
    /// Create a new EVM deployer
    #[allow(dead_code)]
    pub fn new(config: EvmConfig) -> Self {
        Self { config }
    }
    
    /// Generate deployment transaction
    #[allow(dead_code)]
    pub fn generate_deployment_tx(
        &self,
        bytecode: &EvmBytecode,
        constructor_args: &[u8],
    ) -> Result<EvmTransaction, CompilerError> {
        let mut data = Vec::new();
        
        // Add contract bytecode
        for instruction in &bytecode.instructions {
            data.extend_from_slice(&self.instruction_to_bytes(instruction));
        }
        
        // Add constructor arguments
        data.extend_from_slice(constructor_args);
        
        Ok(EvmTransaction {
            to: None, // Contract creation
            value: 0,
            gas_limit: self.config.gas_limit,
            gas_price: 20_000_000_000, // 20 gwei
            data,
            chain_id: self.config.chain_id,
        })
    }
    
    /// Convert instruction to bytes
    #[allow(dead_code)]
    fn instruction_to_bytes(&self, instruction: &EvmInstruction) -> Vec<u8> {
        match instruction {
            EvmInstruction::Push(data) => {
                let mut bytes = vec![0x60 + (data.len() - 1) as u8];
                bytes.extend_from_slice(data);
                bytes
            }
            EvmInstruction::Add => vec![0x01],
            EvmInstruction::Sub => vec![0x03],
            EvmInstruction::Mul => vec![0x02],
            EvmInstruction::Div => vec![0x04],
            EvmInstruction::Mod => vec![0x06],
            EvmInstruction::Eq => vec![0x14],
            EvmInstruction::Lt => vec![0x10],
            EvmInstruction::Gt => vec![0x11],
            EvmInstruction::IsZero => vec![0x15],
            EvmInstruction::And => vec![0x16],
            EvmInstruction::Or => vec![0x17],
            EvmInstruction::SLoad => vec![0x54],
            EvmInstruction::SStore => vec![0x55],
            EvmInstruction::Jump => vec![0x56],
            EvmInstruction::JumpI => vec![0x57],
            EvmInstruction::JumpDest => vec![0x5b],
            EvmInstruction::Return => vec![0xf3],
            EvmInstruction::Revert => vec![0xfd],
            EvmInstruction::Stop => vec![0x00],
            _ => vec![0x00], // Placeholder
        }
    }
}

/// EVM transaction representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvmTransaction {
    pub to: Option<String>, // None for contract creation
    pub value: u64,
    pub gas_limit: u64,
    pub gas_price: u64,
    pub data: Vec<u8>,
    pub chain_id: u64,
}

/// EVM compatibility utilities
pub mod utils {
    use super::*;
    
    /// Check if Augustium contract is EVM compatible
    #[allow(dead_code)]
    pub fn is_evm_compatible(contract: &Contract) -> bool {
        // Check for EVM-incompatible features
        for function in &contract.functions {
            if function.name.name.contains("augustium_specific") {
                return false;
            }
        }
        true
    }
    
    /// Generate Solidity interface from Augustium contract
    #[allow(dead_code)]
    pub fn generate_solidity_interface(contract: &Contract) -> Result<String, CompilerError> {
        let mut interface = format!("interface I{} {{\n", contract.name.name);
        
        for function in &contract.functions {
            interface.push_str(&format!(
                "    function {}({}) external",
                 function.name.name,
                function.parameters.iter()
                    .map(|p| format!("{} {}", 
                        augustium_type_to_solidity_simple(&p.type_annotation),
                         p.name.name
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            
            if let Some(return_type) = &function.return_type {
                interface.push_str(&format!(" returns ({})", augustium_type_to_solidity_simple(return_type)));
            }
            
            interface.push_str(";\n");
        }
        
        interface.push_str("}\n");
        Ok(interface)
    }
    
    /// Simple type conversion helper
    #[allow(dead_code)]
    fn augustium_type_to_solidity_simple(aug_type: &Type) -> String {
        match aug_type {
            Type::U8 => "uint8".to_string(),
            Type::U256 => "uint256".to_string(),
            Type::Bool => "bool".to_string(),
            Type::Address => "address".to_string(),
            Type::String => "string memory".to_string(),
            Type::Array { element_type, size } => {
                let inner_type = augustium_type_to_solidity_simple(element_type);
                if let Some(size) = size {
                    format!("{}[{}] memory", inner_type, size)
                } else {
                    format!("{}[] memory", inner_type)
                }
            }
            Type::Named(name) => name.name.clone(),
            _ => "bytes".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::error::SourceLocation;
    
    #[test]
    fn test_evm_config_default() {
        let config = EvmConfig::default();
        assert_eq!(config.evm_version, "london");
        assert_eq!(config.chain_id, 1);
        assert!(config.enable_precompiles);
    }
    
    #[test]
    fn test_function_selector_generation() {
        let config = EvmConfig::default();
        let translator = EvmTranslator::new(config);
        
        let params = vec![
            Parameter {
                name: Identifier::dummy("amount"),
                type_annotation: Type::U256,
                location: SourceLocation::dummy(),
            }
        ];
        
        let selector = translator.generate_function_selector("transfer", &params).unwrap();
        assert_eq!(selector.len(), 4);
    }
    
    #[test]
    fn test_type_conversion() {
        let config = EvmConfig::default();
        let translator = EvmTranslator::new(config);
        
        assert_eq!(translator.augustium_type_to_solidity(&Type::U256).unwrap(), "uint256");
        assert_eq!(translator.augustium_type_to_solidity(&Type::Bool).unwrap(), "bool");
        assert_eq!(translator.augustium_type_to_solidity(&Type::Address).unwrap(), "address");
    }
    
    #[test]
    fn test_bytecode_generation() {
        let config = EvmConfig::default();
        let translator = EvmTranslator::new(config);
        
        let mut instructions = Vec::new();
        instructions.push(EvmInstruction::Push(vec![0x01]));
        instructions.push(EvmInstruction::Push(vec![0x02]));
        instructions.push(EvmInstruction::Add);
        
        let bytecode = EvmBytecode {
            instructions,
            metadata: EvmMetadata {
                contract_name: "Test".to_string(),
                compiler_version: "augustium-0.1.0".to_string(),
                source_hash: "0x123".to_string(),
                abi: vec![],
                storage_layout: HashMap::new(),
            },
        };
        
        let hex = translator.bytecode_to_hex(&bytecode);
        assert!(hex.contains("6001")); // PUSH1 0x01
        assert!(hex.contains("6002")); // PUSH1 0x02
        assert!(hex.contains("01")); // ADD
    }
}