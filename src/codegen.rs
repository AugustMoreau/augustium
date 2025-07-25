// Code generator - turns AST into bytecode instructions
// Final step before the VM can execute the code

use crate::ast::*;
use crate::error::{Result, CodegenError, CodegenErrorKind, SourceLocation};
use std::collections::HashMap;

// Individual VM instructions that our bytecode is made of
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum Instruction {
    // Stack operations
    Push(Value),
    Pop,
    Dup,
    Swap,
    
    // Arithmetic operations
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    
    // Comparison operations
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    
    // Logical operations
    And,
    Or,
    Not,
    
    // Bitwise operations
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    Shl,
    Shr,
    
    // Memory operations
    Load(u32),   // Load from local variable
    Store(u32),  // Store to local variable
    LoadField(u32), // Load from contract field
    StoreField(u32), // Store to contract field
    
    // Control flow
    Jump(u32),
    JumpIf(u32),
    JumpIfNot(u32),
    Call(u32),
    Return,
    
    // Contract operations
    Deploy,
    Invoke(String), // Function name
    Emit(String),   // Event name
    
    // Blockchain operations
    GetBalance,
    Transfer,
    GetCaller,
    GetValue,
    GetBlockNumber,
    GetTimestamp,
    
    // Safety operations
    Require,
    Assert,
    Revert,
    
    // Machine Learning operations - Phase 1 Enhanced
    MLCreateModel(String),     // Model type ("neural_network", "linear_regression", etc.)
    MLLoadModel(u32),          // Model ID
    MLSaveModel(u32),          // Model ID
    MLTrain(u32),              // Model ID
    MLPredict(u32),            // Model ID
    MLSetHyperparams(u32),     // Model ID
    MLGetMetrics(u32),         // Model ID
    MLForward(u32),            // Neural network forward pass
    MLBackward(u32),           // Neural network backward pass
    MLUpdateWeights(u32),      // Update model weights
    MLNormalize,               // Data normalization
    MLDenormalize,             // Data denormalization
    MLActivation(String),      // Activation function ("relu", "sigmoid", "tanh")
    MLLoss(String),            // Loss function ("mse", "cross_entropy")
    MLOptimizer(String),       // Optimizer ("sgd", "adam", "rmsprop")
    
    // Enhanced ML Instructions - Phase 1
    MLCreateTensor(Vec<usize>), // Create tensor with dimensions
    MLTensorOp(String),        // Tensor operations ("add", "mul", "matmul", "transpose")
    MLReshape(Vec<usize>),     // Reshape tensor
    MLSlice(Vec<(usize, usize)>), // Slice tensor
    MLConcat(usize),           // Concatenate tensors along axis
    MLSplit(usize, usize),     // Split tensor along axis
    MLReduce(String, Option<usize>), // Reduce operations ("sum", "mean", "max", "min")
    MLBroadcast(Vec<usize>),   // Broadcast tensor to shape
    
    // Advanced Neural Network Operations
    MLConv2D(u32, u32, u32, u32), // Convolution (filters, kernel_size, stride, padding)
    MLMaxPool2D(u32, u32),     // Max pooling (pool_size, stride)
    MLDropout(f64),            // Dropout with probability
    MLBatchNorm,               // Batch normalization
    MLLayerNorm,               // Layer normalization
    MLAttention,               // Self-attention mechanism
    MLEmbedding(u32, u32),     // Embedding layer (vocab_size, embed_dim)
    
    // Model Management
    MLCloneModel(u32),         // Clone existing model
    MLMergeModels(Vec<u32>),   // Merge multiple models
    MLQuantizeModel(u32, String), // Quantize model ("int8", "int16")
    MLPruneModel(u32, f64),    // Prune model with threshold
    MLDistillModel(u32, u32),  // Knowledge distillation (teacher, student)
    
    // Training Operations
    MLSetLearningRate(f64),    // Set learning rate
    MLScheduleLR(String),      // Learning rate scheduler
    MLGradientClip(f64),       // Gradient clipping
    MLEarlyStopping(u32, f64), // Early stopping (patience, min_delta)
    MLCheckpoint(String),      // Save training checkpoint
    MLRestoreCheckpoint(String), // Restore from checkpoint
    
    // Data Operations
    MLLoadDataset(String),     // Load dataset from storage
    MLSaveDataset(String),     // Save dataset to storage
    MLSplitDataset(f64, f64),  // Split dataset (train_ratio, val_ratio)
    MLShuffleDataset,          // Shuffle dataset
    MLAugmentData(String),     // Data augmentation
    MLPreprocessData(String),  // Data preprocessing pipeline
    
    // Evaluation and Metrics
    MLEvaluate(u32),           // Evaluate model on test set
    MLConfusionMatrix,         // Generate confusion matrix
    MLROCCurve,                // Generate ROC curve
    MLFeatureImportance,       // Calculate feature importance
    MLExplainPrediction(u32),  // Explain model prediction
    
    // Cross-Chain ML Operations
    MLExportModel(String),     // Export model for cross-chain
    MLImportModel(String),     // Import model from cross-chain
    MLVerifyModel(String),     // Verify model integrity
    MLSyncModel(u32, String),  // Sync model across chains
    
    // Debugging
    Debug(String),
    
    // Halt execution
    Halt,
}

/// Runtime value in the AVM
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    U256([u8; 32]),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    I256([u8; 32]),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Address([u8; 20]),
    Array(Vec<Value>),
    Tuple(Vec<Value>),
    // Machine Learning types - Enhanced Phase 1
    MLModel {
        model_id: u32,
        model_type: String,
        weights: Vec<f64>,
        biases: Vec<f64>,
        hyperparams: std::collections::HashMap<String, f64>,
        architecture: Vec<u32>, // Layer sizes
        version: u32,
        checksum: String,
    },
    MLDataset {
        features: Vec<Vec<f64>>,
        targets: Vec<f64>,
        normalized: bool,
        metadata: std::collections::HashMap<String, String>,
        split_info: Option<(f64, f64)>, // train_ratio, val_ratio
    },
    MLMetrics {
        accuracy: f64,
        loss: f64,
        precision: f64,
        recall: f64,
        f1_score: f64,
        auc_roc: f64,
        confusion_matrix: Vec<Vec<u32>>,
    },
    Tensor {
        data: Vec<f64>,
        shape: Vec<usize>,
        dtype: String, // "f32", "f64", "i32", etc.
    },
    Matrix(Vec<Vec<f64>>),     // 2D matrix
    Vector(Vec<f64>),          // 1D vector
    
    // Advanced ML Types
    MLOptimizer {
        optimizer_type: String, // "sgd", "adam", "rmsprop"
        learning_rate: f64,
        momentum: Option<f64>,
        beta1: Option<f64>,
        beta2: Option<f64>,
        epsilon: Option<f64>,
    },
    MLScheduler {
        scheduler_type: String, // "step", "exponential", "cosine"
        step_size: Option<u32>,
        gamma: Option<f64>,
        t_max: Option<u32>,
    },
    MLCheckpoint {
        model_state: Vec<u8>,
        optimizer_state: Vec<u8>,
        epoch: u32,
        loss: f64,
        timestamp: u64,
    },
    MLExplanation {
         feature_importance: Vec<f64>,
         shap_values: Vec<f64>,
         lime_explanation: std::collections::HashMap<String, f64>,
     },
     Null,
}

impl Value {
    /// Get the type of this value
    #[allow(dead_code)]
    pub fn get_type(&self) -> Type {
        match self {
            Value::U8(_) => Type::U8,
            Value::U16(_) => Type::U16,
            Value::U32(_) => Type::U32,
            Value::U64(_) => Type::U64,
            Value::U128(_) => Type::U128,
            Value::U256(_) => Type::U256,
            Value::I8(_) => Type::I8,
            Value::I16(_) => Type::I16,
            Value::I32(_) => Type::I32,
            Value::I64(_) => Type::I64,
            Value::I128(_) => Type::I128,
            Value::I256(_) => Type::I256,
            Value::F32(_) => Type::F32,
            Value::F64(_) => Type::F64,
            Value::Bool(_) => Type::Bool,
            Value::String(_) => Type::String,
            Value::Address(_) => Type::Address,
            Value::Array(elements) => {
                if elements.is_empty() {
                    Type::Array {
                        element_type: Box::new(Type::U32), // Default
                        size: Some(0),
                    }
                } else {
                    Type::Array {
                        element_type: Box::new(elements[0].get_type()),
                        size: Some(elements.len() as u64),
                    }
                }
            }
            Value::Tuple(elements) => {
                Type::Tuple(elements.iter().map(|v| v.get_type()).collect())
            }
            Value::MLModel { .. } => Type::MLModel {
                model_type: "neural_network".to_string(),
                input_shape: vec![],
                output_shape: vec![],
            },
            Value::MLDataset { .. } => Type::MLDataset {
                feature_types: vec![],
                target_type: Box::new(Type::U32),
            },
            Value::MLMetrics { .. } => Type::MLMetrics,
            Value::Tensor { data: _, shape: _, dtype: _ } => Type::Tensor { element_type: Box::new(Type::U32), dimensions: vec![] },
            Value::Matrix(_) => Type::Matrix { element_type: Box::new(Type::U32), rows: 0, cols: 0 },
            Value::Vector(_) => Type::Vector { element_type: Box::new(Type::U32), size: None },
            Value::MLOptimizer { .. } => Type::U32, // Placeholder
            Value::MLScheduler { .. } => Type::U32, // Placeholder
            Value::MLCheckpoint { .. } => Type::U32, // Placeholder
            Value::MLExplanation { .. } => Type::U32, // Placeholder
            Value::Null => Type::U32, // Placeholder
        }
    }
}

/// Compiled bytecode
#[derive(Debug, Clone)]
pub struct Bytecode {
    pub instructions: Vec<Instruction>,
    #[allow(dead_code)]
    pub constants: Vec<Value>,
    #[allow(dead_code)]
    pub functions: HashMap<String, u32>, // Function name -> instruction offset
    pub contracts: HashMap<String, ContractBytecode>,
}

impl Bytecode {
    /// Serialize bytecode to bytes for file storage
    pub fn to_bytes(&self) -> Vec<u8> {
        // Simple serialization - in a real implementation, you'd use a proper format
        format!("{:?}", self).into_bytes()
    }
    
    /// Get the size of the bytecode in instructions
    pub fn len(&self) -> usize {
        self.instructions.len()
    }
}

/// Contract-specific bytecode
#[derive(Debug, Clone)]
pub struct ContractBytecode {
    pub constructor: Vec<Instruction>,
    pub functions: HashMap<String, Vec<Instruction>>,
    pub fields: HashMap<String, u32>, // Field name -> field index
    #[allow(dead_code)]
    pub events: HashMap<String, Vec<Type>>, // Event name -> parameter types
}

/// Local variable information
#[derive(Debug, Clone)]
struct LocalVariable {
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    var_type: Type,
    index: u32,
    #[allow(dead_code)]
    mutable: bool,
}

/// Code generation context
struct CodegenContext {
    locals: HashMap<String, LocalVariable>,
    next_local_index: u32,
    current_contract: Option<String>,
    current_function: Option<String>,
    break_labels: Vec<u32>,
    continue_labels: Vec<u32>,
    next_label: u32,
}

impl CodegenContext {
    fn new() -> Self {
        Self {
            locals: HashMap::new(),
            next_local_index: 0,
            current_contract: None,
            current_function: None,
            break_labels: Vec::new(),
            continue_labels: Vec::new(),
            next_label: 0,
        }
    }
    
    fn add_local(&mut self, name: String, var_type: Type, mutable: bool) -> u32 {
        let index = self.next_local_index;
        self.next_local_index += 1;
        
        let local = LocalVariable {
            name: name.clone(),
            var_type,
            index,
            mutable,
        };
        
        self.locals.insert(name, local);
        index
    }
    
    fn get_local(&self, name: &str) -> Option<&LocalVariable> {
        self.locals.get(name)
    }
    
    fn new_label(&mut self) -> u32 {
        let label = self.next_label;
        self.next_label += 1;
        label
    }
    
    fn push_loop_labels(&mut self, break_label: u32, continue_label: u32) {
        self.break_labels.push(break_label);
        self.continue_labels.push(continue_label);
    }
    
    fn pop_loop_labels(&mut self) {
        self.break_labels.pop();
        self.continue_labels.pop();
    }
    
    fn current_break_label(&self) -> Option<u32> {
        self.break_labels.last().copied()
    }
    
    fn current_continue_label(&self) -> Option<u32> {
        self.continue_labels.last().copied()
    }
}

/// Code generator
pub struct CodeGenerator {
    context: CodegenContext,
    instructions: Vec<Instruction>,
    constants: Vec<Value>,
    functions: HashMap<String, u32>,
    contracts: HashMap<String, ContractBytecode>,
    labels: HashMap<u32, u32>, // Label -> instruction offset
}

impl CodeGenerator {
    pub fn new() -> Self {
        Self {
            context: CodegenContext::new(),
            instructions: Vec::new(),
            constants: Vec::new(),
            functions: HashMap::new(),
            contracts: HashMap::new(),
            labels: HashMap::new(),
        }
    }
    
    /// Generate bytecode for a source file
    pub fn generate(&mut self, ast: &SourceFile) -> Result<Bytecode> {
        for item in &ast.items {
            self.generate_item(item)?;
        }
        
        // Resolve labels
        self.resolve_labels();
        
        Ok(Bytecode {
            instructions: self.instructions.clone(),
            constants: self.constants.clone(),
            functions: self.functions.clone(),
            contracts: self.contracts.clone(),
        })
    }
    
    /// Generate code for a top-level item
    fn generate_item(&mut self, item: &Item) -> Result<()> {
        match item {
            Item::Contract(contract) => self.generate_contract(contract),
            Item::Function(function) => self.generate_function(function),
            Item::Struct(_) => Ok(()), // Structs don't generate runtime code
            Item::Enum(_) => Ok(()),   // Enums don't generate runtime code
            Item::Trait(_) => Ok(()),  // Traits don't generate runtime code
            Item::Impl(impl_block) => self.generate_impl(impl_block),
            Item::Use(_) => Ok(()),    // Use declarations don't generate code
            Item::Const(const_decl) => self.generate_const(const_decl),
            Item::Module(module) => self.generate_module(module),
        }
    }
    
    /// Generate code for a contract
    fn generate_contract(&mut self, contract: &Contract) -> Result<()> {
        let old_contract = self.context.current_contract.clone();
        self.context.current_contract = Some(contract.name.name.clone());
        
        let mut contract_bytecode = ContractBytecode {
            constructor: Vec::new(),
            functions: HashMap::new(),
            fields: HashMap::new(),
            events: HashMap::new(),
        };
        
        // Map contract fields
        for (index, field) in contract.fields.iter().enumerate() {
            contract_bytecode.fields.insert(field.name.name.clone(), index as u32);
        }
        
        // TODO: Generate constructor if present
        // Constructor handling will be implemented when constructor syntax is added to Contract struct
        
        // Generate functions
        for function in &contract.functions {
            let function_start = self.instructions.len();
            self.generate_function(function)?;
            let function_instructions = self.instructions[function_start..].to_vec();
            contract_bytecode.functions.insert(
                function.name.name.clone(),
                function_instructions,
            );
        }
        
        self.contracts.insert(contract.name.name.clone(), contract_bytecode);
        self.context.current_contract = old_contract;
        
        Ok(())
    }
    
    /// Generate code for a function
    fn generate_function(&mut self, function: &Function) -> Result<()> {
        let old_function = self.context.current_function.clone();
        self.context.current_function = Some(function.name.name.clone());
        
        // Record function start position
        let function_start = self.instructions.len() as u32;
        self.functions.insert(function.name.name.clone(), function_start);
        
        // Clear local variables
        self.context.locals.clear();
        self.context.next_local_index = 0;
        
        // Add parameters as local variables
        for param in &function.parameters {
            self.context.add_local(
                param.name.name.clone(),
                param.type_annotation.clone(),
                false, // Parameters are immutable
            );
        }
        
        // Generate function body
        self.generate_block(&function.body)?;
        
        // Add implicit return if needed
        if function.return_type.is_none() {
            self.emit(Instruction::Return);
        }
        
        self.context.current_function = old_function;
        
        Ok(())
    }
    
    /// Generate code for an impl block
    fn generate_impl(&mut self, impl_block: &Impl) -> Result<()> {
        // Generate functions in impl block
        for function in &impl_block.functions {
            self.generate_function(function)?;
        }
        
        Ok(())
    }
    
    /// Generate code for a const declaration
    fn generate_const(&mut self, const_decl: &ConstDeclaration) -> Result<()> {
        // Constants are handled at compile time
        // Add to constants table
        let value = self.evaluate_const_expression(&const_decl.value)?;
        self.constants.push(value);
        
        Ok(())
    }
    
    /// Generate code for a module
    fn generate_module(&mut self, module: &Module) -> Result<()> {
        // Generate code for module items
        for item in &module.items {
            self.generate_item(item)?;
        }
        
        Ok(())
    }
    
    /// Generate code for a block
    fn generate_block(&mut self, block: &Block) -> Result<()> {
        for statement in &block.statements {
            self.generate_statement(statement)?;
        }
        
        Ok(())
    }
    
    /// Generate code for a statement
    fn generate_statement(&mut self, statement: &Statement) -> Result<()> {
        match statement {
            Statement::Expression(expr) => {
                self.generate_expression(expr)?;
                self.emit(Instruction::Pop); // Discard expression result
                Ok(())
            }
            Statement::Let(let_stmt) => self.generate_let_statement(let_stmt),
            Statement::Return(return_stmt) => self.generate_return_statement(return_stmt),
            Statement::If(if_stmt) => self.generate_if_statement(if_stmt),
            Statement::While(while_stmt) => self.generate_while_statement(while_stmt),
            Statement::For(for_stmt) => self.generate_for_statement(for_stmt),
            Statement::Match(match_stmt) => self.generate_match_statement(match_stmt),
            Statement::Break(_) => self.generate_break_statement(),
            Statement::Continue(_) => self.generate_continue_statement(),
            Statement::Emit(emit_stmt) => self.generate_emit_statement(emit_stmt),
            Statement::Require(require_stmt) => self.generate_require_statement(require_stmt),
            Statement::Assert(assert_stmt) => self.generate_assert_statement(assert_stmt),
            Statement::Revert(revert_stmt) => self.generate_revert_statement(revert_stmt),
        }
    }
    
    /// Generate code for a let statement
    fn generate_let_statement(&mut self, let_stmt: &LetStatement) -> Result<()> {
        // Generate value if present
        if let Some(value) = &let_stmt.value {
            self.generate_expression(value)?;
        } else {
            // Initialize with default value
            let default_value = self.default_value_for_type(&let_stmt.type_annotation.as_ref().unwrap())?;
            self.emit(Instruction::Push(default_value));
        }
        
        // Add local variable
        let var_type = let_stmt.type_annotation.clone().unwrap_or(Type::U32);
        let local_index = self.context.add_local(
            let_stmt.name.name.clone(),
            var_type,
            let_stmt.mutable,
        );
        
        // Store value in local variable
        self.emit(Instruction::Store(local_index));
        
        Ok(())
    }
    
    /// Generate code for a return statement
    fn generate_return_statement(&mut self, return_stmt: &ReturnStatement) -> Result<()> {
        if let Some(value) = &return_stmt.value {
            self.generate_expression(value)?;
        }
        
        self.emit(Instruction::Return);
        
        Ok(())
    }
    
    /// Generate code for an if statement
    fn generate_if_statement(&mut self, if_stmt: &IfStatement) -> Result<()> {
        // Generate condition
        self.generate_expression(&if_stmt.condition)?;
        
        let else_label = self.context.new_label();
        let end_label = self.context.new_label();
        
        // Jump to else if condition is false
        self.emit(Instruction::JumpIfNot(else_label));
        
        // Generate then block
        self.generate_block(&if_stmt.then_block)?;
        
        // Jump to end
        self.emit(Instruction::Jump(end_label));
        
        // Else label
        self.place_label(else_label);
        
        // Generate else block if present
        if let Some(else_block) = &if_stmt.else_block {
            self.generate_statement(else_block)?;
        }
        
        // End label
        self.place_label(end_label);
        
        Ok(())
    }
    
    /// Generate code for a while statement
    fn generate_while_statement(&mut self, while_stmt: &WhileStatement) -> Result<()> {
        let loop_start = self.context.new_label();
        let loop_end = self.context.new_label();
        
        self.context.push_loop_labels(loop_end, loop_start);
        
        // Loop start
        self.place_label(loop_start);
        
        // Generate condition
        self.generate_expression(&while_stmt.condition)?;
        
        // Jump to end if condition is false
        self.emit(Instruction::JumpIfNot(loop_end));
        
        // Generate body
        self.generate_block(&while_stmt.body)?;
        
        // Jump back to start
        self.emit(Instruction::Jump(loop_start));
        
        // Loop end
        self.place_label(loop_end);
        
        self.context.pop_loop_labels();
        
        Ok(())
    }
    
    /// Generate code for a for statement
    fn generate_for_statement(&mut self, for_stmt: &ForStatement) -> Result<()> {
        // Generate iterable
        self.generate_expression(&for_stmt.iterable)?;
        
        // TODO: Implement proper iteration
        // For now, we'll generate a simple loop
        
        let loop_start = self.context.new_label();
        let loop_end = self.context.new_label();
        
        self.context.push_loop_labels(loop_end, loop_start);
        
        // Add loop variable
        let _loop_var_index = self.context.add_local(
            for_stmt.variable.name.clone(),
            Type::U32, // Placeholder
            false,
        );
        
        // Loop start
        self.place_label(loop_start);
        
        // TODO: Check if iteration is complete
        // For now, we'll just generate the body
        
        // Generate body
        self.generate_block(&for_stmt.body)?;
        
        // Jump back to start
        self.emit(Instruction::Jump(loop_start));
        
        // Loop end
        self.place_label(loop_end);
        
        self.context.pop_loop_labels();
        
        Ok(())
    }
    
    /// Generate code for a match statement
    fn generate_match_statement(&mut self, _match_stmt: &MatchStatement) -> Result<()> {
        // TODO: Implement match statement code generation
        Ok(())
    }
    
    /// Generate code for a break statement
    fn generate_break_statement(&mut self) -> Result<()> {
        if let Some(break_label) = self.context.current_break_label() {
            self.emit(Instruction::Jump(break_label));
            Ok(())
        } else {
            Err(CodegenError::new(
                CodegenErrorKind::InvalidBreak,
                SourceLocation::default(),
                "Break statement outside of loop".to_string(),
            ).into())
        }
    }
    
    /// Generate code for a continue statement
    fn generate_continue_statement(&mut self) -> Result<()> {
        if let Some(continue_label) = self.context.current_continue_label() {
            self.emit(Instruction::Jump(continue_label));
            Ok(())
        } else {
            Err(CodegenError::new(
                CodegenErrorKind::InvalidContinue,
                SourceLocation::default(),
                "Continue statement outside of loop".to_string(),
            ).into())
        }
    }
    
    /// Generate code for an emit statement
    fn generate_emit_statement(&mut self, emit_stmt: &EmitStatement) -> Result<()> {
        // Generate arguments
        for arg in &emit_stmt.arguments {
            self.generate_expression(arg)?;
        }
        
        // Emit event
        self.emit(Instruction::Emit(emit_stmt.event.name.clone()));
        
        Ok(())
    }
    
    /// Generate code for a require statement
    fn generate_require_statement(&mut self, require_stmt: &RequireStatement) -> Result<()> {
        // Generate condition
        self.generate_expression(&require_stmt.condition)?;
        
        // Generate message if present
        if let Some(message) = &require_stmt.message {
            self.generate_expression(message)?;
        } else {
            self.emit(Instruction::Push(Value::String("Requirement failed".to_string())));
        }
        
        self.emit(Instruction::Require);
        
        Ok(())
    }
    
    /// Generate code for an assert statement
    fn generate_assert_statement(&mut self, assert_stmt: &AssertStatement) -> Result<()> {
        // Generate condition
        self.generate_expression(&assert_stmt.condition)?;
        
        // Generate message if present
        if let Some(message) = &assert_stmt.message {
            self.generate_expression(message)?;
        } else {
            self.emit(Instruction::Push(Value::String("Assertion failed".to_string())));
        }
        
        self.emit(Instruction::Assert);
        
        Ok(())
    }
    
    /// Generate code for a revert statement
    fn generate_revert_statement(&mut self, revert_stmt: &RevertStatement) -> Result<()> {
        // Generate message if present
        if let Some(message) = &revert_stmt.message {
            self.generate_expression(message)?;
        } else {
            self.emit(Instruction::Push(Value::String("Transaction reverted".to_string())));
        }
        
        self.emit(Instruction::Revert);
        
        Ok(())
    }
    
    /// Generate code for an expression
    fn generate_expression(&mut self, expression: &Expression) -> Result<()> {
        match expression {
            Expression::Literal(literal) => self.generate_literal(literal),
            Expression::Identifier(identifier) => self.generate_identifier(identifier),
            Expression::Binary(binary_expr) => self.generate_binary_expression(binary_expr),
            Expression::Unary(unary_expr) => self.generate_unary_expression(unary_expr),
            Expression::Call(call_expr) => self.generate_call_expression(call_expr),
            Expression::FieldAccess(field_expr) => self.generate_field_access(field_expr),
            Expression::Index(index_expr) => self.generate_index_expression(index_expr),
            Expression::Array(array_expr) => self.generate_array_expression(array_expr),
            Expression::Tuple(tuple_expr) => self.generate_tuple_expression(tuple_expr),
            Expression::Struct(struct_expr) => self.generate_struct_expression(struct_expr),
            Expression::Assignment(assign_expr) => self.generate_assignment_expression(assign_expr),
            Expression::Range(range_expr) => self.generate_range_expression(range_expr),
            Expression::Closure(closure_expr) => self.generate_closure_expression(closure_expr),
            // Machine Learning expressions
            Expression::MLCreateModel(ml_create) => self.generate_ml_create_model(ml_create),
            Expression::MLTrain(ml_train) => self.generate_ml_train(ml_train),
            Expression::MLPredict(ml_predict) => self.generate_ml_predict(ml_predict),
            Expression::MLForward(ml_forward) => self.generate_ml_forward(ml_forward),
            Expression::MLBackward(ml_backward) => self.generate_ml_backward(ml_backward),
            Expression::TensorOp(tensor_op) => self.generate_tensor_op(tensor_op),
            Expression::MatrixOp(matrix_op) => self.generate_matrix_op(matrix_op),
            Expression::Block(block) => {
                self.generate_block(block)?;
                // Block expressions should leave a value on the stack
                self.emit(Instruction::Push(Value::Null));
                Ok(())
            }
        }
    }
    
    /// Generate code for a literal
    fn generate_literal(&mut self, literal: &Literal) -> Result<()> {
        let value = match literal {
            Literal::Integer(n) => Value::U32(*n as u32),
            Literal::Float(f) => Value::F64(*f),
            Literal::String(s) => Value::String(s.clone()),
            Literal::Boolean(b) => Value::Bool(*b),
            Literal::Address(addr) => {
                // Parse hex address
                let mut bytes = [0u8; 20];
                if addr.len() >= 42 && addr.starts_with("0x") {
                    for (i, chunk) in addr[2..].as_bytes().chunks(2).enumerate() {
                        if i < 20 {
                            let hex_str = std::str::from_utf8(chunk).unwrap_or("00");
                            bytes[i] = u8::from_str_radix(hex_str, 16).unwrap_or(0);
                        }
                    }
                }
                Value::Address(bytes)
            }
        };
        
        self.emit(Instruction::Push(value));
        Ok(())
    }
    
    /// Generate code for an identifier
    fn generate_identifier(&mut self, identifier: &Identifier) -> Result<()> {
        if let Some(local) = self.context.get_local(&identifier.name) {
            self.emit(Instruction::Load(local.index));
        } else {
            // Could be a contract field or global
            // For now, we'll assume it's a contract field
            // TODO: Implement proper symbol resolution
            self.emit(Instruction::LoadField(0)); // Placeholder
        }
        
        Ok(())
    }
    
    /// Generate code for a binary expression
    fn generate_binary_expression(&mut self, binary_expr: &BinaryExpression) -> Result<()> {
        // Generate left operand
        self.generate_expression(&binary_expr.left)?;
        
        // Generate right operand
        self.generate_expression(&binary_expr.right)?;
        
        // Generate operation
        let instruction = match binary_expr.operator {
            BinaryOperator::Add => Instruction::Add,
            BinaryOperator::Subtract => Instruction::Sub,
            BinaryOperator::Multiply => Instruction::Mul,
            BinaryOperator::Divide => Instruction::Div,
            BinaryOperator::Modulo => Instruction::Mod,
            BinaryOperator::Equal => Instruction::Eq,
            BinaryOperator::NotEqual => Instruction::Ne,
            BinaryOperator::Less => Instruction::Lt,
            BinaryOperator::LessEqual => Instruction::Le,
            BinaryOperator::Greater => Instruction::Gt,
            BinaryOperator::GreaterEqual => Instruction::Ge,
            BinaryOperator::And => Instruction::And,
            BinaryOperator::Or => Instruction::Or,
            BinaryOperator::BitAnd => Instruction::BitAnd,
            BinaryOperator::BitOr => Instruction::BitOr,
            BinaryOperator::BitXor => Instruction::BitXor,
            BinaryOperator::LeftShift => Instruction::Shl,
            BinaryOperator::RightShift => Instruction::Shr,
        };
        
        self.emit(instruction);
        Ok(())
    }
    
    /// Generate code for a unary expression
    fn generate_unary_expression(&mut self, unary_expr: &UnaryExpression) -> Result<()> {
        // Generate operand
        self.generate_expression(&unary_expr.operand)?;
        
        // Generate operation
        let instruction = match unary_expr.operator {
            UnaryOperator::Not => Instruction::Not,
            UnaryOperator::Minus => {
                // Negate by subtracting from zero
                self.emit(Instruction::Push(Value::U32(0)));
                self.emit(Instruction::Swap);
                Instruction::Sub
            }
            UnaryOperator::BitNot => Instruction::BitNot,
        };
        
        self.emit(instruction);
        Ok(())
    }
    
    /// Generate code for a function call
    fn generate_call_expression(&mut self, call_expr: &CallExpression) -> Result<()> {
        // Generate arguments
        for arg in &call_expr.arguments {
            self.generate_expression(arg)?;
        }
        
        // Generate function call
        match call_expr.function.as_ref() {
            Expression::Identifier(identifier) => {
                if let Some(&function_offset) = self.functions.get(&identifier.name) {
                    self.emit(Instruction::Call(function_offset));
                } else {
                    // Could be a built-in function or contract method
                    self.emit(Instruction::Invoke(identifier.name.clone()));
                }
            }
            Expression::FieldAccess(field_access) => {
                // Method call
                self.generate_expression(&field_access.object)?;
                self.emit(Instruction::Invoke(field_access.field.name.clone()));
            }
            _ => {
                return Err(CodegenError::new(
                    CodegenErrorKind::InvalidFunctionCall,
                    call_expr.location.clone(),
                    "Invalid function call expression".to_string(),
                ).into());
            }
        }
        
        Ok(())
    }
    
    /// Generate code for field access
    fn generate_field_access(&mut self, field_expr: &FieldAccessExpression) -> Result<()> {
        // Generate object
        self.generate_expression(&field_expr.object)?;
        
        // TODO: Look up field index
        let field_index = 0; // Placeholder
        
        self.emit(Instruction::LoadField(field_index));
        Ok(())
    }
    
    /// Generate code for index expression
    fn generate_index_expression(&mut self, index_expr: &IndexExpression) -> Result<()> {
        // Generate object
        self.generate_expression(&index_expr.object)?;
        
        // Generate index
        self.generate_expression(&index_expr.index)?;
        
        // TODO: Implement array indexing instruction
        self.emit(Instruction::Load(0)); // Placeholder
        Ok(())
    }
    
    /// Generate code for array expression
    fn generate_array_expression(&mut self, array_expr: &ArrayExpression) -> Result<()> {
        // Generate elements
        for element in &array_expr.elements {
            self.generate_expression(element)?;
        }
        
        // Create array
        let array_size = array_expr.elements.len() as u32;
        self.emit(Instruction::Push(Value::U32(array_size)));
        
        // TODO: Implement array creation instruction
        Ok(())
    }
    
    /// Generate code for tuple expression
    fn generate_tuple_expression(&mut self, tuple_expr: &TupleExpression) -> Result<()> {
        // Generate elements
        for element in &tuple_expr.elements {
            self.generate_expression(element)?;
        }
        
        // Create tuple
        let tuple_size = tuple_expr.elements.len() as u32;
        self.emit(Instruction::Push(Value::U32(tuple_size)));
        
        // TODO: Implement tuple creation instruction
        Ok(())
    }
    
    /// Generate code for struct expression
    fn generate_struct_expression(&mut self, _struct_expr: &StructExpression) -> Result<()> {
        // TODO: Implement struct expression code generation
        Ok(())
    }
    
    /// Generate code for assignment expression
    fn generate_assignment_expression(&mut self, assign_expr: &AssignmentExpression) -> Result<()> {
        // Generate value
        self.generate_expression(&assign_expr.value)?;
        
        // Generate assignment target
        match assign_expr.target.as_ref() {
            Expression::Identifier(identifier) => {
                if let Some(local) = self.context.get_local(&identifier.name) {
                    if !local.mutable {
                        return Err(CodegenError::new(
                            CodegenErrorKind::ImmutableAssignment,
                            assign_expr.location.clone(),
                            format!("Cannot assign to immutable variable '{}'", identifier.name),
                        ).into());
                    }
                    self.emit(Instruction::Store(local.index));
                } else {
                    // Could be a contract field
                    self.emit(Instruction::StoreField(0)); // Placeholder
                }
            }
            Expression::FieldAccess(field_access) => {
                // Field assignment
                self.generate_expression(&field_access.object)?;
                // TODO: Look up field index
                let field_index = 0; // Placeholder
                self.emit(Instruction::StoreField(field_index));
            }
            Expression::Index(index_expr) => {
                // Array element assignment
                self.generate_expression(&index_expr.object)?;
                self.generate_expression(&index_expr.index)?;
                // TODO: Implement array element assignment
            }
            _ => {
                return Err(CodegenError::new(
                    CodegenErrorKind::InvalidAssignmentTarget,
                    assign_expr.location.clone(),
                    "Invalid assignment target".to_string(),
                ).into());
            }
        }
        
        Ok(())
    }
    
    /// Generate code for range expression
    fn generate_range_expression(&mut self, _range_expr: &RangeExpression) -> Result<()> {
        // TODO: Implement range expression code generation
        Ok(())
    }
    
    /// Generate code for closure expression
    fn generate_closure_expression(&mut self, _closure_expr: &ClosureExpression) -> Result<()> {
        // TODO: Implement closure expression code generation
        Ok(())
    }
    
    /// Generate code for ML create model expression
    fn generate_ml_create_model(&mut self, ml_create: &MLCreateModelExpression) -> Result<()> {
        // Generate model configuration
        for (key, value) in &ml_create.config {
            self.generate_expression(value)?;
            // Store config value with key
            self.emit(Instruction::Push(Value::String(key.clone())));
        }
        
        // Create the model
        self.emit(Instruction::MLCreateModel(ml_create.model_type.clone()));
        
        Ok(())
    }
    
    /// Generate code for ML train expression
    fn generate_ml_train(&mut self, ml_train: &MLTrainExpression) -> Result<()> {
        // Generate model reference
        self.generate_expression(&ml_train.model)?;
        
        // Generate dataset
        self.generate_expression(&ml_train.dataset)?;
        
        // Generate epochs if provided
        if let Some(epochs) = &ml_train.epochs {
            self.generate_expression(epochs)?;
        } else {
            // Default epochs
            self.emit(Instruction::Push(Value::U32(100)));
        }
        
        // Execute training
        self.emit(Instruction::MLTrain(0)); // Model ID will be resolved at runtime
        
        Ok(())
    }
    
    /// Generate code for ML predict expression
    fn generate_ml_predict(&mut self, ml_predict: &MLPredictExpression) -> Result<()> {
        // Generate model reference
        self.generate_expression(&ml_predict.model)?;
        
        // Generate input data
        self.generate_expression(&ml_predict.input)?;
        
        // Execute prediction
        self.emit(Instruction::MLPredict(0)); // Model ID will be resolved at runtime
        
        Ok(())
    }
    
    /// Generate code for ML forward expression
    fn generate_ml_forward(&mut self, ml_forward: &MLForwardExpression) -> Result<()> {
        // Generate model reference
        self.generate_expression(&ml_forward.model)?;
        
        // Generate input data
        self.generate_expression(&ml_forward.input)?;
        
        // Execute forward pass
        self.emit(Instruction::MLForward(0)); // Model ID will be resolved at runtime
        
        Ok(())
    }
    
    /// Generate code for ML backward expression
    fn generate_ml_backward(&mut self, ml_backward: &MLBackwardExpression) -> Result<()> {
        // Generate model reference
        self.generate_expression(&ml_backward.model)?;
        
        // Generate gradients
        self.generate_expression(&ml_backward.gradients)?;
        
        // Execute backward pass
        self.emit(Instruction::MLBackward(0)); // Model ID will be resolved at runtime
        
        Ok(())
    }
    
    /// Generate code for tensor operations
    fn generate_tensor_op(&mut self, tensor_op: &TensorOpExpression) -> Result<()> {
        // Generate operands
        for operand in &tensor_op.operands {
            self.generate_expression(operand)?;
        }
        
        // Generate tensor operation based on type
        match tensor_op.operation {
            TensorOperation::Add => {
                self.emit(Instruction::Add);
            }
            TensorOperation::Subtract => {
                self.emit(Instruction::Sub);
            }
            TensorOperation::Multiply => {
                self.emit(Instruction::Mul);
            }
            TensorOperation::Divide => {
                self.emit(Instruction::Div);
            }
            TensorOperation::MatMul => {
                // Matrix multiplication - custom instruction needed
                self.emit(Instruction::Push(Value::String("matmul".to_string())));
                self.emit(Instruction::MLActivation("matmul".to_string()));
            }
            TensorOperation::Transpose => {
                self.emit(Instruction::Push(Value::String("transpose".to_string())));
                self.emit(Instruction::MLActivation("transpose".to_string()));
            }
            TensorOperation::Reshape => {
                self.emit(Instruction::Push(Value::String("reshape".to_string())));
                self.emit(Instruction::MLActivation("reshape".to_string()));
            }
            TensorOperation::Sum => {
                self.emit(Instruction::Push(Value::String("sum".to_string())));
                self.emit(Instruction::MLActivation("sum".to_string()));
            }
            TensorOperation::Mean => {
                self.emit(Instruction::Push(Value::String("mean".to_string())));
                self.emit(Instruction::MLActivation("mean".to_string()));
            }
            TensorOperation::Max => {
                self.emit(Instruction::Push(Value::String("max".to_string())));
                self.emit(Instruction::MLActivation("max".to_string()));
            }
            TensorOperation::Min => {
                self.emit(Instruction::Push(Value::String("min".to_string())));
                self.emit(Instruction::MLActivation("min".to_string()));
            }
        }
        
        Ok(())
    }
    
    /// Generate code for matrix operations
    fn generate_matrix_op(&mut self, matrix_op: &MatrixOpExpression) -> Result<()> {
        // Generate operands
        for operand in &matrix_op.operands {
            self.generate_expression(operand)?;
        }
        
        // Generate matrix operation based on type
        match matrix_op.operation {
            MatrixOperation::Add => {
                self.emit(Instruction::Add);
            }
            MatrixOperation::Subtract => {
                self.emit(Instruction::Sub);
            }
            MatrixOperation::Multiply => {
                self.emit(Instruction::Mul);
            }
            MatrixOperation::Transpose => {
                self.emit(Instruction::Push(Value::String("transpose".to_string())));
                self.emit(Instruction::MLActivation("transpose".to_string()));
            }
            MatrixOperation::Inverse => {
                self.emit(Instruction::Push(Value::String("inverse".to_string())));
                self.emit(Instruction::MLActivation("inverse".to_string()));
            }
            MatrixOperation::Determinant => {
                self.emit(Instruction::Push(Value::String("determinant".to_string())));
                self.emit(Instruction::MLActivation("determinant".to_string()));
            }
            MatrixOperation::Eigenvalues => {
                self.emit(Instruction::Push(Value::String("eigenvalues".to_string())));
                self.emit(Instruction::MLActivation("eigenvalues".to_string()));
            }
        }
        
        Ok(())
    }
    
    /// Emit an instruction
    fn emit(&mut self, instruction: Instruction) {
        self.instructions.push(instruction);
    }
    
    pub fn get_instructions(&self) -> Vec<Instruction> {
        self.instructions.clone()
    }
    /// Place a label at the current instruction position
    fn place_label(&mut self, label: u32) {
        self.labels.insert(label, self.instructions.len() as u32);
    }
    
    /// Resolve all labels to instruction offsets
    fn resolve_labels(&mut self) {
        for instruction in &mut self.instructions {
            match instruction {
                Instruction::Jump(ref mut label) |
                Instruction::JumpIf(ref mut label) |
                Instruction::JumpIfNot(ref mut label) => {
                    if let Some(&offset) = self.labels.get(label) {
                        *label = offset;
                    }
                }
                _ => {}
            }
        }
    }
    
    /// Get default value for a type
    fn default_value_for_type(&self, type_annotation: &Type) -> Result<Value> {
        let value = match type_annotation {
            Type::U8 => Value::U8(0),
            Type::U16 => Value::U16(0),
            Type::U32 => Value::U32(0),
            Type::U64 => Value::U64(0),
            Type::U128 => Value::U128(0),
            Type::U256 => Value::U256([0; 32]),
            Type::I8 => Value::I8(0),
            Type::I16 => Value::I16(0),
            Type::I32 => Value::I32(0),
            Type::I64 => Value::I64(0),
            Type::I128 => Value::I128(0),
            Type::I256 => Value::I256([0; 32]),
            Type::F32 => Value::F32(0.0),
            Type::F64 => Value::F64(0.0),
            Type::Bool => Value::Bool(false),
            Type::String => Value::String(String::new()),
            Type::Address => Value::Address([0; 20]),
            Type::Array { .. } => Value::Array(Vec::new()),
            Type::Tuple(types) => {
                let mut elements = Vec::new();
                for t in types {
                    elements.push(self.default_value_for_type(t)?);
                }
                Value::Tuple(elements)
            }
            Type::MLModel { .. } => Value::MLModel {
                model_id: 0,
                model_type: "default".to_string(),
                weights: vec![],
                biases: vec![],
                hyperparams: std::collections::HashMap::new(),
                architecture: vec![],
                version: 1,
                checksum: "default".to_string(),
            },
            Type::MLDataset { .. } => Value::MLDataset {
                features: vec![],
                targets: vec![],
                normalized: false,
                metadata: std::collections::HashMap::new(),
                split_info: None,
            },
            Type::MLMetrics => Value::MLMetrics {
                accuracy: 0.0,
                loss: 0.0,
                precision: 0.0,
                recall: 0.0,
                f1_score: 0.0,
                auc_roc: 0.0,
                confusion_matrix: vec![],
            },
            Type::Tensor { .. } => Value::Tensor { data: vec![], shape: vec![], dtype: "f64".to_string() },
            Type::Matrix { .. } => Value::Matrix(vec![]),
            Type::Vector { .. } => Value::Vector(vec![]),
            _ => Value::Null,
        };
        
        Ok(value)
    }
    
    /// Evaluate a constant expression at compile time
    fn evaluate_const_expression(&self, _expression: &Expression) -> Result<Value> {
        // TODO: Implement constant expression evaluation
        Ok(Value::U32(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::semantic::SemanticAnalyzer;
    
    fn compile_source(source: &str) -> Result<Bytecode> {
        let mut lexer = Lexer::new(source, "test.aug");
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(tokens);
        let ast = parser.parse()?;
        let mut analyzer = SemanticAnalyzer::new();
        analyzer.analyze(&ast)?;
        let mut codegen = CodeGenerator::new();
        codegen.generate(&ast)
    }
    
    #[test]
    fn test_simple_function_codegen() {
        let source = r#"
            fn add(a: u32, b: u32) -> u32 {
                return a + b;
            }
        "#;
        
        let result = compile_source(source);
        assert!(result.is_ok());
        
        let bytecode = result.unwrap();
        assert!(!bytecode.instructions.is_empty());
        assert!(bytecode.functions.contains_key("add"));
    }
    
    #[test]
    fn test_contract_codegen() {
        let source = r#"
            contract SimpleToken {
                let balance: u256;
                
                fn get_balance() -> u256 {
                    return self.balance;
                }
            }
        "#;
        
        let result = compile_source(source);
        assert!(result.is_ok());
        
        let bytecode = result.unwrap();
        assert!(bytecode.contracts.contains_key("SimpleToken"));
    }
    
    #[test]
    fn test_control_flow_codegen() {
        let source = r#"
            fn test_if(x: u32) -> u32 {
                if x > 10 {
                    return x * 2;
                } else {
                    return x + 1;
                }
            }
        "#;
        
        let result = compile_source(source);
        assert!(result.is_ok());
        
        let bytecode = result.unwrap();
        assert!(!bytecode.instructions.is_empty());
    }
    
    #[test]
    fn test_loop_codegen() {
        let source = r#"
            fn test_while() {
                let mut i = 0;
                while i < 10 {
                    i = i + 1;
                }
            }
        "#;
        
        let result = compile_source(source);
        assert!(result.is_ok());
        
        let bytecode = result.unwrap();
        assert!(!bytecode.instructions.is_empty());
    }
}
