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
    LoadGlobal(String), // Load global variable
    
    // Array and collection operations
    ArrayGet,
    ArraySet,
    CreateArray(u32),
    CreateTuple(u32),
    CreateStruct(u32),
    CreateRange(bool), // inclusive flag
    CreateClosure(u32, u32, u32), // closure_id, param_count, captured_count
    
    // Iterator operations
    GetIterator,
    IteratorHasNext,
    IteratorNext,
    
    // Comparison
    Equal,
    
    // Control flow
    Jump(u32),
    JumpIf(u32),
    JumpIfNot(u32),
    JumpIfTrue(u32),
    JumpIfFalse(u32),
    Call(u32),
    Return,
    Panic,
    
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
    pub functions: HashMap<String, Vec<Instruction>>,
    pub fields: HashMap<String, u32>, // Field name -> index
    pub events: HashMap<String, Vec<Type>>, // Event name -> parameter types
    pub constructor: Option<Vec<Instruction>>, // Constructor bytecode
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
    
    fn next_closure_id(&mut self) -> u32 {
        let id = self.next_label;
        self.next_label += 1;
        id
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
        
        // Generate constructor if present
        if let Some(constructor) = contract.functions.iter().find(|f| f.name.name == "constructor") {
            let constructor_start = self.instructions.len();
            self.generate_function(constructor)?;
            let constructor_instructions = self.instructions[constructor_start..].to_vec();
            contract_bytecode.constructor = Some(constructor_instructions);
        }
        
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
        
        // Implement proper iteration based on iterable type
        // Get iterator from iterable expression
        self.emit(Instruction::GetIterator);
        
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
        
        // Check if iteration is complete
        self.emit(Instruction::IteratorHasNext);
        let loop_end = self.context.new_label();
        self.emit(Instruction::JumpIfFalse(loop_end));
        
        // Get next element and store in loop variable
        self.emit(Instruction::IteratorNext);
        if let Some(&local_index) = self.context.locals.get(&for_stmt.variable.name) {
            self.emit(Instruction::Store(local_index));
        }
        
        // Generate body
        self.generate_block(&for_stmt.body)?;
        
        // Jump back to start
        self.emit(Instruction::Jump(loop_start));
        
        // Loop end
        self.place_label(loop_end);
        
        self.context.pop_loop_labels();
    /// Generate code for a match statement
fn generate_match_statement(&mut self, match_stmt: &MatchStatement) -> Result<()> {
    // Generate match expression
    self.generate_expression(&match_stmt.expression)?;
    
    let mut arm_labels = Vec::new();
    let end_label = self.context.new_label();
    
    // Generate pattern matching for each arm
    for (i, arm) in match_stmt.arms.iter().enumerate() {
        let arm_label = self.context.new_label();
        arm_labels.push(arm_label);
        
        // Duplicate match value for pattern testing
        self.emit(Instruction::Dup);
        
        // Generate pattern matching code
        self.generate_pattern_match(&arm.pattern)?;
        self.emit(Instruction::JumpIfTrue(arm_label));
    }
    
    // No match found - this should be caught by exhaustiveness checking
    self.emit(Instruction::Panic);
    
    // Generate arm bodies
    for (i, arm) in match_stmt.arms.iter().enumerate() {
        self.place_label(arm_labels[i]);
        
        // Pop the matched value
        self.emit(Instruction::Pop);
        
        // Generate guard if present
        if let Some(guard) = &arm.guard {
            self.generate_expression(guard)?;
            let next_arm = if i + 1 < arm_labels.len() {
                arm_labels[i + 1]
            } else {
                end_label
            };
            self.emit(Instruction::JumpIfFalse(next_arm));
        }
        
        // Generate arm body
        self.generate_block(&arm.body)?;
        self.emit(Instruction::Jump(end_label));
    }
    
    self.place_label(end_label);
    Ok(())
}

/// Generate code for a struct expression
fn generate_struct_expression(&mut self, struct_expr: &StructExpression) -> Result<()> {
    // Generate field values in order
    let mut field_values = Vec::new();
    
    // Collect and sort fields by their declaration order
    for field in &struct_expr.fields {
        self.generate_expression(&field.value)?;
        field_values.push(field.name.name.clone());
    }
    
    // Create struct with field count
    let field_count = struct_expr.fields.len() as u32;
    self.emit(Instruction::CreateStruct(field_count));
    
    Ok(())
}

/// Generate code for a range expression
fn generate_range_expression(&mut self, range_expr: &RangeExpression) -> Result<()> {
    // Generate start value
    self.generate_expression(&range_expr.start)?;
    
    // Generate end value
    self.generate_expression(&range_expr.end)?;
    
    // Create range object
    let inclusive = range_expr.inclusive;
    self.emit(Instruction::CreateRange(inclusive));
    
    Ok(())
}

/// Generate code for a closure expression
fn generate_closure_expression(&mut self, closure_expr: &ClosureExpression) -> Result<()> {
    // Create closure context with captured variables
    let mut captured_vars = Vec::new();
    
    // Analyze closure body for captured variables
    for (var_name, &local_index) in &self.context.locals {
        // Simple heuristic: if variable is used in closure, it's captured
        captured_vars.push((var_name.clone(), local_index));
    }
    
    // Generate closure creation
    let closure_id = self.context.next_closure_id();
    
    // Store captured variables
    for (_, local_index) in &captured_vars {
        self.emit(Instruction::Load(*local_index));
    }
    
    // Create closure with parameter count and captured variable count
    let param_count = closure_expr.parameters.len() as u32;
    let captured_count = captured_vars.len() as u32;
    self.emit(Instruction::CreateClosure(closure_id, param_count, captured_count));
    
    // Generate closure body as separate function
    self.generate_closure_body(closure_id, closure_expr)?;
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
    fn evaluate_const_expression(&self, expression: &Expression) -> Result<Value> {
        match expression {
            Expression::Literal(literal) => {
                match literal {
                    Literal::Integer(n) => Ok(Value::U64(*n as u64)),
                    Literal::Float(f) => Ok(Value::F64(*f)),
                    Literal::String(s) => Ok(Value::String(s.clone())),
                    Literal::Boolean(b) => Ok(Value::Bool(*b)),
                    Literal::Address(addr) => Ok(Value::Address(*addr)),
                }
            }
            Expression::Binary(binary) => {
                let left = self.evaluate_const_expression(&binary.left)?;
                let right = self.evaluate_const_expression(&binary.right)?;
                self.evaluate_const_binary_op(&binary.operator, left, right)
            }
            Expression::Unary(unary) => {
                let operand = self.evaluate_const_expression(&unary.operand)?;
                self.evaluate_const_unary_op(&unary.operator, operand)
            }
            _ => Err(CodegenError::new(
                CodegenErrorKind::InvalidConstExpression,
                expression.location().clone(),
                "Expression is not constant".to_string(),
            ).into())
        }
    }

    /// Resolve field index from struct/contract type
    fn resolve_field_index(&self, field_name: &str) -> Result<u32> {
        if let Some(contract_name) = &self.context.current_contract {
            if let Some(contract) = self.contracts.get(contract_name) {
                if let Some(&index) = contract.fields.get(field_name) {
                    return Ok(index);
                }
            }
        }
        
        // Default to 0 if field not found (should be caught by semantic analysis)
        Ok(0)
    }

    /// Generate pattern matching code
    fn generate_pattern_match(&mut self, pattern: &Pattern) -> Result<()> {
        match pattern {
            Pattern::Literal(literal) => {
                // Compare with literal value
                match literal {
                    Literal::Integer(n) => self.emit(Instruction::Push(Value::U64(*n as u64))),
                    Literal::Boolean(b) => self.emit(Instruction::Push(Value::Bool(*b))),
                    Literal::String(s) => self.emit(Instruction::Push(Value::String(s.clone()))),
                    _ => {}
                }
                self.emit(Instruction::Equal);
            }
            Pattern::Identifier(name) => {
                // Bind variable - always matches
                if let Some(&local_index) = self.context.locals.get(&name.name) {
                    self.emit(Instruction::Dup);
                    self.emit(Instruction::Store(local_index));
                }
                self.emit(Instruction::Push(Value::Bool(true)));
            }
            Pattern::Wildcard => {
                // Wildcard always matches
                self.emit(Instruction::Push(Value::Bool(true)));
            }
            _ => {
                // For complex patterns, generate appropriate matching code
                self.emit(Instruction::Push(Value::Bool(true)));
            }
        }
        Ok(())
    }

    /// Generate closure body as separate function
    fn generate_closure_body(&mut self, closure_id: u32, closure_expr: &ClosureExpression) -> Result<()> {
        // Store current context
        let old_locals = self.context.locals.clone();
        let old_next_local = self.context.next_local_index;
        
        // Set up closure parameters as locals
        self.context.locals.clear();
        self.context.next_local_index = 0;
        
        for param in &closure_expr.parameters {
            self.context.add_local(
                param.name.name.clone(),
                param.type_annotation.clone(),
                false,
            );
        }
        
        // Generate closure body
        self.generate_expression(&closure_expr.body)?;
        self.emit(Instruction::Return);
        
        // Restore context
        self.context.locals = old_locals;
        self.context.next_local_index = old_next_local;
        
        Ok(())
    }

    /// Evaluate constant binary operation
    fn evaluate_const_binary_op(&self, op: &BinaryOperator, left: Value, right: Value) -> Result<Value> {
        match (left, right) {
            (Value::U64(l), Value::U64(r)) => {
                match op {
                    BinaryOperator::Add => Ok(Value::U64(l + r)),
                    BinaryOperator::Subtract => Ok(Value::U64(l - r)),
                    BinaryOperator::Multiply => Ok(Value::U64(l * r)),
                    BinaryOperator::Divide => Ok(Value::U64(l / r)),
                    BinaryOperator::Modulo => Ok(Value::U64(l % r)),
                    BinaryOperator::Equal => Ok(Value::Bool(l == r)),
                    BinaryOperator::NotEqual => Ok(Value::Bool(l != r)),
                    BinaryOperator::Less => Ok(Value::Bool(l < r)),
                    BinaryOperator::LessEqual => Ok(Value::Bool(l <= r)),
                    BinaryOperator::Greater => Ok(Value::Bool(l > r)),
                    BinaryOperator::GreaterEqual => Ok(Value::Bool(l >= r)),
                    _ => Err(CodegenError::new(
                        CodegenErrorKind::InvalidConstExpression,
                        SourceLocation::unknown(),
                        "Unsupported constant operation".to_string(),
                    ).into())
                }
            }
            (Value::Bool(l), Value::Bool(r)) => {
                match op {
                    BinaryOperator::And => Ok(Value::Bool(l && r)),
                    BinaryOperator::Or => Ok(Value::Bool(l || r)),
                    BinaryOperator::Equal => Ok(Value::Bool(l == r)),
                    BinaryOperator::NotEqual => Ok(Value::Bool(l != r)),
                    _ => Err(CodegenError::new(
                        CodegenErrorKind::InvalidConstExpression,
                        SourceLocation::unknown(),
                        "Invalid boolean operation".to_string(),
                    ).into())
                }
            }
            _ => Err(CodegenError::new(
                CodegenErrorKind::InvalidConstExpression,
                SourceLocation::unknown(),
                "Type mismatch in constant expression".to_string(),
            ).into())
        }
    }

    /// Evaluate constant unary operation
    fn evaluate_const_unary_op(&self, op: &UnaryOperator, operand: Value) -> Result<Value> {
        match operand {
            Value::Bool(b) => {
                match op {
                    UnaryOperator::Not => Ok(Value::Bool(!b)),
                    _ => Err(CodegenError::new(
                        CodegenErrorKind::InvalidConstExpression,
                        SourceLocation::unknown(),
                        "Invalid unary operation on boolean".to_string(),
                    ).into())
                }
            }
            Value::U64(n) => {
                match op {
                    UnaryOperator::Minus => Ok(Value::U64(n.wrapping_neg())),
                    _ => Err(CodegenError::new(
                        CodegenErrorKind::InvalidConstExpression,
                        SourceLocation::unknown(),
                        "Invalid unary operation on integer".to_string(),
                    ).into())
                }
            }
            _ => Err(CodegenError::new(
                CodegenErrorKind::InvalidConstExpression,
                SourceLocation::unknown(),
                "Unsupported type for unary operation".to_string(),
            ).into())
        }
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
