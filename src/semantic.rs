// Semantic analyzer - checks types, scopes, and other logical stuff
// Runs after parsing to make sure the code makes sense

use crate::ast::*;
use crate::error::{Result, SemanticError, SemanticErrorKind, SourceLocation};
use std::collections::{HashMap, HashSet};

// Info about each variable/function we've seen
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Symbol {
    pub name: String,
    pub symbol_type: SymbolType,
    pub location: SourceLocation,
    pub mutable: bool,
}

/// Types of symbols in the symbol table
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum SymbolType {
    Variable(Type),
    Function {
        parameters: Vec<Type>,
        return_type: Option<Type>,
    },
    Contract {
        fields: HashMap<String, Type>,
        functions: HashMap<String, (Vec<Type>, Option<Type>)>,
    },
    Struct {
        fields: HashMap<String, Type>,
    },
    Enum {
        variants: HashMap<String, Option<Vec<Type>>>,
    },
    Trait {
        functions: HashMap<String, (Vec<Type>, Option<Type>)>,
    },
    Module,
    Constant(Type),
}

/// Scope for symbol resolution
#[derive(Debug, Clone)]
pub struct Scope {
    symbols: HashMap<String, Symbol>,
    parent: Option<Box<Scope>>,
}

impl Scope {
    pub fn new() -> Self {
        Self {
            symbols: HashMap::new(),
            parent: None,
        }
    }
    
    pub fn with_parent(parent: Scope) -> Self {
        Self {
            symbols: HashMap::new(),
            parent: Some(Box::new(parent)),
        }
    }
    
    pub fn define(&mut self, name: String, symbol: Symbol) -> Result<()> {
        if self.symbols.contains_key(&name) {
            return Err(SemanticError::new(
                SemanticErrorKind::DuplicateSymbol(name.clone()),
                symbol.location.clone(),
                format!("Symbol '{}' is already defined in this scope", name),
            ).into());
        }
        
        self.symbols.insert(name, symbol);
        Ok(())
    }
    
    pub fn lookup(&self, name: &str) -> Option<&Symbol> {
        if let Some(symbol) = self.symbols.get(name) {
            Some(symbol)
        } else if let Some(parent) = &self.parent {
            parent.lookup(name)
        } else {
            None
        }
    }
    
    pub fn lookup_local(&self, name: &str) -> Option<&Symbol> {
        self.symbols.get(name)
    }
}

/// Semantic analyzer
pub struct SemanticAnalyzer {
    current_scope: Scope,
    current_contract: Option<String>,
    current_function: Option<String>,
    return_type: Option<Type>,
    safety_issues: Vec<SafetyIssue>,
    call_graph: HashMap<String, Vec<String>>,
    external_calls: HashSet<String>,
    state_modifications: HashSet<String>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SafetyIssue {
    pub severity: SafetySeverity,
    pub issue_type: SafetyIssueType,
    pub location: SourceLocation,
    pub message: String,
    pub function: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum SafetySeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum SafetyIssueType {
    PotentialReentrancy,
    ArithmeticOverflow,
    UnauthorizedAccess,
    InvalidStateTransition,
    UncheckedExternalCall,
    IntegerUnderflow,
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        let mut global_scope = Scope::new();
        
        // Add built-in types and functions
        Self::add_builtins(&mut global_scope);
        
        Self {
            current_scope: global_scope,
            current_contract: None,
            current_function: None,
            return_type: None,
            safety_issues: Vec::new(),
            call_graph: HashMap::new(),
            external_calls: HashSet::new(),
            state_modifications: HashSet::new(),
        }
    }
    
    /// Add built-in types and functions to the global scope
    fn add_builtins(scope: &mut Scope) {
        use crate::ast::Type;
        use std::collections::HashMap;
        
        // Create msg object with blockchain message context
        let mut msg_fields = HashMap::new();
        msg_fields.insert("sender".to_string(), Type::Address);
        msg_fields.insert("value".to_string(), Type::U256);
        msg_fields.insert("data".to_string(), Type::Bytes);
        msg_fields.insert("gas".to_string(), Type::U256);
        
        let msg_symbol = Symbol {
            name: "msg".to_string(),
            symbol_type: SymbolType::Struct { fields: msg_fields },
            location: SourceLocation { line: 0, column: 0, file: "<builtin>".to_string(), offset: 0 },
            mutable: false,
        };
        scope.define("msg".to_string(), msg_symbol).unwrap();
        
        // Create block object with blockchain block context
        let mut block_fields = HashMap::new();
        block_fields.insert("number".to_string(), Type::U256);
        block_fields.insert("timestamp".to_string(), Type::U256);
        block_fields.insert("difficulty".to_string(), Type::U256);
        block_fields.insert("gas_limit".to_string(), Type::U256);
        block_fields.insert("coinbase".to_string(), Type::Address);
        
        let block_symbol = Symbol {
            name: "block".to_string(),
            symbol_type: SymbolType::Struct { fields: block_fields },
            location: SourceLocation { line: 0, column: 0, file: "<builtin>".to_string(), offset: 0 },
            mutable: false,
        };
        scope.define("block".to_string(), block_symbol).unwrap();
        
        // Create tx object with transaction context
        let mut tx_fields = HashMap::new();
        tx_fields.insert("origin".to_string(), Type::Address);
        tx_fields.insert("gas_price".to_string(), Type::U256);
        
        let tx_symbol = Symbol {
            name: "tx".to_string(),
            symbol_type: SymbolType::Struct { fields: tx_fields },
            location: SourceLocation { line: 0, column: 0, file: "<builtin>".to_string(), offset: 0 },
            mutable: false,
        };
        scope.define("tx".to_string(), tx_symbol).unwrap();
        
        // Add built-in functions
        let require_symbol = Symbol {
            name: "require".to_string(),
            symbol_type: SymbolType::Function {
                parameters: vec![Type::Bool],
                return_type: None, // Unit type
            },
            location: SourceLocation { line: 0, column: 0, file: "<builtin>".to_string(), offset: 0 },
            mutable: false,
        };
        scope.define("require".to_string(), require_symbol).unwrap();
        
        let assert_symbol = Symbol {
            name: "assert".to_string(),
            symbol_type: SymbolType::Function {
                parameters: vec![Type::Bool],
                return_type: None, // Unit type
            },
            location: SourceLocation { line: 0, column: 0, file: "<builtin>".to_string(), offset: 0 },
            mutable: false,
        };
        scope.define("assert".to_string(), assert_symbol).unwrap();
    }
    
    /// Analyze a source file
    pub fn analyze(&mut self, source_file: &SourceFile) -> Result<()> {
        for item in &source_file.items {
            self.analyze_item(item)?;
        }
        Ok(())
    }
    
    /// Analyze a top-level item
    fn analyze_item(&mut self, item: &Item) -> Result<()> {
        match item {
            Item::Contract(contract) => self.analyze_contract(contract),
            Item::Function(function) => self.analyze_function(function),
            Item::Struct(struct_def) => self.analyze_struct(struct_def),
            Item::Enum(enum_def) => self.analyze_enum(enum_def),
            Item::Trait(trait_def) => self.analyze_trait(trait_def),
            Item::Impl(impl_block) => self.analyze_impl(impl_block),
            Item::Use(use_decl) => self.analyze_use(use_decl),
            Item::Const(const_decl) => self.analyze_const(const_decl),
            Item::Module(module) => self.analyze_module(module),
        }
    }
    
    /// Analyze a contract
    fn analyze_contract(&mut self, contract: &Contract) -> Result<()> {
        // Check for duplicate contract names
        if self.current_scope.lookup_local(&contract.name.name).is_some() {
            return Err(SemanticError::new(
                SemanticErrorKind::DuplicateSymbol(contract.name.name.clone()),
                contract.location.clone(),
                format!("Contract '{}' is already defined", contract.name.name),
            ).into());
        }
        
        // Create contract symbol
        let mut fields = HashMap::new();
        let mut functions = HashMap::new();
        
        // Analyze fields
        for field in &contract.fields {
            if fields.contains_key(&field.name.name) {
                return Err(SemanticError::new(
                    SemanticErrorKind::DuplicateField(field.name.name.clone()),
                    field.location.clone(),
                    format!("Field '{}' is already defined in contract", field.name.name),
                ).into());
            }
            
            self.validate_type(&field.type_annotation)?;
            fields.insert(field.name.name.clone(), field.type_annotation.clone());
        }
        
        // Analyze functions
        for function in &contract.functions {
            if functions.contains_key(&function.name.name) {
                return Err(SemanticError::new(
                    SemanticErrorKind::DuplicateFunction(function.name.name.clone()),
                    function.location.clone(),
                    format!("Function '{}' is already defined in contract", function.name.name),
                ).into());
            }
            
            let param_types: Vec<Type> = function.parameters.iter()
                .map(|p| p.type_annotation.clone())
                .collect();
            
            functions.insert(
                function.name.name.clone(),
                (param_types, function.return_type.clone()),
            );
        }
        
        // Add contract to scope
        let contract_symbol = Symbol {
            name: contract.name.name.clone(),
            symbol_type: SymbolType::Contract { fields, functions },
            location: contract.location.clone(),
            mutable: false,
        };
        
        self.current_scope.define(contract.name.name.clone(), contract_symbol)?;
        
        // Analyze contract body in new scope
        let old_contract = self.current_contract.clone();
        self.current_contract = Some(contract.name.name.clone());
        
        let contract_scope = Scope::with_parent(self.current_scope.clone());
        let old_scope = std::mem::replace(&mut self.current_scope, contract_scope);
        
        // Add fields to contract scope
        for field in &contract.fields {
            let field_symbol = Symbol {
                name: field.name.name.clone(),
                symbol_type: SymbolType::Variable(field.type_annotation.clone()),
                location: field.location.clone(),
                mutable: true, // Contract fields are mutable by default
            };
            self.current_scope.define(field.name.name.clone(), field_symbol)?;
        }
        
        // Analyze functions
        for function in &contract.functions {
            self.analyze_function(function)?;
        }
        
        // Restore scope and context
        self.current_scope = old_scope;
        self.current_contract = old_contract;
        
        Ok(())
    }
    
    /// Analyze a function
    fn analyze_function(&mut self, function: &Function) -> Result<()> {
        // Validate parameter types
        for param in &function.parameters {
            self.validate_type(&param.type_annotation)?;
        }
        
        // Validate return type
        if let Some(return_type) = &function.return_type {
            self.validate_type(return_type)?;
        }
        
        // Create function scope
        let function_scope = Scope::with_parent(self.current_scope.clone());
        let old_scope = std::mem::replace(&mut self.current_scope, function_scope);
        let old_function = self.current_function.clone();
        let old_return_type = self.return_type.clone();
        
        self.current_function = Some(function.name.name.clone());
        self.return_type = function.return_type.clone();
        
        // Add 'self' parameter if we're in a contract context
        if let Some(contract_name) = &self.current_contract {
            // Always add 'self' to scope when in a contract context
            let self_symbol = Symbol {
                 name: "self".to_string(),
                 symbol_type: SymbolType::Variable(Type::Named(Identifier::dummy(contract_name))),
                 location: function.location.clone(),
                 mutable: true, // 'self' can be mutable
             };
            self.current_scope.define("self".to_string(), self_symbol)?;
        }
        
        // Add parameters to function scope
        for param in &function.parameters {
            // Skip 'self' parameter as it's handled above
            if param.name.name == "self" {
                continue;
            }
            
            let param_symbol = Symbol {
                name: param.name.name.clone(),
                symbol_type: SymbolType::Variable(param.type_annotation.clone()),
                location: param.location.clone(),
                mutable: false, // Parameters are immutable by default
            };
            self.current_scope.define(param.name.name.clone(), param_symbol)?;
        }
        
        // Analyze function body
        self.analyze_block(&function.body)?;
        
        // Perform safety analysis
        self.analyze_access_control(function);
        
        // Analyze arithmetic safety in function body
        for stmt in &function.body.statements {
            if let Statement::Expression(expr) = stmt {
                self.analyze_arithmetic_safety(expr);
            }
        }
        
        // Analyze reentrancy for this function
        self.analyze_reentrancy(&function.name.name);
        
        // Restore scope and context
        self.current_scope = old_scope;
        self.current_function = old_function;
        self.return_type = old_return_type;
        
        Ok(())
    }
    
    /// Analyze a struct
    fn analyze_struct(&mut self, struct_def: &Struct) -> Result<()> {
        // Check for duplicate struct names
        if self.current_scope.lookup_local(&struct_def.name.name).is_some() {
            return Err(SemanticError::new(
                SemanticErrorKind::DuplicateSymbol(struct_def.name.name.clone()),
                struct_def.location.clone(),
                format!("Struct '{}' is already defined", struct_def.name.name),
            ).into());
        }
        
        let mut fields = HashMap::new();
        
        // Analyze fields
        for field in &struct_def.fields {
            if fields.contains_key(&field.name.name) {
                return Err(SemanticError::new(
                    SemanticErrorKind::DuplicateField(field.name.name.clone()),
                    field.location.clone(),
                    format!("Field '{}' is already defined in struct", field.name.name),
                ).into());
            }
            
            self.validate_type(&field.type_annotation)?;
            fields.insert(field.name.name.clone(), field.type_annotation.clone());
        }
        
        // Add struct to scope
        let struct_symbol = Symbol {
            name: struct_def.name.name.clone(),
            symbol_type: SymbolType::Struct { fields },
            location: struct_def.location.clone(),
            mutable: false,
        };
        
        self.current_scope.define(struct_def.name.name.clone(), struct_symbol)?;
        
        Ok(())
    }
    
    /// Analyze an enum
    fn analyze_enum(&mut self, enum_def: &Enum) -> Result<()> {
        // Check for duplicate enum names
        if self.current_scope.lookup_local(&enum_def.name.name).is_some() {
            return Err(SemanticError::new(
                SemanticErrorKind::DuplicateSymbol(enum_def.name.name.clone()),
                enum_def.location.clone(),
                format!("Enum '{}' is already defined", enum_def.name.name),
            ).into());
        }
        
        let mut variants = HashMap::new();
        
        // Analyze variants
        for variant in &enum_def.variants {
            if variants.contains_key(&variant.name.name) {
                return Err(SemanticError::new(
                    SemanticErrorKind::DuplicateVariant(variant.name.name.clone()),
                    variant.location.clone(),
                    format!("Variant '{}' is already defined in enum", variant.name.name),
                ).into());
            }
            
            let field_types = if let Some(fields) = &variant.fields {
                for field_type in fields {
                    self.validate_type(field_type)?;
                }
                Some(fields.clone())
            } else {
                None
            };
            
            variants.insert(variant.name.name.clone(), field_types);
        }
        
        // Add enum to scope
        let enum_symbol = Symbol {
            name: enum_def.name.name.clone(),
            symbol_type: SymbolType::Enum { variants },
            location: enum_def.location.clone(),
            mutable: false,
        };
        
        self.current_scope.define(enum_def.name.name.clone(), enum_symbol)?;
        
        Ok(())
    }
    
    /// Analyze a trait
    fn analyze_trait(&mut self, trait_def: &Trait) -> Result<()> {
        // Check for duplicate trait names
        if self.current_scope.lookup_local(&trait_def.name.name).is_some() {
            return Err(SemanticError::new(
                SemanticErrorKind::DuplicateSymbol(trait_def.name.name.clone()),
                trait_def.location.clone(),
                format!("Trait '{}' is already defined", trait_def.name.name),
            ).into());
        }
        
        let mut functions = HashMap::new();
        
        // Analyze trait functions
        for function in &trait_def.functions {
            if functions.contains_key(&function.name.name) {
                return Err(SemanticError::new(
                    SemanticErrorKind::DuplicateFunction(function.name.name.clone()),
                    function.location.clone(),
                    format!("Function '{}' is already defined in trait", function.name.name),
                ).into());
            }
            
            // Validate parameter types
            for param in &function.parameters {
                self.validate_type(&param.type_annotation)?;
            }
            
            // Validate return type
            if let Some(return_type) = &function.return_type {
                self.validate_type(return_type)?;
            }
            
            let param_types: Vec<Type> = function.parameters.iter()
                .map(|p| p.type_annotation.clone())
                .collect();
            
            functions.insert(
                function.name.name.clone(),
                (param_types, function.return_type.clone()),
            );
        }
        
        // Add trait to scope
        let trait_symbol = Symbol {
            name: trait_def.name.name.clone(),
            symbol_type: SymbolType::Trait { functions },
            location: trait_def.location.clone(),
            mutable: false,
        };
        
        self.current_scope.define(trait_def.name.name.clone(), trait_symbol)?;
        
        Ok(())
    }
    
    /// Analyze an impl block
    fn analyze_impl(&mut self, impl_block: &Impl) -> Result<()> {
        // Verify that the type being implemented exists
        if self.current_scope.lookup(&impl_block.type_name.name).is_none() {
            return Err(SemanticError::new(
                SemanticErrorKind::UndefinedType(impl_block.type_name.name.clone()),
                impl_block.location.clone(),
                format!("Type '{}' is not defined", impl_block.type_name.name),
            ).into());
        }
        
        // If implementing a trait, verify the trait exists
        if let Some(trait_name) = &impl_block.trait_name {
            if self.current_scope.lookup(&trait_name.name).is_none() {
                return Err(SemanticError::new(
                    SemanticErrorKind::UndefinedTrait(trait_name.name.clone()),
                    impl_block.location.clone(),
                    format!("Trait '{}' is not defined", trait_name.name),
                ).into());
            }
        }
        
        // Analyze functions in impl block
        for function in &impl_block.functions {
            self.analyze_function(function)?;
        }
        
        Ok(())
    }
    
    /// Analyze a use declaration
    fn analyze_use(&mut self, use_decl: &UseDeclaration) -> Result<()> {
        // Validate the module path exists
        let module_path = &use_decl.path;
        
        // For now, we'll do basic validation
        if module_path.is_empty() {
            return Err(SemanticError::new(
                SemanticErrorKind::InvalidOperation,
                use_decl.location.clone(),
                "Empty use declaration path".to_string(),
            ).into());
        }
        
        // Add imported symbols to current scope
        match &use_decl.import_type {
            UseImportType::All => {
                // Import all symbols from module (would need module resolution)
            }
            UseImportType::Specific(symbols) => {
                for symbol_name in symbols {
                    // Add symbol to scope (would need actual symbol resolution)
                    let symbol = Symbol {
                        name: symbol_name.clone(),
                        symbol_type: SymbolType::Variable(Type::U32), // Placeholder
                        location: use_decl.location.clone(),
                        mutable: false,
                    };
                    self.current_scope.define(symbol_name.clone(), symbol)?;
                }
            }
            UseImportType::Alias(original, alias) => {
                // Import with alias
                let symbol = Symbol {
                    name: alias.clone(),
                    symbol_type: SymbolType::Variable(Type::U32), // Placeholder
                    location: use_decl.location.clone(),
                    mutable: false,
                };
                self.current_scope.define(alias.clone(), symbol)?;
            }
        }
        
        Ok(())
    }
    
    /// Analyze a const declaration
    fn analyze_const(&mut self, const_decl: &ConstDeclaration) -> Result<()> {
        // Check for duplicate constant names
        if self.current_scope.lookup_local(&const_decl.name.name).is_some() {
            return Err(SemanticError::new(
                SemanticErrorKind::DuplicateSymbol(const_decl.name.name.clone()),
                const_decl.location.clone(),
                format!("Constant '{}' is already defined", const_decl.name.name),
            ).into());
        }
        
        // Validate type
        self.validate_type(&const_decl.type_annotation)?;
        
        // Analyze value expression
        let value_type = self.analyze_expression(&const_decl.value)?;
        
        // Check type compatibility
        if !self.types_compatible(&const_decl.type_annotation, &value_type) {
            return Err(SemanticError::new(
                SemanticErrorKind::TypeMismatch {
                    expected: format!("{:?}", const_decl.type_annotation),
                    found: format!("{:?}", value_type),
                },
                const_decl.location.clone(),
                format!(
                    "Type mismatch: expected {:?}, found {:?}",
                    const_decl.type_annotation, value_type
                ),
            ).into());
        }
        
        // Add constant to scope
        let const_symbol = Symbol {
            name: const_decl.name.name.clone(),
            symbol_type: SymbolType::Constant(const_decl.type_annotation.clone()),
            location: const_decl.location.clone(),
            mutable: false,
        };
        
        self.current_scope.define(const_decl.name.name.clone(), const_symbol)?;
        
        Ok(())
    }
    
    /// Analyze a module
    fn analyze_module(&mut self, module: &Module) -> Result<()> {
        // Check for duplicate module names
        if self.current_scope.lookup_local(&module.name.name).is_some() {
            return Err(SemanticError::new(
                SemanticErrorKind::DuplicateSymbol(module.name.name.clone()),
                module.location.clone(),
                format!("Module '{}' is already defined", module.name.name),
            ).into());
        }
        
        // Add module to scope
        let module_symbol = Symbol {
            name: module.name.name.clone(),
            symbol_type: SymbolType::Module,
            location: module.location.clone(),
            mutable: false,
        };
        
        self.current_scope.define(module.name.name.clone(), module_symbol)?;
        
        // Analyze module items in new scope
        let module_scope = Scope::with_parent(self.current_scope.clone());
        let old_scope = std::mem::replace(&mut self.current_scope, module_scope);
        
        for item in &module.items {
            self.analyze_item(item)?;
        }
        
        // Restore scope
        self.current_scope = old_scope;
        
        Ok(())
    }
    
    /// Analyze a block of statements
    fn analyze_block(&mut self, block: &Block) -> Result<()> {
        // Create new scope for block
        let block_scope = Scope::with_parent(self.current_scope.clone());
        let old_scope = std::mem::replace(&mut self.current_scope, block_scope);
        
        for statement in &block.statements {
            self.analyze_statement(statement)?;
        }
        
        // Restore scope
        self.current_scope = old_scope;
        
        Ok(())
    }
    
    /// Analyze a statement
    fn analyze_statement(&mut self, statement: &Statement) -> Result<()> {
        match statement {
            Statement::Expression(expr) => {
                self.analyze_expression(expr)?;
                Ok(())
            }
            Statement::Let(let_stmt) => self.analyze_let_statement(let_stmt),
            Statement::Return(return_stmt) => self.analyze_return_statement(return_stmt),
            Statement::If(if_stmt) => self.analyze_if_statement(if_stmt),
            Statement::While(while_stmt) => self.analyze_while_statement(while_stmt),
            Statement::For(for_stmt) => self.analyze_for_statement(for_stmt),
            Statement::Match(match_stmt) => self.analyze_match_statement(match_stmt),
            Statement::Break(_) => {
                // TODO: Check if we're in a loop
                Ok(())
            }
            Statement::Continue(_) => {
                // TODO: Check if we're in a loop
                Ok(())
            }
            Statement::Emit(emit_stmt) => self.analyze_emit_statement(emit_stmt),
            Statement::Require(require_stmt) => self.analyze_require_statement(require_stmt),
            Statement::Assert(assert_stmt) => self.analyze_assert_statement(assert_stmt),
            Statement::Revert(revert_stmt) => self.analyze_revert_statement(revert_stmt),
        }
    }
    
    /// Analyze a let statement
    fn analyze_let_statement(&mut self, let_stmt: &LetStatement) -> Result<()> {
        // Check for duplicate variable names in current scope
        if self.current_scope.lookup_local(&let_stmt.name.name).is_some() {
            return Err(SemanticError::new(
                SemanticErrorKind::DuplicateSymbol(let_stmt.name.name.clone()),
                let_stmt.location.clone(),
                format!("Variable '{}' is already defined in this scope", let_stmt.name.name),
            ).into());
        }
        
        let variable_type = if let Some(value) = &let_stmt.value {
            // Analyze the value expression
            let value_type = self.analyze_expression(value)?;
            
            // If type annotation is provided, check compatibility
            if let Some(declared_type) = &let_stmt.type_annotation {
                self.validate_type(declared_type)?;
                if !self.types_compatible(declared_type, &value_type) {
                    return Err(SemanticError::new(
                        SemanticErrorKind::TypeMismatch {
                            expected: format!("{:?}", declared_type),
                            found: format!("{:?}", value_type),
                        },
                        let_stmt.location.clone(),
                        format!(
                            "Type mismatch: expected {:?}, found {:?}",
                            declared_type, value_type
                        ),
                    ).into());
                }
                declared_type.clone()
            } else {
                value_type
            }
        } else {
            // No value provided, type annotation is required
            if let Some(declared_type) = &let_stmt.type_annotation {
                self.validate_type(declared_type)?;
                declared_type.clone()
            } else {
                return Err(SemanticError::new(
                    SemanticErrorKind::TypeInferenceFailure,
                    let_stmt.location.clone(),
                    "Cannot infer type without value or type annotation".to_string(),
                ).into());
            }
        };
        
        // Add variable to scope
        let variable_symbol = Symbol {
            name: let_stmt.name.name.clone(),
            symbol_type: SymbolType::Variable(variable_type),
            location: let_stmt.location.clone(),
            mutable: let_stmt.mutable,
        };
        
        self.current_scope.define(let_stmt.name.name.clone(), variable_symbol)?;
        
        Ok(())
    }
    
    /// Analyze a return statement
    fn analyze_return_statement(&mut self, return_stmt: &ReturnStatement) -> Result<()> {
        if let Some(value) = &return_stmt.value {
            let value_type = self.analyze_expression(value)?;
            
            // Check if return type matches function return type
            if let Some(expected_type) = &self.return_type {
                if !self.types_compatible(expected_type, &value_type) {
                    return Err(SemanticError::new(
                        SemanticErrorKind::TypeMismatch {
                            expected: format!("{:?}", expected_type),
                            found: format!("{:?}", value_type),
                        },
                        return_stmt.location.clone(),
                        format!(
                            "Return type mismatch: expected {:?}, found {:?}",
                            expected_type, value_type
                        ),
                    ).into());
                }
            } else {
                return Err(SemanticError::new(
                    SemanticErrorKind::UnexpectedReturn,
                    return_stmt.location.clone(),
                    "Cannot return value from function with no return type".to_string(),
                ).into());
            }
        } else {
            // Empty return
            if self.return_type.is_some() {
                return Err(SemanticError::new(
                    SemanticErrorKind::TypeMismatch {
                        expected: "return value".to_string(),
                        found: "empty return".to_string(),
                    },
                    return_stmt.location.clone(),
                    "Function expects return value".to_string(),
                ).into());
            }
        }
        
        Ok(())
    }
    
    /// Analyze an if statement
    fn analyze_if_statement(&mut self, if_stmt: &IfStatement) -> Result<()> {
        // Analyze condition
        let condition_type = self.analyze_expression(&if_stmt.condition)?;
        
        // Condition must be boolean
        if !matches!(condition_type, Type::Bool) {
            return Err(SemanticError::new(
                SemanticErrorKind::TypeMismatch {
                    expected: "bool".to_string(),
                    found: format!("{:?}", condition_type),
                },
                if_stmt.location.clone(),
                format!("If condition must be boolean, found {:?}", condition_type),
            ).into());
        }
        
        // Analyze then block
        self.analyze_block(&if_stmt.then_block)?;
        
        // Analyze else block if present
        if let Some(else_block) = &if_stmt.else_block {
            self.analyze_statement(else_block)?;
        }
        
        Ok(())
    }
    
    /// Analyze a while statement
    fn analyze_while_statement(&mut self, while_stmt: &WhileStatement) -> Result<()> {
        // Analyze condition
        let condition_type = self.analyze_expression(&while_stmt.condition)?;
        
        // Condition must be boolean
        if !matches!(condition_type, Type::Bool) {
            return Err(SemanticError::new(
                SemanticErrorKind::TypeMismatch {
                    expected: "bool".to_string(),
                    found: format!("{:?}", condition_type),
                },
                while_stmt.location.clone(),
                format!("While condition must be boolean, found {:?}", condition_type),
            ).into());
        }
        
        // Analyze body
        self.analyze_block(&while_stmt.body)?;
        
        Ok(())
    }
    
    /// Analyze a for statement
    fn analyze_for_statement(&mut self, for_stmt: &ForStatement) -> Result<()> {
        // Analyze iterable expression
        let iterable_type = self.analyze_expression(&for_stmt.iterable)?;
        
        // Check if iterable_type is actually iterable and infer element type
        let element_type = self.check_iterable_and_get_element_type(&iterable_type, &for_stmt.iterable.location())?;
        
        // Create new scope for loop variable
        let for_scope = Scope::with_parent(self.current_scope.clone());
        let old_scope = std::mem::replace(&mut self.current_scope, for_scope);
        
        // Add loop variable to scope with inferred element type
        let loop_var_symbol = Symbol {
            name: for_stmt.variable.name.clone(),
            symbol_type: SymbolType::Variable(element_type),
            location: for_stmt.variable.location.clone(),
            mutable: false,
        };
        
        self.current_scope.define(for_stmt.variable.name.clone(), loop_var_symbol)?;
        
        // Analyze body
        self.analyze_block(&for_stmt.body)?;
        
        // Restore scope
        self.current_scope = old_scope;
        
        Ok(())
    }
    
    /// Analyze a match statement
    fn analyze_match_statement(&mut self, match_stmt: &MatchStatement) -> Result<()> {
        // Analyze match expression
        let match_type = self.analyze_expression(&match_stmt.expression)?;
        
        // Analyze each match arm
        for arm in &match_stmt.arms {
            // Create new scope for pattern variables
            let arm_scope = Scope::with_parent(self.current_scope.clone());
            let old_scope = std::mem::replace(&mut self.current_scope, arm_scope);
            
            // Analyze pattern and extract variables
            self.analyze_pattern(&arm.pattern, &match_type)?;
            
            // Analyze guard if present
            if let Some(guard) = &arm.guard {
                let guard_type = self.analyze_expression(guard)?;
                if !matches!(guard_type, Type::Bool) {
                    return Err(SemanticError::new(
                        SemanticErrorKind::TypeMismatch {
                            expected: "bool".to_string(),
                            found: format!("{:?}", guard_type),
                        },
                        arm.location.clone(),
                        format!("Match guard must be boolean, found {:?}", guard_type),
                    ).into());
                }
            }
            
            // Analyze arm body
            self.analyze_block(&arm.body)?;
            
            // Restore scope
            self.current_scope = old_scope;
        }
        
        Ok(())
    }
    
    /// Analyze an emit statement
    fn analyze_emit_statement(&mut self, emit_stmt: &EmitStatement) -> Result<()> {
        // Check if event exists in current contract scope
        if let Some(contract_scope) = self.get_current_contract_scope() {
            let event_symbol = contract_scope.lookup(&emit_stmt.event_name.name)
                .ok_or_else(|| SemanticError::new(
                    SemanticErrorKind::UndefinedSymbol(emit_stmt.event_name.name.clone()),
                    emit_stmt.location.clone(),
                    format!("Undefined event '{}'.", emit_stmt.event_name.name),
                ))?;
            
            // Verify it's actually an event
            match &event_symbol.symbol_type {
                SymbolType::Event(param_types) => {
                    // Check argument count and types
                    if emit_stmt.arguments.len() != param_types.len() {
                        return Err(SemanticError::new(
                            SemanticErrorKind::InvalidOperation,
                            emit_stmt.location.clone(),
                            format!("Event '{}' expects {} arguments, got {}", 
                                emit_stmt.event_name.name, param_types.len(), emit_stmt.arguments.len()),
                        ).into());
                    }
                    
                    // Check argument types
                    for (i, arg) in emit_stmt.arguments.iter().enumerate() {
                        let arg_type = self.analyze_expression(arg)?;
                        if !self.types_compatible(&param_types[i], &arg_type) {
                            return Err(SemanticError::new(
                                SemanticErrorKind::TypeMismatch {
                                    expected: format!("{:?}", param_types[i]),
                                    found: format!("{:?}", arg_type),
                                },
                                arg.location().clone(),
                                format!("Argument {} type mismatch", i + 1),
                            ).into());
                        }
                    }
                }
                _ => return Err(SemanticError::new(
                    SemanticErrorKind::InvalidOperation,
                    emit_stmt.location.clone(),
                    format!("'{}' is not an event", emit_stmt.event_name.name),
                ).into())
            }
        }
        
        Ok(())
    }
    
    /// Analyze a require statement
    fn analyze_require_statement(&mut self, require_stmt: &RequireStatement) -> Result<()> {
        // Analyze condition
        let condition_type = self.analyze_expression(&require_stmt.condition)?;
        
        // Condition must be boolean
        if !matches!(condition_type, Type::Bool) {
            return Err(SemanticError::new(
                SemanticErrorKind::TypeMismatch {
                    expected: "bool".to_string(),
                    found: format!("{:?}", condition_type),
                },
                require_stmt.location.clone(),
                format!("Require condition must be boolean, found {:?}", condition_type),
            ).into());
        }
        
        // Analyze message if present
        if let Some(message) = &require_stmt.message {
            let message_type = self.analyze_expression(message)?;
            if !matches!(message_type, Type::String) {
                return Err(SemanticError::new(
                    SemanticErrorKind::TypeMismatch {
                        expected: "string".to_string(),
                        found: format!("{:?}", message_type),
                    },
                    require_stmt.location.clone(),
                    format!("Require message must be string, found {:?}", message_type),
                ).into());
            }
        }
        
        Ok(())
    }
    
    /// Analyze an assert statement
    fn analyze_assert_statement(&mut self, assert_stmt: &AssertStatement) -> Result<()> {
        // Analyze condition
        let condition_type = self.analyze_expression(&assert_stmt.condition)?;
        
        // Condition must be boolean
        if !matches!(condition_type, Type::Bool) {
            return Err(SemanticError::new(
                SemanticErrorKind::TypeMismatch {
                    expected: "bool".to_string(),
                    found: format!("{:?}", condition_type),
                },
                assert_stmt.location.clone(),
                format!("Assert condition must be boolean, found {:?}", condition_type),
            ).into());
        }
        
        // Analyze message if present
        if let Some(message) = &assert_stmt.message {
            let message_type = self.analyze_expression(message)?;
            if !matches!(message_type, Type::String) {
                return Err(SemanticError::new(
                    SemanticErrorKind::TypeMismatch {
                        expected: "string".to_string(),
                        found: format!("{:?}", message_type),
                    },
                    assert_stmt.location.clone(),
                    format!("Assert message must be string, found {:?}", message_type),
                ).into());
            }
        }
        
        Ok(())
    }
    
    /// Analyze a revert statement
    fn analyze_revert_statement(&mut self, revert_stmt: &RevertStatement) -> Result<()> {
        // Analyze message if present
        if let Some(message) = &revert_stmt.message {
            let message_type = self.analyze_expression(message)?;
            if !matches!(message_type, Type::String) {
                return Err(SemanticError::new(
                    SemanticErrorKind::TypeMismatch {
                        expected: "string".to_string(),
                        found: format!("{:?}", message_type),
                    },
                    revert_stmt.location.clone(),
                    format!("Revert message must be string, found {:?}", message_type),
                ).into());
            }
        }
        
        Ok(())
    }
    
    /// Analyze an expression and return its type
    fn analyze_expression(&mut self, expression: &Expression) -> Result<Type> {
        match expression {
            Expression::Literal(literal) => Ok(self.literal_type(literal)),
            Expression::Identifier(identifier) => self.analyze_identifier(identifier),
            Expression::Binary(binary_expr) => self.analyze_binary_expression(binary_expr),
            Expression::Unary(unary_expr) => self.analyze_unary_expression(unary_expr),
            Expression::Call(call_expr) => self.analyze_call_expression(call_expr),
            Expression::FieldAccess(field_expr) => self.analyze_field_access(field_expr),
            Expression::Index(index_expr) => self.analyze_index_expression(index_expr),
            Expression::Array(array_expr) => self.analyze_array_expression(array_expr),
            Expression::Tuple(tuple_expr) => self.analyze_tuple_expression(tuple_expr),
            Expression::Struct(struct_expr) => self.analyze_struct_expression(struct_expr),
            Expression::Assignment(assign_expr) => self.analyze_assignment_expression(assign_expr),
            Expression::Range(range_expr) => self.analyze_range_expression(range_expr),
            Expression::Closure(closure_expr) => self.analyze_closure_expression(closure_expr),
            Expression::Block(block) => {
                self.analyze_block(block)?;
                Ok(Type::U32) // Placeholder
            }
            Expression::MLCreateModel(_) => Ok(Type::MLModel {
                model_type: "neural_network".to_string(),
                input_shape: vec![],
                output_shape: vec![],
            }),
            Expression::MLTrain(_) => Ok(Type::MLMetrics),
            Expression::MLPredict(_) => Ok(Type::Vector {
                element_type: Box::new(Type::U32),
                size: None,
            }),
            Expression::MLForward(_) => Ok(Type::Vector {
                element_type: Box::new(Type::U32),
                size: None,
            }),
            Expression::MLBackward(_) => Ok(Type::MLMetrics),
            Expression::TensorOp(_) => Ok(Type::Tensor {
                element_type: Box::new(Type::U32),
                dimensions: vec![],
            }),
            Expression::MatrixOp(_) => Ok(Type::Matrix {
                element_type: Box::new(Type::U32),
                rows: 0,
                cols: 0,
            }),
        }
    }
    
    /// Get the type of a literal
    fn literal_type(&self, literal: &Literal) -> Type {
        match literal {
            Literal::Integer(_) => Type::U32, // Default integer type
            Literal::Float(_) => Type::F64, // Default float type
            Literal::String(_) => Type::String,
            Literal::Boolean(_) => Type::Bool,
            Literal::Address(_) => Type::Address,
            Literal::Null => Type::Option(Box::new(Type::U32)), // Null as optional type
        }
    }
    
    /// Analyze an identifier expression
    fn analyze_identifier(&self, identifier: &Identifier) -> Result<Type> {
        if let Some(symbol) = self.current_scope.lookup(&identifier.name) {
            match &symbol.symbol_type {
                SymbolType::Variable(var_type) => Ok(var_type.clone()),
                SymbolType::Constant(const_type) => Ok(const_type.clone()),
                SymbolType::Struct { .. } => {
                    // For struct types (like blockchain primitives), return a named type
                    Ok(Type::Named(identifier.clone()))
                },
                _ => Err(SemanticError::new(
                    SemanticErrorKind::InvalidSymbolUsage,
                    identifier.location.clone(),
                    format!("'{}' is not a variable or constant", identifier.name),
                ).into()),
            }
        } else {
            Err(SemanticError::new(
                SemanticErrorKind::UndefinedSymbol(identifier.name.clone()),
                identifier.location.clone(),
                format!("Undefined symbol: {}", identifier.name),
            ).into())
        }
    }
    
    /// Analyze a binary expression
    fn analyze_binary_expression(&mut self, binary_expr: &BinaryExpression) -> Result<Type> {
        let left_type = self.analyze_expression(&binary_expr.left)?;
        let right_type = self.analyze_expression(&binary_expr.right)?;
        
        match binary_expr.operator {
            // Arithmetic operators
            BinaryOperator::Add | BinaryOperator::Subtract | 
            BinaryOperator::Multiply | BinaryOperator::Divide | 
            BinaryOperator::Modulo => {
                if self.is_numeric_type(&left_type) && self.is_numeric_type(&right_type) {
                    // TODO: Implement proper type promotion
                    Ok(left_type)
                } else {
                    Err(SemanticError::new(
                        SemanticErrorKind::TypeMismatch {
                            expected: "numeric types".to_string(),
                            found: format!("{:?} and {:?}", left_type, right_type),
                        },
                        binary_expr.location.clone(),
                        format!("Arithmetic operation requires numeric types, found {:?} and {:?}", left_type, right_type),
                    ).into())
                }
            }
            
            // Comparison operators
            BinaryOperator::Equal | BinaryOperator::NotEqual |
            BinaryOperator::Less | BinaryOperator::LessEqual |
            BinaryOperator::Greater | BinaryOperator::GreaterEqual => {
                if self.types_compatible(&left_type, &right_type) {
                    Ok(Type::Bool)
                } else {
                    Err(SemanticError::new(
                        SemanticErrorKind::TypeMismatch {
                            expected: "compatible types".to_string(),
                            found: format!("{:?} and {:?}", left_type, right_type),
                        },
                        binary_expr.location.clone(),
                        format!("Cannot compare {:?} and {:?}", left_type, right_type),
                    ).into())
                }
            }
            
            // Logical operators
            BinaryOperator::And | BinaryOperator::Or => {
                if matches!(left_type, Type::Bool) && matches!(right_type, Type::Bool) {
                    Ok(Type::Bool)
                } else {
                    Err(SemanticError::new(
                        SemanticErrorKind::TypeMismatch {
                            expected: "boolean types".to_string(),
                            found: format!("{:?} and {:?}", left_type, right_type),
                        },
                        binary_expr.location.clone(),
                        format!("Logical operation requires boolean types, found {:?} and {:?}", left_type, right_type),
                    ).into())
                }
            }
            
            // Bitwise operators
            BinaryOperator::BitAnd | BinaryOperator::BitOr | BinaryOperator::BitXor |
            BinaryOperator::LeftShift | BinaryOperator::RightShift => {
                if self.is_integer_type(&left_type) && self.is_integer_type(&right_type) {
                    Ok(left_type)
                } else {
                    Err(SemanticError::new(
                        SemanticErrorKind::TypeMismatch {
                            expected: "integer types".to_string(),
                            found: format!("{:?} and {:?}", left_type, right_type),
                        },
                        binary_expr.location.clone(),
                        format!("Bitwise operation requires integer types, found {:?} and {:?}", left_type, right_type),
                    ).into())
                }
            }
        }
    }
    
    /// Analyze a unary expression
    fn analyze_unary_expression(&mut self, unary_expr: &UnaryExpression) -> Result<Type> {
        let operand_type = self.analyze_expression(&unary_expr.operand)?;
        
        match unary_expr.operator {
            UnaryOperator::Not => {
                if matches!(operand_type, Type::Bool) {
                    Ok(Type::Bool)
                } else {
                    Err(SemanticError::new(
                        SemanticErrorKind::TypeMismatch {
                            expected: "bool".to_string(),
                            found: format!("{:?}", operand_type),
                        },
                        unary_expr.location.clone(),
                        format!("Logical NOT requires boolean type, found {:?}", operand_type),
                    ).into())
                }
            }
            UnaryOperator::Minus => {
                if self.is_numeric_type(&operand_type) {
                    Ok(operand_type)
                } else {
                    Err(SemanticError::new(
                        SemanticErrorKind::TypeMismatch {
                            expected: "numeric type".to_string(),
                            found: format!("{:?}", operand_type),
                        },
                        unary_expr.location.clone(),
                        format!("Unary minus requires numeric type, found {:?}", operand_type),
                    ).into())
                }
            }
            UnaryOperator::BitNot => {
                if self.is_integer_type(&operand_type) {
                    Ok(operand_type)
                } else {
                    Err(SemanticError::new(
                        SemanticErrorKind::TypeMismatch {
                            expected: "integer type".to_string(),
                            found: format!("{:?}", operand_type),
                        },
                        unary_expr.location.clone(),
                        format!("Bitwise NOT requires integer type, found {:?}", operand_type),
                    ).into())
                }
            }
        }
    }
    
    /// Analyze a function call expression
    fn analyze_call_expression(&mut self, call_expr: &CallExpression) -> Result<Type> {
        // Detect explicit cast like `i64(0)` where the callee is an identifier of a built-in type.
        if let Expression::Identifier(ident) = call_expr.function.as_ref() {
            let type_name = ident.name.as_str();
            if self.is_builtin_type_name(type_name) {
                // For now treat any built-in type identifier followed by one argument as an explicit cast.
                if call_expr.arguments.len() != 1 {
                    return Err(SemanticError::new(
                        SemanticErrorKind::InvalidArguments,
                        call_expr.location.clone(),
                        format!("Type cast {} expects exactly 1 argument", type_name),
                    ).into());
                }

                let arg_type = self.analyze_expression(&call_expr.arguments[0])?;
                if !self.is_numeric_type(&arg_type) {
                    return Err(SemanticError::new(
                        SemanticErrorKind::TypeMismatch {
                        expected: "numeric type".to_string(),
                        found: format!("{:?}", arg_type),
                    },
                        call_expr.location.clone(),
                        format!("Cannot cast non-numeric type {} to {}", arg_type, type_name),
                    ).into());
                }

                // Map the type name string to the corresponding Type variant.
                let cast_type = match type_name {
                    "u8" => Type::U8,
                    "u16" => Type::U16,
                    "u32" => Type::U32,
                    "u64" => Type::U64,
                    "u128" => Type::U128,
                    "u256" => Type::U256,
                    "i8" => Type::I8,
                    "i16" => Type::I16,
                    "i32" => Type::I32,
                    "i64" => Type::I64,
                    "i128" => Type::I128,
                    "i256" => Type::I256,
                    "bool" => Type::Bool,
                    "string" => Type::String,
                    "address" => Type::Address,
                    "bytes" => Type::Bytes,
                    _ => {
                        return Err(SemanticError::new(
                            SemanticErrorKind::InvalidArguments,
                            call_expr.location.clone(),
                            format!("Unsupported cast type {}", type_name),
                        ).into())
                    }
                };

                return Ok(cast_type);
            }
        }

        // Fallback to original placeholder logic (function calls not yet implemented fully).
        Ok(Type::U32)
    }
    
    /// Analyze a field access expression
    fn analyze_field_access(&mut self, field_expr: &FieldAccessExpression) -> Result<Type> {
        let object_type = self.analyze_expression(&field_expr.object)?;
        
        match &object_type {
            Type::Named(type_name) => {
                // Look up the type definition
                if let Some(symbol) = self.current_scope.lookup(&type_name.name) {
                    match &symbol.symbol_type {
                        SymbolType::Contract { fields, .. } => {
                            // Look up field in contract
                            if let Some(field_type) = fields.get(&field_expr.field.name) {
                                Ok(field_type.clone())
                            } else {
                                Err(SemanticError::new(
                                    SemanticErrorKind::UndefinedField(field_expr.field.name.clone()),
                                    field_expr.location.clone(),
                                    format!("Field '{}' not found in contract '{}'", field_expr.field.name, type_name.name),
                                ).into())
                            }
                        }
                        SymbolType::Struct { fields } => {
                            // Look up field in struct
                            if let Some(field_type) = fields.get(&field_expr.field.name) {
                                Ok(field_type.clone())
                            } else {
                                Err(SemanticError::new(
                                    SemanticErrorKind::UndefinedField(field_expr.field.name.clone()),
                                    field_expr.location.clone(),
                                    format!("Field '{}' not found in struct '{}'", field_expr.field.name, type_name.name),
                                ).into())
                            }
                        }
                        _ => {
                            Err(SemanticError::new(
                                SemanticErrorKind::TypeMismatch {
                                    expected: "struct or contract".to_string(),
                                    found: format!("{:?}", object_type),
                                },
                                field_expr.location.clone(),
                                format!("Cannot access field on type {:?}", object_type),
                            ).into())
                        }
                    }
                } else {
                    Err(SemanticError::new(
                        SemanticErrorKind::UndefinedType(type_name.name.clone()),
                        field_expr.location.clone(),
                        format!("Undefined type: {}", type_name.name),
                    ).into())
                }
            }
            _ => {
                Err(SemanticError::new(
                    SemanticErrorKind::TypeMismatch {
                        expected: "struct or contract".to_string(),
                        found: format!("{:?}", object_type),
                    },
                    field_expr.location.clone(),
                    format!("Cannot access field on type {:?}", object_type),
                ).into())
            }
        }
    }
    
    /// Analyze an index expression
    fn analyze_index_expression(&mut self, index_expr: &IndexExpression) -> Result<Type> {
        let object_type = self.analyze_expression(&index_expr.object)?;
        let index_type = self.analyze_expression(&index_expr.index)?;
        
        // Index must be integer
        if !self.is_integer_type(&index_type) {
            return Err(SemanticError::new(
                SemanticErrorKind::TypeMismatch {
                    expected: "integer type".to_string(),
                    found: format!("{:?}", index_type),
                },
                index_expr.location.clone(),
                format!("Array index must be integer, found {:?}", index_type),
            ).into());
        }
        
        // Extract element type from array/vector type
        let element_type = self.extract_element_type(&object_type, &index_expr.location)?;
        Ok(element_type)
    }
    
    /// Analyze an array expression
    fn analyze_array_expression(&mut self, array_expr: &ArrayExpression) -> Result<Type> {
        if array_expr.elements.is_empty() {
            // Empty array - type inference needed
            return Err(SemanticError::new(
                SemanticErrorKind::TypeInferenceFailure,
                array_expr.location.clone(),
                "Cannot infer type of empty array".to_string(),
            ).into());
        }
        
        // Analyze first element to get element type
        let element_type = self.analyze_expression(&array_expr.elements[0])?;
        
        // Check that all elements have the same type
        for (i, element) in array_expr.elements.iter().enumerate().skip(1) {
            let elem_type = self.analyze_expression(element)?;
            if !self.types_compatible(&element_type, &elem_type) {
                return Err(SemanticError::new(
                    SemanticErrorKind::TypeMismatch {
                        expected: format!("{:?}", element_type),
                        found: format!("{:?}", elem_type),
                    },
                    array_expr.location.clone(),
                    format!("Array element {} has type {:?}, expected {:?}", i, elem_type, element_type),
                ).into());
            }
        }
        
        Ok(Type::Array {
            element_type: Box::new(element_type),
            size: Some(array_expr.elements.len() as u64),
        })
    }
    
    /// Analyze a tuple expression
    fn analyze_tuple_expression(&mut self, tuple_expr: &TupleExpression) -> Result<Type> {
        let mut element_types = Vec::new();
        
        for element in &tuple_expr.elements {
            element_types.push(self.analyze_expression(element)?);
        }
        
        Ok(Type::Tuple(element_types))
    }
    
    /// Analyze a struct expression
    fn analyze_struct_expression(&mut self, struct_expr: &StructExpression) -> Result<Type> {
        // Look up struct type definition
        let struct_symbol = self.current_scope.lookup(&struct_expr.name.name)
            .ok_or_else(|| SemanticError::new(
                SemanticErrorKind::UndefinedSymbol(struct_expr.name.name.clone()),
                struct_expr.location.clone(),
                format!("Undefined struct type '{}'.", struct_expr.name.name),
            ))?;
        
        let struct_type = match &struct_symbol.symbol_type {
            SymbolType::Type(t) => t.clone(),
            _ => return Err(SemanticError::new(
                SemanticErrorKind::InvalidOperation,
                struct_expr.location.clone(),
                format!("'{}' is not a struct type", struct_expr.name.name),
            ).into())
        };
        
        // Analyze field expressions and check types
        for field in &struct_expr.fields {
            let field_type = self.analyze_expression(&field.value)?;
            // TODO: Check field type compatibility with struct definition
        }
        
        Ok(struct_type)
    }
    
    /// Analyze an assignment expression
    fn analyze_assignment_expression(&mut self, assign_expr: &AssignmentExpression) -> Result<Type> {
        let target_type = self.analyze_expression(&assign_expr.target)?;
        let value_type = self.analyze_expression(&assign_expr.value)?;
        
        // Check type compatibility
        if !self.types_compatible(&target_type, &value_type) {
            return Err(SemanticError::new(
                SemanticErrorKind::TypeMismatch {
                    expected: format!("{:?}", target_type),
                    found: format!("{:?}", value_type),
                },
                assign_expr.location.clone(),
                format!("Cannot assign {:?} to {:?}", value_type, target_type),
            ).into());
        }
        
        // Check if target is mutable
        self.check_assignment_target_mutability(&assign_expr.target)?;
        
        Ok(target_type)
    }
    
    /// Analyze a range expression
    fn analyze_range_expression(&mut self, range_expr: &RangeExpression) -> Result<Type> {
        let start_type = self.analyze_expression(&range_expr.start)?;
        let end_type = self.analyze_expression(&range_expr.end)?;
        
        // Both bounds must be integers
        if !self.is_integer_type(&start_type) || !self.is_integer_type(&end_type) {
            return Err(SemanticError::new(
                SemanticErrorKind::TypeMismatch {
                    expected: "integer type".to_string(),
                    found: format!("start: {:?}, end: {:?}", start_type, end_type),
                },
                range_expr.location.clone(),
                "Range bounds must be integers".to_string(),
            ).into());
        }
        
        // Return range type
        Ok(Type::Range {
            element_type: Box::new(start_type),
            inclusive: range_expr.inclusive,
        })
    }

    /// Check if a type is iterable and return element type
    fn check_iterable_and_get_element_type(&self, iterable_type: &Type, location: &SourceLocation) -> Result<Type> {
        match iterable_type {
            Type::Array { element_type, .. } => Ok((**element_type).clone()),
            Type::Vector { element_type, .. } => Ok((**element_type).clone()),
            Type::String => Ok(Type::U8), // String iterates over bytes
            Type::Range { element_type, .. } => Ok((**element_type).clone()),
            _ => Err(SemanticError::new(
                SemanticErrorKind::TypeMismatch {
                    expected: "iterable type".to_string(),
                    found: format!("{:?}", iterable_type),
                },
                location.clone(),
                format!("Type {:?} is not iterable", iterable_type),
            ).into())
        }
    }

    /// Extract element type from array/vector type
    fn extract_element_type(&self, container_type: &Type, location: &SourceLocation) -> Result<Type> {
        match container_type {
            Type::Array { element_type, .. } => Ok((**element_type).clone()),
            Type::Vector { element_type, .. } => Ok((**element_type).clone()),
            Type::String => Ok(Type::U8), // String indexing returns byte
            _ => Err(SemanticError::new(
                SemanticErrorKind::TypeMismatch {
                    expected: "indexable type".to_string(),
                    found: format!("{:?}", container_type),
                },
                location.clone(),
                format!("Type {:?} cannot be indexed", container_type),
            ).into())
        }
    }

    /// Get current contract scope for event lookup
    fn get_current_contract_scope(&self) -> Option<&Scope> {
        // In a full implementation, this would track contract scopes
        // For now, return current scope as placeholder
        Some(&self.current_scope)
    }

    /// Analyze pattern and bind variables
    fn analyze_pattern(&mut self, pattern: &Pattern, expected_type: &Type) -> Result<()> {
        match pattern {
            Pattern::Literal(literal) => {
                // Check literal type matches expected
                let literal_type = self.get_literal_type(literal);
                if !self.types_compatible(&literal_type, expected_type) {
                    return Err(SemanticError::new(
                        SemanticErrorKind::TypeMismatch {
                            expected: format!("{:?}", expected_type),
                            found: format!("{:?}", literal_type),
                        },
                        SourceLocation::unknown(),
                        "Pattern type mismatch".to_string(),
                    ).into());
                }
            }
            Pattern::Identifier(name) => {
                // Bind variable with expected type
                let symbol = Symbol {
                    name: name.name.clone(),
                    symbol_type: SymbolType::Variable(expected_type.clone()),
                    location: name.location.clone(),
                    mutable: false,
                };
                self.current_scope.define(name.name.clone(), symbol)?;
            }
            Pattern::Wildcard => {
                // Wildcard matches anything
            }
            Pattern::Tuple(patterns) => {
                if let Type::Tuple(element_types) = expected_type {
                    if patterns.len() != element_types.len() {
                        return Err(SemanticError::new(
                            SemanticErrorKind::InvalidOperation,
                            SourceLocation::unknown(),
                            "Tuple pattern arity mismatch".to_string(),
                        ).into());
                    }
                    for (pattern, expected) in patterns.iter().zip(element_types.iter()) {
                        self.analyze_pattern(pattern, expected)?;
                    }
                }
            }
            _ => {
                // Other pattern types would be implemented here
            }
        }
        Ok(())
    }

    /// Check if assignment target is mutable
    fn check_assignment_target_mutability(&self, target: &Expression) -> Result<()> {
        match target {
            Expression::Identifier(name) => {
                if let Some(symbol) = self.current_scope.lookup(&name.name) {
                    if !symbol.mutable {
                        return Err(SemanticError::new(
                            SemanticErrorKind::InvalidOperation,
                            target.location().clone(),
                            format!("Cannot assign to immutable variable '{}'", name.name),
                        ).into());
                    }
                }
            }
            Expression::FieldAccess(_) => {
                // Field access mutability would depend on the object and field
                // For now, assume it's valid
            }
            Expression::Index(_) => {
                // Array element assignment - check if array is mutable
                // For now, assume it's valid
            }
            _ => {
                return Err(SemanticError::new(
                    SemanticErrorKind::InvalidOperation,
                    target.location().clone(),
                    "Invalid assignment target".to_string(),
                ).into());
            }
        }
        Ok(())
    }

    /// Get type of literal
    fn get_literal_type(&self, literal: &Literal) -> Type {
        match literal {
            Literal::Integer(_) => Type::U64,
            Literal::Float(_) => Type::F64,
            Literal::String(_) => Type::String,
            Literal::Boolean(_) => Type::Bool,
            Literal::Address(_) => Type::Address,
        }
    }
    
    /// Analyze a closure expression
    fn analyze_closure_expression(&mut self, _closure_expr: &ClosureExpression) -> Result<Type> {
        // TODO: Implement closure expression analysis
        Ok(Type::U32) // Placeholder
    }
    
    /// Validate that a type is well-formed
    fn validate_type(&self, type_annotation: &Type) -> Result<()> {
        match type_annotation {
            Type::Named(identifier) => {
                // Check if the named type exists
                if self.current_scope.lookup(&identifier.name).is_none() {
                    return Err(SemanticError::new(
                        SemanticErrorKind::UndefinedType(identifier.name.clone()),
                        identifier.location.clone(),
                        format!("Undefined type: {}", identifier.name),
                    ).into());
                }
            }
            Type::Array { element_type, .. } => {
                self.validate_type(element_type)?;
            }
            Type::Tuple(types) => {
                for t in types {
                    self.validate_type(t)?;
                }
            }
            Type::Option(inner_type) => {
                self.validate_type(inner_type)?;
            }
            Type::Result { ok_type, err_type } => {
                self.validate_type(ok_type)?;
                self.validate_type(err_type)?;
            }
            _ => {} // Built-in types are always valid
        }
        Ok(())
    }
    
    /// Check if a string is a built-in type name
    fn is_builtin_type_name(&self, name: &str) -> bool {
        matches!(
            name,
            "u8" | "u16" | "u32" | "u64" | "u128" | "u256" |
            "i8" | "i16" | "i32" | "i64" | "i128" | "i256" |
            "bool" | "string" | "address" | "bytes"
        )
    }

    /// Check if two types are compatible
    fn types_compatible(&self, expected: &Type, actual: &Type) -> bool {
        // Exact type match
        if std::mem::discriminant(expected) == std::mem::discriminant(actual) {
            return true;
        }
        // Allow any numeric types to be compatible (implicit coercion for now)
        if self.is_numeric_type(expected) && self.is_numeric_type(actual) {
            return true;
        }
        false
    }

    /// Check if a type is numeric
    fn is_numeric_type(&self, t: &Type) -> bool {
        matches!(t,
            Type::U8 | Type::U16 | Type::U32 | Type::U64 | Type::U128 | Type::U256 |
            Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::I128 | Type::I256 |
            Type::F32 | Type::F64
        )
    }

    /// Check if a type is an integer type
    fn is_integer_type(&self, t: &Type) -> bool {
        self.is_numeric_type(t) // For now, all numeric types are integers
    }
    
    /// Perform safety analysis on the analyzed code
    #[allow(dead_code)]
    pub fn get_safety_issues(&self) -> &Vec<SafetyIssue> {
        &self.safety_issues
    }
    
    /// Add a safety issue
    fn add_safety_issue(&mut self, issue: SafetyIssue) {
        self.safety_issues.push(issue);
    }
    
    /// Analyze potential reentrancy vulnerabilities
    fn analyze_reentrancy(&mut self, function_name: &str) {
        if let Some(calls) = self.call_graph.get(function_name) {
            let has_external_calls = calls.iter().any(|call| self.external_calls.contains(call));
            let modifies_state = self.state_modifications.contains(function_name);
            
            if has_external_calls && modifies_state {
                self.add_safety_issue(SafetyIssue {
                    severity: SafetySeverity::High,
                    issue_type: SafetyIssueType::PotentialReentrancy,
                    location: SourceLocation::dummy(), // TODO: Get actual location
                    message: format!("Function '{}' may be vulnerable to reentrancy attacks", function_name),
                    function: Some(function_name.to_string()),
                });
            }
        }
    }
    
    /// Analyze arithmetic operations for potential overflow
    fn analyze_arithmetic_safety(&mut self, expr: &Expression) {
        match expr {
            Expression::Binary(binary_expr) => {
                if self.is_arithmetic_operator(&binary_expr.operator) {
                    let left_type = self.analyze_expression(&binary_expr.left);
                    let right_type = self.analyze_expression(&binary_expr.right);
                    
                    if let (Ok(left), Ok(right)) = (left_type, right_type) {
                        if self.is_numeric_type(&left) && self.is_numeric_type(&right) {
                            // Check for potential overflow in addition/multiplication
                            if matches!(binary_expr.operator, BinaryOperator::Add | BinaryOperator::Multiply) {
                                self.add_safety_issue(SafetyIssue {
                                    severity: SafetySeverity::Medium,
                                    issue_type: SafetyIssueType::ArithmeticOverflow,
                                    location: binary_expr.location.clone(),
                                    message: "Potential arithmetic overflow detected. Consider using checked arithmetic.".to_string(),
                                    function: self.current_function.clone(),
                                });
                            }
                            
                            // Check for potential underflow in subtraction
                            if matches!(binary_expr.operator, BinaryOperator::Subtract) {
                                self.add_safety_issue(SafetyIssue {
                                    severity: SafetySeverity::Medium,
                                    issue_type: SafetyIssueType::IntegerUnderflow,
                                    location: binary_expr.location.clone(),
                                    message: "Potential integer underflow detected. Consider using checked arithmetic.".to_string(),
                                    function: self.current_function.clone(),
                                });
                            }
                        }
                    }
                }
                
                // Recursively analyze operands
                self.analyze_arithmetic_safety(&binary_expr.left);
                self.analyze_arithmetic_safety(&binary_expr.right);
            }
            Expression::Call(call_expr) => {
                 // Track external calls for reentrancy analysis
                 if let Expression::FieldAccess(field_access) = call_expr.function.as_ref() {
                     if let Expression::Identifier(obj) = field_access.object.as_ref() {
                         if obj.name != "self" {
                             // This is an external call
                             if let Some(current_fn) = &self.current_function {
                                 self.external_calls.insert(current_fn.clone());
                                 self.call_graph.entry(current_fn.clone())
                                     .or_insert_with(Vec::new)
                                     .push(format!("{}.{}", obj.name, field_access.field.name));
                             }
                         }
                     }
                 }
                
                // Recursively analyze arguments
                for arg in &call_expr.arguments {
                    self.analyze_arithmetic_safety(arg);
                }
            }
            Expression::Assignment(assign_expr) => {
                 // Track state modifications
                 if let Expression::FieldAccess(field_access) = assign_expr.target.as_ref() {
                     if let Expression::Identifier(obj) = field_access.object.as_ref() {
                         if obj.name == "self" {
                             // This modifies contract state
                             if let Some(current_fn) = &self.current_function {
                                 self.state_modifications.insert(current_fn.clone());
                             }
                         }
                     }
                 }
                
                // Recursively analyze value
                self.analyze_arithmetic_safety(&assign_expr.value);
            }
            Expression::Array(array_expr) => {
                for element in &array_expr.elements {
                    self.analyze_arithmetic_safety(element);
                }
            }
            Expression::Tuple(tuple_expr) => {
                for element in &tuple_expr.elements {
                    self.analyze_arithmetic_safety(element);
                }
            }
            _ => {} // Other expression types don't need arithmetic safety analysis
        }
    }
    
    /// Check if an operator is arithmetic
    fn is_arithmetic_operator(&self, op: &BinaryOperator) -> bool {
        matches!(op, 
            BinaryOperator::Add | 
            BinaryOperator::Subtract | 
            BinaryOperator::Multiply | 
            BinaryOperator::Divide | 
            BinaryOperator::Modulo
        )
    }
    
    /// Analyze access control patterns
    fn analyze_access_control(&mut self, function: &Function) {
        // Check for functions that modify state without access control
        let mut has_access_control = false;
        let mut modifies_state = false;
        
        // Simple heuristic: look for require statements or modifier-like patterns
        for stmt in &function.body.statements {
            if let Statement::Expression(expr) = stmt {
                if let Expression::Call(call_expr) = expr {
                    if let Expression::Identifier(func_name) = call_expr.function.as_ref() {
                        if func_name.name == "require" || func_name.name == "assert" {
                            has_access_control = true;
                        }
                    }
                }
            }
            
            // Check for state modifications
            if self.statement_modifies_state(stmt) {
                modifies_state = true;
            }
        }
        
        // Warn if function modifies state without access control
        if modifies_state && !has_access_control && function.visibility != Visibility::Private {
            self.add_safety_issue(SafetyIssue {
                severity: SafetySeverity::Medium,
                issue_type: SafetyIssueType::UnauthorizedAccess,
                location: function.location.clone(),
                message: format!("Function '{}' modifies state without access control checks", function.name.name),
                function: Some(function.name.name.clone()),
            });
        }
    }
    
    /// Check if a statement modifies contract state
    fn statement_modifies_state(&self, stmt: &Statement) -> bool {
        match stmt {
            Statement::Expression(expr) => self.expression_modifies_state(expr),
            Statement::Let(_) => false, // Local variable declarations don't modify contract state
            Statement::Return(_) => false,
            Statement::If(if_stmt) => {
                self.block_modifies_state(&if_stmt.then_block) ||
                if_stmt.else_block.as_ref().map_or(false, |stmt| self.statement_modifies_state(stmt))
            }
            Statement::While(while_stmt) => self.block_modifies_state(&while_stmt.body),
            Statement::For(for_stmt) => self.block_modifies_state(&for_stmt.body),
            Statement::Match(match_stmt) => {
                match_stmt.arms.iter().any(|arm| self.block_modifies_state(&arm.body))
            }
            Statement::Break(_) | Statement::Continue(_) => false,
            Statement::Emit(_) => true, // Emitting events modifies state
            Statement::Require(_) | Statement::Assert(_) => false, // These don't modify state
            Statement::Revert(_) => false, // Revert doesn't modify state (it reverts it)
        }
    }
    
    /// Check if a block modifies contract state
    fn block_modifies_state(&self, block: &Block) -> bool {
        block.statements.iter().any(|stmt| self.statement_modifies_state(stmt))
    }
    
    /// Check if an expression modifies contract state
    fn expression_modifies_state(&self, expr: &Expression) -> bool {
        match expr {
            Expression::Assignment(assign_expr) => {
                // Check if assignment target is a contract field
                if let Expression::FieldAccess(field_access) = assign_expr.target.as_ref() {
                    if let Expression::Identifier(obj) = field_access.object.as_ref() {
                        return obj.name == "self";
                    }
                }
                false
            }
            Expression::Call(call_expr) => {
                // External calls might modify state
                if let Expression::FieldAccess(field_access) = call_expr.function.as_ref() {
                    if let Expression::Identifier(obj) = field_access.object.as_ref() {
                        return obj.name != "self"; // External calls
                    }
                }
                false
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    
    fn analyze_source(source: &str) -> Result<()> {
        let mut lexer = Lexer::new(source, "test.aug");
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(tokens);
        let ast = parser.parse()?;
        let mut analyzer = SemanticAnalyzer::new();
        analyzer.analyze(&ast)
    }
    
    #[test]
    fn test_simple_contract_analysis() {
        let source = r#"
            contract SimpleToken {
                let balance: u256;
                
                fn get_balance() -> u256 {
                    return self.balance;
                }
            }
        "#;
        
        let result = analyze_source(source);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_type_checking() {
        let source = r#"
            fn test() {
                let x: u32 = 42;
                let y: bool = true;
                let z = x + 10;
            }
        "#;
        
        let result = analyze_source(source);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_type_mismatch_error() {
        let source = r#"
            fn test() {
                let x: u32 = true;
            }
        "#;
        
        let result = analyze_source(source);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_undefined_symbol_error() {
        let source = r#"
            fn test() {
                return undefined_var;
            }
        "#;
        
        let result = analyze_source(source);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_duplicate_symbol_error() {
        let source = r#"
            fn test() {
                let x = 1;
                let x = 2;
            }
        "#;
        
        let result = analyze_source(source);
        assert!(result.is_err());
    }
}