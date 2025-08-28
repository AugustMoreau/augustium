// Semantic analysis extensions for missing features
use crate::ast::*;
use crate::error::{Result, SemanticError, SemanticErrorKind};
use crate::semantic::{SemanticAnalyzer, Symbol, SymbolType, Scope};
use std::collections::HashMap;

impl SemanticAnalyzer {
    /// Analyze generic function with proper type inference
    pub fn analyze_generic_function(&mut self, function: &Function) -> Result<()> {
        // Create new scope for type parameters
        let generic_scope = Scope::with_parent(self.current_scope.clone());
        let old_scope = std::mem::replace(&mut self.current_scope, generic_scope);
        
        // Add type parameters to scope
        for type_param in &function.type_parameters {
            let symbol = Symbol {
                name: type_param.name.name.clone(),
                symbol_type: SymbolType::TypeParameter(type_param.clone()),
                location: type_param.location.clone(),
                mutable: false,
            };
            self.current_scope.define(type_param.name.name.clone(), symbol)?;
        }
        
        // Analyze where clause bounds
        if let Some(where_clause) = &function.where_clause {
            for predicate in &where_clause.predicates {
                self.analyze_where_predicate(predicate)?;
            }
        }
        
        // Analyze function parameters with generic context
        for param in &function.parameters {
            let param_type = self.resolve_type_with_generics(&param.param_type)?;
            let symbol = Symbol {
                name: param.name.name.clone(),
                symbol_type: SymbolType::Variable(param_type),
                location: param.location.clone(),
                mutable: param.mutable,
            };
            self.current_scope.define(param.name.name.clone(), symbol)?;
        }
        
        // Analyze return type
        if let Some(return_type) = &function.return_type {
            self.resolve_type_with_generics(return_type)?;
        }
        
        // Analyze function body
        self.analyze_block(&function.body)?;
        
        // Restore scope
        self.current_scope = old_scope;
        Ok(())
    }
    
    /// Resolve type with generic parameters
    pub fn resolve_type_with_generics(&mut self, ty: &Type) -> Result<Type> {
        match ty {
            Type::Generic { base, args } => {
                let resolved_base = self.resolve_type_with_generics(base)?;
                let mut resolved_args = Vec::new();
                for arg in args {
                    resolved_args.push(self.resolve_type_with_generics(arg)?);
                }
                Ok(Type::Generic {
                    base: Box::new(resolved_base),
                    args: resolved_args,
                })
            }
            Type::Custom(name) => {
                // Check if this is a type parameter
                if let Some(symbol) = self.current_scope.lookup(name) {
                    match &symbol.symbol_type {
                        SymbolType::TypeParameter(_) => Ok(Type::TypeParameter(name.clone())),
                        _ => Ok(ty.clone()),
                    }
                } else {
                    Ok(ty.clone())
                }
            }
            Type::Array(element_type, size) => {
                let resolved_element = self.resolve_type_with_generics(element_type)?;
                Ok(Type::Array(Box::new(resolved_element), *size))
            }
            Type::Tuple(types) => {
                let mut resolved_types = Vec::new();
                for t in types {
                    resolved_types.push(self.resolve_type_with_generics(t)?);
                }
                Ok(Type::Tuple(resolved_types))
            }
            _ => Ok(ty.clone()),
        }
    }
    
    /// Analyze where clause predicate
    fn analyze_where_predicate(&mut self, predicate: &WherePredicate) -> Result<()> {
        // Verify type parameter exists
        if !self.current_scope.lookup(&predicate.type_param.name).is_some() {
            return Err(SemanticError::new(
                SemanticErrorKind::UndefinedVariable(predicate.type_param.name.clone()),
                predicate.location.clone(),
                format!("Undefined type parameter: {}", predicate.type_param.name),
            ).into());
        }
        
        // Analyze bounds
        for bound in &predicate.bounds {
            self.analyze_type_bound(bound)?;
        }
        
        Ok(())
    }
    
    /// Analyze type bound
    fn analyze_type_bound(&mut self, bound: &TypeBound) -> Result<()> {
        // Verify trait exists
        // TODO: Implement trait lookup
        
        // Analyze type arguments
        for arg in &bound.type_args {
            self.resolve_type_with_generics(arg)?;
        }
        
        Ok(())
    }
    
    /// Analyze async function
    pub fn analyze_async_function(&mut self, function: &Function) -> Result<()> {
        if function.is_async {
            // Mark function as async in context
            self.async_functions.insert(function.name.name.clone());
            
            // Analyze return type - must be Future or similar
            if let Some(return_type) = &function.return_type {
                self.validate_async_return_type(return_type)?;
            }
        }
        
        self.analyze_function_body(function)
    }
    
    /// Validate async function return type
    fn validate_async_return_type(&self, return_type: &Type) -> Result<()> {
        match return_type {
            Type::Generic { base, .. } => {
                if let Type::Custom(name) = base.as_ref() {
                    if name == "Future" || name == "Task" {
                        return Ok(());
                    }
                }
            }
            Type::Custom(name) => {
                if name == "Future" || name == "Task" {
                    return Ok(());
                }
            }
            _ => {}
        }
        
        // For now, allow any return type for async functions
        Ok(())
    }
    
    /// Analyze await expression
    pub fn analyze_await_expression(&mut self, await_expr: &AwaitExpression) -> Result<Type> {
        let expr_type = self.analyze_expression(&await_expr.expression)?;
        
        // Verify we're in an async context
        if !self.in_async_context() {
            return Err(SemanticError::new(
                SemanticErrorKind::InvalidOperation,
                await_expr.location.clone(),
                "await can only be used in async functions".to_string(),
            ).into());
        }
        
        // Extract inner type from Future<T>
        match expr_type {
            Type::Generic { base, args } => {
                if let Type::Custom(name) = base.as_ref() {
                    if name == "Future" && !args.is_empty() {
                        return Ok(args[0].clone());
                    }
                }
            }
            _ => {}
        }
        
        // If not a proper Future, return the type as-is
        Ok(expr_type)
    }
    
    /// Check if we're in an async context
    fn in_async_context(&self) -> bool {
        // Check if current function is async
        // This is a simplified check - in practice, we'd track the current function context
        true // For now, allow await anywhere
    }
    
    /// Analyze enhanced pattern matching
    pub fn analyze_enhanced_pattern(&mut self, pattern: &Pattern, matched_type: &Type) -> Result<()> {
        match pattern {
            Pattern::Guard { pattern, condition } => {
                self.analyze_enhanced_pattern(pattern, matched_type)?;
                let condition_type = self.analyze_expression(condition)?;
                if !matches!(condition_type, Type::Bool) {
                    return Err(SemanticError::new(
                        SemanticErrorKind::TypeMismatch {
                            expected: "bool".to_string(),
                            found: format!("{:?}", condition_type),
                        },
                        condition.get_location(),
                        "Pattern guard must be boolean".to_string(),
                    ).into());
                }
            }
            Pattern::Or(patterns) => {
                for pat in patterns {
                    self.analyze_enhanced_pattern(pat, matched_type)?;
                }
            }
            Pattern::Binding { name, pattern } => {
                // Add binding to scope
                let symbol = Symbol {
                    name: name.name.clone(),
                    symbol_type: SymbolType::Variable(matched_type.clone()),
                    location: name.location.clone(),
                    mutable: false,
                };
                self.current_scope.define(name.name.clone(), symbol)?;
                
                self.analyze_enhanced_pattern(pattern, matched_type)?;
            }
            Pattern::Array { patterns, rest } => {
                if let Type::Array(element_type, _) = matched_type {
                    for pat in patterns {
                        self.analyze_enhanced_pattern(pat, element_type)?;
                    }
                    if let Some(rest_pattern) = rest {
                        // Rest pattern matches a slice of the same element type
                        let slice_type = Type::Array(element_type.clone(), None);
                        self.analyze_enhanced_pattern(rest_pattern, &slice_type)?;
                    }
                } else {
                    return Err(SemanticError::new(
                        SemanticErrorKind::TypeMismatch {
                            expected: "array type".to_string(),
                            found: format!("{:?}", matched_type),
                        },
                        SourceLocation::unknown(),
                        "Array pattern requires array type".to_string(),
                    ).into());
                }
            }
            Pattern::Struct { name: _, fields, rest: _ } => {
                // Analyze struct field patterns
                for field in fields {
                    // TODO: Get field type from struct definition
                    let field_type = Type::U32; // Placeholder
                    self.analyze_enhanced_pattern(&field.pattern, &field_type)?;
                }
            }
            Pattern::Range { start, end, inclusive: _ } => {
                if let Some(start_pattern) = start {
                    self.analyze_enhanced_pattern(start_pattern, matched_type)?;
                }
                if let Some(end_pattern) = end {
                    self.analyze_enhanced_pattern(end_pattern, matched_type)?;
                }
            }
            Pattern::Reference { mutable: _, pattern } => {
                // Dereference the matched type
                let inner_type = match matched_type {
                    Type::Reference(inner) => inner.as_ref(),
                    _ => matched_type,
                };
                self.analyze_enhanced_pattern(pattern, inner_type)?;
            }
            Pattern::Deref(pattern) => {
                // Similar to reference pattern
                self.analyze_enhanced_pattern(pattern, matched_type)?;
            }
            _ => {
                // Handle basic patterns
                self.analyze_basic_pattern(pattern, matched_type)?;
            }
        }
        Ok(())
    }
    
    /// Analyze basic pattern (fallback for simple patterns)
    fn analyze_basic_pattern(&mut self, pattern: &Pattern, matched_type: &Type) -> Result<()> {
        match pattern {
            Pattern::Identifier(name) => {
                let symbol = Symbol {
                    name: name.name.clone(),
                    symbol_type: SymbolType::Variable(matched_type.clone()),
                    location: name.location.clone(),
                    mutable: false,
                };
                self.current_scope.define(name.name.clone(), symbol)?;
            }
            Pattern::Literal(literal) => {
                let literal_type = self.get_literal_type(literal);
                if !self.types_compatible(&literal_type, matched_type) {
                    return Err(SemanticError::new(
                        SemanticErrorKind::TypeMismatch {
                            expected: format!("{:?}", matched_type),
                            found: format!("{:?}", literal_type),
                        },
                        SourceLocation::unknown(),
                        "Pattern literal type mismatch".to_string(),
                    ).into());
                }
            }
            Pattern::Tuple(patterns) => {
                if let Type::Tuple(types) = matched_type {
                    if patterns.len() != types.len() {
                        return Err(SemanticError::new(
                            SemanticErrorKind::TypeMismatch {
                                expected: format!("tuple with {} elements", types.len()),
                                found: format!("tuple with {} elements", patterns.len()),
                            },
                            SourceLocation::unknown(),
                            "Tuple pattern length mismatch".to_string(),
                        ).into());
                    }
                    for (pattern, ty) in patterns.iter().zip(types.iter()) {
                        self.analyze_enhanced_pattern(pattern, ty)?;
                    }
                }
            }
            Pattern::Wildcard => {
                // Wildcard matches anything
            }
            _ => {
                // Other patterns handled by enhanced analysis
            }
        }
        Ok(())
    }
    
    /// Get type of literal
    fn get_literal_type(&self, literal: &Literal) -> Type {
        match literal {
            Literal::Integer(_) => Type::I32,
            Literal::Float(_) => Type::F64,
            Literal::String(_) => Type::String,
            Literal::Boolean(_) => Type::Bool,
            Literal::Address(_) => Type::Address,
        }
    }
    
    /// Check if types are compatible
    fn types_compatible(&self, t1: &Type, t2: &Type) -> bool {
        // Simplified type compatibility check
        match (t1, t2) {
            (Type::I32, Type::U32) | (Type::U32, Type::I32) => true,
            (Type::I64, Type::U64) | (Type::U64, Type::I64) => true,
            _ => t1 == t2,
        }
    }
    
    /// Analyze macro invocation
    pub fn analyze_macro_invocation(&mut self, macro_inv: &MacroInvocation) -> Result<Type> {
        // Analyze macro arguments
        for arg in &macro_inv.args {
            self.analyze_expression(arg)?;
        }
        
        // Built-in macro type checking
        match macro_inv.name.name.as_str() {
            "println" | "print" | "debug" => {
                // These macros accept any arguments and return unit
                Ok(Type::Unit)
            }
            "assert" | "require" => {
                // These require boolean first argument
                if macro_inv.args.is_empty() {
                    return Err(SemanticError::new(
                        SemanticErrorKind::InvalidOperation,
                        macro_inv.location.clone(),
                        format!("{} macro requires at least one argument", macro_inv.name.name),
                    ).into());
                }
                
                let first_arg_type = self.analyze_expression(&macro_inv.args[0])?;
                if !matches!(first_arg_type, Type::Bool) {
                    return Err(SemanticError::new(
                        SemanticErrorKind::TypeMismatch {
                            expected: "bool".to_string(),
                            found: format!("{:?}", first_arg_type),
                        },
                        macro_inv.location.clone(),
                        format!("{} macro requires boolean condition", macro_inv.name.name),
                    ).into());
                }
                
                Ok(Type::Unit)
            }
            _ => {
                // Unknown macro - assume it returns unit for now
                Ok(Type::Unit)
            }
        }
    }
    
    /// Analyze operator implementation
    pub fn analyze_operator_impl(&mut self, op_impl: &OperatorImpl) -> Result<()> {
        // Verify target type exists
        self.resolve_type_with_generics(&op_impl.target_type)?;
        
        // Analyze each method in the implementation
        for method in &op_impl.methods {
            // Verify method signature matches operator requirements
            self.validate_operator_method(&op_impl.operator, method)?;
            self.analyze_generic_function(method)?;
        }
        
        Ok(())
    }
    
    /// Validate operator method signature
    fn validate_operator_method(&self, operator: &OverloadableOperator, method: &Function) -> Result<()> {
        match operator {
            OverloadableOperator::Add | OverloadableOperator::Sub | 
            OverloadableOperator::Mul | OverloadableOperator::Div => {
                // Binary operators should have specific signatures
                if method.parameters.len() != 2 {
                    return Err(SemanticError::new(
                        SemanticErrorKind::InvalidOperation,
                        method.location.clone(),
                        format!("Binary operator {:?} requires exactly 2 parameters", operator),
                    ).into());
                }
            }
            OverloadableOperator::Index => {
                // Index operator should have 2 parameters (self, index)
                if method.parameters.len() != 2 {
                    return Err(SemanticError::new(
                        SemanticErrorKind::InvalidOperation,
                        method.location.clone(),
                        "Index operator requires exactly 2 parameters".to_string(),
                    ).into());
                }
            }
            _ => {
                // Other operators have their own requirements
            }
        }
        
        Ok(())
    }
    
    /// Analyze for statement with proper iterator type checking
    pub fn analyze_for_statement_enhanced(&mut self, for_stmt: &ForStatement) -> Result<()> {
        // Analyze iterable expression
        let iterable_type = self.analyze_expression(&for_stmt.iterable)?;
        
        // Infer element type from iterable
        let element_type = self.extract_iterator_element_type(&iterable_type)?;
        
        // Create new scope for loop variable
        let for_scope = Scope::with_parent(self.current_scope.clone());
        let old_scope = std::mem::replace(&mut self.current_scope, for_scope);
        
        // Add loop variable to scope with proper type
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
    
    /// Extract element type from iterator type
    fn extract_iterator_element_type(&self, iterable_type: &Type) -> Result<Type> {
        match iterable_type {
            Type::Array(element_type, _) => Ok(element_type.as_ref().clone()),
            Type::Generic { base, args } => {
                if let Type::Custom(name) = base.as_ref() {
                    if name == "Vec" || name == "Iterator" {
                        if !args.is_empty() {
                            return Ok(args[0].clone());
                        }
                    }
                }
                Err(SemanticError::new(
                    SemanticErrorKind::TypeMismatch {
                        expected: "iterable type".to_string(),
                        found: format!("{:?}", iterable_type),
                    },
                    SourceLocation::unknown(),
                    "Cannot iterate over this type".to_string(),
                ).into())
            }
            Type::Custom(name) => {
                // Check if it's a known iterable type
                match name.as_str() {
                    "String" => Ok(Type::U8), // String iterates over bytes
                    "Range" => Ok(Type::I32), // Range iterates over integers
                    _ => Err(SemanticError::new(
                        SemanticErrorKind::TypeMismatch {
                            expected: "iterable type".to_string(),
                            found: format!("{:?}", iterable_type),
                        },
                        SourceLocation::unknown(),
                        format!("Type '{}' is not iterable", name),
                    ).into()),
                }
            }
            _ => Err(SemanticError::new(
                SemanticErrorKind::TypeMismatch {
                    expected: "iterable type".to_string(),
                    found: format!("{:?}", iterable_type),
                },
                SourceLocation::unknown(),
                "Type is not iterable".to_string(),
            ).into()),
        }
    }
}
