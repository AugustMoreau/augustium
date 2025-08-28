// Code generation extensions for generics, async, macros
use crate::ast::*;
use crate::codegen::{CodeGenerator, Instruction, Value};
use crate::error::{Result, CompilerError, SemanticError, SemanticErrorKind};
use std::collections::HashMap;

impl CodeGenerator {
    /// Generate code for generic functions with monomorphization
    pub fn generate_generic_function(&mut self, function: &Function, type_args: &[Type]) -> Result<Vec<Instruction>> {
        let mut instructions = Vec::new();
        
        // Create monomorphized version of the function
        let monomorphized = self.monomorphize_function(function, type_args)?;
        
        // Generate function prologue
        instructions.push(Instruction::Debug(format!("Generic function: {}<{:?}>", 
            function.name.name, type_args)));
        
        // Generate parameter handling
        for (i, param) in monomorphized.parameters.iter().enumerate() {
            instructions.push(Instruction::Store(i as u32));
        }
        
        // Generate function body
        let body_instructions = self.generate_block(&monomorphized.body)?;
        instructions.extend(body_instructions);
        
        // Generate function epilogue
        if monomorphized.return_type.is_none() {
            instructions.push(Instruction::Push(Value::Unit));
        }
        instructions.push(Instruction::Return);
        
        Ok(instructions)
    }
    
    /// Monomorphize a generic function for specific type arguments
    fn monomorphize_function(&self, function: &Function, type_args: &[Type]) -> Result<Function> {
        let mut type_map = HashMap::new();
        
        // Map type parameters to concrete types
        for (param, arg) in function.type_parameters.iter().zip(type_args.iter()) {
            type_map.insert(param.name.name.clone(), arg.clone());
        }
        
        // Substitute types throughout the function
        let mut monomorphized = function.clone();
        
        // Substitute parameter types
        for param in &mut monomorphized.parameters {
            param.param_type = self.substitute_type(&param.param_type, &type_map);
        }
        
        // Substitute return type
        if let Some(return_type) = &mut monomorphized.return_type {
            *return_type = self.substitute_type(return_type, &type_map);
        }
        
        // Substitute types in function body
        monomorphized.body = self.substitute_block(&monomorphized.body, &type_map)?;
        
        Ok(monomorphized)
    }
    
    /// Substitute types in a block
    fn substitute_block(&self, block: &Block, type_map: &HashMap<String, Type>) -> Result<Block> {
        let mut new_statements = Vec::new();
        
        for stmt in &block.statements {
            new_statements.push(self.substitute_statement(stmt, type_map)?);
        }
        
        Ok(Block {
            statements: new_statements,
            location: block.location.clone(),
        })
    }
    
    /// Substitute types in a statement
    fn substitute_statement(&self, stmt: &Statement, type_map: &HashMap<String, Type>) -> Result<Statement> {
        match stmt {
            Statement::Let(let_stmt) => {
                let mut new_let = let_stmt.clone();
                if let Some(type_ann) = &mut new_let.type_annotation {
                    *type_ann = self.substitute_type(type_ann, type_map);
                }
                if let Some(value) = &mut new_let.value {
                    *value = self.substitute_expression(value, type_map)?;
                }
                Ok(Statement::Let(new_let))
            }
            Statement::Expression(expr) => {
                Ok(Statement::Expression(self.substitute_expression(expr, type_map)?))
            }
            Statement::Return(ret_stmt) => {
                let mut new_ret = ret_stmt.clone();
                if let Some(value) = &mut new_ret.value {
                    *value = self.substitute_expression(value, type_map)?;
                }
                Ok(Statement::Return(new_ret))
            }
            _ => Ok(stmt.clone()), // Other statements don't need type substitution
        }
    }
    
    /// Substitute types in an expression
    fn substitute_expression(&self, expr: &Expression, type_map: &HashMap<String, Type>) -> Result<Expression> {
        match expr {
            Expression::Call(call) => {
                let mut new_call = call.clone();
                new_call.function = Box::new(self.substitute_expression(&new_call.function, type_map)?);
                for arg in &mut new_call.arguments {
                    *arg = self.substitute_expression(arg, type_map)?;
                }
                Ok(Expression::Call(new_call))
            }
            Expression::MethodCall(method_call) => {
                let mut new_method = method_call.clone();
                new_method.object = Box::new(self.substitute_expression(&new_method.object, type_map)?);
                for arg in &mut new_method.arguments {
                    *arg = self.substitute_expression(arg, type_map)?;
                }
                Ok(Expression::MethodCall(new_method))
            }
            _ => Ok(expr.clone()), // Other expressions handled as needed
        }
    }
    
    /// Substitute a type using the type map
    fn substitute_type(&self, ty: &Type, type_map: &HashMap<String, Type>) -> Type {
        match ty {
            Type::Custom(name) => {
                if let Some(concrete_type) = type_map.get(name) {
                    concrete_type.clone()
                } else {
                    ty.clone()
                }
            }
            Type::Generic { base, args } => {
                let new_base = Box::new(self.substitute_type(base, type_map));
                let new_args = args.iter().map(|arg| self.substitute_type(arg, type_map)).collect();
                Type::Generic { base: new_base, args: new_args }
            }
            Type::Array(element_type, size) => {
                Type::Array(Box::new(self.substitute_type(element_type, type_map)), *size)
            }
            Type::Tuple(types) => {
                Type::Tuple(types.iter().map(|t| self.substitute_type(t, type_map)).collect())
            }
            _ => ty.clone(),
        }
    }
    
    /// Generate code for async functions
    pub fn generate_async_function(&mut self, function: &Function) -> Result<Vec<Instruction>> {
        let mut instructions = Vec::new();
        
        if function.is_async {
            // Generate async function wrapper
            instructions.push(Instruction::Debug("Async function start".to_string()));
            
            // Create async context
            instructions.push(Instruction::Push(Value::Struct {
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert("state".to_string(), Value::String("ready".to_string()));
                    fields.insert("result".to_string(), Value::Unit);
                    fields
                },
            }));
            
            // Generate state machine for async execution
            instructions.extend(self.generate_async_state_machine(function)?);
            
            instructions.push(Instruction::Debug("Async function end".to_string()));
        } else {
            // Regular function generation
            instructions.extend(self.generate_function_body(function)?);
        }
        
        Ok(instructions)
    }
    
    /// Generate async state machine
    fn generate_async_state_machine(&mut self, function: &Function) -> Result<Vec<Instruction>> {
        let mut instructions = Vec::new();
        
        // State machine loop
        instructions.push(Instruction::Debug("State machine start".to_string()));
        
        // Generate states for each await point
        let await_points = self.find_await_points(&function.body);
        
        for (i, _) in await_points.iter().enumerate() {
            instructions.push(Instruction::Push(Value::U32(i as u32)));
            instructions.push(Instruction::Debug(format!("State {}", i)));
            
            // Generate code for this state
            instructions.extend(self.generate_async_state(i, function)?);
        }
        
        instructions.push(Instruction::Debug("State machine end".to_string()));
        Ok(instructions)
    }
    
    /// Find await points in function body
    fn find_await_points(&self, block: &Block) -> Vec<SourceLocation> {
        let mut await_points = Vec::new();
        
        for stmt in &block.statements {
            self.find_await_in_statement(stmt, &mut await_points);
        }
        
        await_points
    }
    
    /// Find await expressions in a statement
    fn find_await_in_statement(&self, stmt: &Statement, await_points: &mut Vec<SourceLocation>) {
        match stmt {
            Statement::Expression(expr) => {
                self.find_await_in_expression(expr, await_points);
            }
            Statement::Let(let_stmt) => {
                if let Some(value) = &let_stmt.value {
                    self.find_await_in_expression(value, await_points);
                }
            }
            Statement::Return(ret_stmt) => {
                if let Some(value) = &ret_stmt.value {
                    self.find_await_in_expression(value, await_points);
                }
            }
            _ => {} // Other statements handled as needed
        }
    }
    
    /// Find await expressions in an expression
    fn find_await_in_expression(&self, expr: &Expression, await_points: &mut Vec<SourceLocation>) {
        match expr {
            Expression::Await(await_expr) => {
                await_points.push(await_expr.location.clone());
            }
            Expression::Call(call) => {
                for arg in &call.arguments {
                    self.find_await_in_expression(arg, await_points);
                }
            }
            _ => {} // Other expressions handled as needed
        }
    }
    
    /// Generate code for a specific async state
    fn generate_async_state(&mut self, state_id: usize, function: &Function) -> Result<Vec<Instruction>> {
        let mut instructions = Vec::new();
        
        instructions.push(Instruction::Debug(format!("Generating state {}", state_id)));
        
        // Generate code up to the await point
        instructions.extend(self.generate_function_body(function)?);
        
        // Yield control
        instructions.push(Instruction::Push(Value::String("yield".to_string())));
        
        Ok(instructions)
    }
    
    /// Generate code for await expressions
    pub fn generate_await_expression(&mut self, await_expr: &AwaitExpression) -> Result<Vec<Instruction>> {
        let mut instructions = Vec::new();
        
        // Generate the future expression
        instructions.extend(self.generate_expression(&await_expr.expression)?);
        
        // Generate await instruction
        instructions.push(Instruction::Debug("Await point".to_string()));
        instructions.push(Instruction::Push(Value::String("awaiting".to_string())));
        
        // Suspend execution and yield control
        instructions.push(Instruction::Push(Value::Bool(true))); // Suspend flag
        
        Ok(instructions)
    }
    
    /// Generate code for macro expansions
    pub fn generate_macro_expansion(&mut self, macro_inv: &MacroInvocation) -> Result<Vec<Instruction>> {
        let mut instructions = Vec::new();
        
        match macro_inv.name.name.as_str() {
            "println" => {
                // Generate println macro
                for arg in &macro_inv.args {
                    instructions.extend(self.generate_expression(arg)?);
                }
                instructions.push(Instruction::Debug("println!".to_string()));
                instructions.push(Instruction::Push(Value::Unit));
            }
            "assert" => {
                // Generate assert macro
                if !macro_inv.args.is_empty() {
                    instructions.extend(self.generate_expression(&macro_inv.args[0])?);
                    
                    let message = if macro_inv.args.len() > 1 {
                        // Custom message
                        self.generate_expression(&macro_inv.args[1])?
                    } else {
                        // Default message
                        vec![Instruction::Push(Value::String("Assertion failed".to_string()))]
                    };
                    
                    instructions.extend(message);
                    instructions.push(Instruction::Assert);
                }
            }
            "require" => {
                // Generate require macro (similar to assert but for contracts)
                if !macro_inv.args.is_empty() {
                    instructions.extend(self.generate_expression(&macro_inv.args[0])?);
                    
                    let message = if macro_inv.args.len() > 1 {
                        self.generate_expression(&macro_inv.args[1])?
                    } else {
                        vec![Instruction::Push(Value::String("Requirement failed".to_string()))]
                    };
                    
                    instructions.extend(message);
                    instructions.push(Instruction::Require);
                }
            }
            "debug" => {
                // Generate debug macro
                for arg in &macro_inv.args {
                    instructions.extend(self.generate_expression(arg)?);
                }
                instructions.push(Instruction::Debug("debug!".to_string()));
                instructions.push(Instruction::Push(Value::Unit));
            }
            _ => {
                // Unknown macro - generate error or placeholder
                instructions.push(Instruction::Debug(format!("Unknown macro: {}", macro_inv.name.name)));
                instructions.push(Instruction::Push(Value::Unit));
            }
        }
        
        Ok(instructions)
    }
    
    /// Generate code for enhanced pattern matching
    pub fn generate_enhanced_pattern_match(&mut self, match_stmt: &MatchStatement) -> Result<Vec<Instruction>> {
        let mut instructions = Vec::new();
        
        // Generate match expression
        instructions.extend(self.generate_expression(&match_stmt.expression)?);
        
        // Generate jump table for pattern matching
        let mut jump_labels = Vec::new();
        let mut pattern_code = Vec::new();
        
        for (i, arm) in match_stmt.arms.iter().enumerate() {
            let label = format!("match_arm_{}", i);
            jump_labels.push(label.clone());
            
            // Generate pattern matching code
            let mut arm_instructions = Vec::new();
            arm_instructions.push(Instruction::Debug(format!("Match arm {}", i)));
            
            // Generate pattern check
            arm_instructions.extend(self.generate_pattern_check(&arm.pattern)?);
            
            // Generate guard check if present
            if let Some(guard) = &arm.guard {
                arm_instructions.extend(self.generate_expression(guard)?);
                arm_instructions.push(Instruction::JumpIfNot(format!("match_arm_{}", i + 1)));
            }
            
            // Generate arm body
            arm_instructions.extend(self.generate_block(&arm.body)?);
            arm_instructions.push(Instruction::Jump("match_end".to_string()));
            
            pattern_code.push((label, arm_instructions));
        }
        
        // Generate pattern matching dispatch
        for (label, code) in pattern_code {
            instructions.push(Instruction::Debug(format!("Label: {}", label)));
            instructions.extend(code);
        }
        
        instructions.push(Instruction::Debug("match_end".to_string()));
        Ok(instructions)
    }
    
    /// Generate pattern checking code
    fn generate_pattern_check(&mut self, pattern: &Pattern) -> Result<Vec<Instruction>> {
        let mut instructions = Vec::new();
        
        match pattern {
            Pattern::Literal(literal) => {
                instructions.push(Instruction::Dup); // Duplicate match value
                instructions.extend(self.generate_literal(literal)?);
                instructions.push(Instruction::Eq);
            }
            Pattern::Identifier(_) => {
                // Always matches, bind the value
                instructions.push(Instruction::Push(Value::Bool(true)));
            }
            Pattern::Wildcard => {
                // Always matches
                instructions.push(Instruction::Push(Value::Bool(true)));
            }
            Pattern::Guard { pattern, condition } => {
                // Check base pattern first
                instructions.extend(self.generate_pattern_check(pattern)?);
                
                // Then check guard condition
                instructions.extend(self.generate_expression(condition)?);
                instructions.push(Instruction::And);
            }
            Pattern::Or(patterns) => {
                // Check each pattern with OR logic
                for (i, pat) in patterns.iter().enumerate() {
                    if i > 0 {
                        instructions.push(Instruction::Or);
                    }
                    instructions.extend(self.generate_pattern_check(pat)?);
                }
            }
            Pattern::Array { patterns, rest: _ } => {
                // Array pattern matching
                instructions.push(Instruction::Dup); // Duplicate array
                instructions.push(Instruction::Push(Value::U32(patterns.len() as u32)));
                instructions.push(Instruction::Debug("Array length check".to_string()));
                
                // Check each element
                for (i, pat) in patterns.iter().enumerate() {
                    instructions.push(Instruction::Dup); // Duplicate array
                    instructions.push(Instruction::Push(Value::U32(i as u32)));
                    instructions.push(Instruction::Debug("Array index".to_string()));
                    instructions.extend(self.generate_pattern_check(pat)?);
                }
            }
            Pattern::Range { start, end, inclusive } => {
                // Range pattern matching
                instructions.push(Instruction::Dup); // Duplicate value
                
                if let Some(start_pat) = start {
                    instructions.extend(self.generate_pattern_check(start_pat)?);
                    instructions.push(Instruction::Ge); // Greater or equal
                }
                
                if let Some(end_pat) = end {
                    instructions.push(Instruction::Dup); // Duplicate value again
                    instructions.extend(self.generate_pattern_check(end_pat)?);
                    if *inclusive {
                        instructions.push(Instruction::Le); // Less or equal
                    } else {
                        instructions.push(Instruction::Lt); // Less than
                    }
                    
                    if start.is_some() {
                        instructions.push(Instruction::And);
                    }
                }
            }
            _ => {
                // Other patterns - simplified implementation
                instructions.push(Instruction::Push(Value::Bool(true)));
            }
        }
        
        Ok(instructions)
    }
    
    /// Generate code for operator implementations
    pub fn generate_operator_impl(&mut self, op_impl: &OperatorImpl) -> Result<Vec<Instruction>> {
        let mut instructions = Vec::new();
        
        instructions.push(Instruction::Debug(format!("Operator impl: {:?} for {:?}", 
            op_impl.operator, op_impl.target_type)));
        
        // Generate each method in the operator implementation
        for method in &op_impl.methods {
            instructions.extend(self.generate_function_body(method)?);
        }
        
        Ok(instructions)
    }
    
    /// Generate literal value
    fn generate_literal(&mut self, literal: &Literal) -> Result<Vec<Instruction>> {
        let instruction = match literal {
            Literal::Integer(n) => Instruction::Push(Value::I32(*n as i32)),
            Literal::Float(f) => Instruction::Push(Value::F64(*f)),
            Literal::String(s) => Instruction::Push(Value::String(s.clone())),
            Literal::Boolean(b) => Instruction::Push(Value::Bool(*b)),
            Literal::Address(addr) => Instruction::Push(Value::Address(*addr)),
        };
        
        Ok(vec![instruction])
    }
    
    /// Generate function body (placeholder for existing method)
    fn generate_function_body(&mut self, function: &Function) -> Result<Vec<Instruction>> {
        self.generate_block(&function.body)
    }
}
