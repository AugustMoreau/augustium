// Parser extensions for generics, async, macros, and enhanced patterns
use crate::ast::*;
use crate::error::{ParseError, ParseErrorKind, Result};
use crate::lexer::TokenType;
use crate::parser::Parser;

impl Parser {
    /// Parse generic type parameters: <T, U: Trait, V: Clone + Send>
    pub fn parse_type_parameters(&mut self) -> Result<Vec<TypeParameter>> {
        let mut params = Vec::new();
        
        if !self.check(&TokenType::Greater) {
            loop {
                let name = self.parse_identifier()?;
                
                let mut bounds = Vec::new();
                if self.match_token(&TokenType::Colon) {
                    loop {
                        bounds.push(self.parse_type_bound()?);
                        if !self.match_token(&TokenType::Plus) {
                            break;
                        }
                    }
                }
                
                params.push(TypeParameter {
                    name,
                    bounds,
                    default: None,
                    location: self.current_location(),
                });
                
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
        }
        
        self.consume(&TokenType::Greater, "Expected '>' after type parameters")?;
        Ok(params)
    }
    
    /// Parse type bound: Trait or Trait<T>
    pub fn parse_type_bound(&mut self) -> Result<TypeBound> {
        let trait_name = self.parse_identifier()?;
        
        let type_args = if self.match_token(&TokenType::Less) {
            let mut args = Vec::new();
            if !self.check(&TokenType::Greater) {
                loop {
                    args.push(self.parse_type()?);
                    if !self.match_token(&TokenType::Comma) {
                        break;
                    }
                }
            }
            self.consume(&TokenType::Greater, "Expected '>' after type arguments")?;
            args
        } else {
            Vec::new()
        };
        
        Ok(TypeBound {
            trait_name,
            type_args,
            location: self.current_location(),
        })
    }
    
    /// Parse where clause: where T: Clone, U: Send + Sync
    pub fn parse_where_clause(&mut self) -> Result<WhereClause> {
        let mut predicates = Vec::new();
        
        loop {
            let type_param = self.parse_identifier()?;
            self.consume(&TokenType::Colon, "Expected ':' after type parameter in where clause")?;
            
            let mut bounds = Vec::new();
            loop {
                bounds.push(self.parse_type_bound()?);
                if !self.match_token(&TokenType::Plus) {
                    break;
                }
            }
            
            predicates.push(WherePredicate {
                type_param,
                bounds,
                location: self.current_location(),
            });
            
            if !self.match_token(&TokenType::Comma) {
                break;
            }
        }
        
        Ok(WhereClause {
            predicates,
            location: self.current_location(),
        })
    }
    
    /// Parse generic type with arguments: Vec<T>, HashMap<K, V>
    pub fn parse_generic_type(&mut self, base_type: Type) -> Result<Type> {
        self.consume(&TokenType::Less, "Expected '<' for generic type")?;
        
        let mut type_args = Vec::new();
        if !self.check(&TokenType::Greater) {
            loop {
                type_args.push(self.parse_type()?);
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
        }
        
        self.consume(&TokenType::Greater, "Expected '>' after type arguments")?;
        
        Ok(Type::Generic {
            base: Box::new(base_type),
            args: type_args,
        })
    }
    
    /// Parse await expression: expr.await
    pub fn parse_await_expression(&mut self, expr: Expression) -> Result<Expression> {
        self.consume(&TokenType::Dot, "Expected '.' before await")?;
        self.consume(&TokenType::Await, "Expected 'await'")?;
        
        Ok(Expression::Await(AwaitExpression {
            expression: Box::new(expr),
            location: self.current_location(),
        }))
    }
    
    /// Parse macro invocation: println!("hello"), assert!(condition)
    pub fn parse_macro_invocation(&mut self, name: Identifier) -> Result<Expression> {
        self.consume(&TokenType::Exclamation, "Expected '!' after macro name")?;
        self.consume(&TokenType::LeftParen, "Expected '(' after macro name")?;
        
        let mut args = Vec::new();
        if !self.check(&TokenType::RightParen) {
            loop {
                args.push(self.parse_expression()?);
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
        }
        
        self.consume(&TokenType::RightParen, "Expected ')' after macro arguments")?;
        
        Ok(Expression::MacroInvocation(MacroInvocation {
            name,
            args,
            location: self.current_location(),
        }))
    }
    
    /// Parse enhanced patterns with guards, or-patterns, etc.
    pub fn parse_enhanced_pattern(&mut self) -> Result<Pattern> {
        let base_pattern = self.parse_base_pattern()?;
        
        // Check for guards: pattern if condition
        if self.match_token(&TokenType::If) {
            let condition = self.parse_expression()?;
            return Ok(Pattern::Guard {
                pattern: Box::new(base_pattern),
                condition,
            });
        }
        
        // Check for or-patterns: pattern1 | pattern2
        if self.match_token(&TokenType::Pipe) {
            let mut patterns = vec![base_pattern];
            loop {
                patterns.push(self.parse_base_pattern()?);
                if !self.match_token(&TokenType::Pipe) {
                    break;
                }
            }
            return Ok(Pattern::Or(patterns));
        }
        
        // Check for binding patterns: name @ pattern
        if let Pattern::Identifier(name) = &base_pattern {
            if self.match_token(&TokenType::At) {
                let inner_pattern = self.parse_base_pattern()?;
                return Ok(Pattern::Binding {
                    name: name.clone(),
                    pattern: Box::new(inner_pattern),
                });
            }
        }
        
        Ok(base_pattern)
    }
    
    /// Parse base pattern without enhancements
    pub fn parse_base_pattern(&mut self) -> Result<Pattern> {
        match &self.peek().token_type {
            TokenType::Underscore => {
                self.advance();
                Ok(Pattern::Wildcard)
            }
            TokenType::IntegerLiteral(n) => {
                let n = *n;
                self.advance();
                Ok(Pattern::Literal(Literal::Integer(n)))
            }
            TokenType::StringLiteral(s) => {
                let s = s.clone();
                self.advance();
                Ok(Pattern::Literal(Literal::String(s)))
            }
            TokenType::BooleanLiteral(b) => {
                let b = *b;
                self.advance();
                Ok(Pattern::Literal(Literal::Boolean(b)))
            }
            TokenType::Identifier(name) => {
                let name = name.clone();
                self.advance();
                
                // Check for struct pattern: Point { x, y }
                if self.match_token(&TokenType::LeftBrace) {
                    let mut fields = Vec::new();
                    let mut rest = false;
                    
                    if !self.check(&TokenType::RightBrace) {
                        loop {
                            if self.match_token(&TokenType::DotDot) {
                                rest = true;
                                break;
                            }
                            
                            let field_name = self.parse_identifier()?;
                            let pattern = if self.match_token(&TokenType::Colon) {
                                self.parse_enhanced_pattern()?
                            } else {
                                Pattern::Identifier(field_name.clone())
                            };
                            
                            fields.push(FieldPattern {
                                name: field_name,
                                pattern,
                                shorthand: false,
                            });
                            
                            if !self.match_token(&TokenType::Comma) {
                                break;
                            }
                        }
                    }
                    
                    self.consume(&TokenType::RightBrace, "Expected '}' after struct pattern")?;
                    
                    Ok(Pattern::Struct {
                        name: Identifier { name, location: self.current_location() },
                        fields,
                        rest,
                    })
                } else {
                    Ok(Pattern::Identifier(Identifier { name, location: self.current_location() }))
                }
            }
            TokenType::LeftBracket => {
                self.advance();
                let mut patterns = Vec::new();
                let mut rest = None;
                
                if !self.check(&TokenType::RightBracket) {
                    loop {
                        if self.match_token(&TokenType::DotDot) {
                            if self.check(&TokenType::RightBracket) {
                                // [a, b, ..]
                                break;
                            } else {
                                // [a, ..rest]
                                rest = Some(Box::new(self.parse_enhanced_pattern()?));
                                break;
                            }
                        }
                        
                        patterns.push(self.parse_enhanced_pattern()?);
                        
                        if !self.match_token(&TokenType::Comma) {
                            break;
                        }
                    }
                }
                
                self.consume(&TokenType::RightBracket, "Expected ']' after array pattern")?;
                
                Ok(Pattern::Array { patterns, rest })
            }
            TokenType::LeftParen => {
                self.advance();
                let mut patterns = Vec::new();
                
                if !self.check(&TokenType::RightParen) {
                    loop {
                        patterns.push(self.parse_enhanced_pattern()?);
                        if !self.match_token(&TokenType::Comma) {
                            break;
                        }
                    }
                }
                
                self.consume(&TokenType::RightParen, "Expected ')' after tuple pattern")?;
                Ok(Pattern::Tuple(patterns))
            }
            TokenType::Ampersand => {
                self.advance();
                let mutable = self.match_token(&TokenType::Mut);
                let pattern = self.parse_base_pattern()?;
                Ok(Pattern::Reference {
                    mutable,
                    pattern: Box::new(pattern),
                })
            }
            TokenType::Star => {
                self.advance();
                let pattern = self.parse_base_pattern()?;
                Ok(Pattern::Deref(Box::new(pattern)))
            }
            _ => {
                // Try to parse range patterns: 1..=10, ..=5, 1..
                if self.check_range_pattern() {
                    self.parse_range_pattern()
                } else {
                    Err(ParseError::new(
                        ParseErrorKind::UnexpectedToken,
                        self.current_location(),
                        format!("Expected pattern, found {:?}", self.peek().token_type),
                    ).into())
                }
            }
        }
    }
    
    /// Check if current position starts a range pattern
    fn check_range_pattern(&self) -> bool {
        // Look for patterns like: 1..5, 1..=5, ..5, ..=5
        if let Some(next_token) = self.tokens.get(self.current + 1) {
            matches!(next_token.token_type, TokenType::DotDot | TokenType::DotDotEqual)
        } else {
            false
        }
    }
    
    /// Parse range pattern: 1..5, 1..=5, ..5, ..=5
    fn parse_range_pattern(&mut self) -> Result<Pattern> {
        let start = if self.check(&TokenType::DotDot) || self.check(&TokenType::DotDotEqual) {
            None
        } else {
            Some(Box::new(self.parse_base_pattern()?))
        };
        
        let inclusive = if self.match_token(&TokenType::DotDotEqual) {
            true
        } else if self.match_token(&TokenType::DotDot) {
            false
        } else {
            return Err(ParseError::new(
                ParseErrorKind::UnexpectedToken,
                self.current_location(),
                "Expected '..' or '..=' in range pattern".to_string(),
            ).into());
        };
        
        let end = if self.check(&TokenType::Comma) || self.check(&TokenType::RightParen) || 
                     self.check(&TokenType::RightBrace) || self.check(&TokenType::RightBracket) {
            None
        } else {
            Some(Box::new(self.parse_base_pattern()?))
        };
        
        Ok(Pattern::Range {
            start,
            end,
            inclusive,
        })
    }
    
    /// Parse operator implementation: impl Add for Point
    pub fn parse_operator_impl(&mut self) -> Result<Item> {
        let start_location = self.current_location();
        
        self.consume(&TokenType::Impl, "Expected 'impl'")?;
        
        let operator = self.parse_overloadable_operator()?;
        
        self.consume(&TokenType::For, "Expected 'for' after operator")?;
        
        let target_type = self.parse_type()?;
        
        self.consume(&TokenType::LeftBrace, "Expected '{' after operator impl")?;
        
        let mut methods = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.is_at_end() {
            methods.push(self.parse_function()?);
        }
        
        self.consume(&TokenType::RightBrace, "Expected '}' after operator impl")?;
        
        Ok(Item::OperatorImpl(OperatorImpl {
            operator,
            target_type,
            methods,
            location: start_location,
        }))
    }
    
    /// Parse overloadable operator
    fn parse_overloadable_operator(&mut self) -> Result<OverloadableOperator> {
        match &self.peek().token_type {
            TokenType::Plus => {
                self.advance();
                Ok(OverloadableOperator::Add)
            }
            TokenType::Minus => {
                self.advance();
                Ok(OverloadableOperator::Sub)
            }
            TokenType::Star => {
                self.advance();
                Ok(OverloadableOperator::Mul)
            }
            TokenType::Slash => {
                self.advance();
                Ok(OverloadableOperator::Div)
            }
            TokenType::Percent => {
                self.advance();
                Ok(OverloadableOperator::Rem)
            }
            TokenType::EqualEqual => {
                self.advance();
                Ok(OverloadableOperator::Eq)
            }
            TokenType::NotEqual => {
                self.advance();
                Ok(OverloadableOperator::Ne)
            }
            TokenType::Less => {
                self.advance();
                Ok(OverloadableOperator::Lt)
            }
            TokenType::LessEqual => {
                self.advance();
                Ok(OverloadableOperator::Le)
            }
            TokenType::Greater => {
                self.advance();
                Ok(OverloadableOperator::Gt)
            }
            TokenType::GreaterEqual => {
                self.advance();
                Ok(OverloadableOperator::Ge)
            }
            TokenType::LeftBracket => {
                self.advance();
                self.consume(&TokenType::RightBracket, "Expected ']' after '['")?;
                Ok(OverloadableOperator::Index)
            }
            _ => Err(ParseError::new(
                ParseErrorKind::UnexpectedToken,
                self.current_location(),
                format!("Expected overloadable operator, found {:?}", self.peek().token_type),
            ).into()),
        }
    }
}
