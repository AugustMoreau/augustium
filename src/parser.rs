// Parser - converts tokens into an AST
// Uses recursive descent parsing

use crate::ast::*;
use crate::error::{CompilerError, ParseError, ParseErrorKind, Result, SourceLocation};
use crate::lexer::{Token, TokenType};

// Parser keeps track of tokens and current position
pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
    errors: Vec<ParseError>,
    panic_mode: bool, // when we hit an error, panic mode helps us recover
}

impl Parser {
    // Set up a new parser with tokens from the lexer
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            current: 0,
            errors: Vec::new(),
            panic_mode: false,
        }
    }
    
    /// Get all collected errors
    pub fn get_errors(&self) -> &Vec<ParseError> {
        &self.errors
    }
    
    /// Add an error to the collection
    fn add_error(&mut self, error: ParseError) {
        self.errors.push(error);
        self.panic_mode = true;
    }
    
    /// Synchronize parser state after an error
    fn synchronize(&mut self) {
        self.panic_mode = false;
        
        while !self.is_at_end() {
            if self.peek().token_type == TokenType::Semicolon {
                self.advance();
                return;
            }
            
            match &self.peek().token_type {
                TokenType::Contract | TokenType::Fn | TokenType::Struct |
                TokenType::Enum | TokenType::Trait | TokenType::Impl |
                TokenType::Use | TokenType::Const | TokenType::Mod |
                TokenType::If | TokenType::While | TokenType::For |
                TokenType::Return | TokenType::Let | TokenType::Match => {
                    return;
                }
                _ => {
                    self.advance();
                }
            }
        }
    }
    
    /// Parse the tokens into a complete source file AST with error recovery
    pub fn parse(&mut self) -> Result<SourceFile> {
        let start_location = self.current_location();
        let mut items = Vec::new();
        
        while !self.is_at_end() {
            let start_index = self.current;
            
            // Skip newlines at the top level
            if self.check(&TokenType::Newline) {
                self.advance();
            } else {
                match self.parse_item() {
                    Ok(item) => items.push(item),
                    Err(error) => {
                        if let CompilerError::ParseError(parse_error) = &error {
                            self.add_error(parse_error.clone());
                            self.synchronize();
                        } else {
                            return Err(error);
                        }
                    }
                }
            }
            
            // SAFETY GUARD: Ensure the parser always makes progress to avoid infinite loops.
            // If no tokens were consumed during this iteration, advance by one token to
            // prevent hanging on unexpected input.
            if self.current == start_index {
                self.advance();
            }
        }
        
        // If we collected any errors, return the first one but still provide the AST
        if !self.errors.is_empty() {
            // For now, return the first error. In the future, we could return a special
            // result type that contains both the AST and the errors
            return Err(CompilerError::ParseError(self.errors[0].clone()));
        }
        
        Ok(SourceFile {
            items,
            location: start_location,
        })
    }
    
    /// Parse a top-level item with error recovery
    fn parse_item(&mut self) -> Result<Item> {
        if self.panic_mode {
            self.synchronize();
        }
        
        match &self.peek().token_type {
            TokenType::Contract => self.parse_contract().map(Item::Contract),
            TokenType::Fn => self.parse_function().map(Item::Function),
            TokenType::Struct => self.parse_struct().map(Item::Struct),
            TokenType::Enum => self.parse_enum().map(Item::Enum),
            TokenType::Trait => self.parse_trait().map(Item::Trait),
            TokenType::Impl => self.parse_impl().map(Item::Impl),
            TokenType::Use => self.parse_use().map(Item::Use),
            TokenType::Const => self.parse_const().map(Item::Const),
            TokenType::Mod => self.parse_module().map(Item::Module),
            _ => {
                let token = self.peek();
                Err(ParseError::new(
                    ParseErrorKind::UnexpectedToken,
                    token.location.clone(),
                    format!("Expected item declaration, found {:?}", token.token_type),
                ).into())
            }
        }
    }
    
    /// Parse a contract declaration
    fn parse_contract(&mut self) -> Result<Contract> {
        let start_location = self.current_location();
        self.consume(&TokenType::Contract, "Expected 'contract'")?;
        
        let name = self.parse_identifier()?;
        
        self.consume(&TokenType::LeftBrace, "Expected '{' after contract name")?;
        
        let mut fields = Vec::new();
        let mut functions = Vec::new();
        let mut events = Vec::new();
        let mut modifiers = Vec::new();
        
        while !self.check(&TokenType::RightBrace) && !self.is_at_end() {
            // Skip newlines
            if self.check(&TokenType::Newline) {
                self.advance();
                continue;
            }
            
            match &self.peek().token_type {
                TokenType::Fn => functions.push(self.parse_function()?),
                TokenType::Event => events.push(self.parse_event()?),
                TokenType::Constructor => {
                    // Parse constructor as a special function
                    functions.push(self.parse_constructor()?);
                }
                TokenType::Let => {
                    // Field declaration with let keyword
                    fields.push(self.parse_field()?);
                }
                TokenType::Pub | TokenType::Priv => {
                    // Could be a field or function
                    let checkpoint = self.current;
                    let _visibility = self.parse_visibility();
                    
                    if self.check(&TokenType::Fn) {
                        self.current = checkpoint; // Reset
                        functions.push(self.parse_function()?);
                    } else if self.check(&TokenType::Let) {
                        self.current = checkpoint; // Reset
                        fields.push(self.parse_field()?);
                    } else {
                        self.current = checkpoint; // Reset
                        fields.push(self.parse_field()?);
                    }
                }
                _ => {
                    // Try to parse as field, but provide better error message
                    let token = self.peek();
                    return Err(ParseError::new(
                        ParseErrorKind::UnexpectedToken,
                        token.location.clone(),
                        format!("Unexpected token in contract body: {:?}. Expected function, field, or event declaration.", token.token_type),
                    ).into());
                }
            }
        }
        
        self.consume(&TokenType::RightBrace, "Expected '}' after contract body")?;
        
        Ok(Contract {
            name,
            fields,
            functions,
            events,
            modifiers,
            location: start_location,
        })
    }
    
    /// Parse a function declaration
    fn parse_function(&mut self) -> Result<Function> {
        let start_location = self.current_location();
        
        let visibility = if self.check(&TokenType::Pub) || self.check(&TokenType::Priv) {
            self.parse_visibility()
        } else {
            Visibility::Private
        };
        
        self.consume(&TokenType::Fn, "Expected 'fn'")?;
        
        let name = self.parse_identifier()?;
        
        self.consume(&TokenType::LeftParen, "Expected '(' after function name")?;
        
        let mut parameters = Vec::new();
        if !self.check(&TokenType::RightParen) {
            loop {
                parameters.push(self.parse_parameter()?);
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
        }
        
        self.consume(&TokenType::RightParen, "Expected ')' after parameters")?;
        
        let return_type = if self.match_token(&TokenType::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };
        
        let body = self.parse_block()?;
        
        Ok(Function {
            name,
            parameters,
            return_type,
            body,
            visibility,
            mutability: Mutability::Immutable, // TODO: Parse mutability
            attributes: Vec::new(), // TODO: Parse attributes
            location: start_location,
        })
    }
    
    /// Parse a constructor declaration
    fn parse_constructor(&mut self) -> Result<Function> {
        let start_location = self.current_location();
        
        let visibility = if self.check(&TokenType::Pub) || self.check(&TokenType::Priv) {
            self.parse_visibility()
        } else {
            Visibility::Private
        };
        
        self.consume(&TokenType::Constructor, "Expected 'constructor'")?;
        
        self.consume(&TokenType::LeftParen, "Expected '(' after constructor")?;
        
        let mut parameters = Vec::new();
        if !self.check(&TokenType::RightParen) {
            loop {
                parameters.push(self.parse_parameter()?);
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
        }
        
        self.consume(&TokenType::RightParen, "Expected ')' after parameters")?;
        
        let body = self.parse_block()?;
        
        Ok(Function {
            name: Identifier::new("constructor".to_string(), start_location.clone()),
            parameters,
            return_type: None, // Constructors don't have return types
            body,
            visibility,
            mutability: Mutability::Immutable,
            attributes: Vec::new(),
            location: start_location,
        })
    }
    
    /// Parse a struct declaration
    fn parse_struct(&mut self) -> Result<Struct> {
        let start_location = self.current_location();
        
        let visibility = if self.check(&TokenType::Pub) || self.check(&TokenType::Priv) {
            self.parse_visibility()
        } else {
            Visibility::Private
        };
        
        self.consume(&TokenType::Struct, "Expected 'struct'")?;
        
        let name = self.parse_identifier()?;
        
        self.consume(&TokenType::LeftBrace, "Expected '{' after struct name")?;
        
        let mut fields = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.is_at_end() {
            if self.check(&TokenType::Newline) {
                self.advance();
                continue;
            }
            
            fields.push(self.parse_field()?);
            
            if !self.check(&TokenType::RightBrace) {
                self.consume(&TokenType::Comma, "Expected ',' after struct field")?;
            }
        }
        
        self.consume(&TokenType::RightBrace, "Expected '}' after struct body")?;
        
        Ok(Struct {
            name,
            fields,
            visibility,
            location: start_location,
        })
    }
    
    /// Parse an enum declaration
    fn parse_enum(&mut self) -> Result<Enum> {
        let start_location = self.current_location();
        
        let visibility = if self.check(&TokenType::Pub) || self.check(&TokenType::Priv) {
            self.parse_visibility()
        } else {
            Visibility::Private
        };
        
        self.consume(&TokenType::Enum, "Expected 'enum'")?;
        
        let name = self.parse_identifier()?;
        
        self.consume(&TokenType::LeftBrace, "Expected '{' after enum name")?;
        
        let mut variants = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.is_at_end() {
            if self.check(&TokenType::Newline) {
                self.advance();
                continue;
            }
            
            variants.push(self.parse_enum_variant()?);
            
            if !self.check(&TokenType::RightBrace) {
                self.consume(&TokenType::Comma, "Expected ',' after enum variant")?;
            }
        }
        
        self.consume(&TokenType::RightBrace, "Expected '}' after enum body")?;
        
        Ok(Enum {
            name,
            variants,
            visibility,
            location: start_location,
        })
    }
    
    /// Parse an enum variant
    fn parse_enum_variant(&mut self) -> Result<EnumVariant> {
        let start_location = self.current_location();
        let name = self.parse_identifier()?;
        
        let fields = if self.match_token(&TokenType::LeftParen) {
            let mut field_types = Vec::new();
            if !self.check(&TokenType::RightParen) {
                loop {
                    field_types.push(self.parse_type()?);
                    if !self.match_token(&TokenType::Comma) {
                        break;
                    }
                }
            }
            self.consume(&TokenType::RightParen, "Expected ')' after enum variant fields")?;
            Some(field_types)
        } else {
            None
        };
        
        Ok(EnumVariant {
            name,
            fields,
            location: start_location,
        })
    }
    
    /// Parse a trait declaration
    fn parse_trait(&mut self) -> Result<Trait> {
        let start_location = self.current_location();
        
        let visibility = if self.check(&TokenType::Pub) || self.check(&TokenType::Priv) {
            self.parse_visibility()
        } else {
            Visibility::Private
        };
        
        self.consume(&TokenType::Trait, "Expected 'trait'")?;
        
        let name = self.parse_identifier()?;
        
        self.consume(&TokenType::LeftBrace, "Expected '{' after trait name")?;
        
        let mut functions = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.is_at_end() {
            if self.check(&TokenType::Newline) {
                self.advance();
                continue;
            }
            
            functions.push(self.parse_trait_function()?);
        }
        
        self.consume(&TokenType::RightBrace, "Expected '}' after trait body")?;
        
        Ok(Trait {
            name,
            functions,
            visibility,
            location: start_location,
        })
    }
    
    /// Parse a trait function signature
    fn parse_trait_function(&mut self) -> Result<TraitFunction> {
        let start_location = self.current_location();
        
        self.consume(&TokenType::Fn, "Expected 'fn'")?;
        
        let name = self.parse_identifier()?;
        
        self.consume(&TokenType::LeftParen, "Expected '(' after function name")?;
        
        let mut parameters = Vec::new();
        if !self.check(&TokenType::RightParen) {
            loop {
                parameters.push(self.parse_parameter()?);
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
        }
        
        self.consume(&TokenType::RightParen, "Expected ')' after parameters")?;
        
        let return_type = if self.match_token(&TokenType::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };
        
        self.consume(&TokenType::Semicolon, "Expected ';' after trait function signature")?;
        
        Ok(TraitFunction {
            name,
            parameters,
            return_type,
            location: start_location,
        })
    }
    
    /// Parse an impl block
    fn parse_impl(&mut self) -> Result<Impl> {
        let start_location = self.current_location();
        
        self.consume(&TokenType::Impl, "Expected 'impl'")?;
        
        // Parse optional trait name
        let trait_name = if self.peek_ahead(1).map(|t| &t.token_type) == Some(&TokenType::For) {
            let trait_name = self.parse_identifier()?;
            self.consume(&TokenType::For, "Expected 'for' after trait name")?;
            Some(trait_name)
        } else {
            None
        };
        
        let type_name = self.parse_identifier()?;
        
        self.consume(&TokenType::LeftBrace, "Expected '{' after impl declaration")?;
        
        let mut functions = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.is_at_end() {
            if self.check(&TokenType::Newline) {
                self.advance();
                continue;
            }
            
            functions.push(self.parse_function()?);
        }
        
        self.consume(&TokenType::RightBrace, "Expected '}' after impl body")?;
        
        Ok(Impl {
            trait_name,
            type_name,
            functions,
            location: start_location,
        })
    }
    
    /// Parse a use declaration
    fn parse_use(&mut self) -> Result<UseDeclaration> {
        let start_location = self.current_location();
        
        self.consume(&TokenType::Use, "Expected 'use'")?;
        
        let mut path = vec![self.parse_identifier()?];
        
        while self.match_token(&TokenType::DoubleColon) {
            path.push(self.parse_identifier()?);
        }
        
        let alias = if self.match_token(&TokenType::As) {
            Some(self.parse_identifier()?)
        } else {
            None
        };
        
        self.consume(&TokenType::Semicolon, "Expected ';' after use declaration")?;
        
        Ok(UseDeclaration {
            path,
            alias,
            location: start_location,
        })
    }
    
    /// Parse a const declaration
    fn parse_const(&mut self) -> Result<ConstDeclaration> {
        let start_location = self.current_location();
        
        let visibility = if self.check(&TokenType::Pub) || self.check(&TokenType::Priv) {
            self.parse_visibility()
        } else {
            Visibility::Private
        };
        
        self.consume(&TokenType::Const, "Expected 'const'")?;
        
        let name = self.parse_identifier()?;
        
        self.consume(&TokenType::Colon, "Expected ':' after const name")?;
        
        let type_annotation = self.parse_type()?;
        
        self.consume(&TokenType::Equal, "Expected '=' after const type")?;
        
        let value = self.parse_expression()?;
        
        self.consume(&TokenType::Semicolon, "Expected ';' after const declaration")?;
        
        Ok(ConstDeclaration {
            name,
            type_annotation,
            value,
            visibility,
            location: start_location,
        })
    }
    
    /// Parse a module declaration
    fn parse_module(&mut self) -> Result<Module> {
        let start_location = self.current_location();
        
        let visibility = if self.check(&TokenType::Pub) || self.check(&TokenType::Priv) {
            self.parse_visibility()
        } else {
            Visibility::Private
        };
        
        self.consume(&TokenType::Mod, "Expected 'mod'")?;
        
        let name = self.parse_identifier()?;
        
        self.consume(&TokenType::LeftBrace, "Expected '{' after module name")?;
        
        let mut items = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.is_at_end() {
            if self.check(&TokenType::Newline) {
                self.advance();
                continue;
            }
            
            items.push(self.parse_item()?);
        }
        
        self.consume(&TokenType::RightBrace, "Expected '}' after module body")?;
        
        Ok(Module {
            name,
            items,
            visibility,
            location: start_location,
        })
    }
    
    /// Parse an event declaration
    fn parse_event(&mut self) -> Result<Event> {
        let start_location = self.current_location();
        
        self.consume(&TokenType::Event, "Expected 'event'")?;
        
        let name = self.parse_identifier()?;
        
        self.consume(&TokenType::LeftParen, "Expected '(' after event name")?;
        
        let mut fields = Vec::new();
        if !self.check(&TokenType::RightParen) {
            loop {
                fields.push(self.parse_event_field()?);
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
        }
        
        self.consume(&TokenType::RightParen, "Expected ')' after event fields")?;
        
        self.consume(&TokenType::Semicolon, "Expected ';' after event declaration")?;
        
        Ok(Event {
            name,
            fields,
            location: start_location,
        })
    }
    
    /// Parse an event field
    fn parse_event_field(&mut self) -> Result<EventField> {
        let start_location = self.current_location();
        
        let indexed = self.match_token(&TokenType::Indexed);
        
        let name = self.parse_identifier()?;
        
        self.consume(&TokenType::Colon, "Expected ':' after event field name")?;
        
        let type_annotation = self.parse_type()?;
        
        Ok(EventField {
            name,
            type_annotation,
            indexed,
            location: start_location,
        })
    }
    
    /// Parse a field declaration
    fn parse_field(&mut self) -> Result<Field> {
        let start_location = self.current_location();
        
        let visibility = if self.check(&TokenType::Pub) || self.check(&TokenType::Priv) {
            self.parse_visibility()
        } else {
            Visibility::Private
        };
        
        // Handle 'let' keyword for field declarations
        if self.match_token(&TokenType::Let) {
            // Consume optional 'mut' keyword (not stored in current Field struct)
            self.match_token(&TokenType::Mut);
        }
        
        let name = self.parse_identifier()?;
        
        self.consume(&TokenType::Colon, "Expected ':' after field name")?;
        
        let type_annotation = self.parse_type()?;
        
        // Parse optional default value (not stored in current Field struct)
        if self.match_token(&TokenType::Equal) {
            let _default_value = self.parse_expression()?;
        }
        
        // Consume semicolon if present
        self.match_token(&TokenType::Semicolon);
        
        Ok(Field {
            name,
            type_annotation,
            visibility,
            location: start_location,
        })
    }
    
    /// Parse a function parameter
    fn parse_parameter(&mut self) -> Result<Parameter> {
        let start_location = self.current_location();
        
        let name = self.parse_identifier()?;
        
        self.consume(&TokenType::Colon, "Expected ':' after parameter name")?;
        
        let type_annotation = self.parse_type()?;
        
        Ok(Parameter {
            name,
            type_annotation,
            location: start_location,
        })
    }
    
    /// Parse a type annotation
    fn parse_type(&mut self) -> Result<Type> {
        match &self.peek().token_type {
            TokenType::U8 => { self.advance(); Ok(Type::U8) }
            TokenType::U16 => { self.advance(); Ok(Type::U16) }
            TokenType::U32 => { self.advance(); Ok(Type::U32) }
            TokenType::U64 => { self.advance(); Ok(Type::U64) }
            TokenType::U128 => { self.advance(); Ok(Type::U128) }
            TokenType::U256 => { self.advance(); Ok(Type::U256) }
            TokenType::I8 => { self.advance(); Ok(Type::I8) }
            TokenType::I16 => { self.advance(); Ok(Type::I16) }
            TokenType::I32 => { self.advance(); Ok(Type::I32) }
            TokenType::I64 => { self.advance(); Ok(Type::I64) }
            TokenType::I128 => { self.advance(); Ok(Type::I128) }
            TokenType::I256 => { self.advance(); Ok(Type::I256) }
            TokenType::Bool => { self.advance(); Ok(Type::Bool) }
            TokenType::String => { self.advance(); Ok(Type::String) }
            TokenType::Address => { self.advance(); Ok(Type::Address) }
            TokenType::Bytes => { self.advance(); Ok(Type::Bytes) }
            TokenType::LeftBracket => {
                self.advance(); // consume '['
                let element_type = Box::new(self.parse_type()?);
                
                let size = if self.match_token(&TokenType::Semicolon) {
                    match &self.peek().token_type {
                        TokenType::IntegerLiteral(n) => {
                            let size = *n;
                            self.advance();
                            Some(size)
                        }
                        _ => return Err(ParseError::new(
                            ParseErrorKind::UnexpectedToken,
                            self.current_location(),
                            "Expected array size".to_string(),
                        ).into()),
                    }
                } else {
                    None
                };
                
                self.consume(&TokenType::RightBracket, "Expected ']' after array type")?;
                
                Ok(Type::Array { element_type, size })
            }
            TokenType::LeftParen => {
                self.advance(); // consume '('
                let mut types = Vec::new();
                
                if !self.check(&TokenType::RightParen) {
                    loop {
                        types.push(self.parse_type()?);
                        if !self.match_token(&TokenType::Comma) {
                            break;
                        }
                    }
                }
                
                self.consume(&TokenType::RightParen, "Expected ')' after tuple type")?;
                
                Ok(Type::Tuple(types))
            }
            TokenType::Identifier(_) => {
                let name = self.parse_identifier()?;
                
                // Check for generic types like Option<T> or Result<T, E>
                if name.name == "Option" && self.match_token(&TokenType::Less) {
                    let inner_type = Box::new(self.parse_type()?);
                    self.consume(&TokenType::Greater, "Expected '>' after Option type")?;
                    Ok(Type::Option(inner_type))
                } else if name.name == "Result" && self.match_token(&TokenType::Less) {
                    let ok_type = Box::new(self.parse_type()?);
                    self.consume(&TokenType::Comma, "Expected ',' in Result type")?;
                    let err_type = Box::new(self.parse_type()?);
                    self.consume(&TokenType::Greater, "Expected '>' after Result type")?;
                    Ok(Type::Result { ok_type, err_type })
                } else {
                    Ok(Type::Named(name))
                }
            }
            _ => {
                let token = self.peek();
                Err(ParseError::new(
                    ParseErrorKind::UnexpectedToken,
                    token.location.clone(),
                    format!("Expected type, found {:?}", token.token_type),
                ).into())
            }
        }
    }
    
    /// Parse visibility modifier
    fn parse_visibility(&mut self) -> Visibility {
        if self.match_token(&TokenType::Pub) {
            Visibility::Public
        } else if self.match_token(&TokenType::Priv) {
            Visibility::Private
        } else {
            Visibility::Private
        }
    }
    
    /// Parse a block of statements
    fn parse_block(&mut self) -> Result<Block> {
        let start_location = self.current_location();
        
        self.consume(&TokenType::LeftBrace, "Expected '{'")?;
        
        let mut statements = Vec::new();
        
        while !self.check(&TokenType::RightBrace) && !self.is_at_end() {
            // Skip newlines
            if self.check(&TokenType::Newline) {
                self.advance();
                continue;
            }
            
            statements.push(self.parse_statement()?);
        }
        
        self.consume(&TokenType::RightBrace, "Expected '}'")?;
        
        Ok(Block {
            statements,
            location: start_location,
        })
    }
    
    /// Parse a statement
    fn parse_statement(&mut self) -> Result<Statement> {
        match &self.peek().token_type {
            TokenType::Let => Ok(Statement::Let(self.parse_let_statement()?)),
            TokenType::Return => Ok(Statement::Return(self.parse_return_statement()?)),
            TokenType::If => Ok(Statement::If(self.parse_if_statement()?)),
            TokenType::While => Ok(Statement::While(self.parse_while_statement()?)),
            TokenType::For => Ok(Statement::For(self.parse_for_statement()?)),
            TokenType::Match => Ok(Statement::Match(self.parse_match_statement()?)),
            TokenType::Break => Ok(Statement::Break(self.parse_break_statement()?)),
            TokenType::Continue => Ok(Statement::Continue(self.parse_continue_statement()?)),
            TokenType::Emit => Ok(Statement::Emit(self.parse_emit_statement()?)),
            TokenType::Require => Ok(Statement::Require(self.parse_require_statement()?)),
            TokenType::Assert => Ok(Statement::Assert(self.parse_assert_statement()?)),
            TokenType::Revert => Ok(Statement::Revert(self.parse_revert_statement()?)),
            _ => {
                // Expression statement
                let expr = self.parse_expression()?;
                self.consume(&TokenType::Semicolon, "Expected ';' after expression")?;
                Ok(Statement::Expression(expr))
            }
        }
    }
    
    /// Parse a let statement
    fn parse_let_statement(&mut self) -> Result<LetStatement> {
        let start_location = self.current_location();
        
        self.consume(&TokenType::Let, "Expected 'let'")?;
        
        let mutable = self.match_token(&TokenType::Mut);
        
        let name = self.parse_identifier()?;
        
        let type_annotation = if self.match_token(&TokenType::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        
        let value = if self.match_token(&TokenType::Equal) {
            Some(self.parse_expression()?)
        } else {
            None
        };
        
        self.consume(&TokenType::Semicolon, "Expected ';' after let statement")?;
        
        Ok(LetStatement {
            name,
            type_annotation,
            value,
            mutable,
            location: start_location,
        })
    }
    
    /// Parse a return statement
    fn parse_return_statement(&mut self) -> Result<ReturnStatement> {
        let start_location = self.current_location();
        
        self.consume(&TokenType::Return, "Expected 'return'")?;
        
        let value = if !self.check(&TokenType::Semicolon) {
            Some(self.parse_expression()?)
        } else {
            None
        };
        
        self.consume(&TokenType::Semicolon, "Expected ';' after return statement")?;
        
        Ok(ReturnStatement {
            value,
            location: start_location,
        })
    }
    
    /// Parse an if statement
    fn parse_if_statement(&mut self) -> Result<IfStatement> {
        let start_location = self.current_location();
        
        self.consume(&TokenType::If, "Expected 'if'")?;
        
        let condition = self.parse_expression()?;
        
        let then_block = self.parse_block()?;
        
        let else_block = if self.match_token(&TokenType::Else) {
            if self.check(&TokenType::If) {
                Some(Box::new(Statement::If(self.parse_if_statement()?)))
            } else {
                Some(Box::new(Statement::Expression(Expression::Block(self.parse_block()?))))
            }
        } else {
            None
        };
        
        Ok(IfStatement {
            condition,
            then_block,
            else_block,
            location: start_location,
        })
    }
    
    /// Parse a while statement
    fn parse_while_statement(&mut self) -> Result<WhileStatement> {
        let start_location = self.current_location();
        
        self.consume(&TokenType::While, "Expected 'while'")?;
        
        let condition = self.parse_expression()?;
        
        let body = self.parse_block()?;
        
        Ok(WhileStatement {
            condition,
            body,
            location: start_location,
        })
    }
    
    /// Parse a for statement
    fn parse_for_statement(&mut self) -> Result<ForStatement> {
        let start_location = self.current_location();
        
        self.consume(&TokenType::For, "Expected 'for'")?;
        
        let variable = self.parse_identifier()?;
        
        self.consume(&TokenType::In, "Expected 'in' after for variable")?;
        
        let iterable = self.parse_expression()?;
        
        let body = self.parse_block()?;
        
        Ok(ForStatement {
            variable,
            iterable,
            body,
            location: start_location,
        })
    }
    
    /// Parse a match statement
    fn parse_match_statement(&mut self) -> Result<MatchStatement> {
        let start_location = self.current_location();
        
        self.consume(&TokenType::Match, "Expected 'match'")?;
        
        let expression = self.parse_expression()?;
        
        self.consume(&TokenType::LeftBrace, "Expected '{' after match expression")?;
        
        let mut arms = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.is_at_end() {
            if self.check(&TokenType::Newline) {
                self.advance();
                continue;
            }
            
            arms.push(self.parse_match_arm()?);
        }
        
        self.consume(&TokenType::RightBrace, "Expected '}' after match arms")?;
        
        Ok(MatchStatement {
            expression,
            arms,
            location: start_location,
        })
    }
    
    /// Parse a match arm
    fn parse_match_arm(&mut self) -> Result<MatchArm> {
        let start_location = self.current_location();
        
        let pattern = self.parse_pattern()?;
        
        let guard = if self.match_token(&TokenType::If) {
            Some(self.parse_expression()?)
        } else {
            None
        };
        
        self.consume(&TokenType::FatArrow, "Expected '=>' after match pattern")?;
        
        let body = self.parse_block()?;
        
        Ok(MatchArm {
            pattern,
            guard,
            body,
            location: start_location,
        })
    }
    
    /// Parse a pattern
    fn parse_pattern(&mut self) -> Result<Pattern> {
        match &self.peek().token_type {
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
            TokenType::Identifier(_) => {
                let name = self.parse_identifier()?;
                Ok(Pattern::Identifier(name))
            }
            TokenType::Underscore => {
                self.advance();
                Ok(Pattern::Wildcard)
            }
            _ => {
                let token = self.peek();
                Err(ParseError::new(
                    ParseErrorKind::UnexpectedToken,
                    token.location.clone(),
                    format!("Expected pattern, found {:?}", token.token_type),
                ).into())
            }
        }
    }
    
    /// Parse break statement
    fn parse_break_statement(&mut self) -> Result<BreakStatement> {
        let start_location = self.current_location();
        
        self.consume(&TokenType::Break, "Expected 'break'")?;
        self.consume(&TokenType::Semicolon, "Expected ';' after break")?;
        
        Ok(BreakStatement {
            location: start_location,
        })
    }
    
    /// Parse continue statement
    fn parse_continue_statement(&mut self) -> Result<ContinueStatement> {
        let start_location = self.current_location();
        
        self.consume(&TokenType::Continue, "Expected 'continue'")?;
        self.consume(&TokenType::Semicolon, "Expected ';' after continue")?;
        
        Ok(ContinueStatement {
            location: start_location,
        })
    }
    
    /// Parse emit statement
    fn parse_emit_statement(&mut self) -> Result<EmitStatement> {
        let start_location = self.current_location();
        
        self.consume(&TokenType::Emit, "Expected 'emit'")?;
        
        let event = self.parse_identifier()?;
        
        self.consume(&TokenType::LeftParen, "Expected '(' after event name")?;
        
        let mut arguments = Vec::new();
        if !self.check(&TokenType::RightParen) {
            loop {
                arguments.push(self.parse_expression()?);
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
        }
        
        self.consume(&TokenType::RightParen, "Expected ')' after event arguments")?;
        self.consume(&TokenType::Semicolon, "Expected ';' after emit statement")?;
        
        Ok(EmitStatement {
            event,
            arguments,
            location: start_location,
        })
    }
    
    /// Parse require statement
    fn parse_require_statement(&mut self) -> Result<RequireStatement> {
        let start_location = self.current_location();
        
        self.consume(&TokenType::Require, "Expected 'require'")?;
        
        self.consume(&TokenType::LeftParen, "Expected '(' after require")?;
        
        let condition = self.parse_expression()?;
        
        let message = if self.match_token(&TokenType::Comma) {
            Some(self.parse_expression()?)
        } else {
            None
        };
        
        self.consume(&TokenType::RightParen, "Expected ')' after require arguments")?;
        self.consume(&TokenType::Semicolon, "Expected ';' after require statement")?;
        
        Ok(RequireStatement {
            condition,
            message,
            location: start_location,
        })
    }
    
    /// Parse assert statement
    fn parse_assert_statement(&mut self) -> Result<AssertStatement> {
        let start_location = self.current_location();
        
        self.consume(&TokenType::Assert, "Expected 'assert'")?;
        
        self.consume(&TokenType::LeftParen, "Expected '(' after assert")?;
        
        let condition = self.parse_expression()?;
        
        let message = if self.match_token(&TokenType::Comma) {
            Some(self.parse_expression()?)
        } else {
            None
        };
        
        self.consume(&TokenType::RightParen, "Expected ')' after assert arguments")?;
        self.consume(&TokenType::Semicolon, "Expected ';' after assert statement")?;
        
        Ok(AssertStatement {
            condition,
            message,
            location: start_location,
        })
    }
    
    /// Parse revert statement
    fn parse_revert_statement(&mut self) -> Result<RevertStatement> {
        let start_location = self.current_location();
        
        self.consume(&TokenType::Revert, "Expected 'revert'")?;
        
        let message = if self.match_token(&TokenType::LeftParen) {
            let msg = Some(self.parse_expression()?);
            self.consume(&TokenType::RightParen, "Expected ')' after revert message")?;
            msg
        } else {
            None
        };
        
        self.consume(&TokenType::Semicolon, "Expected ';' after revert statement")?;
        
        Ok(RevertStatement {
            message,
            location: start_location,
        })
    }
    
    /// Parse an expression
    fn parse_expression(&mut self) -> Result<Expression> {
        self.parse_assignment()
    }
    
    /// Parse assignment expression
    fn parse_assignment(&mut self) -> Result<Expression> {
        let expr = self.parse_logical_or()?;
        
        if let Some(op) = self.match_assignment_operator() {
            let value = Box::new(self.parse_assignment()?);
            let location = expr.location().clone();
            
            return Ok(Expression::Assignment(AssignmentExpression {
                target: Box::new(expr),
                operator: op,
                value,
                location,
            }));
        }
        
        Ok(expr)
    }
    
    /// Parse logical OR expression
    fn parse_logical_or(&mut self) -> Result<Expression> {
        let mut expr = self.parse_logical_and()?;
        
        while self.match_token(&TokenType::Or) {
            let location = expr.location().clone();
            let right = Box::new(self.parse_logical_and()?);
            expr = Expression::Binary(BinaryExpression {
                left: Box::new(expr),
                operator: BinaryOperator::Or,
                right,
                location,
            });
        }
        
        Ok(expr)
    }
    
    /// Parse logical AND expression
    fn parse_logical_and(&mut self) -> Result<Expression> {
        let mut expr = self.parse_equality()?;
        
        while self.match_token(&TokenType::And) {
            let location = expr.location().clone();
            let right = Box::new(self.parse_equality()?);
            expr = Expression::Binary(BinaryExpression {
                left: Box::new(expr),
                operator: BinaryOperator::And,
                right,
                location,
            });
        }
        
        Ok(expr)
    }
    
    /// Parse equality expression
    fn parse_equality(&mut self) -> Result<Expression> {
        let mut expr = self.parse_comparison()?;
        
        while let Some(op) = self.match_equality_operator() {
            let location = expr.location().clone();
            let right = Box::new(self.parse_comparison()?);
            expr = Expression::Binary(BinaryExpression {
                left: Box::new(expr),
                operator: op,
                right,
                location,
            });
        }
        
        Ok(expr)
    }
    
    /// Parse comparison expression
    fn parse_comparison(&mut self) -> Result<Expression> {
        let mut expr = self.parse_bitwise_or()?;
        
        while let Some(op) = self.match_comparison_operator() {
            let location = expr.location().clone();
            let right = Box::new(self.parse_bitwise_or()?);
            expr = Expression::Binary(BinaryExpression {
                left: Box::new(expr),
                operator: op,
                right,
                location,
            });
        }
        
        Ok(expr)
    }
    
    /// Parse bitwise OR expression
    fn parse_bitwise_or(&mut self) -> Result<Expression> {
        let mut expr = self.parse_bitwise_xor()?;
        
        while self.match_token(&TokenType::BitOr) {
            let location = expr.location().clone();
            let right = Box::new(self.parse_bitwise_xor()?);
            expr = Expression::Binary(BinaryExpression {
                left: Box::new(expr),
                operator: BinaryOperator::BitOr,
                right,
                location,
            });
        }
        
        Ok(expr)
    }
    
    /// Parse bitwise XOR expression
    fn parse_bitwise_xor(&mut self) -> Result<Expression> {
        let mut expr = self.parse_bitwise_and()?;
        
        while self.match_token(&TokenType::BitXor) {
            let location = expr.location().clone();
            let right = Box::new(self.parse_bitwise_and()?);
            expr = Expression::Binary(BinaryExpression {
                left: Box::new(expr),
                operator: BinaryOperator::BitXor,
                right,
                location,
            });
        }
        
        Ok(expr)
    }
    
    /// Parse bitwise AND expression
    fn parse_bitwise_and(&mut self) -> Result<Expression> {
        let mut expr = self.parse_shift()?;
        
        while self.match_token(&TokenType::BitAnd) {
            let location = expr.location().clone();
            let right = Box::new(self.parse_shift()?);
            expr = Expression::Binary(BinaryExpression {
                left: Box::new(expr),
                operator: BinaryOperator::BitAnd,
                right,
                location,
            });
        }
        
        Ok(expr)
    }
    
    /// Parse shift expression
    fn parse_shift(&mut self) -> Result<Expression> {
        let mut expr = self.parse_term()?;
        
        while let Some(op) = self.match_shift_operator() {
            let location = expr.location().clone();
            let right = Box::new(self.parse_term()?);
            expr = Expression::Binary(BinaryExpression {
                left: Box::new(expr),
                operator: op,
                right,
                location,
            });
        }
        
        Ok(expr)
    }
    
    /// Parse term expression (+ and -)
    fn parse_term(&mut self) -> Result<Expression> {
        let mut expr = self.parse_factor()?;
        
        while let Some(op) = self.match_term_operator() {
            let location = expr.location().clone();
            let right = Box::new(self.parse_factor()?);
            expr = Expression::Binary(BinaryExpression {
                left: Box::new(expr),
                operator: op,
                right,
                location,
            });
        }
        
        Ok(expr)
    }
    
    /// Parse factor expression (*, /, %)
    fn parse_factor(&mut self) -> Result<Expression> {
        let mut expr = self.parse_unary()?;
        
        while let Some(op) = self.match_factor_operator() {
            let location = expr.location().clone();
            let right = Box::new(self.parse_unary()?);
            expr = Expression::Binary(BinaryExpression {
                left: Box::new(expr),
                operator: op,
                right,
                location,
            });
        }
        
        Ok(expr)
    }
    
    /// Parse unary expression
    fn parse_unary(&mut self) -> Result<Expression> {
        if let Some(op) = self.match_unary_operator() {
            let start_location = self.current_location();
            let operand = Box::new(self.parse_unary()?);
            return Ok(Expression::Unary(UnaryExpression {
                operator: op,
                operand,
                location: start_location,
            }));
        }
        
        self.parse_postfix()
    }
    
    /// Parse postfix expression (function calls, field access, indexing)
    fn parse_postfix(&mut self) -> Result<Expression> {
        let mut expr = self.parse_primary()?;
        
        loop {
            match &self.peek().token_type {
                TokenType::LeftParen => {
                    // Function call
                    self.advance();
                    let mut arguments = Vec::new();
                    
                    if !self.check(&TokenType::RightParen) {
                        loop {
                            arguments.push(self.parse_expression()?);
                            if !self.match_token(&TokenType::Comma) {
                                break;
                            }
                        }
                    }
                    
                    self.consume(&TokenType::RightParen, "Expected ')' after function arguments")?;
                    
                    let location = expr.location().clone();
                    expr = Expression::Call(CallExpression {
                        function: Box::new(expr),
                        arguments,
                        location,
                    });
                }
                TokenType::Dot => {
                    // Field access
                    self.advance();
                    let field = self.parse_identifier()?;
                    let location = expr.location().clone();
                    expr = Expression::FieldAccess(FieldAccessExpression {
                        object: Box::new(expr),
                        field,
                        location,
                    });
                }
                TokenType::LeftBracket => {
                    // Index access
                    self.advance();
                    let index = Box::new(self.parse_expression()?);
                    self.consume(&TokenType::RightBracket, "Expected ']' after index")?;
                    let location = expr.location().clone();
                    expr = Expression::Index(IndexExpression {
                        object: Box::new(expr),
                        index,
                        location,
                    });
                }
                _ => break,
            }
        }
        
        Ok(expr)
    }
    
    /// Parse primary expression
    fn parse_primary(&mut self) -> Result<Expression> {
        match &self.peek().token_type {
            // Literal values
            TokenType::IntegerLiteral(n) => {
                let n = *n;
                self.advance();
                Ok(Expression::Literal(Literal::Integer(n)))
            }
            TokenType::StringLiteral(s) => {
                let s = s.clone();
                self.advance();
                Ok(Expression::Literal(Literal::String(s)))
            }
            TokenType::BooleanLiteral(b) => {
                let b = *b;
                self.advance();
                Ok(Expression::Literal(Literal::Boolean(b)))
            }
            TokenType::AddressLiteral(a) => {
                let a = a.clone();
                self.advance();
                Ok(Expression::Literal(Literal::Address(a)))
            }
            // Treat built-in primitive type keywords like identifiers so that expressions
            // such as `i64(0)` or `u128(5)` parse as a call expression in parse_postfix.
            TokenType::I8 | TokenType::I16 | TokenType::I32 | TokenType::I64 |
            TokenType::I128 | TokenType::I256 | TokenType::U8 | TokenType::U16 |
            TokenType::U32 | TokenType::U64 | TokenType::U128 | TokenType::U256 |
            TokenType::Bool | TokenType::String | TokenType::Address | TokenType::Bytes |
            TokenType::Identifier(_) => {
                // Build an Identifier from the raw token text
                let tok = self.peek();
                let ident = Identifier::new(tok.raw.clone(), tok.location.clone());
                self.advance();
                Ok(Expression::Identifier(ident))
            }
            TokenType::LeftParen => {
                self.advance();
                // Check for empty tuple – "()"
                if self.check(&TokenType::RightParen) {
                    self.advance();
                    return Ok(Expression::Tuple(TupleExpression {
                        elements: Vec::new(),
                        location: self.current_location(),
                    }));
                }

                let expr = self.parse_expression()?;

                // Tuple literal of two or more values
                if self.match_token(&TokenType::Comma) {
                    let mut elements = vec![expr];
                    if !self.check(&TokenType::RightParen) {
                        loop {
                            elements.push(self.parse_expression()?);
                            if !self.match_token(&TokenType::Comma) {
                                break;
                            }
                        }
                    }
                    self.consume(&TokenType::RightParen, "Expected ')' after tuple")?;
                    Ok(Expression::Tuple(TupleExpression {
                        elements,
                        location: self.current_location(),
                    }))
                } else {
                    // Simple parenthesised expression
                    self.consume(&TokenType::RightParen, "Expected ')' after expression")?;
                    Ok(expr)
                }
            }
            TokenType::LeftBracket => {
                let start_location = self.current_location();
                self.advance();
                let mut elements = Vec::new();
                if !self.check(&TokenType::RightBracket) {
                    loop {
                        elements.push(self.parse_expression()?);
                        if !self.match_token(&TokenType::Comma) {
                            break;
                        }
                    }
                }
                self.consume(&TokenType::RightBracket, "Expected ']' after array")?;
                Ok(Expression::Array(ArrayExpression { elements, location: start_location }))
            }
            TokenType::LeftBrace => {
                Ok(Expression::Block(self.parse_block()?))
            }
            _ => {
                let token = self.peek();
                Err(ParseError::new(
                    ParseErrorKind::UnexpectedToken,
                    token.location.clone(),
                    format!("Unexpected token in expression: {:?}", token.token_type),
                ).into())
            }
        }
    }
/*

            TokenType::IntegerLiteral(n) => {
                let n = *n;
                self.advance();
                Ok(Expression::Literal(Literal::Integer(n)))
            }
            TokenType::StringLiteral(s) => {
                let s = s.clone();
                self.advance();
                Ok(Expression::Literal(Literal::String(s)))
            }
            TokenType::BooleanLiteral(b) => {
                let b = *b;
                self.advance();
                Ok(Expression::Literal(Literal::Boolean(b)))
            }
                    self.advance();
                    return Ok(Expression::Tuple(TupleExpression {
                        elements: Vec::new(),
                        location: self.current_location(),
                    }));
                }
                
                let expr = self.parse_expression()?;
                
                // Check if it's a tuple
                if self.match_token(&TokenType::Comma) {
                    let mut elements = vec![expr];
                    
                    if !self.check(&TokenType::RightParen) {
                        loop {
                            elements.push(self.parse_expression()?);
                            if !self.match_token(&TokenType::Comma) {
                                break;
                            }
                        }
                    }
                    
                    self.consume(&TokenType::RightParen, "Expected ')' after tuple")?;
                    
                    Ok(Expression::Tuple(TupleExpression {
                        elements,
                        location: self.current_location(),
                    }))
                } else {
                    self.consume(&TokenType::RightParen, "Expected ')' after expression")?;
                    Ok(expr)
                }
            }
            TokenType::LeftBracket => {
                let start_location = self.current_location();
                self.advance();
                
                let mut elements = Vec::new();
                
                if !self.check(&TokenType::RightBracket) {
                    loop {
                        elements.push(self.parse_expression()?);
                        if !self.match_token(&TokenType::Comma) {
                            break;
                        }
                    }
                }
                
                self.consume(&TokenType::RightBracket, "Expected ']' after array")?;
                
                Ok(Expression::Array(ArrayExpression {
                    elements,
                    location: start_location,
                }))
            }
            TokenType::LeftBrace => {
                Ok(Expression::Block(self.parse_block()?))
            }
            _ => {
                let token = self.peek();
                Err(ParseError::new(
                    ParseErrorKind::UnexpectedToken,
                    token.location.clone(),
                    format!("Unexpected token in expression: {:?}", token.token_type),
                ).into())
            }
        }
    }
    
*/
    /// Parse an identifier
    fn parse_identifier(&mut self) -> Result<Identifier> {
        match &self.peek().token_type {
            TokenType::Identifier(name) => {
                let name = name.clone();
                let location = self.current_location();
                self.advance();
                Ok(Identifier::new(name, location))
            }
            _ => {
                let token = self.peek();
                Err(ParseError::new(
                    ParseErrorKind::UnexpectedToken,
                    token.location.clone(),
                    format!("Expected identifier, found {:?}", token.token_type),
                ).into())
            }
        }
    }
    
    // Helper methods for operator matching
    
    fn match_assignment_operator(&mut self) -> Option<AssignmentOperator> {
        match &self.peek().token_type {
            TokenType::Equal => {
                self.advance();
                Some(AssignmentOperator::Assign)
            }
            TokenType::PlusEqual => {
                self.advance();
                Some(AssignmentOperator::AddAssign)
            }
            TokenType::MinusEqual => {
                self.advance();
                Some(AssignmentOperator::SubtractAssign)
            }
            TokenType::StarEqual => {
                self.advance();
                Some(AssignmentOperator::MultiplyAssign)
            }
            TokenType::SlashEqual => {
                self.advance();
                Some(AssignmentOperator::DivideAssign)
            }
            _ => None,
        }
    }
    
    fn match_equality_operator(&mut self) -> Option<BinaryOperator> {
        match &self.peek().token_type {
            TokenType::EqualEqual => {
                self.advance();
                Some(BinaryOperator::Equal)
            }
            TokenType::NotEqual => {
                self.advance();
                Some(BinaryOperator::NotEqual)
            }
            _ => None,
        }
    }
    
    fn match_comparison_operator(&mut self) -> Option<BinaryOperator> {
        match &self.peek().token_type {
            TokenType::Greater => {
                self.advance();
                Some(BinaryOperator::Greater)
            }
            TokenType::GreaterEqual => {
                self.advance();
                Some(BinaryOperator::GreaterEqual)
            }
            TokenType::Less => {
                self.advance();
                Some(BinaryOperator::Less)
            }
            TokenType::LessEqual => {
                self.advance();
                Some(BinaryOperator::LessEqual)
            }
            _ => None,
        }
    }
    
    fn match_shift_operator(&mut self) -> Option<BinaryOperator> {
        match &self.peek().token_type {
            TokenType::LeftShift => {
                self.advance();
                Some(BinaryOperator::LeftShift)
            }
            TokenType::RightShift => {
                self.advance();
                Some(BinaryOperator::RightShift)
            }
            _ => None,
        }
    }
    
    fn match_term_operator(&mut self) -> Option<BinaryOperator> {
        match &self.peek().token_type {
            TokenType::Plus => {
                self.advance();
                Some(BinaryOperator::Add)
            }
            TokenType::Minus => {
                self.advance();
                Some(BinaryOperator::Subtract)
            }
            _ => None,
        }
    }
    
    fn match_factor_operator(&mut self) -> Option<BinaryOperator> {
        match &self.peek().token_type {
            TokenType::Star => {
                self.advance();
                Some(BinaryOperator::Multiply)
            }
            TokenType::Slash => {
                self.advance();
                Some(BinaryOperator::Divide)
            }
            TokenType::Percent => {
                self.advance();
                Some(BinaryOperator::Modulo)
            }
            _ => None,
        }
    }
    
    fn match_unary_operator(&mut self) -> Option<UnaryOperator> {
        match &self.peek().token_type {
            TokenType::Not => {
                self.advance();
                Some(UnaryOperator::Not)
            }
            TokenType::Minus => {
                self.advance();
                Some(UnaryOperator::Minus)
            }
            TokenType::BitNot => {
                self.advance();
                Some(UnaryOperator::BitNot)
            }
            _ => None,
        }
    }
    
    // Utility methods
    
    /// Check if we're at the end of the token stream
    fn is_at_end(&self) -> bool {
        self.current >= self.tokens.len() || self.peek().token_type == TokenType::Eof
    }
    
    /// Get the current token without advancing
    fn peek(&self) -> &Token {
        if self.current >= self.tokens.len() {
            // Return a dummy EOF token
            static EOF_TOKEN: Token = Token {
                token_type: TokenType::Eof,
                location: SourceLocation {
                    file: String::new(),
                    line: 0,
                    column: 0,
                    offset: 0,
                },
                raw: String::new(),
            };
            &EOF_TOKEN
        } else {
            &self.tokens[self.current]
        }
    }
    
    /// Look ahead at the token at the given offset
    fn peek_ahead(&self, offset: usize) -> Option<&Token> {
        let index = self.current + offset;
        if index < self.tokens.len() {
            Some(&self.tokens[index])
        } else {
            None
        }
    }
    
    /// Advance to the next token and return the current one
    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        &self.tokens[self.current - 1]
    }
    
    /// Check if the current token matches the given type
    fn check(&self, token_type: &TokenType) -> bool {
        if self.is_at_end() {
            false
        } else {
            std::mem::discriminant(&self.peek().token_type) == std::mem::discriminant(token_type)
        }
    }
    
    /// If the current token matches, advance and return true
    fn match_token(&mut self, token_type: &TokenType) -> bool {
        if self.check(token_type) {
            self.advance();
            true
        } else {
            false
        }
    }
    
    /// Consume a token of the given type or return an error
    fn consume(&mut self, token_type: &TokenType, message: &str) -> Result<&Token> {
        if self.check(token_type) {
            Ok(self.advance())
        } else {
            let token = self.peek();
            Err(ParseError::new(
                ParseErrorKind::UnexpectedToken,
                token.location.clone(),
                format!("{}: expected {:?}, found {:?}", message, token_type, token.token_type),
            ).into())
        }
    }
    
    /// Consume a token with graceful error recovery
    fn consume_with_recovery(&mut self, token_type: &TokenType, message: &str) -> Option<&Token> {
        if self.check(token_type) {
            Some(self.advance())
        } else {
            let token = self.peek();
            let error = ParseError::new(
                ParseErrorKind::UnexpectedToken,
                token.location.clone(),
                format!("{}: expected {:?}, found {:?}", message, token_type, token.token_type),
            );
            self.add_error(error);
            
            // Try to recover by skipping to the next logical token
            self.advance();
            None
        }
    }
    
    /// Create a checkpoint for backtracking
    fn checkpoint(&self) -> usize {
        self.current
    }
    
    /// Restore to a checkpoint
    fn restore(&mut self, checkpoint: usize) {
        self.current = checkpoint;
    }
    
    /// Get the current source location
    fn current_location(&self) -> SourceLocation {
        self.peek().location.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    
    fn parse_source(source: &str) -> Result<SourceFile> {
        let mut lexer = Lexer::new(source, "test.aug");
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(tokens);
        parser.parse()
    }
    
    #[test]
    fn test_parse_simple_contract() {
        let source = r#"
            contract SimpleToken {
                let balance: u256;
                
                fn get_balance() -> u256 {
                    return self.balance;
                }
            }
        "#;
        
        let result = parse_source(source);
        assert!(result.is_ok());
        
        let ast = result.unwrap();
        assert_eq!(ast.items.len(), 1);
        
        if let Item::Contract(contract) = &ast.items[0] {
            assert_eq!(contract.name.name, "SimpleToken");
            assert_eq!(contract.fields.len(), 1);
            assert_eq!(contract.functions.len(), 1);
        } else {
            panic!("Expected contract item");
        }
    }
    
    #[test]
    fn test_parse_function() {
        let source = r#"
            fn add(a: u32, b: u32) -> u32 {
                return a + b;
            }
        "#;
        
        let result = parse_source(source);
        assert!(result.is_ok());
        
        let ast = result.unwrap();
        assert_eq!(ast.items.len(), 1);
        
        if let Item::Function(function) = &ast.items[0] {
            assert_eq!(function.name.name, "add");
            assert_eq!(function.parameters.len(), 2);
            assert!(function.return_type.is_some());
        } else {
            panic!("Expected function item");
        }
    }
    
    #[test]
    fn test_parse_struct() {
        let source = r#"
            struct Point {
                x: u32,
                y: u32,
            }
        "#;
        
        let result = parse_source(source);
        assert!(result.is_ok());
        
        let ast = result.unwrap();
        assert_eq!(ast.items.len(), 1);
        
        if let Item::Struct(struct_def) = &ast.items[0] {
            assert_eq!(struct_def.name.name, "Point");
            assert_eq!(struct_def.fields.len(), 2);
        } else {
            panic!("Expected struct item");
        }
    }
    
    #[test]
    fn test_parse_expressions() {
        let source = r#"
            fn test() {
                let x = 1 + 2 * 3;
                let y = (a && b) || c;
                let z = arr[0];
                let w = obj.field;
            }
        "#;
        
        let result = parse_source(source);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_parse_control_flow() {
        let source = r#"
            fn test() {
                if x > 0 {
                    return x;
                } else {
                    return 0;
                }
                
                while i < 10 {
                    i = i + 1;
                }
                
                for item in items {
                    process(item);
                }
            }
        "#;
        
        let result = parse_source(source);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_parse_error() {
        let source = "contract { invalid syntax }";
        
        let result = parse_source(source);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_error_recovery() {
        let source = r#"
            contract BadContract {
                invalid_field_syntax
                
                fn valid_function() -> u32 {
                    return 42;
                }
            }
            
            fn another_valid_function() -> u32 {
                return 100;
            }
        "#;
        
        let mut lexer = Lexer::new(source, "test.aug");
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        
        // Parse should fail but collect errors
        let result = parser.parse();
        assert!(result.is_err());
        
        // Should have collected at least one error
        assert!(!parser.get_errors().is_empty());
        
        // The parser should have attempted to recover and continue parsing
        // This test verifies that error recovery mechanisms are in place
    }
    
    #[test]
    fn test_checkpoint_restore() {
        let source = "fn test() {}";
        let mut lexer = Lexer::new(source, "test.aug");
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        
        let checkpoint = parser.checkpoint();
        parser.advance(); // Move forward
        parser.restore(checkpoint); // Restore
        
        // Should be back at the beginning
        assert_eq!(parser.current, 0);
    }
}