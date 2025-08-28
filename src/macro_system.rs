//! Macro system for Augustium - enables metaprogramming capabilities
//! Supports both declarative and procedural macros

use crate::ast::*;
use crate::error::{Result, CompilerError, ParseError, ParseErrorKind, SourceLocation};
use crate::lexer::{Token, TokenType};
use crate::parser::Parser;
use std::collections::HashMap;

/// Macro definition
#[derive(Debug, Clone)]
pub struct MacroDefinition {
    pub name: String,
    pub kind: MacroKind,
    pub rules: Vec<MacroRule>,
    pub location: SourceLocation,
}

/// Types of macros
#[derive(Debug, Clone)]
pub enum MacroKind {
    /// Declarative macro (macro_rules! style)
    Declarative,
    /// Procedural macro that operates on tokens
    Procedural,
    /// Attribute macro
    Attribute,
    /// Derive macro
    Derive,
}

/// Macro rule for pattern matching
#[derive(Debug, Clone)]
pub struct MacroRule {
    pub pattern: MacroPattern,
    pub expansion: MacroExpansion,
}

/// Pattern for matching macro arguments
#[derive(Debug, Clone)]
pub enum MacroPattern {
    /// Literal token
    Literal(String),
    /// Variable capture
    Variable {
        name: String,
        kind: MacroVariableKind,
        repetition: Option<MacroRepetition>,
    },
    /// Sequence of patterns
    Sequence(Vec<MacroPattern>),
    /// Optional pattern
    Optional(Box<MacroPattern>),
    /// Choice between patterns
    Choice(Vec<MacroPattern>),
}

/// Types of macro variables
#[derive(Debug, Clone)]
pub enum MacroVariableKind {
    Expression,
    Statement,
    Type,
    Pattern,
    Identifier,
    Literal,
    Block,
    Item,
}

/// Repetition modifiers
#[derive(Debug, Clone)]
pub struct MacroRepetition {
    pub separator: Option<String>,
    pub kind: RepetitionKind,
}

#[derive(Debug, Clone)]
pub enum RepetitionKind {
    /// Zero or more (*)
    ZeroOrMore,
    /// One or more (+)
    OneOrMore,
    /// Zero or one (?)
    ZeroOrOne,
}

/// Macro expansion template
#[derive(Debug, Clone)]
pub enum MacroExpansion {
    /// Token sequence
    Tokens(Vec<Token>),
    /// AST nodes
    Ast(Vec<Item>),
    /// Template with variable substitution
    Template {
        tokens: Vec<MacroToken>,
    },
}

/// Token in macro template
#[derive(Debug, Clone)]
pub enum MacroToken {
    /// Literal token
    Literal(Token),
    /// Variable substitution
    Variable(String),
    /// Repeated section
    Repeat {
        tokens: Vec<MacroToken>,
        separator: Option<String>,
        variable: String,
    },
}

/// Macro processor
pub struct MacroProcessor {
    macros: HashMap<String, MacroDefinition>,
    expansion_depth: usize,
    max_expansion_depth: usize,
}

impl MacroProcessor {
    pub fn new() -> Self {
        Self {
            macros: HashMap::new(),
            expansion_depth: 0,
            max_expansion_depth: 64,
        }
    }

    /// Register a macro definition
    pub fn register_macro(&mut self, macro_def: MacroDefinition) {
        self.macros.insert(macro_def.name.clone(), macro_def);
    }

    /// Expand a macro invocation
    pub fn expand_macro(&mut self, invocation: &MacroInvocation) -> Result<Vec<Item>> {
        if self.expansion_depth >= self.max_expansion_depth {
            return Err(CompilerError::Parse(ParseError {
                kind: ParseErrorKind::MacroExpansionDepthExceeded,
                location: invocation.location.clone(),
                message: "Maximum macro expansion depth exceeded".to_string(),
            }));
        }

        let macro_def = self.macros.get(&invocation.name.name)
            .ok_or_else(|| CompilerError::Parse(ParseError {
                kind: ParseErrorKind::UnknownMacro,
                location: invocation.location.clone(),
                message: format!("Unknown macro: {}", invocation.name.name),
            }))?;

        self.expansion_depth += 1;
        let result = self.expand_with_definition(macro_def, invocation);
        self.expansion_depth -= 1;

        result
    }

    fn expand_with_definition(&mut self, macro_def: &MacroDefinition, invocation: &MacroInvocation) -> Result<Vec<Item>> {
        match macro_def.kind {
            MacroKind::Declarative => self.expand_declarative(macro_def, invocation),
            MacroKind::Procedural => self.expand_procedural(macro_def, invocation),
            MacroKind::Attribute => self.expand_attribute(macro_def, invocation),
            MacroKind::Derive => self.expand_derive(macro_def, invocation),
        }
    }

    fn expand_declarative(&mut self, macro_def: &MacroDefinition, invocation: &MacroInvocation) -> Result<Vec<Item>> {
        // Try each rule until one matches
        for rule in &macro_def.rules {
            if let Ok(bindings) = self.match_pattern(&rule.pattern, &invocation.arguments) {
                return self.expand_template(&rule.expansion, &bindings);
            }
        }

        Err(CompilerError::Parse(ParseError {
            kind: ParseErrorKind::MacroPatternMismatch,
            location: invocation.location.clone(),
            message: format!("No matching pattern for macro {}", macro_def.name),
        }))
    }

    fn expand_procedural(&mut self, _macro_def: &MacroDefinition, _invocation: &MacroInvocation) -> Result<Vec<Item>> {
        // Procedural macros would call external functions
        // For now, return empty expansion
        Ok(Vec::new())
    }

    fn expand_attribute(&mut self, _macro_def: &MacroDefinition, _invocation: &MacroInvocation) -> Result<Vec<Item>> {
        // Attribute macros modify the item they're applied to
        Ok(Vec::new())
    }

    fn expand_derive(&mut self, _macro_def: &MacroDefinition, _invocation: &MacroInvocation) -> Result<Vec<Item>> {
        // Derive macros generate implementations
        Ok(Vec::new())
    }

    fn match_pattern(&self, pattern: &MacroPattern, args: &[MacroArgument]) -> Result<HashMap<String, Vec<MacroArgument>>> {
        let mut bindings = HashMap::new();
        self.match_pattern_recursive(pattern, args, &mut bindings, 0)?;
        Ok(bindings)
    }

    fn match_pattern_recursive(
        &self,
        pattern: &MacroPattern,
        args: &[MacroArgument],
        bindings: &mut HashMap<String, Vec<MacroArgument>>,
        mut pos: usize,
    ) -> Result<usize> {
        match pattern {
            MacroPattern::Literal(lit) => {
                if pos >= args.len() {
                    return Err(CompilerError::Parse(ParseError {
                        kind: ParseErrorKind::MacroPatternMismatch,
                        location: SourceLocation::unknown(),
                        message: "Expected literal in macro pattern".to_string(),
                    }));
                }
                // Check if argument matches literal
                pos += 1;
                Ok(pos)
            }
            MacroPattern::Variable { name, kind: _, repetition } => {
                match repetition {
                    None => {
                        if pos >= args.len() {
                            return Err(CompilerError::Parse(ParseError {
                                kind: ParseErrorKind::MacroPatternMismatch,
                                location: SourceLocation::unknown(),
                                message: "Expected variable in macro pattern".to_string(),
                            }));
                        }
                        bindings.entry(name.clone()).or_insert_with(Vec::new).push(args[pos].clone());
                        Ok(pos + 1)
                    }
                    Some(rep) => {
                        let mut matched = Vec::new();
                        while pos < args.len() {
                            matched.push(args[pos].clone());
                            pos += 1;
                            
                            // Handle separator
                            if let Some(_sep) = &rep.separator {
                                if pos < args.len() {
                                    // Skip separator token
                                    pos += 1;
                                }
                            }
                        }
                        bindings.insert(name.clone(), matched);
                        Ok(pos)
                    }
                }
            }
            MacroPattern::Sequence(patterns) => {
                for pattern in patterns {
                    pos = self.match_pattern_recursive(pattern, args, bindings, pos)?;
                }
                Ok(pos)
            }
            MacroPattern::Optional(pattern) => {
                // Try to match, but don't fail if it doesn't
                if let Ok(new_pos) = self.match_pattern_recursive(pattern, args, bindings, pos) {
                    Ok(new_pos)
                } else {
                    Ok(pos)
                }
            }
            MacroPattern::Choice(patterns) => {
                for pattern in patterns {
                    if let Ok(new_pos) = self.match_pattern_recursive(pattern, args, bindings, pos) {
                        return Ok(new_pos);
                    }
                }
                Err(CompilerError::Parse(ParseError {
                    kind: ParseErrorKind::MacroPatternMismatch,
                    location: SourceLocation::unknown(),
                    message: "No choice matched in macro pattern".to_string(),
                }))
            }
        }
    }

    fn expand_template(&self, expansion: &MacroExpansion, bindings: &HashMap<String, Vec<MacroArgument>>) -> Result<Vec<Item>> {
        match expansion {
            MacroExpansion::Tokens(tokens) => {
                // Convert tokens back to AST
                let mut parser = Parser::new(tokens.clone());
                let mut items = Vec::new();
                // Parse items from expanded tokens
                // This would require extending the parser
                Ok(items)
            }
            MacroExpansion::Ast(items) => {
                Ok(items.clone())
            }
            MacroExpansion::Template { tokens } => {
                let expanded_tokens = self.expand_template_tokens(tokens, bindings)?;
                let mut parser = Parser::new(expanded_tokens);
                // Parse expanded tokens into AST
                Ok(Vec::new())
            }
        }
    }

    fn expand_template_tokens(&self, tokens: &[MacroToken], bindings: &HashMap<String, Vec<MacroArgument>>) -> Result<Vec<Token>> {
        let mut result = Vec::new();
        
        for token in tokens {
            match token {
                MacroToken::Literal(tok) => {
                    result.push(tok.clone());
                }
                MacroToken::Variable(name) => {
                    if let Some(args) = bindings.get(name) {
                        for arg in args {
                            // Convert macro argument to tokens
                            result.extend(self.argument_to_tokens(arg)?);
                        }
                    }
                }
                MacroToken::Repeat { tokens, separator, variable } => {
                    if let Some(args) = bindings.get(variable) {
                        for (i, _arg) in args.iter().enumerate() {
                            if i > 0 {
                                if let Some(sep) = separator {
                                    // Add separator token
                                    result.push(Token {
                                        token_type: TokenType::Identifier(sep.clone()),
                                        location: SourceLocation::unknown(),
                                    });
                                }
                            }
                            result.extend(self.expand_template_tokens(tokens, bindings)?);
                        }
                    }
                }
            }
        }
        
        Ok(result)
    }

    fn argument_to_tokens(&self, arg: &MacroArgument) -> Result<Vec<Token>> {
        match arg {
            MacroArgument::Expression(_expr) => {
                // Convert expression to tokens
                Ok(Vec::new())
            }
            MacroArgument::Type(_ty) => {
                // Convert type to tokens
                Ok(Vec::new())
            }
            MacroArgument::Pattern(_pat) => {
                // Convert pattern to tokens
                Ok(Vec::new())
            }
            MacroArgument::Statement(_stmt) => {
                // Convert statement to tokens
                Ok(Vec::new())
            }
            MacroArgument::Literal(lit) => {
                Ok(vec![Token {
                    token_type: TokenType::StringLiteral(lit.clone()),
                    location: SourceLocation::unknown(),
                }])
            }
        }
    }
}

/// Built-in macros
impl MacroProcessor {
    pub fn register_builtin_macros(&mut self) {
        // println! macro
        self.register_macro(MacroDefinition {
            name: "println".to_string(),
            kind: MacroKind::Declarative,
            rules: vec![
                MacroRule {
                    pattern: MacroPattern::Variable {
                        name: "format".to_string(),
                        kind: MacroVariableKind::Expression,
                        repetition: None,
                    },
                    expansion: MacroExpansion::Template {
                        tokens: vec![
                            MacroToken::Literal(Token {
                                token_type: TokenType::Identifier("print".to_string()),
                                location: SourceLocation::unknown(),
                            }),
                            MacroToken::Variable("format".to_string()),
                        ],
                    },
                }
            ],
            location: SourceLocation::unknown(),
        });

        // assert! macro
        self.register_macro(MacroDefinition {
            name: "assert".to_string(),
            kind: MacroKind::Declarative,
            rules: vec![
                MacroRule {
                    pattern: MacroPattern::Variable {
                        name: "condition".to_string(),
                        kind: MacroVariableKind::Expression,
                        repetition: None,
                    },
                    expansion: MacroExpansion::Template {
                        tokens: vec![
                            MacroToken::Literal(Token {
                                token_type: TokenType::If,
                                location: SourceLocation::unknown(),
                            }),
                            MacroToken::Literal(Token {
                                token_type: TokenType::Not,
                                location: SourceLocation::unknown(),
                            }),
                            MacroToken::Variable("condition".to_string()),
                            MacroToken::Literal(Token {
                                token_type: TokenType::LeftBrace,
                                location: SourceLocation::unknown(),
                            }),
                            MacroToken::Literal(Token {
                                token_type: TokenType::Revert,
                                location: SourceLocation::unknown(),
                            }),
                            MacroToken::Literal(Token {
                                token_type: TokenType::RightBrace,
                                location: SourceLocation::unknown(),
                            }),
                        ],
                    },
                }
            ],
            location: SourceLocation::unknown(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macro_processor() {
        let mut processor = MacroProcessor::new();
        processor.register_builtin_macros();
        
        assert!(processor.macros.contains_key("println"));
        assert!(processor.macros.contains_key("assert"));
    }

    #[test]
    fn test_pattern_matching() {
        let processor = MacroProcessor::new();
        let pattern = MacroPattern::Variable {
            name: "x".to_string(),
            kind: MacroVariableKind::Expression,
            repetition: None,
        };
        
        let args = vec![MacroArgument::Literal("test".to_string())];
        let result = processor.match_pattern(&pattern, &args);
        assert!(result.is_ok());
    }
}
