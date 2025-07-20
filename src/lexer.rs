// Lexer - breaks source code into tokens
// First step in compilation

use crate::error::{LexError, LexErrorKind, SourceLocation, Result};
use std::collections::HashMap;
use std::fmt;

// All the different types of tokens we can find in Augustium code
#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    // Literals
    IntegerLiteral(u64),
    StringLiteral(String),
    BooleanLiteral(bool),
    AddressLiteral(String),
    
    // Identifiers and keywords
    Identifier(String),
    
    // Keywords
    Contract,
    Constructor,
    Fn,
    Let,
    Mut,
    Const,
    If,
    Else,
    While,
    For,
    In,
    Return,
    Break,
    Continue,
    Match,
    Enum,
    Struct,
    Trait,
    Impl,
    Mod,
    Use,
    Pub,
    Priv,
    Event,
    Emit,
    Require,
    Assert,
    Revert,
    As,
    Indexed,
    
    // Types
    U8, U16, U32, U64, U128, U256,
    I8, I16, I32, I64, I128, I256,
    Bool,
    String,
    Address,
    Bytes,
    
    // Operators
    Plus,           // +
    Minus,          // -
    Star,           // *
    Slash,          // /
    Percent,        // %
    Equal,          // =
    EqualEqual,     // ==
    NotEqual,       // !=
    Less,           // <
    LessEqual,      // <=
    Greater,        // >
    GreaterEqual,   // >=
    And,            // &&
    Or,             // ||
    Not,            // !
    BitAnd,         // &
    BitOr,          // |
    BitXor,         // ^
    BitNot,         // ~
    LeftShift,      // <<
    RightShift,     // >>
    PlusEqual,      // +=
    MinusEqual,     // -=
    StarEqual,      // *=
    SlashEqual,     // /=
    
    // Punctuation
    LeftParen,      // (
    RightParen,     // )
    LeftBrace,      // {
    RightBrace,     // }
    LeftBracket,    // [
    RightBracket,   // ]
    Semicolon,      // ;
    Comma,          // ,
    Dot,            // .
    Colon,          // :
    DoubleColon,    // ::
    Arrow,          // ->
    FatArrow,       // =>
    Question,       // ?
    At,             // @
    Hash,           // #
    Dollar,         // $
    Underscore,     // _
    
    // Special tokens
    Newline,
    Eof,
    
    // Comments (usually filtered out)
    LineComment(String),
    BlockComment(String),
}

/// A token with its type, location, and raw text
#[derive(Debug, Clone)]
pub struct Token {
    pub token_type: TokenType,
    pub location: SourceLocation,
    pub raw: String,
}

/// Lexer state for tokenizing Augustium source code
pub struct Lexer {
    input: Vec<char>,
    position: usize,
    line: usize,
    column: usize,
    file: String,
    keywords: HashMap<String, TokenType>,
}

impl Lexer {
    /// Create a new lexer for the given source code
    pub fn new(source: &str, file: &str) -> Self {
        let mut keywords = HashMap::new();
        
        // Insert keywords
        keywords.insert("contract".to_string(), TokenType::Contract);
        keywords.insert("constructor".to_string(), TokenType::Constructor);
        keywords.insert("fn".to_string(), TokenType::Fn);
        keywords.insert("let".to_string(), TokenType::Let);
        keywords.insert("mut".to_string(), TokenType::Mut);
        keywords.insert("const".to_string(), TokenType::Const);
        keywords.insert("if".to_string(), TokenType::If);
        keywords.insert("else".to_string(), TokenType::Else);
        keywords.insert("while".to_string(), TokenType::While);
        keywords.insert("for".to_string(), TokenType::For);
        keywords.insert("in".to_string(), TokenType::In);
        keywords.insert("return".to_string(), TokenType::Return);
        keywords.insert("break".to_string(), TokenType::Break);
        keywords.insert("continue".to_string(), TokenType::Continue);
        keywords.insert("match".to_string(), TokenType::Match);
        keywords.insert("enum".to_string(), TokenType::Enum);
        keywords.insert("struct".to_string(), TokenType::Struct);
        keywords.insert("trait".to_string(), TokenType::Trait);
        keywords.insert("impl".to_string(), TokenType::Impl);
        keywords.insert("mod".to_string(), TokenType::Mod);
        keywords.insert("use".to_string(), TokenType::Use);
        keywords.insert("pub".to_string(), TokenType::Pub);
        keywords.insert("priv".to_string(), TokenType::Priv);
        keywords.insert("event".to_string(), TokenType::Event);
        keywords.insert("emit".to_string(), TokenType::Emit);
        keywords.insert("require".to_string(), TokenType::Require);
        keywords.insert("assert".to_string(), TokenType::Assert);
        keywords.insert("revert".to_string(), TokenType::Revert);
        keywords.insert("as".to_string(), TokenType::As);
        keywords.insert("indexed".to_string(), TokenType::Indexed);
        
        // Insert type keywords
        keywords.insert("u8".to_string(), TokenType::U8);
        keywords.insert("u16".to_string(), TokenType::U16);
        keywords.insert("u32".to_string(), TokenType::U32);
        keywords.insert("u64".to_string(), TokenType::U64);
        keywords.insert("u128".to_string(), TokenType::U128);
        keywords.insert("u256".to_string(), TokenType::U256);
        keywords.insert("i8".to_string(), TokenType::I8);
        keywords.insert("i16".to_string(), TokenType::I16);
        keywords.insert("i32".to_string(), TokenType::I32);
        keywords.insert("i64".to_string(), TokenType::I64);
        keywords.insert("i128".to_string(), TokenType::I128);
        keywords.insert("i256".to_string(), TokenType::I256);
        keywords.insert("bool".to_string(), TokenType::Bool);
        keywords.insert("string".to_string(), TokenType::String);
        keywords.insert("address".to_string(), TokenType::Address);
        keywords.insert("bytes".to_string(), TokenType::Bytes);
        
        // Insert boolean literals
        keywords.insert("true".to_string(), TokenType::BooleanLiteral(true));
        keywords.insert("false".to_string(), TokenType::BooleanLiteral(false));
        
        Self {
            input: source.chars().collect(),
            position: 0,
            line: 1,
            column: 1,
            file: file.to_string(),
            keywords,
        }
    }
    
    /// Tokenize the entire input and return a vector of tokens
    pub fn tokenize(&mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        
        while !self.is_at_end() {
            match self.next_token()? {
                Some(token) => {
                    // Filter out comments in normal tokenization
                    match token.token_type {
                        TokenType::LineComment(_) | TokenType::BlockComment(_) => {
                            // Skip comments
                        }
                        _ => tokens.push(token),
                    }
                }
                None => break,
            }
        }
        
        // Add EOF token
        tokens.push(Token {
            token_type: TokenType::Eof,
            location: self.current_location(),
            raw: String::new(),
        });
        
        Ok(tokens)
    }
    
    /// Get the next token from the input
    fn next_token(&mut self) -> Result<Option<Token>> {
        self.skip_whitespace();
        
        if self.is_at_end() {
            return Ok(None);
        }
        
        let start_pos = self.position;
        let start_location = self.current_location();
        
        let ch = self.advance();
        
        let token_type = match ch {
            // Single-character tokens
            '(' => TokenType::LeftParen,
            ')' => TokenType::RightParen,
            '{' => TokenType::LeftBrace,
            '}' => TokenType::RightBrace,
            '[' => TokenType::LeftBracket,
            ']' => TokenType::RightBracket,
            ';' => TokenType::Semicolon,
            ',' => TokenType::Comma,
            '.' => TokenType::Dot,
            '?' => TokenType::Question,
            '@' => TokenType::At,
            '#' => TokenType::Hash,
            '$' => TokenType::Dollar,
            '~' => TokenType::BitNot,
            
            // Operators that might be compound
            '+' => {
                if self.match_char('=') {
                    TokenType::PlusEqual
                } else {
                    TokenType::Plus
                }
            }
            '-' => {
                if self.match_char('=') {
                    TokenType::MinusEqual
                } else if self.match_char('>') {
                    TokenType::Arrow
                } else {
                    TokenType::Minus
                }
            }
            '*' => {
                if self.match_char('=') {
                    TokenType::StarEqual
                } else {
                    TokenType::Star
                }
            }
            '/' => {
                if self.match_char('/') {
                    return Ok(Some(self.line_comment(start_location)?));
                } else if self.match_char('*') {
                    return Ok(Some(self.block_comment(start_location)?));
                } else if self.match_char('=') {
                    TokenType::SlashEqual
                } else {
                    TokenType::Slash
                }
            }
            '%' => TokenType::Percent,
            '=' => {
                if self.match_char('=') {
                    TokenType::EqualEqual
                } else if self.match_char('>') {
                    TokenType::FatArrow
                } else {
                    TokenType::Equal
                }
            }
            '!' => {
                if self.match_char('=') {
                    TokenType::NotEqual
                } else {
                    TokenType::Not
                }
            }
            '<' => {
                if self.match_char('=') {
                    TokenType::LessEqual
                } else if self.match_char('<') {
                    TokenType::LeftShift
                } else {
                    TokenType::Less
                }
            }
            '>' => {
                if self.match_char('=') {
                    TokenType::GreaterEqual
                } else if self.match_char('>') {
                    TokenType::RightShift
                } else {
                    TokenType::Greater
                }
            }
            '&' => {
                if self.match_char('&') {
                    TokenType::And
                } else {
                    TokenType::BitAnd
                }
            }
            '|' => {
                if self.match_char('|') {
                    TokenType::Or
                } else {
                    TokenType::BitOr
                }
            }
            '^' => TokenType::BitXor,
            ':' => {
                if self.match_char(':') {
                    TokenType::DoubleColon
                } else {
                    TokenType::Colon
                }
            }
            
            // String literals
            '"' => return Ok(Some(self.string_literal(start_location)?)),
            
            // Character that starts a number
            '0'..='9' => return Ok(Some(self.number_literal(start_location)?)),
            
            // Identifiers and keywords
            'a'..='z' | 'A'..='Z' | '_' => {
                return Ok(Some(self.identifier_or_keyword(start_location)?));
            }
            
            // Newlines
            '\n' => {
                self.line += 1;
                self.column = 1;
                TokenType::Newline
            }
            
            // Unexpected character
            _ => {
                return Err(LexError::new(
                    LexErrorKind::UnexpectedCharacter(ch),
                    start_location,
                    format!("Unexpected character '{}'", ch),
                ).into());
            }
        };
        
        let raw = self.input[start_pos..self.position].iter().collect();
        
        Ok(Some(Token {
            token_type,
            location: start_location,
            raw,
        }))
    }
    
    /// Parse a string literal
    fn string_literal(&mut self, start_location: SourceLocation) -> Result<Token> {
        let mut value = String::new();
        
        while !self.is_at_end() && self.peek() != '"' {
            let ch = self.advance();
            
            if ch == '\\' {
                // Handle escape sequences
                if self.is_at_end() {
                    return Err(LexError::new(
                        LexErrorKind::UnterminatedString,
                        start_location,
                        "Unterminated string literal".to_string(),
                    ).into());
                }
                
                let escaped = self.advance();
                match escaped {
                    'n' => value.push('\n'),
                    't' => value.push('\t'),
                    'r' => value.push('\r'),
                    '\\' => value.push('\\'),
                    '"' => value.push('"'),
                    '0' => value.push('\0'),
                    _ => {
                        return Err(LexError::new(
                            LexErrorKind::InvalidEscapeSequence,
                            self.current_location(),
                            format!("Invalid escape sequence '\\{}'.", escaped),
                        ).into());
                    }
                }
            } else {
                value.push(ch);
            }
        }
        
        if self.is_at_end() {
            return Err(LexError::new(
                LexErrorKind::UnterminatedString,
                start_location,
                "Unterminated string literal".to_string(),
            ).into());
        }
        
        // Consume closing quote
        self.advance();
        
        let raw = format!("\"{}\"", value);
        
        Ok(Token {
            token_type: TokenType::StringLiteral(value),
            location: start_location,
            raw,
        })
    }
    
    /// Parse a number literal
    fn number_literal(&mut self, start_location: SourceLocation) -> Result<Token> {
        let start_pos = self.position - 1; // We already consumed the first digit
        
        // Handle hexadecimal numbers
        if self.input[start_pos] == '0' && self.peek() == 'x' {
            self.advance(); // consume 'x'
            
            while !self.is_at_end() && self.peek().is_ascii_hexdigit() {
                self.advance();
            }
            
            let hex_str = &self.input[start_pos + 2..self.position].iter().collect::<String>();
            
            match u64::from_str_radix(hex_str, 16) {
                Ok(value) => {
                    let raw = self.input[start_pos..self.position].iter().collect();
                    return Ok(Token {
                        token_type: TokenType::IntegerLiteral(value),
                        location: start_location,
                        raw,
                    });
                }
                Err(_) => {
                    return Err(LexError::new(
                        LexErrorKind::InvalidNumber,
                        start_location,
                        "Invalid hexadecimal number".to_string(),
                    ).into());
                }
            }
        }
        
        // Handle decimal numbers
        while !self.is_at_end() && self.peek().is_ascii_digit() {
            self.advance();
        }

        // Optional numeric type suffix (e.g. i64, u128)
        if !self.is_at_end() {
            let c = self.peek();
            if c == 'i' || c == 'u' {
                // consume leading 'i' or 'u'
                self.advance();
                // consume the digits of the suffix
                while !self.is_at_end() && self.peek().is_ascii_digit() {
                    self.advance();
                }
                // We intentionally ignore the suffix for now – semantic analyzer handles type.
            }
        }
        
        let number_raw: String = self.input[start_pos..self.position].iter().collect();
        let digits_only: String = number_raw
            .chars()
            .take_while(|ch| ch.is_ascii_hexdigit()) // stops before optional suffix
            .collect();
        
        match digits_only.parse::<u64>() {
            Ok(value) => {
                Ok(Token {
                    token_type: TokenType::IntegerLiteral(value),
                    location: start_location,
                    raw: number_raw,
                })
            }
            Err(_) => {
                Err(LexError::new(
                    LexErrorKind::InvalidNumber,
                    start_location,
                    format!("Invalid number: {}", number_raw),
                ).into())
            }
        }
    }
    
    /// Parse an identifier or keyword
    fn identifier_or_keyword(&mut self, start_location: SourceLocation) -> Result<Token> {
        let start_pos = self.position - 1; // We already consumed the first character
        
        while !self.is_at_end() && (self.peek().is_alphanumeric() || self.peek() == '_') {
            self.advance();
        }
        
        let text = self.input[start_pos..self.position].iter().collect::<String>();
        
        // Check if it's a keyword
        let token_type = self.keywords.get(&text)
            .cloned()
            .unwrap_or_else(|| {
                // Check if it's an address literal (starts with 0x and is 42 characters)
                if text.starts_with("0x") && text.len() == 42 {
                    TokenType::AddressLiteral(text.clone())
                } else {
                    TokenType::Identifier(text.clone())
                }
            });
        
        Ok(Token {
            token_type,
            location: start_location,
            raw: text,
        })
    }
    
    /// Parse a line comment
    fn line_comment(&mut self, start_location: SourceLocation) -> Result<Token> {
        let start_pos = self.position - 2; // We already consumed '//'
        
        while !self.is_at_end() && self.peek() != '\n' {
            self.advance();
        }
        
        let comment_text = self.input[start_pos + 2..self.position].iter().collect::<String>();
        let raw = self.input[start_pos..self.position].iter().collect();
        
        Ok(Token {
            token_type: TokenType::LineComment(comment_text),
            location: start_location,
            raw,
        })
    }
    
    /// Parse a block comment
    fn block_comment(&mut self, start_location: SourceLocation) -> Result<Token> {
        let start_pos = self.position - 2; // We already consumed '/*'
        let mut nesting_level = 1;
        
        while !self.is_at_end() && nesting_level > 0 {
            let ch = self.advance();
            
            if ch == '/' && self.peek() == '*' {
                self.advance(); // consume '*'
                nesting_level += 1;
            } else if ch == '*' && self.peek() == '/' {
                self.advance(); // consume '/'
                nesting_level -= 1;
            } else if ch == '\n' {
                self.line += 1;
                self.column = 1;
            }
        }
        
        if nesting_level > 0 {
            return Err(LexError::new(
                LexErrorKind::UnterminatedComment,
                start_location,
                "Unterminated block comment".to_string(),
            ).into());
        }
        
        let comment_text = self.input[start_pos + 2..self.position - 2].iter().collect::<String>();
        let raw = self.input[start_pos..self.position].iter().collect();
        
        Ok(Token {
            token_type: TokenType::BlockComment(comment_text),
            location: start_location,
            raw,
        })
    }
    
    /// Skip whitespace characters (except newlines)
    fn skip_whitespace(&mut self) {
        while !self.is_at_end() {
            match self.peek() {
                ' ' | '\r' | '\t' => {
                    self.advance();
                }
                _ => break,
            }
        }
    }
    
    /// Check if we're at the end of input
    fn is_at_end(&self) -> bool {
        self.position >= self.input.len()
    }
    
    /// Get the current character without advancing
    fn peek(&self) -> char {
        if self.is_at_end() {
            '\0'
        } else {
            self.input[self.position]
        }
    }
    
    /// Get the next character without advancing
    fn peek_next(&self) -> char {
        if self.position + 1 >= self.input.len() {
            '\0'
        } else {
            self.input[self.position + 1]
        }
    }
    
    /// Advance to the next character and return the current one
    fn advance(&mut self) -> char {
        if self.is_at_end() {
            '\0'
        } else {
            let ch = self.input[self.position];
            self.position += 1;
            self.column += 1;
            ch
        }
    }
    
    /// Check if the current character matches the expected one and advance if so
    fn match_char(&mut self, expected: char) -> bool {
        if self.is_at_end() || self.peek() != expected {
            false
        } else {
            self.advance();
            true
        }
    }
    
    /// Get the current source location
    fn current_location(&self) -> SourceLocation {
        SourceLocation::new(self.file.clone(), self.line, self.column, self.position)
    }
}

// Display implementation for TokenType
impl fmt::Display for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenType::IntegerLiteral(n) => write!(f, "integer({})", n),
            TokenType::StringLiteral(s) => write!(f, "string(\"{}\")", s),
            TokenType::BooleanLiteral(b) => write!(f, "boolean({})", b),
            TokenType::AddressLiteral(a) => write!(f, "address({})", a),
            TokenType::Identifier(name) => write!(f, "identifier({})", name),
            TokenType::LineComment(text) => write!(f, "line_comment({})", text),
            TokenType::BlockComment(text) => write!(f, "block_comment({})", text),
            _ => write!(f, "{:?}", self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_tokens() {
        let mut lexer = Lexer::new("+ - * / = == != < > <= >= && || !", "test.aug");
        let tokens = lexer.tokenize().unwrap();
        
        let expected = vec![
            TokenType::Plus,
            TokenType::Minus,
            TokenType::Star,
            TokenType::Slash,
            TokenType::Equal,
            TokenType::EqualEqual,
            TokenType::NotEqual,
            TokenType::Less,
            TokenType::Greater,
            TokenType::LessEqual,
            TokenType::GreaterEqual,
            TokenType::And,
            TokenType::Or,
            TokenType::Not,
            TokenType::Eof,
        ];
        
        for (i, expected_type) in expected.iter().enumerate() {
            assert_eq!(&tokens[i].token_type, expected_type);
        }
    }
    
    #[test]
    fn test_keywords() {
        let mut lexer = Lexer::new("contract fn let mut if else", "test.aug");
        let tokens = lexer.tokenize().unwrap();
        
        assert_eq!(tokens[0].token_type, TokenType::Contract);
        assert_eq!(tokens[1].token_type, TokenType::Fn);
        assert_eq!(tokens[2].token_type, TokenType::Let);
        assert_eq!(tokens[3].token_type, TokenType::Mut);
        assert_eq!(tokens[4].token_type, TokenType::If);
        assert_eq!(tokens[5].token_type, TokenType::Else);
    }
    
    #[test]
    fn test_string_literal() {
        let mut lexer = Lexer::new(r#""Hello, World!"
"#, "test.aug");
        let tokens = lexer.tokenize().unwrap();
        
        match &tokens[0].token_type {
            TokenType::StringLiteral(s) => assert_eq!(s, "Hello, World!"),
            _ => panic!("Expected string literal"),
        }
    }
    
    #[test]
    fn test_number_literal() {
        let mut lexer = Lexer::new("42 0x1a", "test.aug");
        let tokens = lexer.tokenize().unwrap();
        
        match &tokens[0].token_type {
            TokenType::IntegerLiteral(n) => assert_eq!(*n, 42),
            _ => panic!("Expected integer literal"),
        }
        
        match &tokens[1].token_type {
            TokenType::IntegerLiteral(n) => assert_eq!(*n, 26), // 0x1a = 26
            _ => panic!("Expected integer literal"),
        }
    }
    
    #[test]
    fn test_identifier() {
        let mut lexer = Lexer::new("my_variable _private __internal", "test.aug");
        let tokens = lexer.tokenize().unwrap();
        
        match &tokens[0].token_type {
            TokenType::Identifier(name) => assert_eq!(name, "my_variable"),
            _ => panic!("Expected identifier"),
        }
    }
    
    #[test]
    fn test_comments() {
        let mut lexer = Lexer::new("// line comment\n/* block comment */", "test.aug");
        let tokens = lexer.tokenize().unwrap();
        
        // Comments should be filtered out in normal tokenization
        assert_eq!(tokens.len(), 2); // Newline + EOF
        assert_eq!(tokens[0].token_type, TokenType::Newline);
        assert_eq!(tokens[1].token_type, TokenType::Eof);
    }
}