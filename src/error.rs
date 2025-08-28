// Error types for the compiler
// Covers lexing, parsing, semantic analysis, and codegen errors

use std::fmt;
use std::error::Error;

// Just a shortcut so we don't have to type Result<T, CompilerError> everywhere
pub type Result<T> = std::result::Result<T, CompilerError>;

// ML-specific error type
#[derive(Debug, Clone)]
pub enum AugustiumError {
    Runtime(String),
    InvalidInput(String),
    DeviceError(String),
    MemoryError(String),
    ComputationError(String),
    IoError(String),
}

impl fmt::Display for AugustiumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AugustiumError::Runtime(msg) => write!(f, "Runtime error: {}", msg),
            AugustiumError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            AugustiumError::DeviceError(msg) => write!(f, "Device error: {}", msg),
            AugustiumError::MemoryError(msg) => write!(f, "Memory error: {}", msg),
            AugustiumError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
            AugustiumError::IoError(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl Error for AugustiumError {}

// Main error enum - covers all the different phases of compilation
#[derive(Debug, Clone)]
pub enum CompilerError {
    // Lexer found invalid tokens
    LexError(LexError),
    
    // Parser couldn't understand the syntax
    ParseError(ParseError),
    
    // Semantic analyzer found logical errors
    SemanticError(SemanticError),
    
    // Code generator had issues
    CodegenError(CodegenError),
    
    /// Virtual machine errors
    VmError(VmError),
    
    /// I/O related errors
    IoError(String),
    
    /// Unsupported compilation target
    UnsupportedTarget(String),
    
    /// Internal compiler errors (bugs)
    InternalError(String),
}

/// Lexical analysis errors
#[derive(Debug, Clone)]
pub struct LexError {
    pub kind: LexErrorKind,
    pub location: SourceLocation,
    pub message: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum LexErrorKind {
    UnexpectedCharacter(char),
    UnterminatedString,
    UnterminatedComment,
    InvalidNumber,
    InvalidEscapeSequence,
    InvalidUnicodeEscape,
}

/// Syntax parsing errors
#[derive(Debug, Clone)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub location: SourceLocation,
    pub message: String,
    pub expected: Option<String>,
    pub found: Option<String>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum ParseErrorKind {
    UnexpectedToken,
    UnexpectedEndOfFile,
    MissingToken,
    InvalidExpression,
    InvalidStatement,
    InvalidDeclaration,
    InvalidType,
    InvalidPattern,
}

/// Semantic analysis errors
#[derive(Debug, Clone)]
pub struct SemanticError {
    pub kind: SemanticErrorKind,
    pub location: SourceLocation,
    pub message: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum SemanticErrorKind {
    /// Type checking errors
    TypeMismatch { expected: String, found: String },
    TypeInferenceFailure,
    UnexpectedReturn,
    InvalidSymbolUsage,
    UndefinedSymbol(String),
    UndefinedVariable(String),
    UndefinedFunction(String),
    UndefinedType(String),
    RedefinedVariable(String),
    RedefinedFunction(String),
    RedefinedType(String),
    DuplicateSymbol(String),
    DuplicateField(String),
    UndefinedField(String),
    DuplicateFunction(String),
    DuplicateVariant(String),
    UndefinedTrait(String),
    
    /// Safety analysis errors
    PotentialReentrancy,
    PotentialOverflow,
    UnauthorizedAccess,
    InvalidStateTransition,
    
    /// Contract-specific errors
    InvalidModifier,
    MissingConstructor,
    InvalidEventDeclaration,
    InvalidStorageAccess,
    
    /// Generic semantic errors
    InvalidOperation,
    InvalidArguments,
    InvalidReturnType,
    UnreachableCode,
}

/// Code generation errors
#[derive(Debug, Clone)]
pub struct CodegenError {
    pub kind: CodegenErrorKind,
    pub location: SourceLocation,
    pub message: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum CodegenErrorKind {
    UnsupportedFeature(String),
    OptimizationFailed,
    BytecodeGenerationFailed,
    InvalidInstruction,
    StackOverflow,
    InvalidJumpTarget,
    ImmutableAssignment,
    InvalidAssignmentTarget,
    InvalidFunctionCall,
    InvalidBreak,
    InvalidContinue,
}

/// Virtual machine errors
#[derive(Debug, Clone)]
pub struct VmError {
    pub kind: VmErrorKind,
    pub message: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum VmErrorKind {
    /// Execution errors
    StackUnderflow,
    StackOverflow,
    InvalidInstruction,
    DivisionByZero,
    IntegerOverflow,
    OutOfGas,
    
    /// Memory errors
    InvalidMemoryAccess,
    OutOfMemory,
    
    /// Security errors
    ReentrancyDetected,
    UnauthorizedAccess,
    InvalidStateTransition,
    
    /// Contract errors
    ContractNotFound,
    FunctionNotFound,
    InvalidCallData,
    ExecutionReverted(String),
    
    /// Additional VM errors
    InvalidAddress,
    InvalidInput,
    ExecutionFailed,
    EmptyCallStack,
    InvalidLocalAccess,
    TypeMismatch,
    RequirementFailed,
    AssertionFailed,
    TransactionReverted,
    CallDepthExceeded,
    
    /// DeFi and financial errors
    InsufficientBalance,
    InvalidAmount,
    InsufficientLiquidity,
    InsufficientCollateral,
    
    /// Governance errors
    InvalidState,
    VotingEnded,
    AlreadyVoted,
    TimelockNotReady,
    Unauthorized,
    InsufficientVotingPower,
    NotFound,
    TransactionExpired,
    
    /// Oracle errors
    InvalidData,
    ExcessiveDeviation,
    StaleData,
    InsufficientData,
    
    /// ML-specific errors
    InvalidOperation,
}

/// Source code location for error reporting
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SourceLocation {
    pub file: String,
    pub line: usize,
    pub column: usize,
    pub offset: usize,
}

impl Default for SourceLocation {
    fn default() -> Self {
        Self::unknown()
    }
}

impl SourceLocation {
    pub fn new(file: String, line: usize, column: usize, offset: usize) -> Self {
        Self { file, line, column, offset }
    }
    
    pub fn unknown() -> Self {
        Self {
            file: "<unknown>".to_string(),
            line: 0,
            column: 0,
            offset: 0,
        }
    }

    pub fn dummy() -> Self {
        Self {
            file: "<dummy>".to_string(),
            line: 0,
            column: 0,
            offset: 0,
        }
    }
}

// Display implementations for user-friendly error messages

impl fmt::Display for CompilerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompilerError::LexError(e) => write!(f, "Lexical error: {}", e),
            CompilerError::ParseError(e) => write!(f, "Parse error: {}", e),
            CompilerError::SemanticError(e) => write!(f, "Semantic error: {}", e),
            CompilerError::CodegenError(e) => write!(f, "Code generation error: {}", e),
            CompilerError::VmError(e) => write!(f, "VM error: {}", e),
            CompilerError::IoError(msg) => write!(f, "I/O error: {}", msg),
            CompilerError::UnsupportedTarget(msg) => write!(f, "Unsupported target: {}", msg),
            CompilerError::InternalError(msg) => write!(f, "Internal compiler error: {}", msg),
        }
    }
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}: {}", self.kind, self.location, self.message)
    }
}

impl fmt::Display for LexErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexErrorKind::UnexpectedCharacter(c) => write!(f, "Unexpected character '{}'", c),
            LexErrorKind::UnterminatedString => write!(f, "Unterminated string literal"),
            LexErrorKind::UnterminatedComment => write!(f, "Unterminated comment"),
            LexErrorKind::InvalidNumber => write!(f, "Invalid number literal"),
            LexErrorKind::InvalidEscapeSequence => write!(f, "Invalid escape sequence"),
            LexErrorKind::InvalidUnicodeEscape => write!(f, "Invalid unicode escape sequence"),
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}: {}", self.kind, self.location, self.message)?;
        if let Some(expected) = &self.expected {
            write!(f, " (expected {})", expected)?;
        }
        if let Some(found) = &self.found {
            write!(f, " (found {})", found)?;
        }
        Ok(())
    }
}

impl fmt::Display for ParseErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseErrorKind::UnexpectedToken => write!(f, "Unexpected token"),
            ParseErrorKind::UnexpectedEndOfFile => write!(f, "Unexpected end of file"),
            ParseErrorKind::MissingToken => write!(f, "Missing token"),
            ParseErrorKind::InvalidExpression => write!(f, "Invalid expression"),
            ParseErrorKind::InvalidStatement => write!(f, "Invalid statement"),
            ParseErrorKind::InvalidDeclaration => write!(f, "Invalid declaration"),
            ParseErrorKind::InvalidType => write!(f, "Invalid type"),
            ParseErrorKind::InvalidPattern => write!(f, "Invalid pattern"),
        }
    }
}

impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}: {}", self.kind, self.location, self.message)
    }
}

impl fmt::Display for SemanticErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SemanticErrorKind::TypeMismatch { expected, found } => {
                write!(f, "Type mismatch: expected {}, found {}", expected, found)
            }
            SemanticErrorKind::TypeInferenceFailure => write!(f, "Type inference failure"),
            SemanticErrorKind::UnexpectedReturn => write!(f, "Unexpected return statement"),
            SemanticErrorKind::InvalidSymbolUsage => write!(f, "Invalid symbol usage"),
            SemanticErrorKind::UndefinedSymbol(name) => write!(f, "Undefined symbol: {}", name),
            SemanticErrorKind::UndefinedVariable(name) => write!(f, "Undefined variable '{}'.", name),
            SemanticErrorKind::UndefinedFunction(name) => write!(f, "Undefined function '{}'.", name),
            SemanticErrorKind::UndefinedType(name) => write!(f, "Undefined type '{}'.", name),
            SemanticErrorKind::RedefinedVariable(name) => write!(f, "Variable '{}' already defined", name),
            SemanticErrorKind::RedefinedFunction(name) => write!(f, "Function '{}' already defined", name),
            SemanticErrorKind::RedefinedType(name) => write!(f, "Type '{}' already defined", name),
            SemanticErrorKind::PotentialReentrancy => write!(f, "Potential reentrancy vulnerability detected"),
            SemanticErrorKind::PotentialOverflow => write!(f, "Potential integer overflow detected"),
            SemanticErrorKind::UnauthorizedAccess => write!(f, "Unauthorized access detected"),
            SemanticErrorKind::InvalidStateTransition => write!(f, "Invalid state transition"),
            SemanticErrorKind::InvalidModifier => write!(f, "Invalid modifier"),
            SemanticErrorKind::MissingConstructor => write!(f, "Missing constructor"),
            SemanticErrorKind::InvalidEventDeclaration => write!(f, "Invalid event declaration"),
            SemanticErrorKind::InvalidStorageAccess => write!(f, "Invalid storage access"),
            SemanticErrorKind::InvalidOperation => write!(f, "Invalid operation"),
            SemanticErrorKind::InvalidArguments => write!(f, "Invalid arguments"),
            SemanticErrorKind::InvalidReturnType => write!(f, "Invalid return type"),
            SemanticErrorKind::UnreachableCode => write!(f, "Unreachable code detected"),
            SemanticErrorKind::DuplicateSymbol(name) => write!(f, "Duplicate symbol: {}", name),
            SemanticErrorKind::DuplicateField(name) => write!(f, "Duplicate field: {}", name),
            SemanticErrorKind::UndefinedField(name) => write!(f, "Undefined field: {}", name),
            SemanticErrorKind::DuplicateFunction(name) => write!(f, "Duplicate function: {}", name),
            SemanticErrorKind::DuplicateVariant(name) => write!(f, "Duplicate variant: {}", name),
            SemanticErrorKind::UndefinedTrait(name) => write!(f, "Undefined trait: {}", name),
        }
    }
}

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}: {}", self.kind, self.location, self.message)
    }
}

impl fmt::Display for CodegenErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodegenErrorKind::UnsupportedFeature(feature) => {
                write!(f, "Unsupported feature: {}", feature)
            }
            CodegenErrorKind::OptimizationFailed => write!(f, "Optimization failed"),
            CodegenErrorKind::BytecodeGenerationFailed => write!(f, "Bytecode generation failed"),
            CodegenErrorKind::InvalidInstruction => write!(f, "Invalid instruction"),
            CodegenErrorKind::StackOverflow => write!(f, "Stack overflow during compilation"),
            CodegenErrorKind::InvalidJumpTarget => write!(f, "Invalid jump target"),
            CodegenErrorKind::ImmutableAssignment => write!(f, "Cannot assign to immutable variable"),
            CodegenErrorKind::InvalidAssignmentTarget => write!(f, "Invalid assignment target"),
            CodegenErrorKind::InvalidFunctionCall => write!(f, "Invalid function call"),
            CodegenErrorKind::InvalidBreak => write!(f, "Invalid break statement"),
            CodegenErrorKind::InvalidContinue => write!(f, "Invalid continue statement"),
        }
    }
}

impl fmt::Display for VmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl fmt::Display for VmErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VmErrorKind::StackUnderflow => write!(f, "Stack underflow"),
            VmErrorKind::StackOverflow => write!(f, "Stack overflow"),
            VmErrorKind::InvalidInstruction => write!(f, "Invalid instruction"),
            VmErrorKind::DivisionByZero => write!(f, "Division by zero"),
            VmErrorKind::IntegerOverflow => write!(f, "Integer overflow"),
            VmErrorKind::OutOfGas => write!(f, "Out of gas"),
            VmErrorKind::InvalidMemoryAccess => write!(f, "Invalid memory access"),
            VmErrorKind::OutOfMemory => write!(f, "Out of memory"),
            VmErrorKind::ReentrancyDetected => write!(f, "Reentrancy detected"),
            VmErrorKind::UnauthorizedAccess => write!(f, "Unauthorized access"),
            VmErrorKind::InvalidStateTransition => write!(f, "Invalid state transition"),
            VmErrorKind::ContractNotFound => write!(f, "Contract not found"),
            VmErrorKind::FunctionNotFound => write!(f, "Function not found"),
            VmErrorKind::InvalidCallData => write!(f, "Invalid call data"),
            VmErrorKind::ExecutionReverted(reason) => write!(f, "Execution reverted: {}", reason),
            VmErrorKind::InvalidAddress => write!(f, "Invalid address"),
            VmErrorKind::InvalidInput => write!(f, "Invalid input"),
            VmErrorKind::ExecutionFailed => write!(f, "Execution failed"),
            VmErrorKind::EmptyCallStack => write!(f, "Empty call stack"),
            VmErrorKind::InvalidLocalAccess => write!(f, "Invalid local variable access"),
            VmErrorKind::TypeMismatch => write!(f, "Type mismatch"),
            VmErrorKind::RequirementFailed => write!(f, "Requirement failed"),
            VmErrorKind::AssertionFailed => write!(f, "Assertion failed"),
            VmErrorKind::TransactionReverted => write!(f, "Transaction reverted"),
            VmErrorKind::InsufficientBalance => write!(f, "Insufficient balance"),
            VmErrorKind::InvalidAmount => write!(f, "Invalid amount"),
            VmErrorKind::InsufficientLiquidity => write!(f, "Insufficient liquidity"),
            VmErrorKind::InsufficientData => write!(f, "Insufficient data"),
            VmErrorKind::InsufficientCollateral => write!(f, "Insufficient collateral"),
            VmErrorKind::InvalidState => write!(f, "Invalid state"),
            VmErrorKind::VotingEnded => write!(f, "Voting period has ended"),
            VmErrorKind::AlreadyVoted => write!(f, "Already voted"),
            VmErrorKind::TimelockNotReady => write!(f, "Timelock not ready"),
            VmErrorKind::Unauthorized => write!(f, "Unauthorized"),
            VmErrorKind::InsufficientVotingPower => write!(f, "Insufficient voting power"),
            VmErrorKind::NotFound => write!(f, "Not found"),
            VmErrorKind::TransactionExpired => write!(f, "Transaction expired"),
            VmErrorKind::InvalidData => write!(f, "Invalid data"),
            VmErrorKind::ExcessiveDeviation => write!(f, "Excessive deviation"),
            VmErrorKind::StaleData => write!(f, "Stale data"),
            VmErrorKind::InvalidOperation => write!(f, "Invalid operation"),
            VmErrorKind::CallDepthExceeded => write!(f, "Call depth exceeded"),
        }
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

// Error conversion implementations

impl From<LexError> for CompilerError {
    fn from(error: LexError) -> Self {
        CompilerError::LexError(error)
    }
}

impl From<ParseError> for CompilerError {
    fn from(error: ParseError) -> Self {
        CompilerError::ParseError(error)
    }
}

impl From<SemanticError> for CompilerError {
    fn from(error: SemanticError) -> Self {
        CompilerError::SemanticError(error)
    }
}

impl From<CodegenError> for CompilerError {
    fn from(error: CodegenError) -> Self {
        CompilerError::CodegenError(error)
    }
}

impl From<VmError> for CompilerError {
    fn from(error: VmError) -> Self {
        CompilerError::VmError(error)
    }
}

// Helper functions for creating errors

impl LexError {
    pub fn new(kind: LexErrorKind, location: SourceLocation, message: String) -> Self {
        Self { kind, location, message }
    }
}

impl ParseError {
    pub fn new(kind: ParseErrorKind, location: SourceLocation, message: String) -> Self {
        Self {
            kind,
            location,
            message,
            expected: None,
            found: None,
        }
    }
    
    #[allow(dead_code)]
    pub fn with_expected(mut self, expected: String) -> Self {
        self.expected = Some(expected);
        self
    }
    
    #[allow(dead_code)]
    pub fn with_found(mut self, found: String) -> Self {
        self.found = Some(found);
        self
    }
}

impl SemanticError {
    pub fn new(kind: SemanticErrorKind, location: SourceLocation, message: String) -> Self {
        Self { kind, location, message }
    }
}

impl CodegenError {
    pub fn new(kind: CodegenErrorKind, location: SourceLocation, message: String) -> Self {
        Self { kind, location, message }
    }
}

impl VmError {
    pub fn new(kind: VmErrorKind, message: String) -> Self {
        Self { kind, message }
    }
}

// std::error::Error implementations

impl Error for CompilerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            CompilerError::LexError(e) => Some(e),
            CompilerError::ParseError(e) => Some(e),
            CompilerError::SemanticError(e) => Some(e),
            CompilerError::CodegenError(e) => Some(e),
            CompilerError::VmError(e) => Some(e),
            CompilerError::IoError(_) => None,
            CompilerError::UnsupportedTarget(_) => None,
            CompilerError::InternalError(_) => None,
        }
    }
}

impl Error for LexError {}

impl Error for ParseError {}

impl Error for SemanticError {}

impl Error for CodegenError {}

impl Error for VmError {}