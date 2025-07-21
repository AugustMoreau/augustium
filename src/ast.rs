//! Abstract Syntax Tree (AST) definitions for the Augustium programming language
//!
//! This module defines the data structures that represent the parsed structure
//! of Augustium source code.

use crate::error::SourceLocation;


/// A complete Augustium source file
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub items: Vec<Item>,
    pub location: SourceLocation,
}

/// Top-level items in an Augustium source file
#[derive(Debug, Clone)]
pub enum Item {
    Contract(Contract),
    Function(Function),
    Struct(Struct),
    Enum(Enum),
    Trait(Trait),
    Impl(Impl),
    Use(UseDeclaration),
    Const(ConstDeclaration),
    Module(Module),
}

/// Contract definition
#[derive(Debug, Clone)]
pub struct Contract {
    pub name: Identifier,
    pub fields: Vec<Field>,
    pub functions: Vec<Function>,
    pub events: Vec<Event>,
    pub modifiers: Vec<Modifier>,
    pub location: SourceLocation,
}

/// Function definition
#[derive(Debug, Clone)]
pub struct Function {
    pub name: Identifier,
    pub parameters: Vec<Parameter>,
    pub return_type: Option<Type>,
    pub body: Block,
    pub visibility: Visibility,
    pub mutability: Mutability,
    pub attributes: Vec<Attribute>,
    pub location: SourceLocation,
}

/// Struct definition
#[derive(Debug, Clone)]
pub struct Struct {
    pub name: Identifier,
    pub fields: Vec<Field>,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// Enum definition
#[derive(Debug, Clone)]
pub struct Enum {
    pub name: Identifier,
    pub variants: Vec<EnumVariant>,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// Enum variant
#[derive(Debug, Clone)]
pub struct EnumVariant {
    pub name: Identifier,
    pub fields: Option<Vec<Type>>,
    pub location: SourceLocation,
}

/// Trait definition
#[derive(Debug, Clone)]
pub struct Trait {
    pub name: Identifier,
    pub functions: Vec<TraitFunction>,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// Trait function signature
#[derive(Debug, Clone)]
pub struct TraitFunction {
    pub name: Identifier,
    pub parameters: Vec<Parameter>,
    pub return_type: Option<Type>,
    pub location: SourceLocation,
}

/// Implementation block
#[derive(Debug, Clone)]
pub struct Impl {
    pub trait_name: Option<Identifier>,
    pub type_name: Identifier,
    pub functions: Vec<Function>,
    pub location: SourceLocation,
}

/// Use declaration
#[derive(Debug, Clone)]
pub struct UseDeclaration {
    pub path: Vec<Identifier>,
    pub alias: Option<Identifier>,
    pub location: SourceLocation,
}

/// Constant declaration
#[derive(Debug, Clone)]
pub struct ConstDeclaration {
    pub name: Identifier,
    pub type_annotation: Type,
    pub value: Expression,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// Module declaration
#[derive(Debug, Clone)]
pub struct Module {
    pub name: Identifier,
    pub items: Vec<Item>,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// Event definition
#[derive(Debug, Clone)]
pub struct Event {
    pub name: Identifier,
    pub fields: Vec<EventField>,
    pub location: SourceLocation,
}

/// Event field
#[derive(Debug, Clone)]
pub struct EventField {
    pub name: Identifier,
    pub type_annotation: Type,
    pub indexed: bool,
    pub location: SourceLocation,
}

/// Function modifier
#[derive(Debug, Clone)]
pub struct Modifier {
    pub name: Identifier,
    pub parameters: Vec<Parameter>,
    pub body: Block,
    pub location: SourceLocation,
}

/// Struct or contract field
#[derive(Debug, Clone)]
pub struct Field {
    pub name: Identifier,
    pub type_annotation: Type,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// Function parameter
#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: Identifier,
    pub type_annotation: Type,
    pub location: SourceLocation,
}

/// Block of statements
#[derive(Debug, Clone)]
pub struct Block {
    pub statements: Vec<Statement>,
    pub location: SourceLocation,
}

/// Statement types
#[derive(Debug, Clone)]
pub enum Statement {
    Expression(Expression),
    Let(LetStatement),
    Return(ReturnStatement),
    If(IfStatement),
    While(WhileStatement),
    For(ForStatement),
    Match(MatchStatement),
    Break(BreakStatement),
    Continue(ContinueStatement),
    Emit(EmitStatement),
    Require(RequireStatement),
    Assert(AssertStatement),
    Revert(RevertStatement),
}

/// Let statement (variable declaration)
#[derive(Debug, Clone)]
pub struct LetStatement {
    pub name: Identifier,
    pub type_annotation: Option<Type>,
    pub value: Option<Expression>,
    pub mutable: bool,
    pub location: SourceLocation,
}

/// Return statement
#[derive(Debug, Clone)]
pub struct ReturnStatement {
    pub value: Option<Expression>,
    pub location: SourceLocation,
}

/// If statement
#[derive(Debug, Clone)]
pub struct IfStatement {
    pub condition: Expression,
    pub then_block: Block,
    pub else_block: Option<Box<Statement>>,
    pub location: SourceLocation,
}

/// While loop
#[derive(Debug, Clone)]
pub struct WhileStatement {
    pub condition: Expression,
    pub body: Block,
    pub location: SourceLocation,
}

/// For loop
#[derive(Debug, Clone)]
pub struct ForStatement {
    pub variable: Identifier,
    pub iterable: Expression,
    pub body: Block,
    pub location: SourceLocation,
}

/// Match statement
#[derive(Debug, Clone)]
pub struct MatchStatement {
    pub expression: Expression,
    pub arms: Vec<MatchArm>,
    pub location: SourceLocation,
}

/// Match arm
#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expression>,
    pub body: Block,
    pub location: SourceLocation,
}

/// Pattern for match statements
#[derive(Debug, Clone)]
pub enum Pattern {
    Literal(Literal),
    Identifier(Identifier),
    Wildcard,
    Tuple(Vec<Pattern>),
    Struct {
        name: Identifier,
        fields: Vec<(Identifier, Pattern)>,
    },
    Enum {
        name: Identifier,
        variant: Identifier,
        fields: Option<Vec<Pattern>>,
    },
}

/// Break statement
#[derive(Debug, Clone)]
pub struct BreakStatement {
    pub location: SourceLocation,
}

/// Continue statement
#[derive(Debug, Clone)]
pub struct ContinueStatement {
    pub location: SourceLocation,
}

/// Emit statement
#[derive(Debug, Clone)]
pub struct EmitStatement {
    pub event: Identifier,
    pub arguments: Vec<Expression>,
    pub location: SourceLocation,
}

/// Require statement
#[derive(Debug, Clone)]
pub struct RequireStatement {
    pub condition: Expression,
    pub message: Option<Expression>,
    pub location: SourceLocation,
}

/// Assert statement
#[derive(Debug, Clone)]
pub struct AssertStatement {
    pub condition: Expression,
    pub message: Option<Expression>,
    pub location: SourceLocation,
}

/// Revert statement
#[derive(Debug, Clone)]
pub struct RevertStatement {
    pub message: Option<Expression>,
    pub location: SourceLocation,
}

/// Expression types
#[derive(Debug, Clone)]
pub enum Expression {
    Literal(Literal),
    Identifier(Identifier),
    Binary(BinaryExpression),
    Unary(UnaryExpression),
    Call(CallExpression),
    FieldAccess(FieldAccessExpression),
    Index(IndexExpression),
    Array(ArrayExpression),
    Tuple(TupleExpression),
    Struct(StructExpression),
    Assignment(AssignmentExpression),
    Range(RangeExpression),
    Closure(ClosureExpression),
    Block(Block),
}

/// Literal values
#[derive(Debug, Clone)]
pub enum Literal {
    Integer(u64),
    String(String),
    Boolean(bool),
    Address(String),
}

/// Binary expression
#[derive(Debug, Clone)]
pub struct BinaryExpression {
    pub left: Box<Expression>,
    pub operator: BinaryOperator,
    pub right: Box<Expression>,
    pub location: SourceLocation,
}

/// Binary operators
#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    // Arithmetic
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    
    // Comparison
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    
    // Logical
    And,
    Or,
    
    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    LeftShift,
    RightShift,
}

/// Unary expression
#[derive(Debug, Clone)]
pub struct UnaryExpression {
    pub operator: UnaryOperator,
    pub operand: Box<Expression>,
    pub location: SourceLocation,
}

/// Unary operators
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Not,
    Minus,
    BitNot,
}

/// Function call expression
#[derive(Debug, Clone)]
pub struct CallExpression {
    pub function: Box<Expression>,
    pub arguments: Vec<Expression>,
    pub location: SourceLocation,
}

/// Field access expression
#[derive(Debug, Clone)]
pub struct FieldAccessExpression {
    pub object: Box<Expression>,
    pub field: Identifier,
    pub location: SourceLocation,
}

/// Index expression
#[derive(Debug, Clone)]
pub struct IndexExpression {
    pub object: Box<Expression>,
    pub index: Box<Expression>,
    pub location: SourceLocation,
}

/// Array expression
#[derive(Debug, Clone)]
pub struct ArrayExpression {
    pub elements: Vec<Expression>,
    pub location: SourceLocation,
}

/// Tuple expression
#[derive(Debug, Clone)]
pub struct TupleExpression {
    pub elements: Vec<Expression>,
    pub location: SourceLocation,
}

/// Struct construction expression
#[derive(Debug, Clone)]
pub struct StructExpression {
    pub name: Identifier,
    pub fields: Vec<(Identifier, Expression)>,
    pub location: SourceLocation,
}

/// Assignment expression
#[derive(Debug, Clone)]
pub struct AssignmentExpression {
    pub target: Box<Expression>,
    pub operator: AssignmentOperator,
    pub value: Box<Expression>,
    pub location: SourceLocation,
}

/// Assignment operators
#[derive(Debug, Clone, PartialEq)]
pub enum AssignmentOperator {
    Assign,
    AddAssign,
    SubtractAssign,
    MultiplyAssign,
    DivideAssign,
}

/// Range expression
#[derive(Debug, Clone)]
pub struct RangeExpression {
    pub start: Option<Box<Expression>>,
    pub end: Option<Box<Expression>>,
    pub inclusive: bool,
    pub location: SourceLocation,
}

/// Closure expression
#[derive(Debug, Clone)]
pub struct ClosureExpression {
    pub parameters: Vec<Parameter>,
    pub return_type: Option<Type>,
    pub body: Box<Expression>,
    pub location: SourceLocation,
}

/// Type annotations
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    // Primitive types
    U8,
    U16,
    U32,
    U64,
    U128,
    U256,
    I8,
    I16,
    I32,
    I64,
    I128,
    I256,
    Bool,
    String,
    Address,
    Bytes,
    
    // Compound types
    Array {
        element_type: Box<Type>,
        size: Option<u64>,
    },
    Tuple(Vec<Type>),
    Option(Box<Type>),
    Result {
        ok_type: Box<Type>,
        err_type: Box<Type>,
    },
    
    // User-defined types
    Named(Identifier),
    
    // Function types
    Function {
        parameters: Vec<Type>,
        return_type: Box<Type>,
    },
}

/// Identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Identifier {
    pub name: String,
    pub location: SourceLocation,
}

/// Visibility modifiers
#[derive(Debug, Clone, PartialEq)]
pub enum Visibility {
    Public,
    Private,
}

/// Mutability modifiers
#[derive(Debug, Clone, PartialEq)]
pub enum Mutability {
    Mutable,
    Immutable,
    View,
    Pure,
}

/// Function attributes
#[derive(Debug, Clone)]
pub struct Attribute {
    pub name: String,
    pub arguments: Vec<String>,
    pub location: SourceLocation,
}

// Utility implementations

impl Identifier {
    pub fn new(name: String, location: SourceLocation) -> Self {
        Self { name, location }
    }
    
    pub fn dummy(name: &str) -> Self {
        Self {
            name: name.to_string(),
            location: SourceLocation::dummy(),
        }
    }
}

impl Default for Visibility {
    fn default() -> Self {
        Visibility::Private
    }
}

impl Default for Mutability {
    fn default() -> Self {
        Mutability::Immutable
    }
}

// Helper functions for AST construction

impl Expression {
    pub fn location(&self) -> &SourceLocation {
        // Static dummy location for literals that don't carry location info
        static DUMMY_LOCATION: std::sync::OnceLock<SourceLocation> = std::sync::OnceLock::new();
        
        match self {
            Expression::Literal(lit) => match lit {
                Literal::Integer(_) | Literal::String(_) | Literal::Boolean(_) | Literal::Address(_) => {
                    // For now, return a dummy location for literals
                    // In a real implementation, literals would carry location info
                    DUMMY_LOCATION.get_or_init(|| SourceLocation::unknown())
                }
            },
            Expression::Identifier(id) => &id.location,
            Expression::Binary(expr) => &expr.location,
            Expression::Unary(expr) => &expr.location,
            Expression::Call(expr) => &expr.location,
            Expression::FieldAccess(expr) => &expr.location,
            Expression::Index(expr) => &expr.location,
            Expression::Array(expr) => &expr.location,
            Expression::Tuple(expr) => &expr.location,
            Expression::Struct(expr) => &expr.location,
            Expression::Assignment(expr) => &expr.location,
            Expression::Range(expr) => &expr.location,
            Expression::Closure(expr) => &expr.location,
            Expression::Block(block) => &block.location,
        }
    }
}

impl Statement {
    pub fn location(&self) -> &SourceLocation {
        match self {
            Statement::Expression(expr) => expr.location(),
            Statement::Let(stmt) => &stmt.location,
            Statement::Return(stmt) => &stmt.location,
            Statement::If(stmt) => &stmt.location,
            Statement::While(stmt) => &stmt.location,
            Statement::For(stmt) => &stmt.location,
            Statement::Match(stmt) => &stmt.location,
            Statement::Break(stmt) => &stmt.location,
            Statement::Continue(stmt) => &stmt.location,
            Statement::Emit(stmt) => &stmt.location,
            Statement::Require(stmt) => &stmt.location,
            Statement::Assert(stmt) => &stmt.location,
            Statement::Revert(stmt) => &stmt.location,
        }
    }
}

// Display implementations for debugging

use std::fmt;

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::U8 => write!(f, "u8"),
            Type::U16 => write!(f, "u16"),
            Type::U32 => write!(f, "u32"),
            Type::U64 => write!(f, "u64"),
            Type::U128 => write!(f, "u128"),
            Type::U256 => write!(f, "u256"),
            Type::I8 => write!(f, "i8"),
            Type::I16 => write!(f, "i16"),
            Type::I32 => write!(f, "i32"),
            Type::I64 => write!(f, "i64"),
            Type::I128 => write!(f, "i128"),
            Type::I256 => write!(f, "i256"),
            Type::Bool => write!(f, "bool"),
            Type::String => write!(f, "string"),
            Type::Address => write!(f, "address"),
            Type::Bytes => write!(f, "bytes"),
            Type::Array { element_type, size } => {
                if let Some(size) = size {
                    write!(f, "[{}; {}]", element_type, size)
                } else {
                    write!(f, "[{}]", element_type)
                }
            }
            Type::Tuple(types) => {
                write!(f, "(")?;
                for (i, ty) in types.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ty)?;
                }
                write!(f, ")")
            }
            Type::Option(inner) => write!(f, "Option<{}>", inner),
            Type::Result { ok_type, err_type } => write!(f, "Result<{}, {}>", ok_type, err_type),
            Type::Named(name) => write!(f, "{}", name.name),
            Type::Function { parameters, return_type } => {
                write!(f, "fn(")?;
                for (i, param) in parameters.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", param)?;
                }
                write!(f, ") -> {}", return_type)
            }
        }
    }
}

impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = match self {
            BinaryOperator::Add => "+",
            BinaryOperator::Subtract => "-",
            BinaryOperator::Multiply => "*",
            BinaryOperator::Divide => "/",
            BinaryOperator::Modulo => "%",
            BinaryOperator::Equal => "==",
            BinaryOperator::NotEqual => "!=",
            BinaryOperator::Less => "<",
            BinaryOperator::LessEqual => "<=",
            BinaryOperator::Greater => ">",
            BinaryOperator::GreaterEqual => ">=",
            BinaryOperator::And => "&&",
            BinaryOperator::Or => "||",
            BinaryOperator::BitAnd => "&",
            BinaryOperator::BitOr => "|",
            BinaryOperator::BitXor => "^",
            BinaryOperator::LeftShift => "<<",
            BinaryOperator::RightShift => ">>",
        };
        write!(f, "{}", op)
    }
}

impl fmt::Display for UnaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = match self {
            UnaryOperator::Not => "!",
            UnaryOperator::Minus => "-",
            UnaryOperator::BitNot => "~",
        };
        write!(f, "{}", op)
    }
}