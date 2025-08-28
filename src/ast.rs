//! Abstract Syntax Tree (AST) definitions for the Augustium programming language
//!
//! This module defines the data structures that represent the parsed structure
//! of Augustium source code.

use crate::error::SourceLocation;
use serde::{Serialize, Deserialize};


/// A complete Augustium source file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceFile {
    pub items: Vec<Item>,
    pub location: SourceLocation,
}

/// Top-level items in an Augustium source file
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    OperatorImpl(OperatorImpl),
}

/// Contract definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contract {
    pub name: Identifier,
    pub fields: Vec<Field>,
    pub functions: Vec<Function>,
    pub events: Vec<Event>,
    pub modifiers: Vec<Modifier>,
    pub location: SourceLocation,
}

/// Function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: Identifier,
    pub type_parameters: Vec<TypeParameter>,
    pub parameters: Vec<Parameter>,
    pub return_type: Option<Type>,
    pub body: Block,
    pub visibility: Visibility,
    pub mutability: Mutability,
    pub attributes: Vec<Attribute>,
    pub is_async: bool,
    pub location: SourceLocation,
}

/// Struct definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Struct {
    pub name: Identifier,
    pub type_parameters: Vec<TypeParameter>,
    pub fields: Vec<Field>,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// Enum definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Enum {
    pub name: Identifier,
    pub variants: Vec<EnumVariant>,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// Enum variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumVariant {
    pub name: Identifier,
    pub fields: Option<Vec<Type>>,
    pub location: SourceLocation,
}

/// Trait definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trait {
    pub name: Identifier,
    pub type_parameters: Vec<TypeParameter>,
    pub functions: Vec<TraitFunction>,
    pub associated_types: Vec<AssociatedType>,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// Trait function signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitFunction {
    pub name: Identifier,
    pub parameters: Vec<Parameter>,
    pub return_type: Option<Type>,
    pub location: SourceLocation,
}

/// Implementation block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Impl {
    pub trait_name: Option<Identifier>,
    pub type_name: Identifier,
    pub type_parameters: Vec<TypeParameter>,
    pub where_clause: Option<WhereClause>,
    pub functions: Vec<Function>,
    pub location: SourceLocation,
}

/// Use declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UseDeclaration {
    pub path: Vec<Identifier>,
    pub alias: Option<Identifier>,
    pub location: SourceLocation,
}

/// Constant declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstDeclaration {
    pub name: Identifier,
    pub type_annotation: Type,
    pub value: Expression,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// Module declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Module {
    pub name: Identifier,
    pub items: Vec<Item>,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// Event definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub name: Identifier,
    pub fields: Vec<EventField>,
    pub location: SourceLocation,
}

/// Event field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventField {
    pub name: Identifier,
    pub type_annotation: Type,
    pub indexed: bool,
    pub location: SourceLocation,
}

/// Function modifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Modifier {
    pub name: Identifier,
    pub parameters: Vec<Parameter>,
    pub body: Block,
    pub location: SourceLocation,
}

/// Struct or contract field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Field {
    pub name: Identifier,
    pub type_annotation: Type,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// Function parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: Identifier,
    pub type_annotation: Type,
    pub location: SourceLocation,
}

/// Block of statements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub statements: Vec<Statement>,
    pub location: SourceLocation,
}

/// Statement types
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    Macro(MacroInvocation),
}

/// Let statement (variable declaration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LetStatement {
    pub name: Identifier,
    pub type_annotation: Option<Type>,
    pub value: Option<Expression>,
    pub mutable: bool,
    pub location: SourceLocation,
}

/// Return statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnStatement {
    pub value: Option<Expression>,
    pub location: SourceLocation,
}

/// If statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IfStatement {
    pub condition: Expression,
    pub then_block: Block,
    pub else_block: Option<Box<Statement>>,
    pub location: SourceLocation,
}

/// While loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhileStatement {
    pub condition: Expression,
    pub body: Block,
    pub location: SourceLocation,
}

/// For loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForStatement {
    pub variable: Identifier,
    pub iterable: Expression,
    pub body: Block,
    pub location: SourceLocation,
}

/// Match statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchStatement {
    pub expression: Expression,
    pub arms: Vec<MatchArm>,
    pub location: SourceLocation,
}

/// Match arm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expression>,
    pub body: Block,
    pub location: SourceLocation,
}

/// Pattern for match statements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Pattern {
    Literal(Literal),
    Identifier(Identifier),
    Wildcard,
    Tuple(Vec<Pattern>),
    Struct {
        name: Identifier,
        fields: Vec<FieldPattern>,
        rest: bool, // for .. patterns
    },
    Enum {
        name: Identifier,
        variant: Identifier,
        fields: Option<Vec<Pattern>>,
    },
    Array {
        patterns: Vec<Pattern>,
        rest: Option<Box<Pattern>>, // for [a, b, ..rest] patterns
    },
    Slice {
        patterns: Vec<Pattern>,
        rest_position: Option<usize>,
    },
    Range {
        start: Option<Box<Pattern>>,
        end: Option<Box<Pattern>>,
        inclusive: bool,
    },
    Or(Vec<Pattern>), // for a | b | c patterns
    Guard {
        pattern: Box<Pattern>,
        condition: Expression,
    },
    Binding {
        name: Identifier,
        pattern: Box<Pattern>,
    },
    Reference {
        mutable: bool,
        pattern: Box<Pattern>,
    },
    Deref(Box<Pattern>),
}

/// Break statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakStatement {
    pub location: SourceLocation,
}

/// Continue statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinueStatement {
    pub location: SourceLocation,
}

/// Emit statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmitStatement {
    pub event: Identifier,
    pub arguments: Vec<Expression>,
    pub location: SourceLocation,
}

/// Require statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequireStatement {
    pub condition: Expression,
    pub message: Option<Expression>,
    pub location: SourceLocation,
}

/// Assert statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssertStatement {
    pub condition: Expression,
    pub message: Option<Expression>,
    pub location: SourceLocation,
}

/// Revert statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevertStatement {
    pub message: Option<Expression>,
    pub location: SourceLocation,
}

/// Expression types
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    Await(AwaitExpression),
    
    // Machine Learning expressions
    MLCreateModel(MLCreateModelExpression),
    MLTrain(MLTrainExpression),
    MLPredict(MLPredictExpression),
    MLForward(MLForwardExpression),
    MLBackward(MLBackwardExpression),
    TensorOp(TensorOpExpression),
    MatrixOp(MatrixOpExpression),
}

/// Literal values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Literal {
    Integer(u64),
    Float(f64),
    String(String),
    Boolean(bool),
    Address(String),
    Null,
}

/// Binary expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryExpression {
    pub left: Box<Expression>,
    pub operator: BinaryOperator,
    pub right: Box<Expression>,
    pub location: SourceLocation,
}

/// Binary operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnaryExpression {
    pub operator: UnaryOperator,
    pub operand: Box<Expression>,
    pub location: SourceLocation,
}

/// Unary operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UnaryOperator {
    Not,
    Minus,
    BitNot,
}

/// Function call expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallExpression {
    pub function: Box<Expression>,
    pub arguments: Vec<Expression>,
    pub location: SourceLocation,
}

/// Field access expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldAccessExpression {
    pub object: Box<Expression>,
    pub field: Identifier,
    pub location: SourceLocation,
}

/// Index expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexExpression {
    pub object: Box<Expression>,
    pub index: Box<Expression>,
    pub location: SourceLocation,
}

/// Array expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayExpression {
    pub elements: Vec<Expression>,
    pub location: SourceLocation,
}

/// Tuple expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TupleExpression {
    pub elements: Vec<Expression>,
    pub location: SourceLocation,
}

/// Struct construction expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructExpression {
    pub name: Identifier,
    pub fields: Vec<(Identifier, Expression)>,
    pub location: SourceLocation,
}

/// Assignment expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignmentExpression {
    pub target: Box<Expression>,
    pub operator: AssignmentOperator,
    pub value: Box<Expression>,
    pub location: SourceLocation,
}

/// Assignment operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AssignmentOperator {
    Assign,
    AddAssign,
    SubtractAssign,
    MultiplyAssign,
    DivideAssign,
}

/// Range expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeExpression {
    pub start: Option<Box<Expression>>,
    pub end: Option<Box<Expression>>,
    pub inclusive: bool,
    pub location: SourceLocation,
}

/// Closure expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosureExpression {
    pub parameters: Vec<Parameter>,
    pub return_type: Option<Type>,
    pub body: Box<Expression>,
    pub location: SourceLocation,
}

/// Type annotations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    F32,
    F64,
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
    Generic {
        base: Box<Type>,
        type_args: Vec<Type>,
    },
    TypeParameter(Identifier),
    
    // Function types
    Function {
        parameters: Vec<Type>,
        return_type: Box<Type>,
    },
    
    // Machine Learning types
    MLModel {
        model_type: String,
        input_shape: Vec<u64>,
        output_shape: Vec<u64>,
    },
    MLDataset {
        feature_types: Vec<Type>,
        target_type: Box<Type>,
    },
    Tensor {
        element_type: Box<Type>,
        dimensions: Vec<u64>,
    },
    Matrix {
        element_type: Box<Type>,
        rows: u64,
        cols: u64,
    },
    Vector {
        element_type: Box<Type>,
        size: Option<u64>,
    },
    MLMetrics,
    
    // Async types
    Future(Box<Type>),
    Stream(Box<Type>),
}

/// Identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Identifier {
    pub name: String,
    pub location: SourceLocation,
}

/// Visibility modifiers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Visibility {
    Public,
    Private,
}

/// Mutability modifiers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Mutability {
    Mutable,
    Immutable,
    View,
    Pure,
}

/// Function attributes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attribute {
    pub name: String,
    pub arguments: Vec<String>,
    pub location: SourceLocation,
}

/// Type parameter for generics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeParameter {
    pub name: Identifier,
    pub bounds: Vec<TypeBound>,
    pub default: Option<Type>,
    pub location: SourceLocation,
}

/// Type bounds for generics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeBound {
    Trait(Identifier),
    Lifetime(Identifier),
}

/// Associated type in traits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociatedType {
    pub name: Identifier,
    pub bounds: Vec<TypeBound>,
    pub default: Option<Type>,
    pub location: SourceLocation,
}

/// Where clause for complex bounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhereClause {
    pub predicates: Vec<WherePredicate>,
    pub location: SourceLocation,
}

/// Where predicate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WherePredicate {
    Type {
        ty: Type,
        bounds: Vec<TypeBound>,
    },
    Lifetime {
        lifetime: Identifier,
        bounds: Vec<Identifier>,
    },
}

/// Await expression for async
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwaitExpression {
    pub expression: Box<Expression>,
    pub location: SourceLocation,
}

/// Macro invocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroInvocation {
    pub name: Identifier,
    pub arguments: Vec<MacroArgument>,
    pub location: SourceLocation,
}

/// Macro argument
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MacroArgument {
    Expression(Expression),
    Type(Type),
    Pattern(Pattern),
    Statement(Statement),
    Literal(String),
}

/// Field pattern for struct destructuring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldPattern {
    pub name: Identifier,
    pub pattern: Pattern,
    pub shorthand: bool, // for { x } instead of { x: x }
}

/// Operator implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorImpl {
    pub operator: OverloadableOperator,
    pub type_name: Identifier,
    pub parameters: Vec<Parameter>,
    pub return_type: Option<Type>,
    pub body: Block,
    pub location: SourceLocation,
}

/// Operators that can be overloaded
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OverloadableOperator {
    // Arithmetic operators
    Add,        // +
    Subtract,   // -
    Multiply,   // *
    Divide,     // /
    Modulo,     // %
    
    // Comparison operators
    Equal,      // ==
    NotEqual,   // !=
    Less,       // <
    LessEqual,  // <=
    Greater,    // >
    GreaterEqual, // >=
    
    // Bitwise operators
    BitAnd,     // &
    BitOr,      // |
    BitXor,     // ^
    LeftShift,  // <<
    RightShift, // >>
    
    // Unary operators
    Negate,     // -x
    Not,        // !x
    BitNot,     // ~x
    
    // Assignment operators
    AddAssign,     // +=
    SubtractAssign, // -=
    MultiplyAssign, // *=
    DivideAssign,   // /=
    
    // Index operators
    Index,      // []
    IndexMut,   // []= (mutable indexing)
    
    // Call operator
    Call,       // ()
    
    // Conversion operators
    Into,       // into()
    From,       // from()
    
    // Display operators
    Display,    // fmt::Display
    Debug,      // fmt::Debug
}

/// ML Create Model expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLCreateModelExpression {
    pub model_type: String,
    pub config: Vec<(String, Expression)>,
    pub location: SourceLocation,
}

/// ML Train expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLTrainExpression {
    pub model: Box<Expression>,
    pub dataset: Box<Expression>,
    pub epochs: Option<Box<Expression>>,
    pub location: SourceLocation,
}

/// ML Predict expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPredictExpression {
    pub model: Box<Expression>,
    pub input: Box<Expression>,
    pub location: SourceLocation,
}

/// ML Forward pass expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLForwardExpression {
    pub model: Box<Expression>,
    pub input: Box<Expression>,
    pub location: SourceLocation,
}

/// ML Backward pass expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLBackwardExpression {
    pub model: Box<Expression>,
    pub gradients: Box<Expression>,
    pub location: SourceLocation,
}

/// Tensor operation expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorOpExpression {
    pub operation: TensorOperation,
    pub operands: Vec<Expression>,
    pub location: SourceLocation,
}

/// Matrix operation expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixOpExpression {
    pub operation: MatrixOperation,
    pub operands: Vec<Expression>,
    pub location: SourceLocation,
}

/// Tensor operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TensorOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
    MatMul,
    Transpose,
    Reshape,
    Sum,
    Mean,
    Max,
    Min,
}

/// Matrix operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MatrixOperation {
    Add,
    Subtract,
    Multiply,
    Transpose,
    Inverse,
    Determinant,
    Eigenvalues,
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
                Literal::Integer(_) | Literal::Float(_) | Literal::String(_) | Literal::Boolean(_) | Literal::Address(_) | Literal::Null => {
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
            Expression::Await(expr) => &expr.location,
            
            // ML expressions
            Expression::MLCreateModel(expr) => &expr.location,
            Expression::MLTrain(expr) => &expr.location,
            Expression::MLPredict(expr) => &expr.location,
            Expression::MLForward(expr) => &expr.location,
            Expression::MLBackward(expr) => &expr.location,
            Expression::TensorOp(expr) => &expr.location,
            Expression::MatrixOp(expr) => &expr.location,
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
            Statement::Macro(stmt) => &stmt.location,
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
            Type::F32 => write!(f, "f32"),
            Type::F64 => write!(f, "f64"),
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
            
            // ML types
            Type::MLModel { model_type, input_shape, output_shape } => {
                write!(f, "MLModel<{}, {:?} -> {:?}>", model_type, input_shape, output_shape)
            }
            Type::MLDataset { feature_types, target_type } => {
                write!(f, "MLDataset<{:?}, {}>", feature_types, target_type)
            }
            Type::Tensor { element_type, dimensions } => {
                write!(f, "Tensor<{}, {:?}>", element_type, dimensions)
            }
            Type::Matrix { element_type, rows, cols } => {
                write!(f, "Matrix<{}, {}x{}>", element_type, rows, cols)
            }
            Type::Vector { element_type, size } => {
                if let Some(size) = size {
                    write!(f, "Vector<{}, {}>", element_type, size)
                } else {
                    write!(f, "Vector<{}>", element_type)
                }
            }
            Type::MLMetrics => write!(f, "MLMetrics"),
            Type::Future(inner) => write!(f, "Future<{}>", inner),
            Type::Stream(inner) => write!(f, "Stream<{}>", inner),
            Type::Generic { base, type_args } => {
                write!(f, "{}<", base)?;
                for (i, arg) in type_args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ">")
            }
            Type::TypeParameter(name) => write!(f, "{}", name.name),
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