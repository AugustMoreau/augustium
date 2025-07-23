//! Syntax Standardization Module for Augustium
//!
//! This module provides tools and utilities to standardize Augustium syntax,
//! ensuring consistent type annotations, keyword usage, and function definitions
//! across the entire codebase.

use crate::error::{Result, CompilerError};
use crate::ast::{Expression, Statement};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Syntax style configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntaxStyle {
    pub type_annotation_style: TypeAnnotationStyle,
    pub keyword_style: KeywordStyle,
    pub function_style: FunctionStyle,
    pub naming_convention: NamingConvention,
    pub indentation: IndentationStyle,
}

/// Type annotation style preferences
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TypeAnnotationStyle {
    /// Rust-style: `variable: Type`
    RustStyle,
    /// C-style: `Type variable`
    CStyle,
    /// TypeScript-style: `variable: Type`
    TypeScriptStyle,
    /// Auto-detect and standardize
    Auto,
}

/// Keyword style preferences
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KeywordStyle {
    /// Rust-style keywords (fn, let, mut, pub)
    RustStyle,
    /// Solidity-style keywords (function, var, public)
    SolidityStyle,
    /// JavaScript-style keywords (function, let, const)
    JavaScriptStyle,
    /// Unified Augustium style
    AugustiumStyle,
}

/// Function definition style
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FunctionStyle {
    /// `fn name(params) -> return_type`
    RustStyle,
    /// `function name(params) returns (return_type)`
    SolidityStyle,
    /// `function name(params): return_type`
    TypeScriptStyle,
    /// Unified Augustium style
    AugustiumStyle,
}

/// Naming convention preferences
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NamingConvention {
    /// snake_case for variables and functions
    SnakeCase,
    /// camelCase for variables and functions
    CamelCase,
    /// PascalCase for types and contracts
    PascalCase,
    /// Mixed conventions based on context
    Mixed,
}

/// Indentation style
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndentationStyle {
    pub use_spaces: bool,
    pub size: usize,
}

/// Syntax transformation rules
#[derive(Debug, Clone)]
pub struct TransformationRule {
    pub name: String,
    pub description: String,
    pub pattern: SyntaxPattern,
    pub replacement: SyntaxPattern,
    pub priority: u32,
}

/// Syntax pattern for matching and replacement
#[derive(Debug, Clone)]
pub enum SyntaxPattern {
    /// Match a specific keyword
    Keyword(String),
    /// Match a type annotation pattern
    TypeAnnotation {
        variable: String,
        type_name: String,
        style: TypeAnnotationStyle,
    },
    /// Match a function definition pattern
    FunctionDefinition {
        name: String,
        parameters: Vec<(String, String)>, // (name, type)
        return_type: Option<String>,
        style: FunctionStyle,
    },
    /// Match any expression
    Expression(Box<Expression>),
    /// Match any statement
    Statement(Box<Statement>),
}

/// Syntax standardizer
pub struct SyntaxStandardizer {
    style: SyntaxStyle,
    rules: Vec<TransformationRule>,
    type_mappings: HashMap<String, String>,
    keyword_mappings: HashMap<String, String>,
}

/// Syntax analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntaxAnalysis {
    pub inconsistencies: Vec<SyntaxInconsistency>,
    pub style_violations: Vec<StyleViolation>,
    pub suggestions: Vec<SyntaxSuggestion>,
    pub compatibility_score: f64, // 0.0 to 1.0
}

/// Syntax inconsistency report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntaxInconsistency {
    pub location: (usize, usize), // (line, column)
    pub inconsistency_type: InconsistencyType,
    pub current_syntax: String,
    pub expected_syntax: String,
    pub severity: Severity,
}

/// Style violation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleViolation {
    pub location: (usize, usize),
    pub violation_type: ViolationType,
    pub description: String,
    pub suggested_fix: String,
}

/// Syntax improvement suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntaxSuggestion {
    pub location: (usize, usize),
    pub suggestion_type: SuggestionType,
    pub description: String,
    pub before: String,
    pub after: String,
    pub impact: Impact,
}

/// Types of syntax inconsistencies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InconsistencyType {
    TypeAnnotationMismatch,
    KeywordMismatch,
    FunctionDefinitionMismatch,
    NamingConventionMismatch,
    IndentationMismatch,
}

/// Types of style violations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ViolationType {
    DeprecatedSyntax,
    NonStandardKeyword,
    InconsistentNaming,
    PoorReadability,
    PerformanceImpact,
}

/// Types of suggestions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SuggestionType {
    Modernization,
    Standardization,
    Optimization,
    Readability,
    BestPractice,
}

/// Severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum Severity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Impact levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Impact {
    Low,
    Medium,
    High,
    Critical,
}

impl Default for SyntaxStyle {
    fn default() -> Self {
        Self {
            type_annotation_style: TypeAnnotationStyle::RustStyle,
            keyword_style: KeywordStyle::AugustiumStyle,
            function_style: FunctionStyle::AugustiumStyle,
            naming_convention: NamingConvention::SnakeCase,
            indentation: IndentationStyle {
                use_spaces: true,
                size: 4,
            },
        }
    }
}

impl SyntaxStandardizer {
    /// Create a new syntax standardizer with default Augustium style
    pub fn new() -> Self {
        let mut standardizer = Self {
            style: SyntaxStyle::default(),
            rules: Vec::new(),
            type_mappings: HashMap::new(),
            keyword_mappings: HashMap::new(),
        };
        
        standardizer.initialize_default_rules();
        standardizer.initialize_type_mappings();
        standardizer.initialize_keyword_mappings();
        
        standardizer
    }
    
    /// Create a standardizer with custom style
    pub fn with_style(style: SyntaxStyle) -> Self {
        let mut standardizer = Self {
            style,
            rules: Vec::new(),
            type_mappings: HashMap::new(),
            keyword_mappings: HashMap::new(),
        };
        
        standardizer.initialize_default_rules();
        standardizer.initialize_type_mappings();
        standardizer.initialize_keyword_mappings();
        
        standardizer
    }
    
    /// Initialize default transformation rules
    fn initialize_default_rules(&mut self) {
        // Type annotation standardization rules
        self.add_rule(TransformationRule {
            name: "standardize_type_annotations".to_string(),
            description: "Convert all type annotations to Rust-style format".to_string(),
            pattern: SyntaxPattern::TypeAnnotation {
                variable: "*".to_string(),
                type_name: "*".to_string(),
                style: TypeAnnotationStyle::CStyle,
            },
            replacement: SyntaxPattern::TypeAnnotation {
                variable: "*".to_string(),
                type_name: "*".to_string(),
                style: TypeAnnotationStyle::RustStyle,
            },
            priority: 100,
        });
        
        // Keyword standardization rules
        self.add_rule(TransformationRule {
            name: "standardize_function_keyword".to_string(),
            description: "Convert 'function' keyword to 'fn'".to_string(),
            pattern: SyntaxPattern::Keyword("function".to_string()),
            replacement: SyntaxPattern::Keyword("fn".to_string()),
            priority: 90,
        });
        
        self.add_rule(TransformationRule {
            name: "standardize_variable_keyword".to_string(),
            description: "Convert 'var' keyword to 'let'".to_string(),
            pattern: SyntaxPattern::Keyword("var".to_string()),
            replacement: SyntaxPattern::Keyword("let".to_string()),
            priority: 90,
        });
        
        // Function definition standardization
        self.add_rule(TransformationRule {
            name: "standardize_function_definition".to_string(),
            description: "Convert function definitions to Augustium style".to_string(),
            pattern: SyntaxPattern::FunctionDefinition {
                name: "*".to_string(),
                parameters: vec![],
                return_type: None,
                style: FunctionStyle::SolidityStyle,
            },
            replacement: SyntaxPattern::FunctionDefinition {
                name: "*".to_string(),
                parameters: vec![],
                return_type: None,
                style: FunctionStyle::AugustiumStyle,
            },
            priority: 80,
        });
    }
    
    /// Initialize type mappings for standardization
    fn initialize_type_mappings(&mut self) {
        // Solidity to Augustium type mappings
        self.type_mappings.insert("uint".to_string(), "u256".to_string());
        self.type_mappings.insert("uint256".to_string(), "u256".to_string());
        self.type_mappings.insert("uint128".to_string(), "u128".to_string());
        self.type_mappings.insert("uint64".to_string(), "u64".to_string());
        self.type_mappings.insert("uint32".to_string(), "u32".to_string());
        self.type_mappings.insert("uint16".to_string(), "u16".to_string());
        self.type_mappings.insert("uint8".to_string(), "u8".to_string());
        
        self.type_mappings.insert("int".to_string(), "i256".to_string());
        self.type_mappings.insert("int256".to_string(), "i256".to_string());
        self.type_mappings.insert("int128".to_string(), "i128".to_string());
        self.type_mappings.insert("int64".to_string(), "i64".to_string());
        self.type_mappings.insert("int32".to_string(), "i32".to_string());
        self.type_mappings.insert("int16".to_string(), "i16".to_string());
        self.type_mappings.insert("int8".to_string(), "i8".to_string());
        
        // JavaScript to Augustium type mappings
        self.type_mappings.insert("number".to_string(), "u64".to_string());
        self.type_mappings.insert("bigint".to_string(), "u256".to_string());
        self.type_mappings.insert("boolean".to_string(), "bool".to_string());
        
        // C++ to Augustium type mappings
        self.type_mappings.insert("unsigned int".to_string(), "u32".to_string());
        self.type_mappings.insert("unsigned long".to_string(), "u64".to_string());
        self.type_mappings.insert("size_t".to_string(), "usize".to_string());
    }
    
    /// Initialize keyword mappings
    fn initialize_keyword_mappings(&mut self) {
        // Solidity to Augustium keyword mappings
        self.keyword_mappings.insert("function".to_string(), "fn".to_string());
        self.keyword_mappings.insert("returns".to_string(), "->".to_string());
        self.keyword_mappings.insert("public".to_string(), "pub".to_string());
        self.keyword_mappings.insert("private".to_string(), "priv".to_string());
        self.keyword_mappings.insert("internal".to_string(), "internal".to_string());
        self.keyword_mappings.insert("external".to_string(), "external".to_string());
        
        // JavaScript to Augustium keyword mappings
        self.keyword_mappings.insert("var".to_string(), "let".to_string());
        self.keyword_mappings.insert("const".to_string(), "let".to_string()); // In Augustium, immutability is default
        
        // C++ to Augustium keyword mappings
        self.keyword_mappings.insert("void".to_string(), "()".to_string());
        self.keyword_mappings.insert("nullptr".to_string(), "None".to_string());
    }
    
    /// Add a transformation rule
    pub fn add_rule(&mut self, rule: TransformationRule) {
        self.rules.push(rule);
        // Sort rules by priority (higher priority first)
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }
    
    /// Analyze syntax for inconsistencies and violations
    pub fn analyze_syntax(&self, source_code: &str) -> Result<SyntaxAnalysis> {
        let mut inconsistencies = Vec::new();
        let mut style_violations = Vec::new();
        let mut suggestions = Vec::new();
        
        let lines: Vec<&str> = source_code.lines().collect();
        
        for (line_num, line) in lines.iter().enumerate() {
            // Check for type annotation inconsistencies
            self.check_type_annotations(line, line_num, &mut inconsistencies);
            
            // Check for keyword inconsistencies
            self.check_keywords(line, line_num, &mut inconsistencies);
            
            // Check for style violations
            self.check_style_violations(line, line_num, &mut style_violations);
            
            // Generate suggestions
            self.generate_suggestions(line, line_num, &mut suggestions);
        }
        
        let compatibility_score = self.calculate_compatibility_score(&inconsistencies, &style_violations);
        
        Ok(SyntaxAnalysis {
            inconsistencies,
            style_violations,
            suggestions,
            compatibility_score,
        })
    }
    
    /// Check for type annotation inconsistencies
    fn check_type_annotations(&self, line: &str, line_num: usize, inconsistencies: &mut Vec<SyntaxInconsistency>) {
        // Check for C-style type annotations (Type variable)
        if let Some(captures) = regex::Regex::new(r"\b([A-Z][a-zA-Z0-9_]*|u\d+|i\d+)\s+([a-z_][a-zA-Z0-9_]*)\b")
            .unwrap()
            .captures(line) {
            
            if let (Some(type_match), Some(var_match)) = (captures.get(1), captures.get(2)) {
                let current_syntax = format!("{} {}", type_match.as_str(), var_match.as_str());
                let expected_syntax = format!("{}: {}", var_match.as_str(), type_match.as_str());
                
                inconsistencies.push(SyntaxInconsistency {
                    location: (line_num + 1, type_match.start()),
                    inconsistency_type: InconsistencyType::TypeAnnotationMismatch,
                    current_syntax,
                    expected_syntax,
                    severity: Severity::Warning,
                });
            }
        }
    }
    
    /// Check for keyword inconsistencies
    fn check_keywords(&self, line: &str, line_num: usize, inconsistencies: &mut Vec<SyntaxInconsistency>) {
        for (old_keyword, new_keyword) in &self.keyword_mappings {
            if line.contains(old_keyword) {
                if let Some(pos) = line.find(old_keyword) {
                    inconsistencies.push(SyntaxInconsistency {
                        location: (line_num + 1, pos),
                        inconsistency_type: InconsistencyType::KeywordMismatch,
                        current_syntax: old_keyword.clone(),
                        expected_syntax: new_keyword.clone(),
                        severity: Severity::Info,
                    });
                }
            }
        }
    }
    
    /// Check for style violations
    fn check_style_violations(&self, line: &str, line_num: usize, violations: &mut Vec<StyleViolation>) {
        // Check indentation
        if !line.trim().is_empty() {
            let leading_spaces = line.len() - line.trim_start().len();
            if self.style.indentation.use_spaces {
                if leading_spaces % self.style.indentation.size != 0 {
                    violations.push(StyleViolation {
                        location: (line_num + 1, 0),
                        violation_type: ViolationType::InconsistentNaming,
                        description: "Inconsistent indentation".to_string(),
                        suggested_fix: format!("Use {} spaces for indentation", self.style.indentation.size),
                    });
                }
            }
        }
        
        // Check for deprecated syntax patterns
        if line.contains("msg.sender") && !line.contains("tx.origin") {
            if let Some(pos) = line.find("msg.sender") {
                violations.push(StyleViolation {
                    location: (line_num + 1, pos),
                    violation_type: ViolationType::DeprecatedSyntax,
                    description: "Consider using tx.origin for better clarity".to_string(),
                    suggested_fix: "Replace msg.sender with tx.origin where appropriate".to_string(),
                });
            }
        }
    }
    
    /// Generate syntax improvement suggestions
    fn generate_suggestions(&self, line: &str, line_num: usize, suggestions: &mut Vec<SyntaxSuggestion>) {
        // Suggest modern syntax patterns
        if line.contains("require(") {
            if let Some(pos) = line.find("require(") {
                suggestions.push(SyntaxSuggestion {
                    location: (line_num + 1, pos),
                    suggestion_type: SuggestionType::Modernization,
                    description: "Consider using assert! macro for better error handling".to_string(),
                    before: "require(condition, \"message\")".to_string(),
                    after: "assert!(condition, \"message\")".to_string(),
                    impact: Impact::Medium,
                });
            }
        }
        
        // Suggest type improvements
        if line.contains("uint") && !line.contains("u256") {
            if let Some(pos) = line.find("uint") {
                suggestions.push(SyntaxSuggestion {
                    location: (line_num + 1, pos),
                    suggestion_type: SuggestionType::Standardization,
                    description: "Use explicit bit-width types for better clarity".to_string(),
                    before: "uint".to_string(),
                    after: "u256".to_string(),
                    impact: Impact::Low,
                });
            }
        }
    }
    
    /// Calculate compatibility score based on analysis results
    fn calculate_compatibility_score(&self, inconsistencies: &[SyntaxInconsistency], violations: &[StyleViolation]) -> f64 {
        let total_issues = inconsistencies.len() + violations.len();
        
        if total_issues == 0 {
            return 1.0;
        }
        
        // Weight issues by severity
        let weighted_score = inconsistencies.iter().map(|inc| {
            match inc.severity {
                Severity::Critical => 4.0,
                Severity::Error => 3.0,
                Severity::Warning => 2.0,
                Severity::Info => 1.0,
            }
        }).sum::<f64>() + violations.len() as f64;
        
        // Normalize to 0-1 scale (assuming max 10 weighted points per issue)
        let max_possible_score = total_issues as f64 * 10.0;
        1.0 - (weighted_score / max_possible_score).min(1.0)
    }
    
    /// Apply automatic syntax standardization
    pub fn standardize_syntax(&self, source_code: &str) -> Result<String> {
        let mut standardized = source_code.to_string();
        
        // Apply transformation rules in priority order
        for rule in &self.rules {
            standardized = self.apply_transformation_rule(&standardized, rule)?;
        }
        
        // Apply type mappings
        for (old_type, new_type) in &self.type_mappings {
            standardized = standardized.replace(old_type, new_type);
        }
        
        // Apply keyword mappings
        for (old_keyword, new_keyword) in &self.keyword_mappings {
            standardized = standardized.replace(old_keyword, new_keyword);
        }
        
        Ok(standardized)
    }
    
    /// Apply a single transformation rule
    fn apply_transformation_rule(&self, source: &str, rule: &TransformationRule) -> Result<String> {
        // Simplified transformation - in a real implementation, this would use
        // proper AST transformation based on the pattern matching
        match (&rule.pattern, &rule.replacement) {
            (SyntaxPattern::Keyword(old), SyntaxPattern::Keyword(new)) => {
                Ok(source.replace(old, new))
            }
            _ => {
                // For more complex patterns, we would need AST-based transformation
                Ok(source.to_string())
            }
        }
    }
    
    /// Get current style configuration
    pub fn get_style(&self) -> &SyntaxStyle {
        &self.style
    }
    
    /// Update style configuration
    pub fn set_style(&mut self, style: SyntaxStyle) {
        self.style = style;
    }
    
    /// Export style configuration to file
    pub fn export_style(&self, path: &str) -> Result<()> {
        let style_json = serde_json::to_string_pretty(&self.style)
            .map_err(|e| CompilerError::InternalError(format!("Failed to serialize style: {}", e)))?;
        
        std::fs::write(path, style_json)
            .map_err(|e| CompilerError::InternalError(format!("Failed to write style file: {}", e)))?;
        
        Ok(())
    }
    
    /// Import style configuration from file
    pub fn import_style(&mut self, path: &str) -> Result<()> {
        let style_json = std::fs::read_to_string(path)
            .map_err(|e| CompilerError::InternalError(format!("Failed to read style file: {}", e)))?;
        
        self.style = serde_json::from_str(&style_json)
            .map_err(|e| CompilerError::InternalError(format!("Failed to parse style file: {}", e)))?;
        
        Ok(())
    }
}

/// Syntax formatter for consistent code formatting
pub struct SyntaxFormatter {
    style: SyntaxStyle,
}

impl SyntaxFormatter {
    /// Create a new syntax formatter
    pub fn new(style: SyntaxStyle) -> Self {
        Self { style }
    }
    
    /// Format source code according to style guidelines
    pub fn format(&self, source_code: &str) -> Result<String> {
        let lines: Vec<&str> = source_code.lines().collect();
        let mut formatted_lines = Vec::new();
        let mut indent_level: i32 = 0;
        
        for line in lines {
            let trimmed = line.trim();
            
            // Adjust indent level for closing braces
            if trimmed.starts_with('}') || trimmed.starts_with(']') || trimmed.starts_with(')') {
                indent_level = indent_level.saturating_sub(1);
            }
            
            // Apply indentation
            let indent = if self.style.indentation.use_spaces {
                " ".repeat((indent_level * self.style.indentation.size as i32) as usize)
            } else {
                "\t".repeat(indent_level as usize)
            };
            
            let formatted_line = if trimmed.is_empty() {
                String::new()
            } else {
                format!("{}{}", indent, trimmed)
            };
            
            formatted_lines.push(formatted_line);
            
            // Adjust indent level for opening braces
            if trimmed.ends_with('{') || trimmed.ends_with('[') || trimmed.ends_with('(') {
                indent_level += 1;
            }
        }
        
        Ok(formatted_lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_syntax_standardizer_creation() {
        let standardizer = SyntaxStandardizer::new();
        assert!(!standardizer.rules.is_empty());
        assert!(!standardizer.type_mappings.is_empty());
        assert!(!standardizer.keyword_mappings.is_empty());
    }
    
    #[test]
    fn test_syntax_analysis() {
        let standardizer = SyntaxStandardizer::new();
        let source_code = r#"
            function transfer(uint amount, address to) {
                require(amount > 0, "Invalid amount");
                uint balance = balances[msg.sender];
            }
        "#;
        
        let analysis = standardizer.analyze_syntax(source_code).unwrap();
        assert!(!analysis.inconsistencies.is_empty());
        assert!(analysis.compatibility_score < 1.0);
    }
    
    #[test]
    fn test_syntax_standardization() {
        let standardizer = SyntaxStandardizer::new();
        let source_code = "function test() { var x = 5; }";
        
        let standardized = standardizer.standardize_syntax(source_code).unwrap();
        assert!(standardized.contains("fn"));
        assert!(standardized.contains("let"));
        assert!(!standardized.contains("function"));
        assert!(!standardized.contains("var"));
    }
    
    #[test]
    fn test_syntax_formatter() {
        let style = SyntaxStyle::default();
        let formatter = SyntaxFormatter::new(style);
        
        let source_code = r#"
fn test() {
let x = 5;
if x > 0 {
return x;
}
}
        "#;
        
        let formatted = formatter.format(source_code).unwrap();
        assert!(formatted.contains("    let x = 5;"));
        assert!(formatted.contains("        return x;"));
    }
    
    #[test]
    fn test_type_mappings() {
        let standardizer = SyntaxStandardizer::new();
        
        assert_eq!(standardizer.type_mappings.get("uint"), Some(&"u256".to_string()));
        assert_eq!(standardizer.type_mappings.get("boolean"), Some(&"bool".to_string()));
        assert_eq!(standardizer.type_mappings.get("number"), Some(&"u64".to_string()));
    }
    
    #[test]
    fn test_keyword_mappings() {
        let standardizer = SyntaxStandardizer::new();
        
        assert_eq!(standardizer.keyword_mappings.get("function"), Some(&"fn".to_string()));
        assert_eq!(standardizer.keyword_mappings.get("var"), Some(&"let".to_string()));
        assert_eq!(standardizer.keyword_mappings.get("public"), Some(&"pub".to_string()));
    }
}