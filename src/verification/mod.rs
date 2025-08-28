//! Formal verification and security auditing framework for Augustium
//! Provides mathematical proof capabilities and automated security analysis

use crate::ast::*;
use crate::error::{Result, CompilerError};
use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

/// Verification condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationCondition {
    pub id: String,
    pub description: String,
    pub precondition: LogicalFormula,
    pub postcondition: LogicalFormula,
    pub location: crate::error::SourceLocation,
}

/// Logical formula for verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicalFormula {
    /// Boolean literal
    Bool(bool),
    /// Variable reference
    Variable(String),
    /// Arithmetic expression
    Arithmetic(ArithmeticExpr),
    /// Comparison
    Comparison {
        left: Box<LogicalFormula>,
        op: ComparisonOp,
        right: Box<LogicalFormula>,
    },
    /// Logical conjunction (AND)
    And(Vec<LogicalFormula>),
    /// Logical disjunction (OR)
    Or(Vec<LogicalFormula>),
    /// Logical negation (NOT)
    Not(Box<LogicalFormula>),
    /// Implication
    Implies {
        premise: Box<LogicalFormula>,
        conclusion: Box<LogicalFormula>,
    },
    /// Universal quantification
    ForAll {
        variable: String,
        domain: String,
        formula: Box<LogicalFormula>,
    },
    /// Existential quantification
    Exists {
        variable: String,
        domain: String,
        formula: Box<LogicalFormula>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArithmeticExpr {
    Constant(i64),
    Variable(String),
    Add(Box<ArithmeticExpr>, Box<ArithmeticExpr>),
    Subtract(Box<ArithmeticExpr>, Box<ArithmeticExpr>),
    Multiply(Box<ArithmeticExpr>, Box<ArithmeticExpr>),
    Divide(Box<ArithmeticExpr>, Box<ArithmeticExpr>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOp {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

/// Security vulnerability types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VulnerabilityType {
    ReentrancyAttack,
    IntegerOverflow,
    IntegerUnderflow,
    UnauthorizedAccess,
    GasLimitDoS,
    TimestampDependence,
    RandomnessVulnerability,
    FrontRunning,
    FlashLoanAttack,
    OracleManipulation,
    PrivilegeEscalation,
    DataLeakage,
}

/// Security issue found during analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityIssue {
    pub id: String,
    pub vulnerability_type: VulnerabilityType,
    pub severity: Severity,
    pub title: String,
    pub description: String,
    pub location: crate::error::SourceLocation,
    pub recommendation: String,
    pub cwe_id: Option<u32>, // Common Weakness Enumeration ID
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Formal verification engine
pub struct VerificationEngine {
    conditions: Vec<VerificationCondition>,
    axioms: Vec<LogicalFormula>,
    proof_cache: HashMap<String, ProofResult>,
}

#[derive(Debug, Clone)]
pub enum ProofResult {
    Proven,
    Disproven(String), // Counter-example
    Unknown,
    Timeout,
}

impl VerificationEngine {
    pub fn new() -> Self {
        Self {
            conditions: Vec::new(),
            axioms: Vec::new(),
            proof_cache: HashMap::new(),
        }
    }

    /// Add a verification condition
    pub fn add_condition(&mut self, condition: VerificationCondition) {
        self.conditions.push(condition);
    }

    /// Add an axiom (assumed to be true)
    pub fn add_axiom(&mut self, axiom: LogicalFormula) {
        self.axioms.push(axiom);
    }

    /// Verify all conditions
    pub fn verify_all(&mut self) -> Vec<(String, ProofResult)> {
        let mut results = Vec::new();
        
        for condition in &self.conditions {
            let result = self.verify_condition(condition);
            results.push((condition.id.clone(), result));
        }
        
        results
    }

    fn verify_condition(&mut self, condition: &VerificationCondition) -> ProofResult {
        // Check cache first
        if let Some(cached) = self.proof_cache.get(&condition.id) {
            return cached.clone();
        }

        // Create implication: precondition -> postcondition
        let implication = LogicalFormula::Implies {
            premise: Box::new(condition.precondition.clone()),
            conclusion: Box::new(condition.postcondition.clone()),
        };

        // Try to prove the implication
        let result = self.prove_formula(&implication);
        self.proof_cache.insert(condition.id.clone(), result.clone());
        result
    }

    fn prove_formula(&self, formula: &LogicalFormula) -> ProofResult {
        // Simplified proof engine - in practice, this would use SMT solvers
        match formula {
            LogicalFormula::Bool(true) => ProofResult::Proven,
            LogicalFormula::Bool(false) => ProofResult::Disproven("Formula is false".to_string()),
            LogicalFormula::Implies { premise, conclusion } => {
                // Try to find counter-example
                if self.is_satisfiable(premise) && !self.is_satisfiable(conclusion) {
                    ProofResult::Disproven("Counter-example found".to_string())
                } else {
                    // For now, assume unknown
                    ProofResult::Unknown
                }
            }
            _ => ProofResult::Unknown,
        }
    }

    fn is_satisfiable(&self, _formula: &LogicalFormula) -> bool {
        // Simplified satisfiability check
        // In practice, this would use SAT/SMT solvers
        true
    }
}

/// Security auditor
pub struct SecurityAuditor {
    issues: Vec<SecurityIssue>,
    rules: Vec<SecurityRule>,
}

/// Security analysis rule
pub struct SecurityRule {
    pub name: String,
    pub vulnerability_type: VulnerabilityType,
    pub check: Box<dyn Fn(&SourceFile) -> Vec<SecurityIssue>>,
}

impl SecurityAuditor {
    pub fn new() -> Self {
        let mut auditor = Self {
            issues: Vec::new(),
            rules: Vec::new(),
        };
        auditor.load_default_rules();
        auditor
    }

    /// Audit a contract for security issues
    pub fn audit_contract(&mut self, source: &SourceFile) -> Vec<SecurityIssue> {
        self.issues.clear();
        
        for rule in &self.rules {
            let mut rule_issues = (rule.check)(source);
            self.issues.append(&mut rule_issues);
        }
        
        self.issues.clone()
    }

    fn load_default_rules(&mut self) {
        // Reentrancy check
        self.rules.push(SecurityRule {
            name: "Reentrancy Check".to_string(),
            vulnerability_type: VulnerabilityType::ReentrancyAttack,
            check: Box::new(|source| {
                let mut issues = Vec::new();
                
                for item in &source.items {
                    if let Item::Contract(contract) = item {
                        for function in &contract.functions {
                            if Self::has_external_call(&function.body) && 
                               Self::modifies_state_after_call(&function.body) {
                                issues.push(SecurityIssue {
                                    id: format!("reentrancy_{}", function.name.name),
                                    vulnerability_type: VulnerabilityType::ReentrancyAttack,
                                    severity: Severity::High,
                                    title: "Potential Reentrancy Vulnerability".to_string(),
                                    description: format!("Function '{}' makes external calls before state changes", function.name.name),
                                    location: function.location.clone(),
                                    recommendation: "Use the checks-effects-interactions pattern or reentrancy guards".to_string(),
                                    cwe_id: Some(841),
                                });
                            }
                        }
                    }
                }
                
                issues
            }),
        });

        // Integer overflow check
        self.rules.push(SecurityRule {
            name: "Integer Overflow Check".to_string(),
            vulnerability_type: VulnerabilityType::IntegerOverflow,
            check: Box::new(|source| {
                let mut issues = Vec::new();
                
                for item in &source.items {
                    if let Item::Contract(contract) = item {
                        for function in &contract.functions {
                            let overflows = Self::find_potential_overflows(&function.body);
                            for location in overflows {
                                issues.push(SecurityIssue {
                                    id: format!("overflow_{}_{}", function.name.name, issues.len()),
                                    vulnerability_type: VulnerabilityType::IntegerOverflow,
                                    severity: Severity::Medium,
                                    title: "Potential Integer Overflow".to_string(),
                                    description: "Arithmetic operation may overflow".to_string(),
                                    location,
                                    recommendation: "Use safe math libraries or explicit overflow checks".to_string(),
                                    cwe_id: Some(190),
                                });
                            }
                        }
                    }
                }
                
                issues
            }),
        });

        // Access control check
        self.rules.push(SecurityRule {
            name: "Access Control Check".to_string(),
            vulnerability_type: VulnerabilityType::UnauthorizedAccess,
            check: Box::new(|source| {
                let mut issues = Vec::new();
                
                for item in &source.items {
                    if let Item::Contract(contract) = item {
                        for function in &contract.functions {
                            if function.visibility == Visibility::Public && 
                               Self::modifies_critical_state(&function.body) &&
                               !Self::has_access_control(&function.body) {
                                issues.push(SecurityIssue {
                                    id: format!("access_control_{}", function.name.name),
                                    vulnerability_type: VulnerabilityType::UnauthorizedAccess,
                                    severity: Severity::High,
                                    title: "Missing Access Control".to_string(),
                                    description: format!("Public function '{}' modifies critical state without access control", function.name.name),
                                    location: function.location.clone(),
                                    recommendation: "Add proper access control modifiers or checks".to_string(),
                                    cwe_id: Some(284),
                                });
                            }
                        }
                    }
                }
                
                issues
            }),
        });
    }

    fn has_external_call(block: &Block) -> bool {
        // Simplified check for external calls
        for stmt in &block.statements {
            if let Statement::Expression(Expression::Call(_)) = stmt {
                return true;
            }
        }
        false
    }

    fn modifies_state_after_call(block: &Block) -> bool {
        // Simplified check for state modifications after calls
        let mut found_call = false;
        for stmt in &block.statements {
            if let Statement::Expression(Expression::Call(_)) = stmt {
                found_call = true;
            } else if found_call && Self::is_state_modification(stmt) {
                return true;
            }
        }
        false
    }

    fn is_state_modification(stmt: &Statement) -> bool {
        matches!(stmt, Statement::Expression(Expression::Assignment(_)))
    }

    fn find_potential_overflows(block: &Block) -> Vec<crate::error::SourceLocation> {
        let mut locations = Vec::new();
        
        for stmt in &block.statements {
            if let Statement::Expression(Expression::Binary(binary)) = stmt {
                if matches!(binary.operator, BinaryOperator::Add | BinaryOperator::Multiply) {
                    locations.push(binary.location.clone());
                }
            }
        }
        
        locations
    }

    fn modifies_critical_state(block: &Block) -> bool {
        // Check if function modifies critical state (balances, ownership, etc.)
        for stmt in &block.statements {
            if let Statement::Expression(Expression::Assignment(assignment)) = stmt {
                if let Expression::FieldAccess(field_access) = &*assignment.target {
                    let field_name = &field_access.field.name;
                    if field_name.contains("balance") || 
                       field_name.contains("owner") || 
                       field_name.contains("admin") {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn has_access_control(block: &Block) -> bool {
        // Check for access control patterns
        for stmt in &block.statements {
            if let Statement::Require(require_stmt) = stmt {
                // Look for common access control patterns
                return true;
            }
        }
        false
    }
}

/// Contract invariant checker
pub struct InvariantChecker {
    invariants: Vec<ContractInvariant>,
}

#[derive(Debug, Clone)]
pub struct ContractInvariant {
    pub name: String,
    pub description: String,
    pub formula: LogicalFormula,
    pub applies_to: Vec<String>, // Function names this invariant applies to
}

impl InvariantChecker {
    pub fn new() -> Self {
        Self {
            invariants: Vec::new(),
        }
    }

    pub fn add_invariant(&mut self, invariant: ContractInvariant) {
        self.invariants.push(invariant);
    }

    /// Check if all invariants hold for a contract
    pub fn check_invariants(&self, contract: &Contract) -> Vec<InvariantViolation> {
        let mut violations = Vec::new();
        
        for invariant in &self.invariants {
            for function in &contract.functions {
                if invariant.applies_to.is_empty() || 
                   invariant.applies_to.contains(&function.name.name) {
                    
                    if let Some(violation) = self.check_function_invariant(function, invariant) {
                        violations.push(violation);
                    }
                }
            }
        }
        
        violations
    }

    fn check_function_invariant(&self, function: &Function, invariant: &ContractInvariant) -> Option<InvariantViolation> {
        // Simplified invariant checking
        // In practice, this would use symbolic execution or model checking
        
        // For now, just report potential violations
        Some(InvariantViolation {
            invariant_name: invariant.name.clone(),
            function_name: function.name.name.clone(),
            description: format!("Potential violation of invariant '{}' in function '{}'", 
                               invariant.name, function.name.name),
            location: function.location.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct InvariantViolation {
    pub invariant_name: String,
    pub function_name: String,
    pub description: String,
    pub location: crate::error::SourceLocation,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_engine() {
        let mut engine = VerificationEngine::new();
        
        let condition = VerificationCondition {
            id: "test_condition".to_string(),
            description: "Test condition".to_string(),
            precondition: LogicalFormula::Bool(true),
            postcondition: LogicalFormula::Bool(true),
            location: crate::error::SourceLocation::unknown(),
        };
        
        engine.add_condition(condition);
        let results = engine.verify_all();
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "test_condition");
    }

    #[test]
    fn test_security_auditor() {
        let mut auditor = SecurityAuditor::new();
        
        // Create a simple source file for testing
        let source = SourceFile {
            items: vec![],
            location: crate::error::SourceLocation::unknown(),
        };
        
        let issues = auditor.audit_contract(&source);
        // Should not find issues in empty contract
        assert_eq!(issues.len(), 0);
    }

    #[test]
    fn test_invariant_checker() {
        let mut checker = InvariantChecker::new();
        
        let invariant = ContractInvariant {
            name: "Balance Non-negative".to_string(),
            description: "Balance should always be non-negative".to_string(),
            formula: LogicalFormula::Comparison {
                left: Box::new(LogicalFormula::Variable("balance".to_string())),
                op: ComparisonOp::GreaterEqual,
                right: Box::new(LogicalFormula::Arithmetic(ArithmeticExpr::Constant(0))),
            },
            applies_to: vec![],
        };
        
        checker.add_invariant(invariant);
        assert_eq!(checker.invariants.len(), 1);
    }
}
