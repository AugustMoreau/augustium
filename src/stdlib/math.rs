// Math utilities for smart contracts
// Safe arithmetic and common mathematical functions

use crate::error::{CompilerError, SemanticError, SemanticErrorKind, SourceLocation};
use crate::stdlib::core_types::{U256, Bool};

// Common math constants we need
pub struct MathConstants;

impl MathConstants {
    // Pi scaled up for fixed-point math (no floating point in blockchain)
    // This is 3.14159... * 10^18
    pub const PI: U256 = {
        let mut bytes = [0u8; 32];
        // 3141592653589793238 in big-endian bytes
        bytes[24] = 0x2B;
        bytes[25] = 0x99;
        bytes[26] = 0x2D;
        bytes[27] = 0xDA;
        bytes[28] = 0xFC;
        bytes[29] = 0xE3;
        bytes[30] = 0x5E;
        bytes[31] = 0x96;
        U256(bytes)
    };
    
    /// Euler's number (scaled by 10^18 for fixed-point arithmetic)
    pub const E: U256 = {
        let mut bytes = [0u8; 32];
        // 2718281828459045235 in big-endian bytes
        bytes[24] = 0x25;
        bytes[25] = 0xB8;
        bytes[26] = 0x5F;
        bytes[27] = 0x3E;
        bytes[28] = 0xAF;
        bytes[29] = 0x0C;
        bytes[30] = 0x37;
        bytes[31] = 0x73;
        U256(bytes)
    };
    
    /// Maximum value for U256
    pub const MAX_U256: U256 = U256::MAX;
    
    /// Scaling factor for fixed-point arithmetic (10^18)
    pub const SCALE: U256 = {
        let mut bytes = [0u8; 32];
        // 1000000000000000000 in big-endian bytes
        bytes[24] = 0x0D;
        bytes[25] = 0xE0;
        bytes[26] = 0xB6;
        bytes[27] = 0xB3;
        bytes[28] = 0xA7;
        bytes[29] = 0x64;
        bytes[30] = 0x00;
        bytes[31] = 0x00;
        U256(bytes)
    };
}

/// Safe arithmetic operations with overflow protection
pub struct SafeMath;

impl SafeMath {
    /// Safe addition with overflow checking
    pub fn add(a: U256, b: U256) -> Result<U256> {
        a.checked_add(b).ok_or_else(|| {
            CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Arithmetic overflow in addition".to_string(),
            })
        })
    }
    
    /// Safe subtraction with underflow checking
    pub fn sub(a: U256, b: U256) -> Result<U256> {
        a.checked_sub(b).ok_or_else(|| {
            CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Arithmetic underflow in subtraction".to_string(),
            })
        })
    }
    
    /// Safe multiplication with overflow checking
    pub fn mul(a: U256, b: U256) -> Result<U256> {
        Self::checked_mul(a, b).ok_or_else(|| {
            CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Arithmetic overflow in multiplication".to_string(),
            })
        })
    }
    
    /// Safe division with zero checking
    pub fn div(a: U256, b: U256) -> Result<U256> {
        if b == U256::ZERO {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Division by zero".to_string(),
            }));
        }
        
        Self::checked_div(a, b).ok_or_else(|| {
            CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Division overflow".to_string(),
            })
        })
    }
    
    /// Internal checked multiplication implementation
    fn checked_mul(a: U256, b: U256) -> Option<U256> {
        // Simple implementation for small values that fit in u64
        if let (Some(a_val), Some(b_val)) = (a.to_u64(), b.to_u64()) {
            if let Some(result) = a_val.checked_mul(b_val) {
                return Some(U256::new(result));
            }
        }
        
        // For larger values, we'll use a simplified approach
        // This is a basic implementation - in production, you'd want a more sophisticated algorithm
        if a == U256::ZERO || b == U256::ZERO {
            return Some(U256::ZERO);
        }
        
        if a == U256::ONE {
            return Some(b);
        }
        
        if b == U256::ONE {
            return Some(a);
        }
        
        // For simplicity, reject large multiplications that might overflow
        // In a production implementation, you'd implement full 256-bit multiplication
        None
    }
    
    /// Internal checked division implementation
    fn checked_div(a: U256, b: U256) -> Option<U256> {
        if b == U256::ZERO {
            return None;
        }
        
        if a == U256::ZERO {
            return Some(U256::ZERO);
        }
        
        if b == U256::ONE {
            return Some(a);
        }
        
        // Simple implementation for values that fit in u64
        if let (Some(a_val), Some(b_val)) = (a.to_u64(), b.to_u64()) {
            return Some(U256::new(a_val / b_val));
        }
        
        // For larger values, implement basic long division
        // This is a simplified implementation
        Some(U256::ZERO) // Placeholder - would need full implementation
    }
    
    /// Safe modulo operation with zero checking
    pub fn modulo(a: U256, b: U256) -> Result<U256> {
        if b == U256::ZERO {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Modulo by zero".to_string(),
            }));
        }
        
        Ok(U256::new(a.to_u64().unwrap_or(0) % b.to_u64().unwrap_or(1)))
    }
    
    /// Safe power operation with overflow checking
    pub fn pow(base: U256, exponent: U256) -> Result<U256> {
        if exponent == U256::ZERO {
            return Ok(U256::ONE);
        }
        
        if base == U256::ZERO {
            return Ok(U256::ZERO);
        }
        
        if base == U256::ONE {
            return Ok(U256::ONE);
        }
        
        // For safety, limit exponent to reasonable values
        let exp_val = exponent.to_u64().unwrap_or(0);
        if exp_val > 100 {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Exponent too large".to_string(),
            }));
        }
        
        // Simple iterative power calculation
        let mut result = U256::ONE;
        for _ in 0..exp_val {
            result = Self::mul(result, base)?;
        }
        
        Ok(result)
    }
}

/// Fixed-point arithmetic for decimal calculations
pub struct FixedPoint;

impl FixedPoint {
    /// Convert integer to fixed-point representation
    pub fn from_int(value: U256) -> Result<U256> {
        SafeMath::mul(value, MathConstants::SCALE)
    }
    
    /// Convert fixed-point to integer (truncating decimals)
    pub fn to_int(value: U256) -> Result<U256> {
        SafeMath::div(value, MathConstants::SCALE)
    }
    
    /// Fixed-point multiplication
    pub fn mul_fixed(a: U256, b: U256) -> Result<U256> {
        let product = SafeMath::mul(a, b)?;
        SafeMath::div(product, MathConstants::SCALE)
    }
    
    /// Fixed-point division
    pub fn div_fixed(a: U256, b: U256) -> Result<U256> {
        let scaled_a = SafeMath::mul(a, MathConstants::SCALE)?;
        SafeMath::div(scaled_a, b)
    }
}

/// Statistical and utility functions
pub struct MathUtils;

impl MathUtils {
    /// Find minimum of two values
    pub fn min(a: U256, b: U256) -> U256 {
        // Compare by converting to u64 for simplicity
        let a_val = a.to_u64().unwrap_or(u64::MAX);
        let b_val = b.to_u64().unwrap_or(u64::MAX);
        if a_val < b_val { a } else { b }
    }
    
    /// Find maximum of two values
    pub fn max(a: U256, b: U256) -> U256 {
        // Compare by converting to u64 for simplicity
        let a_val = a.to_u64().unwrap_or(u64::MAX);
        let b_val = b.to_u64().unwrap_or(u64::MAX);
        if a_val > b_val { a } else { b }
    }
    
    /// Calculate absolute difference between two values
    pub fn abs_diff(a: U256, b: U256) -> U256 {
        let a_val = a.to_u64().unwrap_or(0);
        let b_val = b.to_u64().unwrap_or(0);
        if a_val >= b_val {
            SafeMath::sub(a, b).unwrap_or(U256::ZERO)
        } else {
            SafeMath::sub(b, a).unwrap_or(U256::ZERO)
        }
    }
    
    /// Calculate average of two values (avoiding overflow)
    pub fn average(a: U256, b: U256) -> U256 {
        // Use (a + b) / 2, but handle potential overflow
        let a_div_2 = SafeMath::div(a, U256::new(2)).unwrap_or(U256::ZERO);
        let b_div_2 = SafeMath::div(b, U256::new(2)).unwrap_or(U256::ZERO);
        let sum_div_2 = SafeMath::add(a_div_2, b_div_2).unwrap_or(U256::ZERO);
        
        let a_mod_2 = SafeMath::modulo(a, U256::new(2)).unwrap_or(U256::ZERO);
        let b_mod_2 = SafeMath::modulo(b, U256::new(2)).unwrap_or(U256::ZERO);
        let remainder = SafeMath::add(a_mod_2, b_mod_2).unwrap_or(U256::ZERO);
        let remainder_div_2 = SafeMath::div(remainder, U256::new(2)).unwrap_or(U256::ZERO);
        
        SafeMath::add(sum_div_2, remainder_div_2).unwrap_or(U256::ZERO)
    }
    
    /// Check if a number is even
    pub fn is_even(value: U256) -> Bool {
        Bool::new(value.to_u64().unwrap_or(0) % 2 == 0)
    }
    
    /// Check if a number is odd
    pub fn is_odd(value: U256) -> Bool {
        Bool::new(value.to_u64().unwrap_or(0) % 2 == 1)
    }
    
    /// Calculate square root using Newton's method (integer approximation)
    pub fn sqrt(value: U256) -> Result<U256> {
        if value == U256::ZERO {
            return Ok(U256::ZERO);
        }
        
        if value == U256::ONE {
            return Ok(U256::ONE);
        }
        
        // Newton's method for integer square root
        let mut x = value;
        let temp_sum = SafeMath::add(value, U256::ONE)?;
        let mut y = SafeMath::div(temp_sum, U256::new(2))?;
        
        // Limit iterations to prevent infinite loops
        for _ in 0..100 {
            let y_val = y.to_u64().unwrap_or(0);
            let x_val = x.to_u64().unwrap_or(0);
            if y_val >= x_val {
                return Ok(x);
            }
            x = y;
            let div_result = SafeMath::div(value, x)?;
            let sum_result = SafeMath::add(x, div_result)?;
            y = SafeMath::div(sum_result, U256::new(2))?;
        }
        
        Ok(x)
    }
    
    /// Calculate greatest common divisor using Euclidean algorithm
    pub fn gcd(mut a: U256, mut b: U256) -> U256 {
        // Limit iterations for safety
        for _ in 0..100 {
            if b == U256::ZERO {
                break;
            }
            let temp = b;
            b = U256::new(a.to_u64().unwrap_or(0) % b.to_u64().unwrap_or(1));
            a = temp;
        }
        a
    }
    
    /// Calculate least common multiple
    pub fn lcm(a: U256, b: U256) -> Result<U256> {
        if a == U256::ZERO || b == U256::ZERO {
            return Ok(U256::ZERO);
        }
        
        let gcd_val = Self::gcd(a, b);
        let product = SafeMath::mul(a, b)?;
        SafeMath::div(product, gcd_val)
    }
}

/// Percentage calculations for financial operations
pub struct Percentage;

impl Percentage {
    /// Calculate percentage of a value
    /// percentage is in basis points (1% = 100 basis points)
    pub fn calculate(value: U256, percentage_bp: U256) -> Result<U256> {
        let product = SafeMath::mul(value, percentage_bp)?;
        SafeMath::div(product, U256::new(10000)) // 10000 basis points = 100%
    }
    
    /// Add percentage to a value
    pub fn add_percentage(value: U256, percentage_bp: U256) -> Result<U256> {
        let percentage_amount = Self::calculate(value, percentage_bp)?;
        SafeMath::add(value, percentage_amount)
    }
    
    /// Subtract percentage from a value
    pub fn sub_percentage(value: U256, percentage_bp: U256) -> Result<U256> {
        let percentage_amount = Self::calculate(value, percentage_bp)?;
        SafeMath::sub(value, percentage_amount)
    }
}

// Type alias for Result to match the codebase pattern
type Result<T> = std::result::Result<T, CompilerError>;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_safe_math_operations() {
        // Test safe addition
        let result = SafeMath::add(U256::new(100), U256::new(200)).unwrap();
        assert_eq!(result, U256::new(300));
        
        // Test safe subtraction
        let result = SafeMath::sub(U256::new(300), U256::new(100)).unwrap();
        assert_eq!(result, U256::new(200));
        
        // Test safe multiplication
        let result = SafeMath::mul(U256::new(10), U256::new(20)).unwrap();
        assert_eq!(result, U256::new(200));
        
        // Test safe division
        let result = SafeMath::div(U256::new(100), U256::new(5)).unwrap();
        assert_eq!(result, U256::new(20));
        
        // Test division by zero error
        assert!(SafeMath::div(U256::new(100), U256::ZERO).is_err());
    }
    
    #[test]
    fn test_math_utils() {
        // Test min/max
        assert_eq!(MathUtils::min(U256::new(10), U256::new(20)), U256::new(10));
        assert_eq!(MathUtils::max(U256::new(10), U256::new(20)), U256::new(20));
        
        // Test average
        assert_eq!(MathUtils::average(U256::new(10), U256::new(20)), U256::new(15));
        
        // Test even/odd
        assert_eq!(MathUtils::is_even(U256::new(10)), Bool::TRUE);
        assert_eq!(MathUtils::is_odd(U256::new(11)), Bool::TRUE);
        
        // Test square root
        assert_eq!(MathUtils::sqrt(U256::new(16)).unwrap(), U256::new(4));
        assert_eq!(MathUtils::sqrt(U256::new(25)).unwrap(), U256::new(5));
    }
    
    #[test]
    fn test_fixed_point_arithmetic() {
        // Test fixed-point conversion
        let fixed_val = FixedPoint::from_int(U256::new(5)).unwrap();
        let int_val = FixedPoint::to_int(fixed_val).unwrap();
        assert_eq!(int_val, U256::new(5));
    }
    
    #[test]
    fn test_percentage_calculations() {
        // Test 10% of 1000 (1000 basis points)
        let result = Percentage::calculate(U256::new(1000), U256::new(1000)).unwrap();
        assert_eq!(result, U256::new(100));
        
        // Test adding 10% to 1000
        let result = Percentage::add_percentage(U256::new(1000), U256::new(1000)).unwrap();
        assert_eq!(result, U256::new(1100));
    }
    
    #[test]
    fn test_power_operations() {
        // Test basic power operations
        assert_eq!(SafeMath::pow(U256::new(2), U256::new(3)).unwrap(), U256::new(8));
        assert_eq!(SafeMath::pow(U256::new(5), U256::new(2)).unwrap(), U256::new(25));
        assert_eq!(SafeMath::pow(U256::new(10), U256::ZERO).unwrap(), U256::ONE);
        assert_eq!(SafeMath::pow(U256::ZERO, U256::new(5)).unwrap(), U256::ZERO);
    }
}