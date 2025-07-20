//! Core Types Module
//!
//! This module provides implementations for Augustium's primitive types,
//! including integer types, boolean, string, address, and bytes types
//! with their associated methods and operations.

use std::fmt;
use std::str::FromStr;
use std::ops::{Add, Sub, Mul, Div, Rem, AddAssign, SubAssign};
use crate::error::{Result, CompilerError, SemanticError, SemanticErrorKind, SourceLocation};

/// Trait for all Augustium core types
pub trait AugustiumType: Clone + fmt::Debug + fmt::Display {
    /// Get the type name as a string
    fn type_name() -> &'static str;
    
    /// Get the size in bytes
    fn size_bytes() -> usize;
    
    /// Convert to bytes representation
    fn to_bytes(&self) -> Vec<u8>;
    
    /// Create from bytes representation
    fn from_bytes(bytes: &[u8]) -> Result<Self> where Self: Sized;
    
    /// Check if the value is zero/empty
    fn is_zero(&self) -> bool;
}

/// Unsigned 8-bit integer
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct U8(pub u8);

impl U8 {
    pub const MIN: U8 = U8(u8::MIN);
    pub const MAX: U8 = U8(u8::MAX);
    
    pub fn new(value: u8) -> Self {
        U8(value)
    }
    
    pub fn checked_add(self, other: U8) -> Option<U8> {
        self.0.checked_add(other.0).map(U8)
    }
    
    pub fn checked_sub(self, other: U8) -> Option<U8> {
        self.0.checked_sub(other.0).map(U8)
    }
    
    pub fn checked_mul(self, other: U8) -> Option<U8> {
        self.0.checked_mul(other.0).map(U8)
    }
    
    pub fn checked_div(self, other: U8) -> Option<U8> {
        self.0.checked_div(other.0).map(U8)
    }
    
    pub fn saturating_add(self, other: U8) -> U8 {
        U8(self.0.saturating_add(other.0))
    }
    
    pub fn saturating_sub(self, other: U8) -> U8 {
        U8(self.0.saturating_sub(other.0))
    }
}

impl AugustiumType for U8 {
    fn type_name() -> &'static str { "u8" }
    fn size_bytes() -> usize { 1 }
    
    fn to_bytes(&self) -> Vec<u8> {
        vec![self.0]
    }
    
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 1 {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::TypeMismatch {
                    expected: "1 byte".to_string(),
                    found: format!("{} bytes", bytes.len()),
                },
                location: SourceLocation::unknown(),
                message: "Invalid byte length for u8".to_string(),
            }));
        }
        Ok(U8(bytes[0]))
    }
    
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl Default for U8 {
    fn default() -> Self {
        U8(0)
    }
}

impl fmt::Display for U8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for U8 {
    type Err = CompilerError;
    
    fn from_str(s: &str) -> Result<Self> {
        s.parse::<u8>()
            .map(U8)
            .map_err(|_| CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::TypeMismatch {
                    expected: "valid u8".to_string(),
                    found: s.to_string(),
                },
                location: SourceLocation::unknown(),
                message: "Failed to parse u8".to_string(),
            }))
    }
}

/// Unsigned 256-bit integer (primary blockchain type)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub struct U256(pub [u8; 32]);

impl U256 {
    pub const ZERO: U256 = U256([0u8; 32]);
    pub const ONE: U256 = {
        let mut bytes = [0u8; 32];
        bytes[31] = 1;
        U256(bytes)
    };
    pub const MAX: U256 = U256([0xFFu8; 32]);
    
    pub fn new(value: u64) -> Self {
        let mut bytes = [0u8; 32];
        bytes[24..32].copy_from_slice(&value.to_be_bytes());
        U256(bytes)
    }
    
    pub fn from_hex(hex: &str) -> Result<Self> {
        let hex = hex.strip_prefix("0x").unwrap_or(hex);
        if hex.len() > 64 {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::TypeMismatch {
                    expected: "hex string <= 64 chars".to_string(),
                    found: format!("{} chars", hex.len()),
                },
                location: SourceLocation::unknown(),
                message: "Hex string too long for u256".to_string(),
            }));
        }
        
        let mut bytes = [0u8; 32];
        let hex_bytes = hex::decode(format!("{:0>64}", hex))
            .map_err(|_| CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::TypeMismatch {
                    expected: "valid hex string".to_string(),
                    found: hex.to_string(),
                },
                location: SourceLocation::unknown(),
                message: "Invalid hex string".to_string(),
            }))?;
        
        bytes.copy_from_slice(&hex_bytes);
        Ok(U256(bytes))
    }
    
    pub fn to_hex(&self) -> String {
        format!("0x{}", hex::encode(self.0))
    }
    
    /// Convert to u64 (truncates if value is larger)
    pub fn as_u64(&self) -> u64 {
        u64::from_be_bytes([
            self.0[24], self.0[25], self.0[26], self.0[27],
            self.0[28], self.0[29], self.0[30], self.0[31],
        ])
    }
    
    pub fn to_u64(&self) -> Option<u64> {
        // Check if the value fits in u64 (only last 8 bytes should be non-zero)
        if self.0[0..24].iter().any(|&b| b != 0) {
            return None;
        }
        Some(u64::from_be_bytes([
            self.0[24], self.0[25], self.0[26], self.0[27],
            self.0[28], self.0[29], self.0[30], self.0[31],
        ]))
    }
    
    pub fn checked_add(self, other: U256) -> Option<U256> {
        let mut result = [0u8; 32];
        let mut carry = 0u16;
        
        for i in (0..32).rev() {
            let sum = self.0[i] as u16 + other.0[i] as u16 + carry;
            result[i] = sum as u8;
            carry = sum >> 8;
        }
        
        if carry > 0 {
            None // Overflow
        } else {
            Some(U256(result))
        }
    }
    
    pub fn checked_sub(self, other: U256) -> Option<U256> {
        if self < other {
            return None; // Underflow
        }
        
        let mut result = [0u8; 32];
        let mut borrow = 0i16;
        
        for i in (0..32).rev() {
            let diff = self.0[i] as i16 - other.0[i] as i16 - borrow;
            if diff < 0 {
                result[i] = (diff + 256) as u8;
                borrow = 1;
            } else {
                result[i] = diff as u8;
                borrow = 0;
            }
        }
        
        Some(U256(result))
    }
    
    pub fn zero() -> Self {
        U256([0u8; 32])
    }
    
    pub fn one() -> Self {
        let mut bytes = [0u8; 32];
        bytes[31] = 1;
        U256(bytes)
    }
    
    /// Calculate power (self^exponent)
    pub fn pow(self, exponent: U256) -> U256 {
        if exponent.is_zero() {
            return U256::one();
        }
        
        if self.is_zero() {
            return U256::zero();
        }
        
        // Simple implementation using repeated multiplication
        // For large exponents, this could be optimized with binary exponentiation
        let exp_u64 = exponent.as_u64();
        let base_u64 = self.as_u64();
        
        if exp_u64 > 64 {
            // For very large exponents, return max value to prevent overflow
            return U256::MAX;
        }
        
        let mut result = 1u64;
        for _ in 0..exp_u64 {
            if let Some(new_result) = result.checked_mul(base_u64) {
                result = new_result;
            } else {
                // Overflow occurred, return max value
                return U256::MAX;
            }
        }
        
        U256::new(result)
    }
}

// Arithmetic trait implementations for U256
impl Add for U256 {
    type Output = U256;
    
    fn add(self, other: U256) -> U256 {
        // Simple implementation - in practice would need proper big integer arithmetic
        let a = self.as_u64();
        let b = other.as_u64();
        U256::new(a.wrapping_add(b))
    }
}

impl Sub for U256 {
    type Output = U256;
    
    fn sub(self, other: U256) -> U256 {
        let a = self.as_u64();
        let b = other.as_u64();
        U256::new(a.wrapping_sub(b))
    }
}

impl Mul for U256 {
    type Output = U256;
    
    fn mul(self, other: U256) -> U256 {
        let a = self.as_u64();
        let b = other.as_u64();
        U256::new(a.wrapping_mul(b))
    }
}

impl Div for U256 {
    type Output = U256;
    
    fn div(self, other: U256) -> U256 {
        let a = self.as_u64();
        let b = other.as_u64();
        if b == 0 {
            U256::zero()
        } else {
            U256::new(a / b)
        }
    }
}

impl Rem for U256 {
    type Output = U256;
    
    fn rem(self, other: U256) -> U256 {
        let a = self.as_u64();
        let b = other.as_u64();
        if b == 0 {
            U256::zero()
        } else {
            U256::new(a % b)
        }
    }
}

impl AddAssign for U256 {
    fn add_assign(&mut self, other: U256) {
        *self = *self + other;
    }
}

impl SubAssign for U256 {
    fn sub_assign(&mut self, other: U256) {
        *self = *self - other;
    }
}

impl AugustiumType for U256 {
    fn type_name() -> &'static str { "u256" }
    fn size_bytes() -> usize { 32 }
    
    fn to_bytes(&self) -> Vec<u8> {
        self.0.to_vec()
    }
    
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 32 {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::TypeMismatch {
                    expected: "32 bytes".to_string(),
                    found: format!("{} bytes", bytes.len()),
                },
                location: SourceLocation::unknown(),
                message: "Invalid byte length for u256".to_string(),
            }));
        }
        let mut array = [0u8; 32];
        array.copy_from_slice(bytes);
        Ok(U256(array))
    }
    
    fn is_zero(&self) -> bool {
        self.0.iter().all(|&b| b == 0)
    }
}

impl Default for U256 {
    fn default() -> Self {
        U256::ZERO
    }
}

impl fmt::Display for U256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display as decimal if it fits in u64, otherwise as hex
        if let Some(value) = self.to_u64() {
            write!(f, "{}", value)
        } else {
            write!(f, "{}", self.to_hex())
        }
    }
}

/// Boolean type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bool(pub bool);

impl Bool {
    pub const TRUE: Bool = Bool(true);
    pub const FALSE: Bool = Bool(false);
    
    pub fn new(value: bool) -> Self {
        Bool(value)
    }
}

impl AugustiumType for Bool {
    fn type_name() -> &'static str { "bool" }
    fn size_bytes() -> usize { 1 }
    
    fn to_bytes(&self) -> Vec<u8> {
        vec![if self.0 { 1 } else { 0 }]
    }
    
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 1 {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::TypeMismatch {
                    expected: "1 byte".to_string(),
                    found: format!("{} bytes", bytes.len()),
                },
                location: SourceLocation::unknown(),
                message: "Invalid byte length for bool".to_string(),
            }));
        }
        Ok(Bool(bytes[0] != 0))
    }
    
    fn is_zero(&self) -> bool {
        !self.0
    }
}

impl fmt::Display for Bool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Blockchain address type (20 bytes)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Address(pub [u8; 20]);

impl Address {
    pub const ZERO: Address = Address([0u8; 20]);
    
    pub fn new(bytes: [u8; 20]) -> Self {
        Address(bytes)
    }
    
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
    
    pub fn from_hex(hex: &str) -> Result<Self> {
        let hex = hex.strip_prefix("0x").unwrap_or(hex);
        if hex.len() != 40 {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::TypeMismatch {
                    expected: "40 character hex string".to_string(),
                    found: format!("{} characters", hex.len()),
                },
                location: SourceLocation::unknown(),
                message: "Invalid hex length for address".to_string(),
            }));
        }
        
        let bytes = hex::decode(hex)
            .map_err(|_| CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::TypeMismatch {
                    expected: "valid hex string".to_string(),
                    found: hex.to_string(),
                },
                location: SourceLocation::unknown(),
                message: "Invalid hex string for address".to_string(),
            }))?;
        
        let mut array = [0u8; 20];
        array.copy_from_slice(&bytes);
        Ok(Address(array))
    }
    
    pub fn to_hex(&self) -> String {
        format!("0x{}", hex::encode(self.0))
    }
    
    pub fn checksum(&self) -> String {
        // Implement EIP-55 checksum encoding
        let hex = hex::encode(self.0);
        let hash = keccak256(hex.as_bytes());
        
        let mut result = String::with_capacity(42);
        result.push_str("0x");
        
        for (i, c) in hex.chars().enumerate() {
            if c.is_ascii_digit() {
                result.push(c);
            } else if hash[i / 2] & (if i % 2 == 0 { 0x80 } else { 0x08 }) != 0 {
                result.push(c.to_ascii_uppercase());
            } else {
                result.push(c.to_ascii_lowercase());
            }
        }
        
        result
    }
}

impl AugustiumType for Address {
    fn type_name() -> &'static str { "address" }
    fn size_bytes() -> usize { 20 }
    
    fn to_bytes(&self) -> Vec<u8> {
        self.0.to_vec()
    }
    
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 20 {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::TypeMismatch {
                    expected: "20 bytes".to_string(),
                    found: format!("{} bytes", bytes.len()),
                },
                location: SourceLocation::unknown(),
                message: "Invalid byte length for address".to_string(),
            }));
        }
        let mut array = [0u8; 20];
        array.copy_from_slice(bytes);
        Ok(Address(array))
    }
    
    fn is_zero(&self) -> bool {
        self.0.iter().all(|&b| b == 0)
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.checksum())
    }
}

/// Simple keccak256 implementation (placeholder)
/// In a real implementation, this would use a proper crypto library
fn keccak256(data: &[u8]) -> [u8; 32] {
    // This is a placeholder implementation
    // In production, use a proper keccak256 implementation
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    let hash = hasher.finish();
    
    let mut result = [0u8; 32];
    result[24..32].copy_from_slice(&hash.to_be_bytes());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u8_operations() {
        let a = U8::new(100);
        let b = U8::new(50);
        
        assert_eq!(a.checked_add(b), Some(U8::new(150)));
        assert_eq!(a.checked_sub(b), Some(U8::new(50)));
        assert_eq!(a.checked_mul(U8::new(2)), Some(U8::new(200)));
        assert_eq!(a.checked_div(U8::new(2)), Some(U8::new(50)));
        
        // Test overflow
        assert_eq!(U8::new(200).checked_add(U8::new(100)), None);
    }
    
    #[test]
    fn test_u256_operations() {
        let a = U256::new(1000);
        let b = U256::new(500);
        
        assert_eq!(a.checked_add(b), Some(U256::new(1500)));
        assert_eq!(a.checked_sub(b), Some(U256::new(500)));
        
        assert!(a.to_u64().is_some());
        assert_eq!(a.to_u64().unwrap(), 1000);
    }
    
    #[test]
    fn test_address_operations() {
        let addr = Address::from_hex("0x742d35Cc6634C0532925a3b8D4C9db4C4C4b3f8e").unwrap();
        assert!(!addr.is_zero());
        assert_eq!(addr.to_hex().len(), 42); // 0x + 40 chars
        
        let zero_addr = Address::ZERO;
        assert!(zero_addr.is_zero());
    }
    
    #[test]
    fn test_bool_operations() {
        let t = Bool::TRUE;
        let f = Bool::FALSE;
        
        assert!(!t.is_zero());
        assert!(f.is_zero());
        assert_eq!(t.to_string(), "true");
        assert_eq!(f.to_string(), "false");
    }
}