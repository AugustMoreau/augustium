//! Address Module
//!
//! This module provides utilities for handling blockchain addresses,
//! including validation, formatting, and address-related operations.

use crate::error::{Result, VmError, VmErrorKind};
use std::fmt;
use serde::{Serialize, Deserialize};

/// Represents a blockchain address (20 bytes)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Address {
    bytes: [u8; 20],
}

impl Address {
    /// Zero address constant
    pub const ZERO: Address = Address { bytes: [0u8; 20] };
    
    /// Create a new address from 20 bytes
    pub fn new(bytes: [u8; 20]) -> Self {
        Self { bytes }
    }
    
    /// Create an address from a hex string
    pub fn from_hex(hex: &str) -> Result<Self> {
        let hex = hex.trim_start_matches("0x");
        
        if hex.len() != 40 {
            return Err(crate::error::CompilerError::VmError(VmError {
                kind: VmErrorKind::InvalidInput,
                message: format!("Address hex string must be 40 characters, got {}", hex.len()),
            }));
        }
        
        let mut bytes = [0u8; 20];
        for i in 0..20 {
            let byte_str = &hex[i * 2..i * 2 + 2];
            bytes[i] = u8::from_str_radix(byte_str, 16)
                .map_err(|_| crate::error::CompilerError::VmError(VmError {
                    kind: VmErrorKind::InvalidInput,
                    message: format!("Invalid hex character in address: {}", byte_str),
                }))?;
        }
        
        Ok(Self { bytes })
    }
    
    /// Create an address from a byte slice
    pub fn from_slice(slice: &[u8]) -> Result<Self> {
        if slice.len() != 20 {
            return Err(crate::error::CompilerError::VmError(VmError {
                kind: VmErrorKind::InvalidInput,
                message: format!("Address must be 20 bytes, got {}", slice.len()),
            }));
        }
        
        let mut bytes = [0u8; 20];
        bytes.copy_from_slice(slice);
        Ok(Self { bytes })
    }
    
    /// Get the raw bytes of the address
    pub fn as_bytes(&self) -> &[u8; 20] {
        &self.bytes
    }
    
    /// Convert to hex string with 0x prefix
    pub fn to_hex(&self) -> String {
        format!("0x{}", hex::encode(&self.bytes))
    }
    
    /// Convert to hex string without 0x prefix
    pub fn to_hex_no_prefix(&self) -> String {
        hex::encode(&self.bytes)
    }
    
    /// Check if this is the zero address
    pub fn is_zero(&self) -> bool {
        self.bytes == [0u8; 20]
    }
    
    /// Generate a checksum address (EIP-55)
    pub fn to_checksum(&self) -> String {
        let hex = self.to_hex_no_prefix();
        let hash = keccak256(hex.as_bytes());
        
        let mut result = String::with_capacity(42);
        result.push_str("0x");
        
        for (i, c) in hex.chars().enumerate() {
            if c.is_ascii_digit() {
                result.push(c);
            } else {
                // Check if the corresponding hash bit is set
                let hash_byte = hash[i / 2];
                let bit_position = if i % 2 == 0 { 4 } else { 0 };
                let bit_set = (hash_byte >> bit_position) & 0x08 != 0;
                
                if bit_set {
                    result.push(c.to_ascii_uppercase());
                } else {
                    result.push(c.to_ascii_lowercase());
                }
            }
        }
        
        result
    }
    
    /// Validate a checksum address (EIP-55)
    pub fn validate_checksum(address: &str) -> bool {
        if let Ok(addr) = Address::from_hex(address) {
            addr.to_checksum() == address
        } else {
            false
        }
    }
}

/// Simple Keccak-256 implementation for checksum calculation
fn keccak256(input: &[u8]) -> [u8; 32] {
    // Simplified hash function for demonstration
    // In a real implementation, you would use a proper Keccak-256 library
    let mut hash = [0u8; 32];
    let mut state = 0x6a09e667f3bcc908u64;
    
    for &byte in input {
        state = state.wrapping_mul(0x100000001b3).wrapping_add(byte as u64);
    }
    
    for i in 0..4 {
        let chunk = (state >> (i * 8)) as u8;
        for j in 0..8 {
            hash[i * 8 + j] = chunk.wrapping_add(j as u8);
        }
    }
    
    hash
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl From<[u8; 20]> for Address {
    fn from(bytes: [u8; 20]) -> Self {
        Self::new(bytes)
    }
}

impl From<Address> for [u8; 20] {
    fn from(address: Address) -> Self {
        address.bytes
    }
}

/// Address utilities for common operations
pub struct AddressUtils;

impl AddressUtils {
    /// Generate a contract address from deployer address and nonce
    pub fn contract_address(deployer: &Address, nonce: u64) -> Address {
        // Simplified contract address generation
        // Real implementation would use RLP encoding and Keccak-256
        let mut combined = deployer.bytes.to_vec();
        combined.extend_from_slice(&nonce.to_be_bytes());
        
        let hash = keccak256(&combined);
        let mut addr_bytes = [0u8; 20];
        addr_bytes.copy_from_slice(&hash[12..32]);
        
        Address::new(addr_bytes)
    }
    
    /// Generate a CREATE2 address
    pub fn create2_address(
        deployer: &Address,
        salt: &[u8; 32],
        code_hash: &[u8; 32],
    ) -> Address {
        // CREATE2 address generation: keccak256(0xff ++ deployer ++ salt ++ keccak256(code))
        let mut data = Vec::with_capacity(85);
        data.push(0xff);
        data.extend_from_slice(&deployer.bytes);
        data.extend_from_slice(salt);
        data.extend_from_slice(code_hash);
        
        let hash = keccak256(&data);
        let mut addr_bytes = [0u8; 20];
        addr_bytes.copy_from_slice(&hash[12..32]);
        
        Address::new(addr_bytes)
    }
    
    /// Check if an address is a valid Ethereum address format
    pub fn is_valid_format(address: &str) -> bool {
        if !address.starts_with("0x") {
            return false;
        }
        
        let hex_part = &address[2..];
        if hex_part.len() != 40 {
            return false;
        }
        
        hex_part.chars().all(|c| c.is_ascii_hexdigit())
    }
    
    /// Convert an address to a shorter display format
    pub fn to_short_string(address: &Address, prefix_len: usize, suffix_len: usize) -> String {
        let hex = address.to_hex();
        if prefix_len + suffix_len >= hex.len() - 2 {
            return hex;
        }
        
        format!(
            "{}...{}",
            &hex[..2 + prefix_len],
            &hex[hex.len() - suffix_len..]
        )
    }
}

/// Address book for managing known addresses
#[derive(Debug, Clone, Default)]
pub struct AddressBook {
    addresses: std::collections::HashMap<String, Address>,
    names: std::collections::HashMap<Address, String>,
}

impl AddressBook {
    /// Create a new address book
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add an address with a name
    pub fn add(&mut self, name: String, address: Address) {
        self.addresses.insert(name.clone(), address);
        self.names.insert(address, name);
    }
    
    /// Get an address by name
    pub fn get_address(&self, name: &str) -> Option<Address> {
        self.addresses.get(name).copied()
    }
    
    /// Get a name by address
    pub fn get_name(&self, address: &Address) -> Option<&str> {
        self.names.get(address).map(|s| s.as_str())
    }
    
    /// Remove an address by name
    pub fn remove(&mut self, name: &str) -> Option<Address> {
        if let Some(address) = self.addresses.remove(name) {
            self.names.remove(&address);
            Some(address)
        } else {
            None
        }
    }
    
    /// List all names
    pub fn names(&self) -> Vec<&str> {
        self.addresses.keys().map(|s| s.as_str()).collect()
    }
    
    /// List all addresses
    pub fn addresses(&self) -> Vec<Address> {
        self.addresses.values().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_address_creation() {
        let bytes = [1u8; 20];
        let addr = Address::new(bytes);
        assert_eq!(addr.as_bytes(), &bytes);
        assert!(!addr.is_zero());
        
        let zero_addr = Address::ZERO;
        assert!(zero_addr.is_zero());
    }
    
    #[test]
    fn test_address_from_hex() {
        let hex = "0x742d35Cc6634C0532925a3b8D4C9db96590c4C87";
        let addr = Address::from_hex(hex).unwrap();
        assert_eq!(addr.to_hex().to_lowercase(), hex.to_lowercase());
        
        // Test without 0x prefix
        let hex_no_prefix = "742d35Cc6634C0532925a3b8D4C9db96590c4C87";
        let addr2 = Address::from_hex(hex_no_prefix).unwrap();
        assert_eq!(addr, addr2);
        
        // Test invalid hex
        assert!(Address::from_hex("invalid").is_err());
        assert!(Address::from_hex("0x123").is_err()); // Too short
    }
    
    #[test]
    fn test_address_from_slice() {
        let bytes = [42u8; 20];
        let addr = Address::from_slice(&bytes).unwrap();
        assert_eq!(addr.as_bytes(), &bytes);
        
        // Test invalid length
        assert!(Address::from_slice(&[1, 2, 3]).is_err());
    }
    
    #[test]
    fn test_address_display() {
        let addr = Address::from_hex("0x742d35Cc6634C0532925a3b8D4C9db96590c4C87").unwrap();
        let display = format!("{}", addr);
        assert!(display.starts_with("0x"));
        assert_eq!(display.len(), 42);
    }
    
    #[test]
    fn test_address_utils() {
        let deployer = Address::from_hex("0x742d35Cc6634C0532925a3b8D4C9db96590c4C87").unwrap();
        let contract_addr = AddressUtils::contract_address(&deployer, 1);
        assert_ne!(contract_addr, deployer);
        assert!(!contract_addr.is_zero());
        
        let salt = [1u8; 32];
        let code_hash = [2u8; 32];
        let create2_addr = AddressUtils::create2_address(&deployer, &salt, &code_hash);
        assert_ne!(create2_addr, deployer);
        assert_ne!(create2_addr, contract_addr);
        
        // Test format validation
        assert!(AddressUtils::is_valid_format("0x742d35Cc6634C0532925a3b8D4C9db96590c4C87"));
        assert!(!AddressUtils::is_valid_format("742d35Cc6634C0532925a3b8D4C9db96590c4C87"));
        assert!(!AddressUtils::is_valid_format("0x123"));
        
        // Test short string
        let short = AddressUtils::to_short_string(&deployer, 4, 4);
        assert!(short.contains("..."));
    }
    
    #[test]
    fn test_address_book() {
        let mut book = AddressBook::new();
        let addr = Address::from_hex("0x742d35Cc6634C0532925a3b8D4C9db96590c4C87").unwrap();
        
        book.add("test_contract".to_string(), addr);
        
        assert_eq!(book.get_address("test_contract"), Some(addr));
        assert_eq!(book.get_name(&addr), Some("test_contract"));
        
        let removed = book.remove("test_contract");
        assert_eq!(removed, Some(addr));
        assert_eq!(book.get_address("test_contract"), None);
    }
}