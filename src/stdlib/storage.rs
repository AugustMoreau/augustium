//! Storage Module
//!
//! This module provides utilities for persistent storage in smart contracts,
//! including key-value storage, structured storage, and storage optimization.

use crate::error::Result;
use crate::stdlib::core_types::{U256, Address, AugustiumType};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Storage key type for contract storage
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StorageKey {
    /// The storage slot (32 bytes)
    pub slot: U256,
}

impl StorageKey {
    /// Create a new storage key from a U256 slot
    pub fn new(slot: U256) -> Self {
        Self { slot }
    }
    
    /// Create a storage key from a string (hashed)
    pub fn from_string(key: &str) -> Self {
        // Simple hash function for demonstration
        let hash = key.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        Self {
            slot: U256::new(hash),
        }
    }
    
    /// Create a storage key from an address and offset
    pub fn from_address_offset(addr: &Address, offset: u64) -> Self {
        // Combine address bytes with offset
        let mut slot_value = 0u64;
        for (i, &byte) in addr.as_bytes().iter().take(8).enumerate() {
            slot_value |= (byte as u64) << (i * 8);
        }
        slot_value = slot_value.wrapping_add(offset);
        
        Self {
            slot: U256::new(slot_value),
        }
    }
}

/// Storage value that can be stored in contract storage
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StorageValue {
    /// 256-bit unsigned integer
    U256(U256),
    /// Boolean value
    Bool(bool),
    /// Address value
    Address(Address),
    /// Byte array
    Bytes(Vec<u8>),
    /// String value
    String(String),
}

impl StorageValue {
    /// Convert to U256 if possible
    pub fn as_u256(&self) -> Option<U256> {
        match self {
            StorageValue::U256(val) => Some(*val),
            StorageValue::Bool(true) => Some(U256::new(1)),
            StorageValue::Bool(false) => Some(U256::new(0)),
            _ => None,
        }
    }
    
    /// Convert to boolean if possible
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            StorageValue::Bool(val) => Some(*val),
            StorageValue::U256(val) => Some(val.as_u64() != 0),
            _ => None,
        }
    }
    
    /// Convert to address if possible
    pub fn as_address(&self) -> Option<Address> {
        match self {
            StorageValue::Address(addr) => Some(*addr),
            _ => None,
        }
    }
    
    /// Convert to bytes
    pub fn as_bytes(&self) -> Vec<u8> {
        match self {
            StorageValue::U256(val) => val.to_bytes(),
            StorageValue::Bool(val) => vec![if *val { 1 } else { 0 }],
            StorageValue::Address(addr) => addr.as_bytes().to_vec(),
            StorageValue::Bytes(bytes) => bytes.clone(),
            StorageValue::String(s) => s.as_bytes().to_vec(),
        }
    }
}

/// Storage interface for contract storage operations
pub trait Storage {
    /// Store a value at the given key
    fn store(&mut self, key: StorageKey, value: StorageValue) -> Result<()>;
    
    /// Load a value from the given key
    fn load(&self, key: &StorageKey) -> Result<Option<StorageValue>>;
    
    /// Check if a key exists in storage
    fn exists(&self, key: &StorageKey) -> bool;
    
    /// Remove a value from storage
    fn remove(&mut self, key: &StorageKey) -> Result<Option<StorageValue>>;
    
    /// Get all keys in storage
    fn keys(&self) -> Vec<StorageKey>;
    
    /// Clear all storage
    fn clear(&mut self) -> Result<()>;
}

/// In-memory storage implementation for testing
#[derive(Debug, Clone, Default)]
pub struct MemoryStorage {
    data: HashMap<StorageKey, StorageValue>,
}

impl MemoryStorage {
    /// Create a new empty memory storage
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
    
    /// Get the number of stored items
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Storage for MemoryStorage {
    fn store(&mut self, key: StorageKey, value: StorageValue) -> Result<()> {
        self.data.insert(key, value);
        Ok(())
    }
    
    fn load(&self, key: &StorageKey) -> Result<Option<StorageValue>> {
        Ok(self.data.get(key).cloned())
    }
    
    fn exists(&self, key: &StorageKey) -> bool {
        self.data.contains_key(key)
    }
    
    fn remove(&mut self, key: &StorageKey) -> Result<Option<StorageValue>> {
        Ok(self.data.remove(key))
    }
    
    fn keys(&self) -> Vec<StorageKey> {
        self.data.keys().cloned().collect()
    }
    
    fn clear(&mut self) -> Result<()> {
        self.data.clear();
        Ok(())
    }
}

/// Storage utilities for common patterns
pub struct StorageUtils;

impl StorageUtils {
    /// Store a mapping value (key -> value)
    pub fn store_mapping<S: Storage>(
        storage: &mut S,
        mapping_slot: U256,
        key: &[u8],
        value: StorageValue,
    ) -> Result<()> {
        let storage_key = Self::compute_mapping_key(mapping_slot, key);
        storage.store(storage_key, value)
    }
    
    /// Load a mapping value
    pub fn load_mapping<S: Storage>(
        storage: &S,
        mapping_slot: U256,
        key: &[u8],
    ) -> Result<Option<StorageValue>> {
        let storage_key = Self::compute_mapping_key(mapping_slot, key);
        storage.load(&storage_key)
    }
    
    /// Compute storage key for mapping
    fn compute_mapping_key(mapping_slot: U256, key: &[u8]) -> StorageKey {
        // Simple key derivation: hash(mapping_slot || key)
        let mut combined = mapping_slot.to_bytes();
        combined.extend_from_slice(key);
        
        let hash = combined.iter().fold(0u64, |acc, &b| {
            acc.wrapping_mul(31).wrapping_add(b as u64)
        });
        
        StorageKey::new(U256::new(hash))
    }
    
    /// Store an array element
    pub fn store_array_element<S: Storage>(
        storage: &mut S,
        array_slot: U256,
        index: u64,
        value: StorageValue,
    ) -> Result<()> {
        let storage_key = StorageKey::new(array_slot.checked_add(U256::new(index)).unwrap_or(array_slot));
        storage.store(storage_key, value)
    }
    
    /// Load an array element
    pub fn load_array_element<S: Storage>(
        storage: &S,
        array_slot: U256,
        index: u64,
    ) -> Result<Option<StorageValue>> {
        let storage_key = StorageKey::new(array_slot.checked_add(U256::new(index)).unwrap_or(array_slot));
        storage.load(&storage_key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_storage_key_creation() {
        let key1 = StorageKey::new(U256::new(42));
        let key2 = StorageKey::from_string("test_key");
        let addr = Address::from_hex("0x742d35Cc6634C0532925a3b8D4C9db96590c4C87").unwrap();
        let key3 = StorageKey::from_address_offset(&addr, 10);
        
        assert_eq!(key1.slot, U256::new(42));
        assert_ne!(key2.slot, U256::new(0));
        assert_ne!(key3.slot, U256::new(0));
    }
    
    #[test]
    fn test_storage_value_conversions() {
        let val1 = StorageValue::U256(U256::new(42));
        let val2 = StorageValue::Bool(true);
        let val3 = StorageValue::Address(Address::ZERO);
        
        assert_eq!(val1.as_u256(), Some(U256::new(42)));
        assert_eq!(val2.as_bool(), Some(true));
        assert_eq!(val2.as_u256(), Some(U256::new(1)));
        assert_eq!(val3.as_address(), Some(Address::ZERO));
    }
    
    #[test]
    fn test_memory_storage() {
        let mut storage = MemoryStorage::new();
        let key = StorageKey::new(U256::new(1));
        let value = StorageValue::U256(U256::new(42));
        
        // Test store and load
        storage.store(key.clone(), value.clone()).unwrap();
        let loaded = storage.load(&key).unwrap();
        assert_eq!(loaded, Some(value));
        
        // Test exists
        assert!(storage.exists(&key));
        
        // Test remove
        let removed = storage.remove(&key).unwrap();
        assert_eq!(removed, Some(StorageValue::U256(U256::new(42))));
        assert!(!storage.exists(&key));
    }
    
    #[test]
    fn test_storage_utils_mapping() {
        let mut storage = MemoryStorage::new();
        let mapping_slot = U256::new(0);
        let key = b"user1";
        let value = StorageValue::U256(U256::new(100));
        
        // Store mapping value
        StorageUtils::store_mapping(&mut storage, mapping_slot, key, value.clone()).unwrap();
        
        // Load mapping value
        let loaded = StorageUtils::load_mapping(&storage, mapping_slot, key).unwrap();
        assert_eq!(loaded, Some(value));
    }
    
    #[test]
    fn test_storage_utils_array() {
        let mut storage = MemoryStorage::new();
        let array_slot = U256::new(1);
        let index = 5;
        let value = StorageValue::String("test".to_string());
        
        // Store array element
        StorageUtils::store_array_element(&mut storage, array_slot, index, value.clone()).unwrap();
        
        // Load array element
        let loaded = StorageUtils::load_array_element(&storage, array_slot, index).unwrap();
        assert_eq!(loaded, Some(value));
    }
}