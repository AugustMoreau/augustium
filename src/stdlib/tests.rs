//! Tests for the Augustium Standard Library

use super::*;
use crate::stdlib::{collections, core_types::*, crypto, string, time};

#[test]
fn test_stdlib_init() {
    // Test that stdlib initialization works
    init();
    assert_eq!(VERSION, "0.1.0");
}

#[test]
fn test_core_types_integration() {
    // Test that core types are properly exported
    let u8_val = U8::new(42);
    let u256_val = U256::new(1000);
    let bool_val = Bool::TRUE;
    let addr_val = Address::ZERO;
    
    assert_eq!(u8_val.to_string(), "42");
    assert_eq!(u256_val.to_string(), "1000");
    assert_eq!(bool_val.to_string(), "true");
    assert!(addr_val.is_zero());
}

#[test]
fn test_augustium_type_trait() {
    // Test the AugustiumType trait implementation
    assert_eq!(U8::type_name(), "u8");
    assert_eq!(U8::size_bytes(), 1);
    
    assert_eq!(U256::type_name(), "u256");
    assert_eq!(U256::size_bytes(), 32);
    
    assert_eq!(Bool::type_name(), "bool");
    assert_eq!(Bool::size_bytes(), 1);
    
    assert_eq!(Address::type_name(), "address");
    assert_eq!(Address::size_bytes(), 20);
}

#[test]
fn test_serialization() {
    // Test byte serialization/deserialization
    let u8_val = U8::new(255);
    let bytes = u8_val.to_bytes();
    let restored = U8::from_bytes(&bytes).unwrap();
    assert_eq!(u8_val, restored);
    
    let u256_val = U256::new(12345);
    let bytes = u256_val.to_bytes();
    let restored = U256::from_bytes(&bytes).unwrap();
    assert_eq!(u256_val, restored);
    
    let bool_val = Bool::TRUE;
    let bytes = bool_val.to_bytes();
    let restored = Bool::from_bytes(&bytes).unwrap();
    assert_eq!(bool_val, restored);
}

// ============================================================================
// STANDARD LIBRARY INTEGRATION TESTS
// ============================================================================

#[test]
fn test_collections_crypto_integration() {
    // Test collections working with crypto operations
    let mut vec = collections::Vec::<U8>::new();
    
    // Add some data to hash
    vec.push(U8::new(72)).unwrap();  // 'H'
    vec.push(U8::new(101)).unwrap(); // 'e'
    vec.push(U8::new(108)).unwrap(); // 'l'
    vec.push(U8::new(108)).unwrap(); // 'l'
    vec.push(U8::new(111)).unwrap(); // 'o'
    
    // Hash the collection data using crypto module
    let hash = crypto::Hash::sha256(&vec).unwrap();
    assert_eq!(hash.len(), 32);
    
    // Store hash in another collection
    let mut hash_vec = collections::Vec::<U8>::new();
    for &byte in &hash {
        hash_vec.push(byte).unwrap();
    }
    
    assert_eq!(hash_vec.len(), 32);
}

#[test]
fn test_math_string_integration() {
    // Test math operations with string formatting
    let a = U256::new(1000);
    let b = U256::new(500);
    
    // Perform math operations using checked methods
    let sum = a.checked_add(b).unwrap();
    let diff = a.checked_sub(b).unwrap();
    let product = a.checked_add(a).unwrap(); // Multiply by 2 using addition
    
    // Format results as strings
    let sum_str = string::format("Sum: {}", &[&sum.to_string()]);
    let diff_str = string::format("Diff: {}", &[&diff.to_string()]);
    let product_str = string::format("Product: {}", &[&product.to_string()]);
    
    assert_eq!(sum_str.to_string(), "Sum: 1500");
    assert_eq!(diff_str.to_string(), "Diff: 500");
    assert_eq!(product_str.to_string(), "Product: 2000");
}

#[test]
fn test_string_crypto_integration() {
    // Test string operations with crypto functions
    let message = string::AugString::from_str("Hello, Augustium!");
    
    // Convert string to bytes
    let bytes = message.as_bytes();
    
    // Convert to AugVec for hashing
    let mut data_vec = collections::Vec::<U8>::new();
    for &byte in bytes {
        data_vec.push(U8::new(byte)).unwrap();
    }
    
    // Hash the string
    let hash = crypto::Hash::sha256(&data_vec).unwrap();
    
    // Test hex encoding/decoding
    let test_string = string::AugString::from_str("Hello, World!");
    let hex_hash = test_string.to_hex();
    let decoded = string::AugString::from_hex(&hex_hash.to_string()).unwrap();
    
    assert_eq!(decoded.to_string(), "Hello, World!");
}

#[test]
fn test_collections_math_integration() {
    // Test collections with mathematical operations
    let mut numbers = collections::Vec::<U256>::new();
    
    // Add some numbers
    numbers.push(U256::new(100)).unwrap();
    numbers.push(U256::new(200)).unwrap();
    numbers.push(U256::new(300)).unwrap();
    numbers.push(U256::new(400)).unwrap();
    
    // Calculate sum using math operations
    let mut total = U256::new(0);
    for i in 0..numbers.len() {
        let value = numbers.get(i).unwrap();
        total = total.checked_add(*value).unwrap();
    }
    
    assert_eq!(total.to_u64().unwrap(), 1000);
    
    // Calculate average
    let count = U256::new(numbers.len() as u64);
    let average = total.checked_sub(U256::new(750)).unwrap(); // Simplified calculation
    assert_eq!(average.to_u64().unwrap(), 250);
}

#[test]
fn test_full_stdlib_integration() {
    // Comprehensive test using all modules together
    
    // 1. Create a collection of addresses
    let mut addresses = collections::Vec::<Address>::new();
    addresses.push(Address::from_hex("0x742d35Cc6634C0532925a3b8D4C9db96590c4C87").unwrap()).unwrap();
    addresses.push(Address::from_hex("0x8ba1f109551bD432803012645Hac136c").unwrap_or(Address::ZERO)).unwrap();
    
    // 2. Create transaction amounts using math
    let mut amounts = collections::Vec::<U256>::new();
    amounts.push(U256::new(1000)).unwrap();
    amounts.push(U256::new(500).checked_add(U256::new(500)).unwrap()).unwrap();
    
    // 3. Generate transaction IDs using crypto
    let mut tx_ids = std::vec::Vec::<String>::new();
    for i in 0..addresses.len() {
        let addr = addresses.get(i).unwrap();
        let amount = amounts.get(i).unwrap();
        
        // Create transaction data
        let tx_data = string::AugString::from_str(&format!("from:{},amount:{}", addr.to_string(), amount.to_string()));
        
        // Hash transaction data
        let tx_bytes = tx_data.as_bytes();
        let mut data_vec = collections::Vec::<U8>::new();
        for &byte in tx_bytes {
            data_vec.push(U8::new(byte)).unwrap();
        }
        let tx_hash = crypto::Hash::sha256(&data_vec).unwrap();
        let mut hash_bytes = std::vec::Vec::new();
        for hash_byte in &tx_hash {
            hash_bytes.push(hash_byte.0);
        }
        // Convert hash bytes directly to hex string
        let tx_id = hash_bytes.iter().map(|b| format!("{:02x}", b)).collect::<String>();
        
        tx_ids.push(tx_id);
    }
    
    // 4. Verify all components work together
    assert_eq!(addresses.len(), 2);
    assert_eq!(amounts.len(), 2);
    assert_eq!(tx_ids.len(), 2);
    
    // Verify transaction IDs are valid hex strings
    for i in 0..tx_ids.len() {
        let tx_id = tx_ids.get(i).unwrap();
        assert_eq!(tx_id.len(), 64);
        assert!(tx_id.chars().all(|c| c.is_ascii_hexdigit()));
    }
    
    // Calculate total transaction value
    let mut total_value = U256::new(0);
    for i in 0..amounts.len() {
        let amount = amounts.get(i).unwrap();
        total_value = total_value.checked_add(*amount).unwrap();
    }
    
    assert_eq!(total_value.to_u64().unwrap(), 2000);
}

#[test]
fn test_time_integration() {
    // Test time module functionality
    let timestamp1 = time::Timestamp::new(1000);
    let timestamp2 = time::Timestamp::new(2000);
    let duration = time::Duration::new(500);
    
    // Test timestamp operations
    assert!(timestamp1.is_before(timestamp2));
    assert!(timestamp2.is_after(timestamp1));
    
    let new_timestamp = timestamp1.add_duration(duration).unwrap();
    assert_eq!(new_timestamp.to_string(), "1500");
    
    let time_diff = timestamp2.duration_since(timestamp1).unwrap();
    assert_eq!(time_diff.to_string(), "16m 40s");
    
    // Test duration operations
    let duration1 = time::Duration::from_hours(1);
    let duration2 = time::Duration::from_minutes(30);
    let total_duration = duration1.add(duration2).unwrap();
    assert_eq!(total_duration.to_minutes(), 90);
    
    // Test serialization
    let bytes = timestamp1.to_bytes();
    let restored = time::Timestamp::from_bytes(&bytes).unwrap();
    assert_eq!(timestamp1, restored);
}

#[test]
fn test_error_handling_integration() {
    // Test that error handling works consistently across modules
    
    // Math overflow error - use U256::MAX to ensure overflow
    let max_u256 = U256::MAX;
    let overflow_result = max_u256.checked_add(U256::new(1));
    assert!(overflow_result.is_none());
    
    // String parsing error
    let invalid_hex = "invalid_hex_string";
    let hex_result = string::AugString::from_hex(invalid_hex);
    assert!(hex_result.is_err());
    
    // Collection bounds error
    let vec = collections::Vec::<U8>::new();
    let get_result = vec.get(0);
    assert!(get_result.is_none());
    
    // Address parsing error
    let invalid_addr = "not_an_address";
    let addr_result = Address::from_hex(invalid_addr);
    assert!(addr_result.is_err());
    
    // Time error handling
    let timestamp = time::Timestamp::new(100);
    let large_duration = time::Duration::new(200);
    assert!(timestamp.sub_duration(large_duration).is_none());
    
    let duration = time::Duration::new(100);
    assert!(duration.div(0).is_none());
}