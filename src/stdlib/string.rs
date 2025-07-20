// String utilities for text processing
// Basic string operations that smart contracts might need

use crate::stdlib::core_types::U256;
use crate::stdlib::math::SafeMath;
use std::vec::Vec;

// Our string type (UTF-8 bytes under the hood)
#[derive(Debug, Clone, PartialEq)]
pub struct AugString {
    data: Vec<u8>,
}

impl AugString {
    /// Create new empty string
    pub fn new() -> Self {
        AugString { data: Vec::new() }
    }

    /// Create string from bytes
    pub fn from_bytes(bytes: &[u8]) -> std::result::Result<Self, &'static str> {
        // Validate UTF-8
        if !is_valid_utf8(bytes) {
            return Err("Invalid UTF-8 sequence");
        }

        Ok(AugString {
            data: bytes.to_vec()
        })
    }

    /// Create string from Rust string slice
    pub fn from_str(s: &str) -> Self {
        AugString {
            data: s.as_bytes().to_vec()
        }
    }

    /// Get string length in bytes
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if string is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Append string
    pub fn push_str(&mut self, s: &str) {
        self.data.extend_from_slice(s.as_bytes());
    }

    /// Append character
    pub fn push(&mut self, ch: char) {
        let mut buffer = [0u8; 4];
        let bytes = ch.encode_utf8(&mut buffer).as_bytes();
        self.data.extend_from_slice(bytes);
    }

    /// Get substring
    pub fn substring(&self, start: usize, end: usize) -> std::result::Result<AugString, &'static str> {
        if start > end || end > self.data.len() {
            return Err("Invalid substring range");
        }

        AugString::from_bytes(&self.data[start..end])
    }

    /// Find substring
    pub fn find(&self, pattern: &str) -> Option<usize> {
        let pattern_bytes = pattern.as_bytes();
        let data = &self.data;

        if pattern_bytes.is_empty() {
            return Some(0);
        }

        if pattern_bytes.len() > data.len() {
            return None;
        }

        for i in 0..=(data.len() - pattern_bytes.len()) {
            if &data[i..i + pattern_bytes.len()] == pattern_bytes {
                return Some(i);
            }
        }

        None
    }

    /// Replace substring
    pub fn replace(&self, from: &str, to: &str) -> AugString {
        let mut result = AugString::new();
        let mut last_end = 0;

        while let Some(start) = self.substring(last_end, self.len())
            .unwrap()
            .find(from) {
            let actual_start = last_end + start;

            // Add text before match
            if let Ok(before) = self.substring(last_end, actual_start) {
                result.push_str(&before.to_string());
            }

            // Add replacement
            result.push_str(to);

            last_end = actual_start + from.len();
        }

        // Add remaining text
        if let Ok(remaining) = self.substring(last_end, self.len()) {
            result.push_str(&remaining.to_string());
        }
        result
    }

    /// Split string by delimiter
    pub fn split(&self, delimiter: &str) -> Vec<AugString> {
        let mut result = Vec::new();
        let mut start = 0;

        while let Some(pos) = self.substring(start, self.len())
            .unwrap()
            .find(delimiter) {
            let actual_pos = start + pos;
            if let Ok(part) = self.substring(start, actual_pos) {
                result.push(part);
            }
            start = actual_pos + delimiter.len();
        }

        // Add last part
        if let Ok(last_part) = self.substring(start, self.len()) {
            result.push(last_part);
        }
        result
    }

    /// Convert to lowercase
    pub fn to_lowercase(&self) -> AugString {
        let mut result = AugString::new();
        for byte in &self.data {
            if *byte >= b'A' && *byte <= b'Z' {
                result.data.push(*byte + 32);
            } else {
                result.data.push(*byte);
            }
        }
        result
    }

    /// Convert to uppercase
    pub fn to_uppercase(&self) -> AugString {
        let mut result = AugString::new();
        for byte in &self.data {
            if *byte >= b'a' && *byte <= b'z' {
                result.data.push(*byte - 32);
            } else {
                result.data.push(*byte);
            }
        }
        result
    }

    /// Trim whitespace
    pub fn trim(&self) -> AugString {
        let start = self.data.iter().position(|&b| !is_whitespace(b)).unwrap_or(0);
        let end = self.data.iter().rposition(|&b| !is_whitespace(b))
            .map(|i| i + 1)
            .unwrap_or(0);

        self.substring(start, end).unwrap_or_else(|_| AugString::new())
    }

    /// Check if string starts with prefix
    pub fn starts_with(&self, prefix: &str) -> bool {
        let prefix_bytes = prefix.as_bytes();
        if prefix_bytes.len() > self.data.len() {
            return false;
        }

        &self.data[..prefix_bytes.len()] == prefix_bytes
    }

    /// Check if string ends with suffix
    pub fn ends_with(&self, suffix: &str) -> bool {
        let suffix_bytes = suffix.as_bytes();
        if suffix_bytes.len() > self.data.len() {
            return false;
        }

        let start = self.data.len() - suffix_bytes.len();
        &self.data[start..] == suffix_bytes
    }

    /// Parse string as U256 number
    pub fn parse_u256(&self) -> std::result::Result<U256, &'static str> {
        let s = self.trim();
        if s.is_empty() {
            return Err("Empty string");
        }

        let mut result = U256::ZERO;
        let ten = U256::new(10u64);
        
        for byte in &s.data {
            if *byte >= b'0' && *byte <= b'9' {
                let digit = U256::new((*byte - b'0') as u64);
                result = SafeMath::mul(result, ten).map_err(|_| "Arithmetic overflow")?;
                result = SafeMath::add(result, digit).map_err(|_| "Arithmetic overflow")?;
            } else {
                return Err("Invalid digit");
            }
        }

        Ok(result)
    }

    /// Convert to Rust string
    pub fn to_string(&self) -> String {
        String::from_utf8_lossy(&self.data).to_string()
    }

    /// Get bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Convert to hex string
    pub fn to_hex(&self) -> AugString {
        let mut hex = AugString::new();
        hex.push_str("0x");
        for byte in &self.data {
            hex.push_str(&format!("{:02x}", byte));
        }
        hex
    }

    /// Create from hex string
    pub fn from_hex(hex: &str) -> std::result::Result<AugString, &'static str> {
        let hex = if hex.starts_with("0x") {
            &hex[2..]
        } else {
            hex
        };

        if hex.len() % 2 != 0 {
            return Err("Invalid hex length");
        }

        let mut bytes = Vec::new();
        for chunk in hex.as_bytes().chunks(2) {
            let hex_str = std::str::from_utf8(chunk).map_err(|_| "Invalid hex character")?;
            let byte = u8::from_str_radix(hex_str, 16).map_err(|_| "Invalid hex character")?;
            bytes.push(byte);
        }

        AugString::from_bytes(&bytes)
    }

    /// Encode to Base64
    pub fn to_base64(&self) -> AugString {
        let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut result = AugString::new();
        
        for chunk in self.data.chunks(3) {
            let mut buf = [0u8; 3];
            for (i, &byte) in chunk.iter().enumerate() {
                buf[i] = byte;
            }
            
            let b = ((buf[0] as u32) << 16) | ((buf[1] as u32) << 8) | (buf[2] as u32);
            
            result.data.push(alphabet[((b >> 18) & 63) as usize]);
            result.data.push(alphabet[((b >> 12) & 63) as usize]);
            
            if chunk.len() > 1 {
                result.data.push(alphabet[((b >> 6) & 63) as usize]);
            } else {
                result.data.push(b'=');
            }
            
            if chunk.len() > 2 {
                result.data.push(alphabet[(b & 63) as usize]);
            } else {
                result.data.push(b'=');
            }
        }
        
        result
    }

    /// Decode from Base64
    pub fn from_base64(base64: &str) -> std::result::Result<AugString, &'static str> {
        let mut bytes = Vec::new();
        let input = base64.as_bytes();
        
        for chunk in input.chunks(4) {
            if chunk.len() != 4 {
                return Err("Invalid base64 length");
            }
            
            let mut values = [0u8; 4];
            for (i, &byte) in chunk.iter().enumerate() {
                values[i] = match byte {
                    b'A'..=b'Z' => byte - b'A',
                    b'a'..=b'z' => byte - b'a' + 26,
                    b'0'..=b'9' => byte - b'0' + 52,
                    b'+' => 62,
                    b'/' => 63,
                    b'=' => 0,
                    _ => return Err("Invalid base64 character"),
                };
            }
            
            let b = ((values[0] as u32) << 18) | ((values[1] as u32) << 12) | 
                    ((values[2] as u32) << 6) | (values[3] as u32);
            
            bytes.push((b >> 16) as u8);
            if chunk[2] != b'=' {
                bytes.push((b >> 8) as u8);
            }
            if chunk[3] != b'=' {
                bytes.push(b as u8);
            }
        }
        
        AugString::from_bytes(&bytes)
    }
}

/// Utility functions
fn is_valid_utf8(bytes: &[u8]) -> bool {
    // Simplified UTF-8 validation
    // In practice, would implement full UTF-8 validation
    std::str::from_utf8(bytes).is_ok()
}

fn is_whitespace(byte: u8) -> bool {
    byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r'
}

/// String formatting function
pub fn format(template: &str, args: &[&str]) -> AugString {
    let mut result = String::from(template);
    
    // Replace {} placeholders with arguments in order
    for arg in args {
        if let Some(pos) = result.find("{}") {
            result.replace_range(pos..pos+2, arg);
        }
    }
    
    AugString::from_str(&result)
}

/// String concatenation
pub fn concat(strings: &[&AugString]) -> AugString {
    let mut result = AugString::new();
    for s in strings {
        result.push_str(&s.to_string());
    }
    result
}

/// Join strings with delimiter
pub fn join(strings: &[&AugString], delimiter: &str) -> AugString {
    let mut result = AugString::new();
    for (i, s) in strings.iter().enumerate() {
        if i > 0 {
            result.push_str(delimiter);
        }
        result.push_str(&s.to_string());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_creation() {
        let s = AugString::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);

        let s2 = AugString::from_str("Hello");
        assert_eq!(s2.len(), 5);
        assert!(!s2.is_empty());
    }

    #[test]
    fn test_string_manipulation() {
        let mut s = AugString::from_str("Hello");
        s.push(' ');
        s.push_str("World");
        assert_eq!(s.to_string(), "Hello World");
    }

    #[test]
    fn test_substring() {
        let s = AugString::from_str("Hello World");
        let sub = s.substring(0, 5).unwrap();
        assert_eq!(sub.to_string(), "Hello");

        let sub2 = s.substring(6, 11).unwrap();
        assert_eq!(sub2.to_string(), "World");
    }

    #[test]
    fn test_find() {
        let s = AugString::from_str("Hello World");
        assert_eq!(s.find("World"), Some(6));
        assert_eq!(s.find("xyz"), None);
        assert_eq!(s.find(""), Some(0));
    }

    #[test]
    fn test_replace() {
        let s = AugString::from_str("Hello World");
        let replaced = s.replace("World", "Rust");
        assert_eq!(replaced.to_string(), "Hello Rust");
    }

    #[test]
    fn test_split() {
        let s = AugString::from_str("a,b,c");
        let parts = s.split(",");
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].to_string(), "a");
        assert_eq!(parts[1].to_string(), "b");
        assert_eq!(parts[2].to_string(), "c");
    }

    #[test]
    fn test_case_conversion() {
        let s = AugString::from_str("Hello World");
        assert_eq!(s.to_lowercase().to_string(), "hello world");
        assert_eq!(s.to_uppercase().to_string(), "HELLO WORLD");
    }

    #[test]
    fn test_trim() {
        let s = AugString::from_str("  Hello World  ");
        assert_eq!(s.trim().to_string(), "Hello World");
    }

    #[test]
    fn test_starts_ends_with() {
        let s = AugString::from_str("Hello World");
        assert!(s.starts_with("Hello"));
        assert!(s.ends_with("World"));
        assert!(!s.starts_with("World"));
        assert!(!s.ends_with("Hello"));
    }

    #[test]
    fn test_parse_u256() {
        let s = AugString::from_str("12345");
        let num = s.parse_u256().unwrap();
        assert_eq!(num, U256::new(12345u64));

        let invalid = AugString::from_str("abc");
        assert!(invalid.parse_u256().is_err());
    }

    #[test]
    fn test_hex_encoding() {
        let s = AugString::from_str("Hello");
        let hex = s.to_hex();
        assert_eq!(hex.to_string(), "0x48656c6c6f");

        let decoded = AugString::from_hex("0x48656c6c6f").unwrap();
        assert_eq!(decoded.to_string(), "Hello");
    }

    #[test]
    fn test_base64_encoding() {
        let s = AugString::from_str("Hello");
        let b64 = s.to_base64();
        let decoded = AugString::from_base64(&b64.to_string()).unwrap();
        assert_eq!(decoded.to_string(), "Hello");
    }

    #[test]
    fn test_format() {
        let formatted = format("Hello, {}! You have {} tokens.", &["Alice", "100"]);
        assert_eq!(formatted.to_string(), "Hello, Alice! You have 100 tokens.");
    }

    #[test]
    fn test_concat() {
        let s1 = AugString::from_str("Hello");
        let s2 = AugString::from_str(" ");
        let s3 = AugString::from_str("World");
        let result = concat(&[&s1, &s2, &s3]);
        assert_eq!(result.to_string(), "Hello World");
    }

    #[test]
    fn test_join() {
        let s1 = AugString::from_str("a");
        let s2 = AugString::from_str("b");
        let s3 = AugString::from_str("c");
        let result = join(&[&s1, &s2, &s3], ",");
        assert_eq!(result.to_string(), "a,b,c");
    }
}