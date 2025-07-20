// Collection types - vectors, arrays, maps etc.
// Basic data structures that contracts need

use crate::error::{CompilerError, SemanticError, SemanticErrorKind, SourceLocation};
use crate::stdlib::core_types::AugustiumType;
use std::collections::HashMap as StdHashMap;
use std::fmt;

// Dynamic array (like Rust's Vec but for Augustium types)
#[derive(Debug, Clone, PartialEq)]
pub struct Vec<T: AugustiumType> {
    data: std::vec::Vec<T>,
    capacity: usize,
}

impl<T: AugustiumType> Vec<T> {
    // Create new empty vector
    pub fn new() -> Self {
        Vec {
            data: std::vec::Vec::new(),
            capacity: 0,
        }
    }

    /// Create a new vector with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Vec {
            data: std::vec::Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Push an element to the end of the vector
    pub fn push(&mut self, value: T) -> Result<(), CompilerError> {
        // Gas-conscious capacity check
        if self.data.len() >= 1024 {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Vector capacity limit exceeded".to_string(),
            }));
        }
        
        self.data.push(value);
        Ok(())
    }

    /// Pop an element from the end of the vector
    pub fn pop(&mut self) -> Option<T> {
        self.data.pop()
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    /// Get mutable element at index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.get_mut(index)
    }

    /// Get element at index with bounds checking
    pub fn at(&self, index: usize) -> Result<&T, CompilerError> {
        self.data.get(index).ok_or_else(|| {
            CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: format!("Index {} out of bounds for vector of length {}", index, self.data.len()),
            })
        })
    }

    /// Set element at index
    pub fn set(&mut self, index: usize, value: T) -> Result<(), CompilerError> {
        if index >= self.data.len() {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: format!("Index {} out of bounds for vector of length {}", index, self.data.len()),
            }));
        }
        self.data[index] = value;
        Ok(())
    }

    /// Get the length of the vector
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clear all elements from the vector
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Check if vector contains a value
    pub fn contains(&self, value: &T) -> bool
    where
        T: PartialEq,
    {
        self.data.contains(value)
    }

    /// Find index of first occurrence of value
    pub fn find(&self, value: &T) -> Option<usize>
    where
        T: PartialEq,
    {
        self.data.iter().position(|x| x == value)
    }

    /// Remove element at index
    pub fn remove(&mut self, index: usize) -> Result<T, CompilerError> {
        if index >= self.data.len() {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: format!("Index {} out of bounds for vector of length {}", index, self.data.len()),
            }));
        }
        Ok(self.data.remove(index))
    }

    /// Insert element at index
    pub fn insert(&mut self, index: usize, value: T) -> Result<(), CompilerError> {
        if index > self.data.len() {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: format!("Index {} out of bounds for insertion in vector of length {}", index, self.data.len()),
            }));
        }
        
        // Gas-conscious capacity check
        if self.data.len() >= 1024 {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Vector capacity limit exceeded".to_string(),
            }));
        }
        
        self.data.insert(index, value);
        Ok(())
    }
}

impl<T: AugustiumType> fmt::Display for Vec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vec[{}]", self.data.len())
    }
}

impl<T: AugustiumType> AugustiumType for Vec<T> {
    fn type_name() -> &'static str {
        "Vec"
    }

    fn size_bytes() -> usize {
        // Base size of vector structure
        std::mem::size_of::<std::vec::Vec<T>>()
    }

    fn to_bytes(&self) -> std::vec::Vec<u8> {
        let mut bytes = std::vec::Vec::new();
        
        // Encode length as 4 bytes
        bytes.extend_from_slice(&(self.data.len() as u32).to_be_bytes());
        // Encode capacity as 4 bytes
        bytes.extend_from_slice(&(self.capacity as u32).to_be_bytes());
        
        // Encode each element
        for item in &self.data {
            bytes.extend_from_slice(&item.to_bytes());
        }
        
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, CompilerError> {
        if bytes.len() < 8 {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Insufficient bytes for Vec deserialization".to_string(),
            }));
        }

        let len = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let capacity = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let mut data = std::vec::Vec::with_capacity(len);
        let mut offset = 8;

        for _ in 0..len {
            if offset >= bytes.len() {
                return Err(CompilerError::SemanticError(SemanticError {
                    kind: SemanticErrorKind::InvalidOperation,
                    location: SourceLocation::unknown(),
                    message: "Insufficient bytes for Vec element deserialization".to_string(),
                }));
            }

            // For simplicity, assume fixed-size elements (this would need to be more sophisticated for variable-size types)
            let element_size = std::mem::size_of::<T>();
            if offset + element_size > bytes.len() {
                return Err(CompilerError::SemanticError(SemanticError {
                    kind: SemanticErrorKind::InvalidOperation,
                    location: SourceLocation::unknown(),
                    message: "Insufficient bytes for Vec element".to_string(),
                }));
            }

            let element = T::from_bytes(&bytes[offset..offset + element_size])?;
            data.push(element);
            offset += element_size;
        }

        Ok(Vec {
            data,
            capacity,
        })
    }

    fn is_zero(&self) -> bool {
        self.data.is_empty()
    }
}

/// Fixed-size array type for Augustium
#[derive(Debug, Clone, PartialEq)]
pub struct Array<T: AugustiumType, const N: usize> {
    data: [T; N],
}

impl<T: AugustiumType + Default + Copy, const N: usize> Array<T, N> {
    /// Create a new array with default values
    pub fn new() -> Self {
        Array {
            data: [T::default(); N],
        }
    }

    /// Create array from slice
    pub fn from_slice(slice: &[T]) -> Result<Self, CompilerError> {
        if slice.len() != N {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: format!("Slice length {} does not match array size {}", slice.len(), N),
            }));
        }

        let mut data = [T::default(); N];
        data.copy_from_slice(slice);
        Ok(Array { data })
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    /// Get mutable element at index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.get_mut(index)
    }

    /// Get element at index with bounds checking
    pub fn at(&self, index: usize) -> Result<&T, CompilerError> {
        self.data.get(index).ok_or_else(|| {
            CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: format!("Index {} out of bounds for array of size {}", index, N),
            })
        })
    }

    /// Set element at index
    pub fn set(&mut self, index: usize, value: T) -> Result<(), CompilerError> {
        if index >= N {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: format!("Index {} out of bounds for array of size {}", index, N),
            }));
        }
        self.data[index] = value;
        Ok(())
    }

    /// Get the length of the array
    pub fn len(&self) -> usize {
        N
    }

    /// Check if array contains a value
    pub fn contains(&self, value: &T) -> bool
    where
        T: PartialEq,
    {
        self.data.contains(value)
    }

    /// Find index of first occurrence of value
    pub fn find(&self, value: &T) -> Option<usize>
    where
        T: PartialEq,
    {
        self.data.iter().position(|x| x == value)
    }

    /// Get slice of the entire array
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get mutable slice of the entire array
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T: AugustiumType + Default + Copy, const N: usize> fmt::Display for Array<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Array[{}]", N)
    }
}

impl<T: AugustiumType + Default + Copy, const N: usize> AugustiumType for Array<T, N> {
    fn type_name() -> &'static str {
        "Array"
    }

    fn size_bytes() -> usize {
        N * std::mem::size_of::<T>()
    }

    fn to_bytes(&self) -> std::vec::Vec<u8> {
        let mut bytes = std::vec::Vec::new();
        
        for item in &self.data {
            bytes.extend_from_slice(&item.to_bytes());
        }
        
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, CompilerError> {
        let element_size = std::mem::size_of::<T>();
        let expected_size = N * element_size;
        
        if bytes.len() != expected_size {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: format!("Expected {} bytes for Array<{}, {}>, got {}", expected_size, std::any::type_name::<T>(), N, bytes.len()),
            }));
        }

        let mut data = [T::default(); N];
        
        for i in 0..N {
            let start = i * element_size;
            let end = start + element_size;
            data[i] = T::from_bytes(&bytes[start..end])?;
        }

        Ok(Array { data })
    }

    fn is_zero(&self) -> bool {
        self.data.iter().all(|x| x.is_zero())
    }
}

/// Hash map type for Augustium
#[derive(Debug, Clone, PartialEq)]
pub struct Map<K: AugustiumType + Eq + std::hash::Hash, V: AugustiumType> {
    data: StdHashMap<K, V>,
}

impl<K: AugustiumType + Eq + std::hash::Hash, V: AugustiumType> Map<K, V> {
    /// Create a new empty map
    pub fn new() -> Self {
        Map {
            data: StdHashMap::new(),
        }
    }

    /// Create a new map with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Map {
            data: StdHashMap::with_capacity(capacity),
        }
    }

    /// Insert a key-value pair
    pub fn insert(&mut self, key: K, value: V) -> Result<Option<V>, CompilerError> {
        // Gas-conscious size check
        if self.data.len() >= 256 {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Map capacity limit exceeded".to_string(),
            }));
        }
        
        Ok(self.data.insert(key, value))
    }

    /// Get value by key
    pub fn get(&self, key: &K) -> Option<&V> {
        self.data.get(key)
    }

    /// Get mutable value by key
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.data.get_mut(key)
    }

    /// Remove value by key
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.data.remove(key)
    }

    /// Check if map contains key
    pub fn contains_key(&self, key: &K) -> bool {
        self.data.contains_key(key)
    }

    /// Get the number of key-value pairs
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the map is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clear all key-value pairs
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get all keys
    pub fn keys(&self) -> std::vec::Vec<&K> {
        self.data.keys().collect()
    }

    /// Get all values
    pub fn values(&self) -> std::vec::Vec<&V> {
        self.data.values().collect()
    }

    /// Iterate over key-value pairs
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.data.iter()
    }
}

impl<K: AugustiumType + Eq + std::hash::Hash, V: AugustiumType> fmt::Display for Map<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Map[{}]", self.data.len())
    }
}

impl<K: AugustiumType + Eq + std::hash::Hash, V: AugustiumType> AugustiumType for Map<K, V> {
    fn type_name() -> &'static str {
        "Map"
    }

    fn size_bytes() -> usize {
        // Base size of hash map structure
        std::mem::size_of::<StdHashMap<K, V>>()
    }

    fn to_bytes(&self) -> std::vec::Vec<u8> {
        let mut bytes = std::vec::Vec::new();
        
        // Encode number of pairs as 4 bytes
        bytes.extend_from_slice(&(self.data.len() as u32).to_be_bytes());
        
        // Encode each key-value pair
        for (key, value) in &self.data {
            let key_bytes = key.to_bytes();
            let value_bytes = value.to_bytes();
            
            // Encode key length, key, value length, value
            bytes.extend_from_slice(&(key_bytes.len() as u32).to_be_bytes());
            bytes.extend_from_slice(&key_bytes);
            bytes.extend_from_slice(&(value_bytes.len() as u32).to_be_bytes());
            bytes.extend_from_slice(&value_bytes);
        }
        
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, CompilerError> {
        if bytes.len() < 4 {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Insufficient bytes for Map deserialization".to_string(),
            }));
        }

        let len = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let mut data = StdHashMap::with_capacity(len);
        let mut offset = 4;

        for _ in 0..len {
            // Read key length
            if offset + 4 > bytes.len() {
                return Err(CompilerError::SemanticError(SemanticError {
                    kind: SemanticErrorKind::InvalidOperation,
                    location: SourceLocation::unknown(),
                    message: "Insufficient bytes for key length".to_string(),
                }));
            }
            let key_len = u32::from_be_bytes([
                bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
            ]) as usize;
            offset += 4;

            // Read key
            if offset + key_len > bytes.len() {
                return Err(CompilerError::SemanticError(SemanticError {
                    kind: SemanticErrorKind::InvalidOperation,
                    location: SourceLocation::unknown(),
                    message: "Insufficient bytes for key".to_string(),
                }));
            }
            let key = K::from_bytes(&bytes[offset..offset + key_len])?;
            offset += key_len;

            // Read value length
            if offset + 4 > bytes.len() {
                return Err(CompilerError::SemanticError(SemanticError {
                    kind: SemanticErrorKind::InvalidOperation,
                    location: SourceLocation::unknown(),
                    message: "Insufficient bytes for value length".to_string(),
                }));
            }
            let value_len = u32::from_be_bytes([
                bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
            ]) as usize;
            offset += 4;

            // Read value
            if offset + value_len > bytes.len() {
                return Err(CompilerError::SemanticError(SemanticError {
                    kind: SemanticErrorKind::InvalidOperation,
                    location: SourceLocation::unknown(),
                    message: "Insufficient bytes for value".to_string(),
                }));
            }
            let value = V::from_bytes(&bytes[offset..offset + value_len])?;
            offset += value_len;

            data.insert(key, value);
        }

        Ok(Map { data })
    }

    fn is_zero(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::core_types::{U8, U256};

    #[test]
    fn test_vec_operations() {
        let mut vec = Vec::<U8>::new();
        assert!(vec.is_empty());
        assert_eq!(vec.len(), 0);

        // Test push
        vec.push(U8::new(42)).unwrap();
        vec.push(U8::new(100)).unwrap();
        assert_eq!(vec.len(), 2);
        assert!(!vec.is_empty());

        // Test get
        assert_eq!(vec.get(0).unwrap().0, 42);
        assert_eq!(vec.get(1).unwrap().0, 100);
        assert!(vec.get(2).is_none());

        // Test pop
        let popped = vec.pop().unwrap();
        assert_eq!(popped.0, 100);
        assert_eq!(vec.len(), 1);

        // Test contains
        assert!(vec.contains(&U8::new(42)));
        assert!(!vec.contains(&U8::new(100)));
    }

    #[test]
    fn test_array_operations() {
        let mut arr = Array::<U8, 3>::new();
        assert_eq!(arr.len(), 3);

        // Test set and get
        arr.set(0, U8::new(10)).unwrap();
        arr.set(1, U8::new(20)).unwrap();
        arr.set(2, U8::new(30)).unwrap();

        assert_eq!(arr.get(0).unwrap().0, 10);
        assert_eq!(arr.get(1).unwrap().0, 20);
        assert_eq!(arr.get(2).unwrap().0, 30);

        // Test bounds checking
        assert!(arr.set(3, U8::new(40)).is_err());
        assert!(arr.get(3).is_none());

        // Test contains
        assert!(arr.contains(&U8::new(20)));
        assert!(!arr.contains(&U8::new(40)));
    }

    #[test]
    fn test_map_operations() {
        let mut map = Map::<U8, U256>::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);

        // Test insert and get
        map.insert(U8::new(1), U256::new(100)).unwrap();
        map.insert(U8::new(2), U256::new(200)).unwrap();
        assert_eq!(map.len(), 2);
        assert!(!map.is_empty());

        assert_eq!(map.get(&U8::new(1)).unwrap(), &U256::new(100));
        assert_eq!(map.get(&U8::new(2)).unwrap(), &U256::new(200));
        assert!(map.get(&U8::new(3)).is_none());

        // Test contains_key
        assert!(map.contains_key(&U8::new(1)));
        assert!(!map.contains_key(&U8::new(3)));

        // Test remove
        let removed = map.remove(&U8::new(1)).unwrap();
        assert_eq!(removed, U256::new(100));
        assert_eq!(map.len(), 1);
        assert!(!map.contains_key(&U8::new(1)));
    }

    #[test]
    fn test_vec_serialization() {
        let mut vec = Vec::<U8>::new();
        vec.push(U8::new(42)).unwrap();
        vec.push(U8::new(100)).unwrap();

        let bytes = vec.to_bytes();
        let deserialized = Vec::<U8>::from_bytes(&bytes).unwrap();

        assert_eq!(vec, deserialized);
        assert_eq!(deserialized.len(), 2);
        assert_eq!(deserialized.get(0).unwrap().0, 42);
        assert_eq!(deserialized.get(1).unwrap().0, 100);
    }

    #[test]
    fn test_augustium_type_trait() {
        let vec = Vec::<U8>::new();
        assert_eq!(Vec::<U8>::type_name(), "Vec");
        assert!(vec.is_zero());

        let arr = Array::<U8, 3>::new();
        assert_eq!(Array::<U8, 3>::type_name(), "Array");
        assert!(arr.is_zero());

        let map = Map::<U8, U256>::new();
        assert_eq!(Map::<U8, U256>::type_name(), "Map");
        assert!(map.is_zero());
    }
}