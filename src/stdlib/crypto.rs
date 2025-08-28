// Crypto module - hash functions, signatures, and randomness
// Used throughout the blockchain for transaction verification
// and secure operations

use crate::error::{Result, CompilerError, SemanticError, SemanticErrorKind, SourceLocation};
use crate::stdlib::core_types::{U8, U256, Bool, Address, AugustiumType};
use crate::stdlib::collections::Vec as AugVec;
use sha2::{Sha256, Digest};
use std::vec::Vec as StdVec;

// Common hash functions used in blockchain
pub struct Hash;

impl Hash {
    // Standard SHA-256 - most common hash function
    pub fn sha256(data: &AugVec<U8>) -> Result<[U8; 32]> {
        let mut hasher = Sha256::new();
        
        // need to convert our vector to bytes first
        let mut bytes = StdVec::new();
        for i in 0..data.len() {
            bytes.push(data.get(i).unwrap().0);
        }
        hasher.update(&bytes);
        
        let result = hasher.finalize();
        let mut hash_bytes = [U8(0); 32];
        
        // copy the hash result into our format
        for (i, &byte) in result.iter().enumerate() {
            hash_bytes[i] = U8(byte);
        }
        
        Ok(hash_bytes)
    }
    
    /// Raw keccak256 function that works with byte slices
    fn keccak256_raw(data: &[u8]) -> Result<[u8; 32]> {
        use sha3::{Keccak256, Digest};
        
        let mut hasher = Keccak256::new();
        hasher.update(data);
        let result = hasher.finalize();
        
        let mut hash_bytes = [0u8; 32];
        hash_bytes.copy_from_slice(&result);
        
        Ok(hash_bytes)
    }
    
    // Keccak-256 - what Ethereum uses for hashing
    pub fn keccak256(data: &AugVec<U8>) -> Result<[U8; 32]> {
        use sha3::{Keccak256, Digest};
        
        let mut hasher = Keccak256::new();
        
        // same conversion process as sha256
        let mut bytes = StdVec::new();
        for i in 0..data.len() {
            bytes.push(data.get(i).unwrap().0);
        }
        hasher.update(&bytes);
        
        let result = hasher.finalize();
        let mut hash_bytes = [U8(0); 32];
        
        for (i, &byte) in result.iter().enumerate() {
            hash_bytes[i] = U8(byte);
        }
        
        Ok(hash_bytes)
    }
    
    /// Computes RIPEMD-160 hash of input bytes
    /// 
    /// # Arguments
    /// * `data` - Input data to hash
    /// 
    /// # Returns
    /// * `Result<[U8; 20], CompilerError>` - 20-byte RIPEMD-160 hash
    pub fn ripemd160(data: &AugVec<U8>) -> Result<[U8; 20]> {
        use ripemd::{Ripemd160, Digest};
        
        let mut hasher = Ripemd160::new();
        
        // Convert AugVec<U8> to bytes
        let mut bytes = StdVec::new();
        for i in 0..data.len() {
            bytes.push(data.get(i).unwrap().0);
        }
        hasher.update(&bytes);
        
        let result = hasher.finalize();
        let mut hash_bytes = [U8(0); 20];
        
        for (i, &byte) in result.iter().enumerate() {
            hash_bytes[i] = U8(byte);
        }
        
        Ok(hash_bytes)
    }
    
    /// Double SHA-256 hash (Bitcoin standard)
    /// 
    /// # Arguments
    /// * `data` - Input data to hash
    /// 
    /// # Returns
    /// * `Result<[U8; 32], CompilerError>` - 32-byte double SHA-256 hash
    pub fn double_sha256(data: &AugVec<U8>) -> Result<[U8; 32]> {
        let first_hash = Self::sha256(data)?;
        
        // Convert first hash to AugVec for second hash
        let mut first_hash_vec = AugVec::new();
        for &byte in &first_hash {
            first_hash_vec.push(byte)?;
        }
        
        Self::sha256(&first_hash_vec)
    }
}

// Signature verification stuff - important for validating transactions
pub struct Signature;

impl Signature {
    // Recover the public key from a signature (like Ethereum does)
    // This is pretty complex cryptography but essential for blockchain
    #[cfg(all(feature = "crypto", not(target_arch = "wasm32")))]
    pub fn ecrecover(
        message_hash: &AugVec<U8>,
        signature: &AugVec<U8>,
        recovery_id: U8,
    ) -> Result<Address> {
        use secp256k1::{Secp256k1, Message, ecdsa::{RecoveryId, RecoverableSignature}};
        
        if recovery_id.0 > 3 {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Invalid recovery ID for ECDSA signature".to_string(),
            }));
        }
        
        // Validate input lengths
        if message_hash.len() != 32 {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Message hash must be 32 bytes".to_string(),
            }));
        }
        
        if signature.len() != 64 {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Signature must be 64 bytes".to_string(),
            }));
        }
        
        // Convert AugVec to bytes
        let mut hash_bytes = [0u8; 32];
        for i in 0..32 {
            hash_bytes[i] = message_hash.get(i).unwrap().0;
        }
        
        let mut sig_bytes = [0u8; 64];
        for i in 0..64 {
            sig_bytes[i] = signature.get(i).unwrap().0;
        }
        
        // Create secp256k1 context
        let secp = Secp256k1::new();
        
        // Parse message
        let message = Message::from_digest_slice(&hash_bytes)
            .map_err(|_| CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Invalid message hash".to_string(),
            }))?;
        
        // Parse recovery ID
        let recovery_id = RecoveryId::from_i32(recovery_id.0 as i32)
            .map_err(|_| CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Invalid recovery ID".to_string(),
            }))?;
        
        // Parse signature
        let recoverable_sig = RecoverableSignature::from_compact(&sig_bytes, recovery_id)
            .map_err(|_| CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Invalid signature format".to_string(),
            }))?;
        
        // Recover public key
        let public_key = secp.recover_ecdsa(&message, &recoverable_sig)
            .map_err(|_| CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Failed to recover public key".to_string(),
            }))?;
        
        // Convert public key to Ethereum address
        let public_key_bytes = public_key.serialize_uncompressed();
        let public_key_hash = Hash::keccak256_raw(&public_key_bytes[1..65])?; // Skip first byte (0x04)
        
        // Take last 20 bytes as address
        let mut address_bytes = [0u8; 20];
        address_bytes.copy_from_slice(&public_key_hash[12..32]);
        
        Ok(Address::from_bytes(&address_bytes)?)
    }
    
    /// Verifies an ECDSA signature against a known public key
    /// 
    /// # Arguments
    /// * `message_hash` - 32-byte hash of the message
    /// * `signature` - 64-byte signature (r + s)
    /// * `public_key` - 64-byte uncompressed public key
    /// 
    /// # Returns
    /// * `Result<Bool, CompilerError>` - True if signature is valid
    #[cfg(all(feature = "crypto", not(target_arch = "wasm32")))]
    pub fn verify_signature(
        message_hash: &[U8; 32],
        signature: &[U8; 64],
        public_key: &[U8; 64],
    ) -> Result<Bool> {
        use secp256k1::{Secp256k1, Message, PublicKey, ecdsa::Signature as Secp256k1Signature};
        
        // Basic validation: ensure no zero signatures
        let is_zero_sig = signature.iter().all(|&byte| byte.0 == 0);
        let is_zero_pubkey = public_key.iter().all(|&byte| byte.0 == 0);
        
        if is_zero_sig || is_zero_pubkey {
            return Ok(Bool(false));
        }
        
        // Convert U8 arrays to byte arrays
        let mut hash_bytes = [0u8; 32];
        for (i, &byte) in message_hash.iter().enumerate() {
            hash_bytes[i] = byte.0;
        }
        
        let mut sig_bytes = [0u8; 64];
        for (i, &byte) in signature.iter().enumerate() {
            sig_bytes[i] = byte.0;
        }
        
        let mut pubkey_bytes = [0u8; 65]; // Uncompressed public key with 0x04 prefix
        pubkey_bytes[0] = 0x04;
        for (i, &byte) in public_key.iter().enumerate() {
            pubkey_bytes[i + 1] = byte.0;
        }
        
        // Create secp256k1 context
        let secp = Secp256k1::verification_only();
        
        // Parse message
        let message = match Message::from_digest_slice(&hash_bytes) {
            Ok(msg) => msg,
            Err(_) => return Ok(Bool(false)),
        };
        
        // Parse public key
        let public_key = match PublicKey::from_slice(&pubkey_bytes) {
            Ok(pk) => pk,
            Err(_) => return Ok(Bool(false)),
        };
        
        // Parse signature
        let signature = match Secp256k1Signature::from_compact(&sig_bytes) {
            Ok(sig) => sig,
            Err(_) => return Ok(Bool(false)),
        };
        
        // Verify signature
        match secp.verify_ecdsa(&message, &signature, &public_key) {
            Ok(_) => Ok(Bool(true)),
            Err(_) => Ok(Bool(false)),
        }
    }
    
    // Stub implementations when crypto feature is disabled
    #[cfg(any(target_arch = "wasm32", not(feature = "crypto")))]
    pub fn ecrecover(
        _message_hash: &AugVec<U8>,
        _signature: &AugVec<U8>,
        _recovery_id: U8,
    ) -> Result<Address> {
        Err(CompilerError::SemanticError(SemanticError {
            kind: SemanticErrorKind::InvalidOperation,
            location: SourceLocation::unknown(),
            message: "ECDSA signature recovery not available (crypto feature disabled)".to_string(),
        }))
    }
    
    #[cfg(any(target_arch = "wasm32", not(feature = "crypto")))]
    pub fn verify_signature(
        _message_hash: &[U8; 32],
        _signature: &[U8; 64],
        _public_key: &[U8; 64],
    ) -> Result<Bool> {
        Err(CompilerError::SemanticError(SemanticError {
            kind: SemanticErrorKind::InvalidOperation,
            location: SourceLocation::unknown(),
            message: "ECDSA signature verification not available (crypto feature disabled)".to_string(),
        }))
    }
}

/// Secure random number generation for blockchain applications
pub struct Random;

impl Random {
    /// Generates a cryptographically secure random U256
    /// 
    /// Note: In a real blockchain environment, this would use
    /// deterministic randomness based on block hash and other factors
    /// 
    /// # Returns
    /// * `Result<U256, CompilerError>` - Random 256-bit number
    #[cfg(all(feature = "crypto", not(target_arch = "wasm32")))]
    pub fn secure_u256() -> Result<U256> {
        use rand::Rng;
        
        let mut rng = rand::thread_rng();
        let mut bytes = [0u8; 32];
        rng.fill(&mut bytes);
        
        Ok(<U256 as AugustiumType>::from_bytes(&bytes)?)
    }
    
    /// Generates a cryptographically secure random U8
    /// 
    /// # Returns
    /// * `Result<U8, CompilerError>` - Random 8-bit number
    #[cfg(all(feature = "crypto", not(target_arch = "wasm32")))]
    pub fn secure_u8() -> Result<U8> {
        use rand::Rng;
        
        let mut rng = rand::thread_rng();
        let byte = rng.gen::<u8>();
        
        Ok(U8(byte))
    }
    
    /// Generates random bytes of specified length
    /// 
    /// # Arguments
    /// * `length` - Number of random bytes to generate
    /// 
    /// # Returns
    /// * `Result<AugVec<U8>, CompilerError>` - Vector of random bytes
    #[cfg(all(feature = "crypto", not(target_arch = "wasm32")))]
    pub fn secure_bytes(length: usize) -> Result<AugVec<U8>> {
        use rand::Rng;
        
        if length > 1024 {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Random byte generation limited to 1024 bytes".to_string(),
            }));
        }
        
        let mut rng = rand::thread_rng();
        let mut result = AugVec::new();
        
        for _ in 0..length {
            let byte = rng.gen::<u8>();
            result.push(U8(byte))?;
        }
        
        Ok(result)
    }
    
    // Stub implementations when crypto feature is disabled
    #[cfg(any(target_arch = "wasm32", not(feature = "crypto")))]
    pub fn secure_u256() -> Result<U256> {
        Err(CompilerError::SemanticError(SemanticError {
            kind: SemanticErrorKind::InvalidOperation,
            location: SourceLocation::unknown(),
            message: "Secure random number generation not available (crypto feature disabled)".to_string(),
        }))
    }
    
    #[cfg(any(target_arch = "wasm32", not(feature = "crypto")))]
    pub fn secure_u8() -> Result<U8> {
        Err(CompilerError::SemanticError(SemanticError {
            kind: SemanticErrorKind::InvalidOperation,
            location: SourceLocation::unknown(),
            message: "Secure random number generation not available (crypto feature disabled)".to_string(),
        }))
    }
    
    #[cfg(any(target_arch = "wasm32", not(feature = "crypto")))]
    pub fn secure_bytes(_length: usize) -> Result<AugVec<U8>> {
        Err(CompilerError::SemanticError(SemanticError {
            kind: SemanticErrorKind::InvalidOperation,
            location: SourceLocation::unknown(),
            message: "Secure random number generation not available (crypto feature disabled)".to_string(),
        }))
    }
}

/// Merkle tree operations for efficient data verification
pub struct MerkleTree;

impl MerkleTree {
    /// Computes Merkle root from a list of leaf hashes
    /// 
    /// # Arguments
    /// * `leaves` - Vector of 32-byte leaf hashes (each as AugVec<U8>)
    /// 
    /// # Returns
    /// * `Result<AugVec<U8>, CompilerError>` - 32-byte Merkle root
    pub fn compute_root(leaves: &AugVec<AugVec<U8>>) -> Result<AugVec<U8>> {
        if leaves.is_empty() {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Cannot compute Merkle root of empty leaf set".to_string(),
            }));
        }
        
        let mut current_level = std::vec::Vec::new();
        for i in 0..leaves.len() {
            current_level.push(leaves.get(i).unwrap().clone());
        }
        
        while current_level.len() > 1 {
            let mut next_level = StdVec::new();
            
            // Process pairs of hashes
            for chunk in current_level.chunks(2) {
                let combined_hash = if chunk.len() == 2 {
                    Self::hash_pair(&chunk[0], &chunk[1])?
                } else {
                    // Odd number of elements, duplicate the last one
                    Self::hash_pair(&chunk[0], &chunk[0])?
                };
                next_level.push(combined_hash);
            }
            
            current_level = next_level;
        }
        
        Ok(current_level[0].clone())
    }
    
    /// Verifies a Merkle proof for a given leaf
    /// 
    /// # Arguments
    /// * `leaf` - The leaf hash to verify
    /// * `proof` - Vector of sibling hashes in the proof path
    /// * `root` - The expected Merkle root
    /// * `index` - The index of the leaf in the tree
    /// 
    /// # Returns
    /// * `Result<Bool, CompilerError>` - True if proof is valid
    pub fn verify_proof(
        leaf: &AugVec<U8>,
        proof: &AugVec<AugVec<U8>>,
        root: &AugVec<U8>,
        index: usize,
    ) -> Result<Bool> {
        let mut current_hash = leaf.clone();
        let mut current_index = index;
        
        for i in 0..proof.len() {
            let sibling = proof.get(i).unwrap();
            current_hash = if current_index % 2 == 0 {
                // Current node is left child
                Self::hash_pair(&current_hash, sibling)?
            } else {
                // Current node is right child
                Self::hash_pair(sibling, &current_hash)?
            };
            current_index /= 2;
        }
        
        // Compare hash vectors
        let hashes_equal = current_hash.len() == root.len() &&
            (0..current_hash.len()).all(|i| {
                current_hash.get(i).unwrap().0 == root.get(i).unwrap().0
            });
        
        Ok(Bool(hashes_equal))
    }
    
    /// Helper function to hash two 32-byte values together
    fn hash_pair(left: &AugVec<U8>, right: &AugVec<U8>) -> Result<AugVec<U8>> {
        let mut combined = AugVec::new();
        
        // Add left hash
        for i in 0..left.len() {
            combined.push(*left.get(i).unwrap())?;
        }
        
        // Add right hash
        for i in 0..right.len() {
            combined.push(*right.get(i).unwrap())?;
        }
        
        let hash_array = Hash::sha256(&combined)?;
        let mut result = AugVec::new();
        for &byte in &hash_array {
            result.push(byte)?;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sha256_hash() {
        let mut data = AugVec::new();
        data.push(U8(b'h')).unwrap();
        data.push(U8(b'e')).unwrap();
        data.push(U8(b'l')).unwrap();
        data.push(U8(b'l')).unwrap();
        data.push(U8(b'o')).unwrap();
        
        let result = Hash::sha256(&data).unwrap();
        
        // SHA-256 of "hello" should be consistent
        assert_eq!(result.len(), 32);
        // First few bytes of SHA-256("hello")
        assert_eq!(result[0].0, 0x2c);
        assert_eq!(result[1].0, 0xf2);
        assert_eq!(result[2].0, 0x4d);
    }
    
    #[test]
    fn test_double_sha256() {
        let mut data = AugVec::new();
        data.push(U8(b'h')).unwrap();
        data.push(U8(b'i')).unwrap();
        
        let result = Hash::double_sha256(&data).unwrap();
        assert_eq!(result.len(), 32);
    }
    
    #[test]
    fn test_signature_verification() {
        let message_hash = [U8(1); 32];
        let signature = [U8(2); 64];
        let public_key = [U8(3); 64];
        
        let result = Signature::verify_signature(&message_hash, &signature, &public_key).unwrap();
        // The signature verification will fail for invalid test data, which is expected
        assert_eq!(result.0, false); // Invalid signature data should return false
    }
    
    #[test]
    fn test_signature_verification_zero_inputs() {
        let message_hash = [U8(1); 32];
        let signature = [U8(0); 64]; // Zero signature
        let public_key = [U8(3); 64];
        
        let result = Signature::verify_signature(&message_hash, &signature, &public_key).unwrap();
        assert_eq!(result.0, false); // Zero signature should return false
    }
    
    #[test]
    fn test_random_generation() {
        let _random_u256 = Random::secure_u256().unwrap();
        let _random_u8 = Random::secure_u8().unwrap();
        let random_bytes = Random::secure_bytes(10).unwrap();
        
        // Basic validation - should not panic and should have correct sizes
        assert_eq!(random_bytes.len(), 10);
    }
    
    #[test]
    fn test_random_bytes_limit() {
        let result = Random::secure_bytes(2000); // Over limit
        assert!(result.is_err());
    }
    
    #[test]
    fn test_merkle_root_single_leaf() {
        let mut leaves = AugVec::new();
        let mut leaf = AugVec::new();
        for _ in 0..32 {
            leaf.push(U8(1)).unwrap();
        }
        leaves.push(leaf.clone()).unwrap();
        
        let root = MerkleTree::compute_root(&leaves).unwrap();
        assert_eq!(root.len(), 32); // Root should be 32 bytes
    }
    
    #[test]
    fn test_merkle_root_empty() {
        let leaves = AugVec::new();
        let result = MerkleTree::compute_root(&leaves);
        assert!(result.is_err()); // Empty leaves should error
    }
    
    #[test]
    fn test_ecrecover_invalid_recovery_id() {
        let mut message_hash = AugVec::new();
        for _ in 0..32 {
            message_hash.push(U8(1)).unwrap();
        }
        let mut signature = AugVec::new();
        for _ in 0..64 {
            signature.push(U8(2)).unwrap();
        }
        let recovery_id = U8(4); // Invalid recovery ID
        
        let result = Signature::ecrecover(&message_hash, &signature, recovery_id);
        assert!(result.is_err());
    }
}