//! Native Augustium Blockchain Consensus Engine
//!
//! This module implements a Proof-of-Stake consensus mechanism for the native Augustium blockchain.
//! It includes validator management, block production, and finality mechanisms.

use crate::error::{Result, CompilerError};
use crate::stdlib::core_types::{U256, AugustiumType};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use serde_bytes;

/// Minimum stake required to become a validator (in native tokens)
pub const MIN_VALIDATOR_STAKE: u64 = 32_000_000_000_000_000; // 32 tokens with 15 decimals (fits in u64)

/// Maximum number of validators in the active set
pub const MAX_VALIDATORS: usize = 100;

/// Block time in seconds
pub const BLOCK_TIME: u64 = 12;

/// Finality threshold (2/3 + 1)
pub const FINALITY_THRESHOLD: f64 = 0.6667;

/// Validator information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Validator {
    pub address: [u8; 20],
    pub stake: U256,
    pub commission: u8, // Commission rate as percentage (0-100)
    pub is_active: bool,
    pub last_block_produced: u64,
    pub total_blocks_produced: u64,
    pub slashing_count: u32,
    pub delegated_stake: U256,
}

/// Block header for the native blockchain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockHeader {
    pub number: u64,
    pub parent_hash: [u8; 32],
    pub state_root: [u8; 32],
    pub transactions_root: [u8; 32],
    pub timestamp: u64,
    pub validator: [u8; 20],
    #[serde(with = "serde_bytes")]
    pub signature: [u8; 65],
    pub gas_limit: u64,
    pub gas_used: u64,
}

/// Block structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub header: BlockHeader,
    pub transactions: Vec<Transaction>,
    pub attestations: Vec<Attestation>,
}

/// Transaction structure for native blockchain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub from: [u8; 20],
    pub to: Option<[u8; 20]>, // None for contract creation
    pub value: U256,
    pub gas_limit: u64,
    pub gas_price: U256,
    pub nonce: u64,
    pub data: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub signature: [u8; 65],
    pub hash: [u8; 32],
}

/// Attestation for block finality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attestation {
    pub validator: [u8; 20],
    pub block_hash: [u8; 32],
    #[serde(with = "serde_bytes")]
    pub signature: [u8; 65],
    pub timestamp: u64,
}

/// Validator set management
#[derive(Debug, Clone)]
pub struct ValidatorSet {
    validators: HashMap<[u8; 20], Validator>,
    active_validators: Vec<[u8; 20]>,
    total_stake: U256,
    epoch: u64,
}

/// Consensus state
#[derive(Debug, Clone)]
pub struct ConsensusState {
    pub current_block: u64,
    pub current_epoch: u64,
    pub finalized_block: u64,
    pub validator_set: ValidatorSet,
    pub pending_blocks: VecDeque<Block>,
    pub attestations: HashMap<[u8; 32], Vec<Attestation>>,
}

/// Proof-of-Stake consensus engine
pub struct PoSConsensus {
    state: ConsensusState,
    chain: Vec<Block>,
    block_cache: HashMap<[u8; 32], Block>,
    validator_rewards: HashMap<[u8; 20], U256>,
}

/// Delegation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delegation {
    pub delegator: [u8; 20],
    pub validator: [u8; 20],
    pub amount: U256,
    pub rewards: U256,
}

/// Slashing conditions
#[derive(Debug, Clone)]
pub enum SlashingCondition {
    DoubleSign,
    Inactivity,
    InvalidBlock,
}

impl Validator {
    /// Create a new validator
    pub fn new(address: [u8; 20], stake: U256, commission: u8) -> Self {
        Self {
            address,
            stake,
            commission,
            is_active: false,
            last_block_produced: 0,
            total_blocks_produced: 0,
            slashing_count: 0,
            delegated_stake: U256::zero(),
        }
    }
    
    /// Get total stake (own + delegated)
    pub fn total_stake(&self) -> U256 {
        self.stake + self.delegated_stake
    }
    
    /// Check if validator meets minimum stake requirement
    pub fn meets_min_stake(&self) -> bool {
        self.total_stake().as_u64() >= MIN_VALIDATOR_STAKE
    }
    
    /// Calculate validator power (voting weight)
    pub fn voting_power(&self, total_stake: U256) -> f64 {
        if total_stake.is_zero() {
            0.0
        } else {
            self.total_stake().as_u64() as f64 / total_stake.as_u64() as f64
        }
    }
}

impl ValidatorSet {
    /// Create a new validator set
    pub fn new() -> Self {
        Self {
            validators: HashMap::new(),
            active_validators: Vec::new(),
            total_stake: U256::zero(),
            epoch: 0,
        }
    }
    
    /// Add a new validator
    pub fn add_validator(&mut self, validator: Validator) -> Result<()> {
        if !validator.meets_min_stake() {
            return Err(CompilerError::InternalError(
                "Validator does not meet minimum stake requirement".to_string()
            ));
        }
        
        self.total_stake = self.total_stake + validator.total_stake();
        self.validators.insert(validator.address, validator);
        self.update_active_set();
        
        Ok(())
    }
    
    /// Remove a validator
    pub fn remove_validator(&mut self, address: &[u8; 20]) -> Result<()> {
        if let Some(validator) = self.validators.remove(address) {
            self.total_stake = self.total_stake - validator.total_stake();
            self.active_validators.retain(|addr| addr != address);
            self.update_active_set();
            Ok(())
        } else {
            Err(CompilerError::InternalError("Validator not found".to_string()))
        }
    }
    
    /// Update the active validator set
    fn update_active_set(&mut self) {
        // Sort validators by stake (descending) and take top MAX_VALIDATORS
        let mut sorted_validators: Vec<_> = self.validators.iter()
            .filter(|(_, v)| v.is_active && v.meets_min_stake())
            .collect();
        
        sorted_validators.sort_by(|a, b| b.1.total_stake().cmp(&a.1.total_stake()));
        
        self.active_validators = sorted_validators
            .into_iter()
            .take(MAX_VALIDATORS)
            .map(|(addr, _)| *addr)
            .collect();
    }
    
    /// Get validator by address
    pub fn get_validator(&self, address: &[u8; 20]) -> Option<&Validator> {
        self.validators.get(address)
    }
    
    /// Get mutable validator by address
    pub fn get_validator_mut(&mut self, address: &[u8; 20]) -> Option<&mut Validator> {
        self.validators.get_mut(address)
    }
    
    /// Check if address is an active validator
    pub fn is_active_validator(&self, address: &[u8; 20]) -> bool {
        self.active_validators.contains(address)
    }
    
    /// Get the next validator for block production (round-robin)
    pub fn get_next_validator(&self, block_number: u64) -> Option<[u8; 20]> {
        if self.active_validators.is_empty() {
            return None;
        }
        
        let index = (block_number as usize) % self.active_validators.len();
        Some(self.active_validators[index])
    }
    
    /// Calculate total voting power for a set of validators
    pub fn calculate_voting_power(&self, validators: &[[u8; 20]]) -> f64 {
        let total_stake: u64 = validators.iter()
            .filter_map(|addr| self.get_validator(addr))
            .map(|v| v.total_stake().as_u64())
            .sum();
        
        total_stake as f64 / self.total_stake.as_u64() as f64
    }
}

impl ConsensusState {
    /// Create a new consensus state
    pub fn new() -> Self {
        Self {
            current_block: 0,
            current_epoch: 0,
            finalized_block: 0,
            validator_set: ValidatorSet::new(),
            pending_blocks: VecDeque::new(),
            attestations: HashMap::new(),
        }
    }
    
    /// Add an attestation
    pub fn add_attestation(&mut self, attestation: Attestation) {
        self.attestations
            .entry(attestation.block_hash)
            .or_insert_with(Vec::new)
            .push(attestation);
    }
    
    /// Check if a block has enough attestations for finality
    pub fn has_finality(&self, block_hash: &[u8; 32]) -> bool {
        if let Some(attestations) = self.attestations.get(block_hash) {
            let attesting_validators: Vec<[u8; 20]> = attestations
                .iter()
                .map(|a| a.validator)
                .collect();
            
            let voting_power = self.validator_set.calculate_voting_power(&attesting_validators);
            voting_power >= FINALITY_THRESHOLD
        } else {
            false
        }
    }
}

impl PoSConsensus {
    /// Create a new PoS consensus engine
    pub fn new() -> Self {
        Self {
            state: ConsensusState::new(),
            chain: Vec::new(),
            block_cache: HashMap::new(),
            validator_rewards: HashMap::new(),
        }
    }
    
    /// Initialize the genesis block
    pub fn initialize_genesis(&mut self, genesis_validators: Vec<Validator>) -> Result<()> {
        // Add genesis validators
        for validator in genesis_validators {
            self.state.validator_set.add_validator(validator)?;
        }
        
        // Create genesis block
        let genesis_block = Block {
            header: BlockHeader {
                number: 0,
                parent_hash: [0; 32],
                state_root: [0; 32],
                transactions_root: [0; 32],
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                validator: [0; 20], // Genesis block has no validator
                signature: [0; 65],
                gas_limit: 10_000_000,
                gas_used: 0,
            },
            transactions: Vec::new(),
            attestations: Vec::new(),
        };
        
        self.chain.push(genesis_block.clone());
        self.block_cache.insert(self.calculate_block_hash(&genesis_block), genesis_block);
        
        Ok(())
    }
    
    /// Propose a new block
    pub fn propose_block(&mut self, validator: [u8; 20], transactions: Vec<Transaction>) -> Result<Block> {
        // Verify validator is eligible
        if !self.state.validator_set.is_active_validator(&validator) {
            return Err(CompilerError::InternalError("Invalid validator".to_string()));
        }
        
        let expected_validator = self.state.validator_set.get_next_validator(self.state.current_block + 1);
        if expected_validator != Some(validator) {
            return Err(CompilerError::InternalError("Not validator's turn".to_string()));
        }
        
        let parent_block = self.chain.last().unwrap();
        let parent_hash = self.calculate_block_hash(parent_block);
        
        // Calculate gas used
        let gas_used = transactions.iter().map(|tx| tx.gas_limit).sum();
        
        let block = Block {
            header: BlockHeader {
                number: self.state.current_block + 1,
                parent_hash,
                state_root: [0; 32], // TODO: Calculate actual state root
                transactions_root: self.calculate_transactions_root(&transactions),
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                validator,
                signature: [0; 65], // TODO: Sign with validator's key
                gas_limit: 10_000_000,
                gas_used,
            },
            transactions,
            attestations: Vec::new(),
        };
        
        Ok(block)
    }
    
    /// Add a block to the chain
    pub fn add_block(&mut self, block: Block) -> Result<()> {
        // Validate block
        self.validate_block(&block)?;
        
        // Add to pending blocks
        self.state.pending_blocks.push_back(block.clone());
        
        // Update state
        self.state.current_block = block.header.number;
        
        // Update validator stats
        if let Some(validator) = self.state.validator_set.get_validator_mut(&block.header.validator) {
            validator.last_block_produced = block.header.number;
            validator.total_blocks_produced += 1;
        }
        
        // Add to chain and cache
        let block_hash = self.calculate_block_hash(&block);
        self.chain.push(block.clone());
        self.block_cache.insert(block_hash, block);
        
        // Distribute rewards
        self.distribute_block_rewards(&block_hash)?;
        
        Ok(())
    }
    
    /// Validate a block
    fn validate_block(&self, block: &Block) -> Result<()> {
        // Check block number
        if block.header.number != self.state.current_block + 1 {
            return Err(CompilerError::InternalError("Invalid block number".to_string()));
        }
        
        // Check parent hash
        if let Some(parent) = self.chain.last() {
            let expected_parent_hash = self.calculate_block_hash(parent);
            if block.header.parent_hash != expected_parent_hash {
                return Err(CompilerError::InternalError("Invalid parent hash".to_string()));
            }
        }
        
        // Check validator
        if !self.state.validator_set.is_active_validator(&block.header.validator) {
            return Err(CompilerError::InternalError("Invalid validator".to_string()));
        }
        
        // Check timestamp
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        if block.header.timestamp > now + 60 { // Allow 1 minute clock drift
            return Err(CompilerError::InternalError("Block timestamp too far in future".to_string()));
        }
        
        // Validate transactions
        for transaction in &block.transactions {
            self.validate_transaction(transaction)?;
        }
        
        Ok(())
    }
    
    /// Validate a transaction
    fn validate_transaction(&self, _transaction: &Transaction) -> Result<()> {
        // TODO: Implement transaction validation
        // - Check signature
        // - Check nonce
        // - Check balance
        // - Check gas limit
        Ok(())
    }
    
    /// Process attestations and finalize blocks
    pub fn process_attestations(&mut self) -> Result<()> {
        let mut finalized_blocks = Vec::new();
        
        // Check each pending block for finality
        for block in &self.state.pending_blocks {
            let block_hash = self.calculate_block_hash(block);
            if self.state.has_finality(&block_hash) {
                finalized_blocks.push(block.header.number);
            }
        }
        
        // Update finalized block number
        if let Some(&max_finalized) = finalized_blocks.iter().max() {
            self.state.finalized_block = max_finalized;
            
            // Remove finalized blocks from pending
            self.state.pending_blocks.retain(|block| block.header.number > max_finalized);
        }
        
        Ok(())
    }
    
    /// Slash a validator for misbehavior
    pub fn slash_validator(&mut self, validator: [u8; 20], condition: SlashingCondition) -> Result<()> {
        if let Some(val) = self.state.validator_set.get_validator_mut(&validator) {
            val.slashing_count += 1;
            
            let slash_amount = match condition {
                SlashingCondition::DoubleSign => val.total_stake() / U256::new(20), // 5%
            SlashingCondition::Inactivity => val.total_stake() / U256::new(100), // 1%
            SlashingCondition::InvalidBlock => val.total_stake() / U256::new(10), // 10%
            };
            
            // Reduce stake
            if val.stake >= slash_amount {
                val.stake = val.stake - slash_amount;
            } else {
                val.stake = U256::zero();
            }
            
            // Deactivate if slashed too many times
            if val.slashing_count >= 3 {
                val.is_active = false;
            }
            
            // Update validator set
            self.state.validator_set.update_active_set();
        }
        
        Ok(())
    }
    
    /// Distribute block rewards to validator and delegators
    fn distribute_block_rewards(&mut self, block_hash: &[u8; 32]) -> Result<()> {
        if let Some(block) = self.block_cache.get(block_hash) {
            let validator_addr = block.header.validator;
            
            // Base reward + transaction fees
            let base_reward = U256::new(2_000_000_000_000_000_000u64); // 2 tokens
            let tx_fees: U256 = block.transactions.iter()
                .map(|tx| tx.gas_price * U256::new(tx.gas_limit))
                .fold(U256::zero(), |acc, fee| acc + fee);
            
            let total_reward = base_reward + tx_fees;
            
            // Add to validator rewards
            *self.validator_rewards.entry(validator_addr).or_insert(U256::zero()) += total_reward;
        }
        
        Ok(())
    }
    
    /// Calculate block hash
    fn calculate_block_hash(&self, block: &Block) -> [u8; 32] {
        // TODO: Implement proper block hashing
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        block.header.number.hash(&mut hasher);
        block.header.parent_hash.hash(&mut hasher);
        block.header.timestamp.hash(&mut hasher);
        
        let hash = hasher.finish();
        let mut result = [0u8; 32];
        result[..8].copy_from_slice(&hash.to_be_bytes());
        result
    }
    
    /// Calculate transactions root (Merkle root)
    fn calculate_transactions_root(&self, transactions: &[Transaction]) -> [u8; 32] {
        // TODO: Implement proper Merkle tree
        if transactions.is_empty() {
            return [0; 32];
        }
        
        // Simplified hash of all transaction hashes
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        for tx in transactions {
            tx.hash.hash(&mut hasher);
        }
        
        let hash = hasher.finish();
        let mut result = [0u8; 32];
        result[..8].copy_from_slice(&hash.to_be_bytes());
        result
    }
    
    /// Get current chain state
    pub fn get_state(&self) -> &ConsensusState {
        &self.state
    }
    
    /// Get block by number
    pub fn get_block(&self, number: u64) -> Option<&Block> {
        self.chain.get(number as usize)
    }
    
    /// Get latest block
    pub fn get_latest_block(&self) -> Option<&Block> {
        self.chain.last()
    }
    
    /// Get validator rewards
    pub fn get_validator_rewards(&self, validator: &[u8; 20]) -> U256 {
        self.validator_rewards.get(validator).cloned().unwrap_or(U256::zero())
    }
}

/// Delegation manager for handling stake delegation
pub struct DelegationManager {
    delegations: HashMap<[u8; 20], Vec<Delegation>>, // validator -> delegations
    delegator_stakes: HashMap<[u8; 20], U256>, // delegator -> total delegated
}

impl DelegationManager {
    /// Create a new delegation manager
    pub fn new() -> Self {
        Self {
            delegations: HashMap::new(),
            delegator_stakes: HashMap::new(),
        }
    }
    
    /// Delegate stake to a validator
    pub fn delegate(&mut self, delegator: [u8; 20], validator: [u8; 20], amount: U256) -> Result<()> {
        let delegation = Delegation {
            delegator,
            validator,
            amount,
            rewards: U256::zero(),
        };
        
        self.delegations.entry(validator).or_insert_with(Vec::new).push(delegation);
        *self.delegator_stakes.entry(delegator).or_insert(U256::zero()) += amount;
        
        Ok(())
    }
    
    /// Undelegate stake from a validator
    pub fn undelegate(&mut self, delegator: [u8; 20], validator: [u8; 20], amount: U256) -> Result<()> {
        if let Some(delegations) = self.delegations.get_mut(&validator) {
            if let Some(delegation) = delegations.iter_mut().find(|d| d.delegator == delegator) {
                if delegation.amount >= amount {
                    delegation.amount = delegation.amount - amount;
                    *self.delegator_stakes.entry(delegator).or_insert(U256::zero()) -= amount;
                    
                    // Remove delegation if amount becomes zero
                    if delegation.amount.is_zero() {
                        delegations.retain(|d| d.delegator != delegator);
                    }
                    
                    return Ok(());
                }
            }
        }
        
        Err(CompilerError::InternalError("Insufficient delegation".to_string()))
    }
    
    /// Get total delegated stake for a validator
    pub fn get_validator_delegated_stake(&self, validator: &[u8; 20]) -> U256 {
        self.delegations.get(validator)
            .map(|delegations| delegations.iter().map(|d| d.amount).fold(U256::zero(), |acc, amt| acc + amt))
            .unwrap_or(U256::zero())
    }
    
    /// Get delegator's total stake
    pub fn get_delegator_stake(&self, delegator: &[u8; 20]) -> U256 {
        self.delegator_stakes.get(delegator).cloned().unwrap_or(U256::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validator_creation() {
        let validator = Validator::new([1; 20], U256::new(MIN_VALIDATOR_STAKE), 5);
        assert!(validator.meets_min_stake());
        assert_eq!(validator.commission, 5);
    }
    
    #[test]
    fn test_validator_set() {
        let mut validator_set = ValidatorSet::new();
        let validator = Validator::new([1; 20], U256::new(MIN_VALIDATOR_STAKE), 5);
        
        assert!(validator_set.add_validator(validator).is_ok());
        assert!(validator_set.get_validator(&[1; 20]).is_some());
    }
    
    #[test]
    fn test_consensus_initialization() {
        let mut consensus = PoSConsensus::new();
        let validators = vec![
            Validator::new([1; 20], U256::new(MIN_VALIDATOR_STAKE), 5),
            Validator::new([2; 20], U256::new(MIN_VALIDATOR_STAKE), 3),
        ];
        
        assert!(consensus.initialize_genesis(validators).is_ok());
        assert_eq!(consensus.chain.len(), 1); // Genesis block
    }
    
    #[test]
    fn test_delegation() {
        let mut delegation_manager = DelegationManager::new();
        let delegator = [1; 20];
        let validator = [2; 20];
        let amount = U256::new(1000);
        
        assert!(delegation_manager.delegate(delegator, validator, amount).is_ok());
        assert_eq!(delegation_manager.get_delegator_stake(&delegator), amount);
        assert_eq!(delegation_manager.get_validator_delegated_stake(&validator), amount);
    }
}