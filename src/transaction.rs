//! Native Augustium Transaction System
//!
//! This module implements the native transaction handling system for the Augustium blockchain,
//! including transaction validation, execution, and state management.

use crate::error::{Result, CompilerError};
use crate::stdlib::core_types::{U256, AugustiumType};
use crate::consensus::{Transaction, Block};
use std::collections::{HashMap, BTreeMap};
use serde::{Serialize, Deserialize};

/// Transaction types supported by the native blockchain
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TransactionType {
    Transfer,
    ContractCreation,
    ContractCall,
    ValidatorStake,
    ValidatorUnstake,
    Delegation,
    Undelegation,
    Governance,
}

/// Transaction status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TransactionStatus {
    Pending,
    Included,
    Confirmed,
    Failed,
    Reverted,
}

/// Transaction receipt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionReceipt {
    pub transaction_hash: [u8; 32],
    pub block_number: u64,
    pub block_hash: [u8; 32],
    pub transaction_index: u32,
    pub from: [u8; 20],
    pub to: Option<[u8; 20]>,
    pub gas_used: u64,
    pub status: TransactionStatus,
    pub logs: Vec<Log>,
    pub contract_address: Option<[u8; 20]>, // For contract creation
    pub return_data: Vec<u8>,
}

/// Event log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Log {
    pub address: [u8; 20],
    pub topics: Vec<[u8; 32]>,
    pub data: Vec<u8>,
}

/// Account state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub nonce: u64,
    pub balance: U256,
    pub code_hash: [u8; 32],
    pub storage_root: [u8; 32],
    pub is_contract: bool,
}

/// Contract storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Storage {
    pub slots: HashMap<[u8; 32], [u8; 32]>,
}

/// World state
#[derive(Debug, Clone)]
pub struct WorldState {
    pub accounts: HashMap<[u8; 20], Account>,
    pub storage: HashMap<[u8; 20], Storage>,
    pub code: HashMap<[u8; 32], Vec<u8>>, // code_hash -> bytecode
    pub state_root: [u8; 32],
}

/// Transaction pool for managing pending transactions
#[derive(Debug, Clone)]
pub struct TransactionPool {
    pending: HashMap<[u8; 32], Transaction>,
    by_sender: HashMap<[u8; 20], Vec<[u8; 32]>>, // sender -> tx hashes
    by_nonce: BTreeMap<([u8; 20], u64), [u8; 32]>, // (sender, nonce) -> tx hash
    gas_price_queue: BTreeMap<U256, Vec<[u8; 32]>>, // gas_price -> tx hashes
}

/// Transaction executor
pub struct TransactionExecutor {
    state: WorldState,
    receipts: HashMap<[u8; 32], TransactionReceipt>,
    pool: TransactionPool,
    gas_limit: u64,
    base_fee: U256,
}

/// Gas costs for different operations
pub struct GasCosts {
    pub base: u64,
    pub transfer: u64,
    pub contract_creation: u64,
    pub contract_call: u64,
    pub storage_set: u64,
    pub storage_clear: u64,
    pub log: u64,
    pub copy: u64,
}

/// Execution context
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub block_number: u64,
    pub block_timestamp: u64,
    pub block_gas_limit: u64,
    pub block_validator: [u8; 20],
    pub tx_origin: [u8; 20],
    pub gas_price: U256,
    pub gas_limit: u64,
    pub gas_used: u64,
}

/// Transaction validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub error: Option<String>,
    pub gas_estimate: u64,
}

impl Default for GasCosts {
    fn default() -> Self {
        Self {
            base: 21000,
            transfer: 21000,
            contract_creation: 53000,
            contract_call: 25000,
            storage_set: 20000,
            storage_clear: 5000,
            log: 375,
            copy: 3,
        }
    }
}

impl Account {
    /// Create a new account
    pub fn new() -> Self {
        Self {
            nonce: 0,
            balance: U256::zero(),
            code_hash: [0; 32],
            storage_root: [0; 32],
            is_contract: false,
        }
    }
    
    /// Create a contract account
    pub fn new_contract(code_hash: [u8; 32]) -> Self {
        Self {
            nonce: 1, // Contract accounts start with nonce 1
            balance: U256::zero(),
            code_hash,
            storage_root: [0; 32],
            is_contract: true,
        }
    }
    
    /// Check if account has sufficient balance
    pub fn has_sufficient_balance(&self, amount: U256) -> bool {
        self.balance >= amount
    }
    
    /// Deduct balance
    pub fn deduct_balance(&mut self, amount: U256) -> Result<()> {
        if self.balance >= amount {
            self.balance = self.balance - amount;
            Ok(())
        } else {
            Err(CompilerError::InternalError("Insufficient balance".to_string()))
        }
    }
    
    /// Add balance
    pub fn add_balance(&mut self, amount: U256) {
        self.balance = self.balance + amount;
    }
    
    /// Increment nonce
    pub fn increment_nonce(&mut self) {
        self.nonce += 1;
    }
}

impl Storage {
    /// Create new storage
    pub fn new() -> Self {
        Self {
            slots: HashMap::new(),
        }
    }
    
    /// Get storage value
    pub fn get(&self, key: &[u8; 32]) -> [u8; 32] {
        self.slots.get(key).cloned().unwrap_or([0; 32])
    }
    
    /// Set storage value
    pub fn set(&mut self, key: [u8; 32], value: [u8; 32]) {
        if value == [0; 32] {
            self.slots.remove(&key);
        } else {
            self.slots.insert(key, value);
        }
    }
}

impl WorldState {
    /// Create a new world state
    pub fn new() -> Self {
        Self {
            accounts: HashMap::new(),
            storage: HashMap::new(),
            code: HashMap::new(),
            state_root: [0; 32],
        }
    }
    
    /// Get account (creates if doesn't exist)
    pub fn get_account(&mut self, address: &[u8; 20]) -> &mut Account {
        self.accounts.entry(*address).or_insert_with(Account::new)
    }
    
    /// Get account (read-only)
    pub fn get_account_readonly(&self, address: &[u8; 20]) -> Option<&Account> {
        self.accounts.get(address)
    }
    
    /// Get storage for account
    pub fn get_storage(&mut self, address: &[u8; 20]) -> &mut Storage {
        self.storage.entry(*address).or_insert_with(Storage::new)
    }
    
    /// Set contract code
    pub fn set_code(&mut self, address: [u8; 20], code: Vec<u8>) -> [u8; 32] {
        let code_hash = self.calculate_code_hash(&code);
        self.code.insert(code_hash, code);
        
        // Update account
        let account = self.get_account(&address);
        account.code_hash = code_hash;
        account.is_contract = true;
        
        code_hash
    }
    
    /// Get contract code
    pub fn get_code(&self, code_hash: &[u8; 32]) -> Option<&Vec<u8>> {
        self.code.get(code_hash)
    }
    
    /// Transfer value between accounts
    pub fn transfer(&mut self, from: [u8; 20], to: [u8; 20], value: U256) -> Result<()> {
        if value.is_zero() {
            return Ok(());
        }
        
        // Check sender balance
        {
            let sender = self.get_account(&from);
            if !sender.has_sufficient_balance(value) {
                return Err(CompilerError::InternalError("Insufficient balance".to_string()));
            }
        }
        
        // Perform transfer
        self.get_account(&from).deduct_balance(value)?;
        self.get_account(&to).add_balance(value);
        
        Ok(())
    }
    
    /// Calculate code hash
    fn calculate_code_hash(&self, code: &[u8]) -> [u8; 32] {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        code.hash(&mut hasher);
        
        let hash = hasher.finish();
        let mut result = [0u8; 32];
        result[..8].copy_from_slice(&hash.to_be_bytes());
        result
    }
    
    /// Calculate state root
    pub fn calculate_state_root(&mut self) -> [u8; 32] {
        // TODO: Implement proper Merkle Patricia Trie
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash all account states
        for (address, account) in &self.accounts {
            address.hash(&mut hasher);
            account.nonce.hash(&mut hasher);
            account.balance.as_u64().hash(&mut hasher);
            account.code_hash.hash(&mut hasher);
        }
        
        let hash = hasher.finish();
        let mut result = [0u8; 32];
        result[..8].copy_from_slice(&hash.to_be_bytes());
        
        self.state_root = result;
        result
    }
}

impl TransactionPool {
    /// Create a new transaction pool
    pub fn new() -> Self {
        Self {
            pending: HashMap::new(),
            by_sender: HashMap::new(),
            by_nonce: BTreeMap::new(),
            gas_price_queue: BTreeMap::new(),
        }
    }
    
    /// Add transaction to pool
    pub fn add_transaction(&mut self, tx: Transaction) -> Result<()> {
        let tx_hash = tx.hash;
        
        // Check if transaction already exists
        if self.pending.contains_key(&tx_hash) {
            return Err(CompilerError::InternalError("Transaction already exists".to_string()));
        }
        
        // Add to various indexes
        self.pending.insert(tx_hash, tx.clone());
        
        self.by_sender.entry(tx.from).or_insert_with(Vec::new).push(tx_hash);
        
        self.by_nonce.insert((tx.from, tx.nonce), tx_hash);
        
        self.gas_price_queue.entry(tx.gas_price).or_insert_with(Vec::new).push(tx_hash);
        
        Ok(())
    }
    
    /// Remove transaction from pool
    pub fn remove_transaction(&mut self, tx_hash: &[u8; 32]) -> Option<Transaction> {
        if let Some(tx) = self.pending.remove(tx_hash) {
            // Remove from indexes
            if let Some(sender_txs) = self.by_sender.get_mut(&tx.from) {
                sender_txs.retain(|hash| hash != tx_hash);
                if sender_txs.is_empty() {
                    self.by_sender.remove(&tx.from);
                }
            }
            
            self.by_nonce.remove(&(tx.from, tx.nonce));
            
            if let Some(gas_price_txs) = self.gas_price_queue.get_mut(&tx.gas_price) {
                gas_price_txs.retain(|hash| hash != tx_hash);
                if gas_price_txs.is_empty() {
                    self.gas_price_queue.remove(&tx.gas_price);
                }
            }
            
            Some(tx)
        } else {
            None
        }
    }
    
    /// Get transactions for block (sorted by gas price)
    pub fn get_transactions_for_block(&self, gas_limit: u64) -> Vec<Transaction> {
        let mut selected = Vec::new();
        let mut total_gas = 0u64;
        
        // Sort by gas price (descending)
        for (_, tx_hashes) in self.gas_price_queue.iter().rev() {
            for tx_hash in tx_hashes {
                if let Some(tx) = self.pending.get(tx_hash) {
                    if total_gas + tx.gas_limit <= gas_limit {
                        selected.push(tx.clone());
                        total_gas += tx.gas_limit;
                    }
                }
            }
        }
        
        selected
    }
    
    /// Get pending transaction count
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
    
    /// Get transaction by hash
    pub fn get_transaction(&self, tx_hash: &[u8; 32]) -> Option<&Transaction> {
        self.pending.get(tx_hash)
    }
}

impl TransactionExecutor {
    /// Create a new transaction executor
    pub fn new(gas_limit: u64, base_fee: U256) -> Self {
        Self {
            state: WorldState::new(),
            receipts: HashMap::new(),
            pool: TransactionPool::new(),
            gas_limit,
            base_fee,
        }
    }
    
    /// Validate a transaction
    pub fn validate_transaction(&self, tx: &Transaction) -> ValidationResult {
        // Check basic format
        if tx.gas_limit == 0 {
            return ValidationResult {
                is_valid: false,
                error: Some("Gas limit cannot be zero".to_string()),
                gas_estimate: 0,
            };
        }
        
        // Check gas price
        if tx.gas_price < self.base_fee {
            return ValidationResult {
                is_valid: false,
                error: Some("Gas price too low".to_string()),
                gas_estimate: 0,
            };
        }
        
        // Check sender account
        if let Some(account) = self.state.get_account_readonly(&tx.from) {
            // Check nonce
            if tx.nonce != account.nonce {
                return ValidationResult {
                    is_valid: false,
                    error: Some("Invalid nonce".to_string()),
                    gas_estimate: 0,
                };
            }
            
            // Check balance for gas + value
            let total_cost = tx.value + (tx.gas_price * U256::new(tx.gas_limit));
            if !account.has_sufficient_balance(total_cost) {
                return ValidationResult {
                    is_valid: false,
                    error: Some("Insufficient balance".to_string()),
                    gas_estimate: 0,
                };
            }
        } else {
            return ValidationResult {
                is_valid: false,
                error: Some("Sender account not found".to_string()),
                gas_estimate: 0,
            };
        }
        
        // Estimate gas
        let gas_estimate = self.estimate_gas(tx);
        
        ValidationResult {
            is_valid: true,
            error: None,
            gas_estimate,
        }
    }
    
    /// Execute a transaction
    pub fn execute_transaction(&mut self, tx: &Transaction, context: &ExecutionContext) -> Result<TransactionReceipt> {
        // Validate transaction
        let validation = self.validate_transaction(tx);
        if !validation.is_valid {
            return Err(CompilerError::InternalError(
                validation.error.unwrap_or("Transaction validation failed".to_string())
            ));
        }
        
        let mut receipt = TransactionReceipt {
            transaction_hash: tx.hash,
            block_number: context.block_number,
            block_hash: [0; 32], // Will be set by block processor
            transaction_index: 0, // Will be set by block processor
            from: tx.from,
            to: tx.to,
            gas_used: 0,
            status: TransactionStatus::Pending,
            logs: Vec::new(),
            contract_address: None,
            return_data: Vec::new(),
        };
        
        let gas_costs = GasCosts::default();
        let mut gas_used = gas_costs.base;
        
        // Deduct gas cost from sender
        let gas_cost = tx.gas_price * U256::new(tx.gas_limit);
        self.state.get_account(&tx.from).deduct_balance(gas_cost)?;
        
        // Increment sender nonce
        self.state.get_account(&tx.from).increment_nonce();
        
        // Execute based on transaction type
        let execution_result = if tx.to.is_none() {
            // Contract creation
            self.execute_contract_creation(tx, &mut gas_used, &gas_costs)
        } else if let Some(to) = tx.to {
            if self.state.get_account_readonly(&to).map_or(false, |acc| acc.is_contract) {
                // Contract call
                self.execute_contract_call(tx, &mut gas_used, &gas_costs)
            } else {
                // Simple transfer
                self.execute_transfer(tx, &mut gas_used, &gas_costs)
            }
        } else {
            Ok(Vec::new())
        };
        
        match execution_result {
            Ok(return_data) => {
                receipt.status = TransactionStatus::Confirmed;
                receipt.return_data = return_data;
            }
            Err(_) => {
                receipt.status = TransactionStatus::Failed;
                // Revert state changes (simplified - in practice would use snapshots)
            }
        }
        
        receipt.gas_used = gas_used;
        
        // Refund unused gas
        let gas_refund = tx.gas_limit.saturating_sub(gas_used);
        if gas_refund > 0 {
            let refund_amount = tx.gas_price * U256::new(gas_refund);
            self.state.get_account(&tx.from).add_balance(refund_amount);
        }
        
        // Store receipt
        self.receipts.insert(tx.hash, receipt.clone());
        
        Ok(receipt)
    }
    
    /// Execute a simple transfer
    fn execute_transfer(&mut self, tx: &Transaction, gas_used: &mut u64, gas_costs: &GasCosts) -> Result<Vec<u8>> {
        *gas_used += gas_costs.transfer;
        
        if let Some(to) = tx.to {
            self.state.transfer(tx.from, to, tx.value)?;
        }
        
        Ok(Vec::new())
    }
    
    /// Execute contract creation
    fn execute_contract_creation(&mut self, tx: &Transaction, gas_used: &mut u64, gas_costs: &GasCosts) -> Result<Vec<u8>> {
        *gas_used += gas_costs.contract_creation;
        
        // Calculate contract address
        let contract_address = self.calculate_contract_address(&tx.from, tx.nonce);
        
        // Deploy contract code
        let _code_hash = self.state.set_code(contract_address, tx.data.clone());
        
        // Transfer value to contract
        if !tx.value.is_zero() {
            self.state.transfer(tx.from, contract_address, tx.value)?;
        }
        
        Ok(contract_address.to_vec())
    }
    
    /// Execute contract call
    fn execute_contract_call(&mut self, tx: &Transaction, gas_used: &mut u64, gas_costs: &GasCosts) -> Result<Vec<u8>> {
        *gas_used += gas_costs.contract_call;
        
        // TODO: Implement actual contract execution (AVM integration)
        // For now, just transfer value if any
        if let Some(to) = tx.to {
            if !tx.value.is_zero() {
                self.state.transfer(tx.from, to, tx.value)?;
            }
        }
        
        Ok(Vec::new())
    }
    
    /// Estimate gas for a transaction
    fn estimate_gas(&self, tx: &Transaction) -> u64 {
        let gas_costs = GasCosts::default();
        let mut estimate = gas_costs.base;
        
        if tx.to.is_none() {
            // Contract creation
            estimate += gas_costs.contract_creation;
            estimate += (tx.data.len() as u64) * gas_costs.copy;
        } else if let Some(to) = tx.to {
            if self.state.get_account_readonly(&to).map_or(false, |acc| acc.is_contract) {
                // Contract call
                estimate += gas_costs.contract_call;
            } else {
                // Simple transfer
                estimate += gas_costs.transfer;
            }
        }
        
        estimate
    }
    
    /// Calculate contract address
    fn calculate_contract_address(&self, sender: &[u8; 20], nonce: u64) -> [u8; 20] {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        sender.hash(&mut hasher);
        nonce.hash(&mut hasher);
        
        let hash = hasher.finish();
        let mut result = [0u8; 20];
        result[..8].copy_from_slice(&hash.to_be_bytes());
        result
    }
    
    /// Execute a block of transactions
    pub fn execute_block(&mut self, block: &Block) -> Result<Vec<TransactionReceipt>> {
        let mut receipts = Vec::new();
        
        let context = ExecutionContext {
            block_number: block.header.number,
            block_timestamp: block.header.timestamp,
            block_gas_limit: block.header.gas_limit,
            block_validator: block.header.validator,
            tx_origin: [0; 20], // Will be set per transaction
            gas_price: U256::zero(), // Will be set per transaction
            gas_limit: 0, // Will be set per transaction
            gas_used: 0,
        };
        
        for (index, tx) in block.transactions.iter().enumerate() {
            let mut tx_context = context.clone();
            tx_context.tx_origin = tx.from;
            tx_context.gas_price = tx.gas_price;
            tx_context.gas_limit = tx.gas_limit;
            
            match self.execute_transaction(tx, &tx_context) {
                Ok(mut receipt) => {
                    receipt.transaction_index = index as u32;
                    receipt.block_hash = [0; 32]; // TODO: Set actual block hash
                    receipts.push(receipt);
                }
                Err(_e) => {
                    // Create failed receipt
                    let receipt = TransactionReceipt {
                        transaction_hash: tx.hash,
                        block_number: block.header.number,
                        block_hash: [0; 32],
                        transaction_index: index as u32,
                        from: tx.from,
                        to: tx.to,
                        gas_used: tx.gas_limit, // Use full gas on failure
                        status: TransactionStatus::Failed,
                        logs: Vec::new(),
                        contract_address: None,
                        return_data: Vec::new(),
                    };
                    receipts.push(receipt);
                }
            }
        }
        
        // Update state root
        self.state.calculate_state_root();
        
        Ok(receipts)
    }
    
    /// Get transaction receipt
    pub fn get_receipt(&self, tx_hash: &[u8; 32]) -> Option<&TransactionReceipt> {
        self.receipts.get(tx_hash)
    }
    
    /// Get account balance
    pub fn get_balance(&self, address: &[u8; 20]) -> U256 {
        self.state.get_account_readonly(address)
            .map(|acc| acc.balance)
            .unwrap_or(U256::zero())
    }
    
    /// Get account nonce
    pub fn get_nonce(&self, address: &[u8; 20]) -> u64 {
        self.state.get_account_readonly(address)
            .map(|acc| acc.nonce)
            .unwrap_or(0)
    }
    
    /// Get world state
    pub fn get_state(&self) -> &WorldState {
        &self.state
    }
    
    /// Get mutable world state
    pub fn get_state_mut(&mut self) -> &mut WorldState {
        &mut self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_account_creation() {
        let account = Account::new();
        assert_eq!(account.nonce, 0);
        assert_eq!(account.balance, U256::zero());
        assert!(!account.is_contract);
    }
    
    #[test]
    fn test_world_state_transfer() {
        let mut state = WorldState::new();
        let from = [1; 20];
        let to = [2; 20];
        let amount = U256::new(1000);
        
        // Add balance to sender
        state.get_account(&from).add_balance(amount);
        
        // Transfer
        assert!(state.transfer(from, to, amount).is_ok());
        
        // Check balances
        assert_eq!(state.get_account_readonly(&from).unwrap().balance, U256::zero());
        assert_eq!(state.get_account_readonly(&to).unwrap().balance, amount);
    }
    
    #[test]
    fn test_transaction_pool() {
        let mut pool = TransactionPool::new();
        
        let tx = Transaction {
            from: [1; 20],
            to: Some([2; 20]),
            value: U256::new(1000),
            gas_limit: 21000,
            gas_price: U256::new(20_000_000_000u64),
            nonce: 0,
            data: Vec::new(),
            signature: [0; 65],
            hash: [1; 32],
        };
        
        assert!(pool.add_transaction(tx).is_ok());
        assert_eq!(pool.pending_count(), 1);
        assert!(pool.get_transaction(&[1; 32]).is_some());
    }
    
    #[test]
    fn test_transaction_validation() {
        let mut executor = TransactionExecutor::new(10_000_000, U256::new(1_000_000_000u64));
        
        // Add balance to sender
        executor.state.get_account(&[1; 20]).add_balance(U256::new(1_000_000_000_000_000_000u64));
        
        let tx = Transaction {
            from: [1; 20],
            to: Some([2; 20]),
            value: U256::new(1000),
            gas_limit: 21000,
            gas_price: U256::new(2_000_000_000u64),
            nonce: 0,
            data: Vec::new(),
            signature: [0; 65],
            hash: [1; 32],
        };
        
        let result = executor.validate_transaction(&tx);
        assert!(result.is_valid);
        assert!(result.gas_estimate > 0);
    }
}