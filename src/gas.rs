//! Comprehensive Gas System for Augustium Blockchain
//!
//! This module implements a complete gas metering and pricing system for the Augustium blockchain,
//! including dynamic gas pricing, gas optimization, and detailed gas accounting.

use crate::error::{Result, CompilerError};
use crate::stdlib::core_types::U256;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Gas price tiers for different network conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GasPriceTier {
    Slow,
    Standard,
    Fast,
    Instant,
}

/// Gas cost categories for different operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GasCategory {
    // Basic operations
    Base,
    VeryLow,
    Low,
    Mid,
    High,
    
    // Memory operations
    Memory,
    MemoryExpansion,
    
    // Storage operations
    StorageSet,
    StorageClear,
    StorageLoad,
    
    // Contract operations
    ContractCreation,
    ContractCall,
    ContractSelfDestruct,
    
    // Cryptographic operations
    Sha256,
    Keccak256,
    Ripemd160,
    EcRecover,
    EcAdd,
    EcMul,
    EcPairing,
    
    // Machine learning operations
    MLInference,
    MLTraining,
    MLDataProcessing,
    
    // System operations
    Log,
    Copy,
    Jump,
    JumpConditional,
    
    // Blockchain operations
    BlockHash,
    Balance,
    ExtCodeSize,
    ExtCodeCopy,
    ExtCodeHash,
    
    // Transaction operations
    TxDataZero,
    TxDataNonZero,
    TxCreate,
    
    // Precompiled contracts
    Precompiled,
}

/// Detailed gas costs for all operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasCosts {
    // Basic operation costs
    pub base: u64,
    pub very_low: u64,
    pub low: u64,
    pub mid: u64,
    pub high: u64,
    
    // Memory costs
    pub memory: u64,
    pub memory_expansion_quad_divisor: u64,
    pub memory_expansion_linear_coeff: u64,
    
    // Storage costs
    pub storage_set: u64,
    pub storage_clear: u64,
    pub storage_clear_refund: u64,
    pub storage_load: u64,
    
    // Contract costs
    pub contract_creation: u64,
    pub contract_call: u64,
    pub contract_call_value: u64,
    pub contract_call_stipend: u64,
    pub contract_self_destruct: u64,
    pub contract_self_destruct_refund: u64,
    
    // Cryptographic costs
    pub sha256_base: u64,
    pub sha256_per_word: u64,
    pub keccak256_base: u64,
    pub keccak256_per_word: u64,
    pub ripemd160_base: u64,
    pub ripemd160_per_word: u64,
    pub ec_recover: u64,
    pub ec_add: u64,
    pub ec_mul: u64,
    pub ec_pairing_base: u64,
    pub ec_pairing_per_point: u64,
    
    // Machine learning costs
    pub ml_inference_base: u64,
    pub ml_inference_per_param: u64,
    pub ml_training_base: u64,
    pub ml_training_per_epoch: u64,
    pub ml_data_processing: u64,
    
    // System costs
    pub log_base: u64,
    pub log_per_topic: u64,
    pub log_per_byte: u64,
    pub copy_per_word: u64,
    pub jump: u64,
    pub jump_conditional: u64,
    
    // Blockchain costs
    pub block_hash: u64,
    pub balance: u64,
    pub ext_code_size: u64,
    pub ext_code_copy: u64,
    pub ext_code_hash: u64,
    
    // Transaction costs
    pub tx_base: u64,
    pub tx_data_zero: u64,
    pub tx_data_non_zero: u64,
    pub tx_create: u64,
}

/// Gas meter for tracking gas usage during execution
#[derive(Debug, Clone)]
pub struct GasMeter {
    gas_limit: u64,
    gas_used: u64,
    gas_refund: u64,
    costs: GasCosts,
    memory_size: u64,
    call_stack_depth: u32,
}

/// Gas price oracle for dynamic pricing
#[derive(Debug, Clone)]
pub struct GasPriceOracle {
    base_fee: U256,
    priority_fees: HashMap<GasPriceTier, U256>,
    network_congestion: f64, // 0.0 to 1.0
    block_utilization: f64,  // 0.0 to 1.0
    historical_prices: Vec<U256>,
}

/// Gas estimation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasEstimate {
    pub estimated_gas: u64,
    pub gas_price_tiers: HashMap<GasPriceTier, U256>,
    pub execution_time_estimate: u64, // in milliseconds
    pub confidence: f64, // 0.0 to 1.0
}

/// Gas optimization suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasOptimization {
    pub current_gas: u64,
    pub optimized_gas: u64,
    pub savings: u64,
    pub suggestions: Vec<String>,
}

impl Default for GasCosts {
    fn default() -> Self {
        Self {
            // Basic operations (EIP-150 compliant)
            base: 2,
            very_low: 3,
            low: 5,
            mid: 8,
            high: 10,
            
            // Memory operations
            memory: 3,
            memory_expansion_quad_divisor: 512,
            memory_expansion_linear_coeff: 3,
            
            // Storage operations (EIP-2929 compliant)
            storage_set: 20000,
            storage_clear: 5000,
            storage_clear_refund: 4800,
            storage_load: 2100,
            
            // Contract operations
            contract_creation: 32000,
            contract_call: 2100,
            contract_call_value: 9000,
            contract_call_stipend: 2300,
            contract_self_destruct: 5000,
            contract_self_destruct_refund: 24000,
            
            // Cryptographic operations
            sha256_base: 60,
            sha256_per_word: 12,
            keccak256_base: 30,
            keccak256_per_word: 6,
            ripemd160_base: 600,
            ripemd160_per_word: 120,
            ec_recover: 3000,
            ec_add: 500,
            ec_mul: 40000,
            ec_pairing_base: 45000,
            ec_pairing_per_point: 34000,
            
            // Machine learning operations (Augustium-specific)
            ml_inference_base: 10000,
            ml_inference_per_param: 10,
            ml_training_base: 100000,
            ml_training_per_epoch: 50000,
            ml_data_processing: 100,
            
            // System operations
            log_base: 375,
            log_per_topic: 375,
            log_per_byte: 8,
            copy_per_word: 3,
            jump: 8,
            jump_conditional: 10,
            
            // Blockchain operations
            block_hash: 20,
            balance: 2100,
            ext_code_size: 2600,
            ext_code_copy: 2600,
            ext_code_hash: 2600,
            
            // Transaction operations
            tx_base: 21000,
            tx_data_zero: 4,
            tx_data_non_zero: 16,
            tx_create: 32000,
        }
    }
}

impl GasMeter {
    /// Create a new gas meter
    pub fn new(gas_limit: u64) -> Self {
        Self {
            gas_limit,
            gas_used: 0,
            gas_refund: 0,
            costs: GasCosts::default(),
            memory_size: 0,
            call_stack_depth: 0,
        }
    }
    
    /// Create a gas meter with custom costs
    pub fn with_costs(gas_limit: u64, costs: GasCosts) -> Self {
        Self {
            gas_limit,
            gas_used: 0,
            gas_refund: 0,
            costs,
            memory_size: 0,
            call_stack_depth: 0,
        }
    }
    
    /// Consume gas for an operation
    pub fn consume_gas(&mut self, category: GasCategory, amount: u64) -> Result<()> {
        let gas_cost = self.calculate_gas_cost(category, amount)?;
        
        if self.gas_used + gas_cost > self.gas_limit {
            return Err(CompilerError::InternalError("Out of gas".to_string()));
        }
        
        self.gas_used += gas_cost;
        Ok(())
    }
    
    /// Calculate gas cost for a specific operation
    pub fn calculate_gas_cost(&self, category: GasCategory, amount: u64) -> Result<u64> {
        let base_cost = match category {
            GasCategory::Base => self.costs.base,
            GasCategory::VeryLow => self.costs.very_low,
            GasCategory::Low => self.costs.low,
            GasCategory::Mid => self.costs.mid,
            GasCategory::High => self.costs.high,
            
            GasCategory::Memory => self.calculate_memory_cost(amount)?,
            GasCategory::MemoryExpansion => self.calculate_memory_expansion_cost(amount)?,
            
            GasCategory::StorageSet => self.costs.storage_set,
            GasCategory::StorageClear => self.costs.storage_clear,
            GasCategory::StorageLoad => self.costs.storage_load,
            
            GasCategory::ContractCreation => self.costs.contract_creation,
            GasCategory::ContractCall => self.calculate_call_cost(amount)?,
            GasCategory::ContractSelfDestruct => self.costs.contract_self_destruct,
            
            GasCategory::Sha256 => self.costs.sha256_base + (amount / 32) * self.costs.sha256_per_word,
            GasCategory::Keccak256 => self.costs.keccak256_base + (amount / 32) * self.costs.keccak256_per_word,
            GasCategory::Ripemd160 => self.costs.ripemd160_base + (amount / 32) * self.costs.ripemd160_per_word,
            GasCategory::EcRecover => self.costs.ec_recover,
            GasCategory::EcAdd => self.costs.ec_add,
            GasCategory::EcMul => self.costs.ec_mul,
            GasCategory::EcPairing => self.costs.ec_pairing_base + amount * self.costs.ec_pairing_per_point,
            
            GasCategory::MLInference => self.costs.ml_inference_base + amount * self.costs.ml_inference_per_param,
            GasCategory::MLTraining => self.costs.ml_training_base + amount * self.costs.ml_training_per_epoch,
            GasCategory::MLDataProcessing => amount * self.costs.ml_data_processing,
            
            GasCategory::Log => self.costs.log_base + amount * self.costs.log_per_byte,
            GasCategory::Copy => (amount / 32) * self.costs.copy_per_word,
            GasCategory::Jump => self.costs.jump,
            GasCategory::JumpConditional => self.costs.jump_conditional,
            
            GasCategory::BlockHash => self.costs.block_hash,
            GasCategory::Balance => self.costs.balance,
            GasCategory::ExtCodeSize => self.costs.ext_code_size,
            GasCategory::ExtCodeCopy => self.costs.ext_code_copy + (amount / 32) * self.costs.copy_per_word,
            GasCategory::ExtCodeHash => self.costs.ext_code_hash,
            
            GasCategory::TxDataZero => amount * self.costs.tx_data_zero,
            GasCategory::TxDataNonZero => amount * self.costs.tx_data_non_zero,
            GasCategory::TxCreate => self.costs.tx_create,
            
            GasCategory::Precompiled => amount, // Custom amount for precompiled contracts
        };
        
        Ok(base_cost)
    }
    
    /// Calculate memory expansion cost
    fn calculate_memory_cost(&self, size: u64) -> Result<u64> {
        if size <= self.memory_size {
            return Ok(0);
        }
        
        let new_cost = self.calculate_memory_expansion_cost(size)?;
        let old_cost = self.calculate_memory_expansion_cost(self.memory_size)?;
        
        Ok(new_cost - old_cost)
    }
    
    /// Calculate memory expansion cost using quadratic formula
    fn calculate_memory_expansion_cost(&self, size: u64) -> Result<u64> {
        let size_in_words = (size + 31) / 32;
        let linear_cost = size_in_words * self.costs.memory_expansion_linear_coeff;
        let quadratic_cost = (size_in_words * size_in_words) / self.costs.memory_expansion_quad_divisor;
        
        Ok(linear_cost + quadratic_cost)
    }
    
    /// Calculate call cost including value transfer and new account creation
    fn calculate_call_cost(&self, value: u64) -> Result<u64> {
        let mut cost = self.costs.contract_call;
        
        // Add cost for value transfer
        if value > 0 {
            cost += self.costs.contract_call_value;
        }
        
        // Add cost for deep call stack
        if self.call_stack_depth >= 1024 {
            return Err(CompilerError::InternalError("Call stack too deep".to_string()));
        }
        
        Ok(cost)
    }
    
    /// Add gas refund
    pub fn add_refund(&mut self, amount: u64) {
        self.gas_refund += amount;
    }
    
    /// Calculate final gas refund (capped at 1/5 of gas used)
    pub fn finalize_refund(&self) -> u64 {
        std::cmp::min(self.gas_refund, self.gas_used / 5)
    }
    
    /// Get remaining gas
    pub fn gas_remaining(&self) -> u64 {
        self.gas_limit - self.gas_used
    }
    
    /// Get gas used
    pub fn gas_used(&self) -> u64 {
        self.gas_used
    }
    
    /// Get gas limit
    pub fn gas_limit(&self) -> u64 {
        self.gas_limit
    }
    
    /// Update memory size
    pub fn update_memory_size(&mut self, new_size: u64) {
        if new_size > self.memory_size {
            self.memory_size = new_size;
        }
    }
    
    /// Increment call stack depth
    pub fn enter_call(&mut self) {
        self.call_stack_depth += 1;
    }
    
    /// Decrement call stack depth
    pub fn exit_call(&mut self) {
        if self.call_stack_depth > 0 {
            self.call_stack_depth -= 1;
        }
    }
}

impl GasPriceOracle {
    /// Create a new gas price oracle
    pub fn new(base_fee: U256) -> Self {
        let mut priority_fees = HashMap::new();
        priority_fees.insert(GasPriceTier::Slow, U256::new(1_000_000_000u64)); // 1 gwei
        priority_fees.insert(GasPriceTier::Standard, U256::new(2_000_000_000u64)); // 2 gwei
        priority_fees.insert(GasPriceTier::Fast, U256::new(5_000_000_000u64)); // 5 gwei
        priority_fees.insert(GasPriceTier::Instant, U256::new(10_000_000_000u64)); // 10 gwei
        
        Self {
            base_fee,
            priority_fees,
            network_congestion: 0.5,
            block_utilization: 0.5,
            historical_prices: Vec::new(),
        }
    }
    
    /// Get gas price for a specific tier
    pub fn get_gas_price(&self, tier: GasPriceTier) -> U256 {
        let priority_fee = self.priority_fees.get(&tier).cloned().unwrap_or(U256::zero());
        let congestion_multiplier = 1.0 + self.network_congestion;
        let adjusted_priority = U256::new((priority_fee.as_u64() as f64 * congestion_multiplier) as u64);
        
        self.base_fee + adjusted_priority
    }
    
    /// Update base fee based on block utilization (EIP-1559)
    pub fn update_base_fee(&mut self, block_gas_used: u64, block_gas_limit: u64) {
        let target_gas = block_gas_limit / 2;
        self.block_utilization = block_gas_used as f64 / block_gas_limit as f64;
        
        if block_gas_used > target_gas {
            // Increase base fee
            let increase = self.base_fee / U256::new(8); // 12.5% max increase
            let actual_increase = increase * U256::new(block_gas_used - target_gas) / U256::new(target_gas);
            self.base_fee = self.base_fee + actual_increase;
        } else if block_gas_used < target_gas {
            // Decrease base fee
            let decrease = self.base_fee / U256::new(8); // 12.5% max decrease
            let actual_decrease = decrease * U256::new(target_gas - block_gas_used) / U256::new(target_gas);
            self.base_fee = if self.base_fee > actual_decrease {
                self.base_fee - actual_decrease
            } else {
                U256::new(1) // Minimum base fee
            };
        }
        
        // Store historical price
        self.historical_prices.push(self.base_fee);
        if self.historical_prices.len() > 100 {
            self.historical_prices.remove(0);
        }
    }
    
    /// Update network congestion based on mempool size
    pub fn update_congestion(&mut self, mempool_size: usize, max_mempool_size: usize) {
        self.network_congestion = (mempool_size as f64 / max_mempool_size as f64).min(1.0);
    }
    
    /// Get gas price estimate with confidence intervals
    pub fn estimate_gas_price(&self, target_blocks: u32) -> HashMap<GasPriceTier, U256> {
        let mut estimates = HashMap::new();
        
        for &tier in &[GasPriceTier::Slow, GasPriceTier::Standard, GasPriceTier::Fast, GasPriceTier::Instant] {
            let base_price = self.get_gas_price(tier);
            
            // Adjust based on target confirmation time
            let time_multiplier = match (tier, target_blocks) {
                (GasPriceTier::Slow, _) => 0.8,
                (GasPriceTier::Standard, 1..=3) => 1.0,
                (GasPriceTier::Fast, 1) => 1.2,
                (GasPriceTier::Instant, 1) => 1.5,
                _ => 1.0,
            };
            
            let adjusted_price = U256::new((base_price.as_u64() as f64 * time_multiplier) as u64);
            estimates.insert(tier, adjusted_price);
        }
        
        estimates
    }
    
    /// Get historical gas price statistics
    pub fn get_price_statistics(&self) -> (U256, U256, U256) { // (min, avg, max)
        if self.historical_prices.is_empty() {
            return (self.base_fee, self.base_fee, self.base_fee);
        }
        
        let min = self.historical_prices.iter().min().cloned().unwrap_or(self.base_fee);
        let max = self.historical_prices.iter().max().cloned().unwrap_or(self.base_fee);
        let sum: u64 = self.historical_prices.iter().map(|p| p.as_u64()).sum();
        let avg = sum / self.historical_prices.len() as u64;
        let avg_u256 = U256::new(avg);
        
        (min, avg_u256, max)
    }
}

/// Gas estimation engine
pub struct GasEstimator {
    costs: GasCosts,
    oracle: GasPriceOracle,
}

impl GasEstimator {
    /// Create a new gas estimator
    pub fn new(base_fee: U256) -> Self {
        Self {
            costs: GasCosts::default(),
            oracle: GasPriceOracle::new(base_fee),
        }
    }
    
    /// Estimate gas for a transaction
    pub fn estimate_transaction_gas(&self, data: &[u8], to: Option<&[u8; 20]>) -> GasEstimate {
        let mut gas = self.costs.tx_base;
        
        // Add data costs
        for &byte in data {
            if byte == 0 {
                gas += self.costs.tx_data_zero;
            } else {
                gas += self.costs.tx_data_non_zero;
            }
        }
        
        // Add contract creation cost
        if to.is_none() {
            gas += self.costs.tx_create;
        }
        
        // Add buffer for execution (20%)
        gas = gas * 120 / 100;
        
        let gas_price_tiers = self.oracle.estimate_gas_price(3);
        
        GasEstimate {
            estimated_gas: gas,
            gas_price_tiers,
            execution_time_estimate: self.estimate_execution_time(gas),
            confidence: 0.85, // 85% confidence
        }
    }
    
    /// Estimate execution time based on gas usage
    fn estimate_execution_time(&self, gas: u64) -> u64 {
        // Rough estimate: 1 gas = 1 microsecond
        gas / 1000 // Convert to milliseconds
    }
    
    /// Analyze gas usage and provide optimization suggestions
    pub fn analyze_gas_usage(&self, operations: &[(GasCategory, u64)]) -> GasOptimization {
        let mut current_gas = 0u64;
        let mut suggestions = Vec::new();
        
        for &(category, amount) in operations {
            let cost = self.costs.calculate_gas_cost(category, amount).unwrap_or(0);
            current_gas += cost;
            
            // Provide optimization suggestions
            match category {
                GasCategory::StorageSet if cost > 15000 => {
                    suggestions.push("Consider batching storage operations to reduce gas costs".to_string());
                }
                GasCategory::ContractCall if amount > 0 => {
                    suggestions.push("Avoid unnecessary value transfers in contract calls".to_string());
                }
                GasCategory::MLTraining if cost > 100000 => {
                    suggestions.push("Consider off-chain ML training with on-chain inference only".to_string());
                }
                GasCategory::Log if amount > 1000 => {
                    suggestions.push("Reduce log data size to minimize gas costs".to_string());
                }
                _ => {}
            }
        }
        
        // Calculate potential savings (simplified)
        let optimized_gas = current_gas * 85 / 100; // Assume 15% optimization potential
        let savings = current_gas - optimized_gas;
        
        if suggestions.is_empty() {
            suggestions.push("Gas usage is already optimized".to_string());
        }
        
        GasOptimization {
            current_gas,
            optimized_gas,
            savings,
            suggestions,
        }
    }
}

impl GasCosts {
    /// Calculate gas cost for a specific operation with amount
    pub fn calculate_gas_cost(&self, category: GasCategory, amount: u64) -> Result<u64> {
        let meter = GasMeter::with_costs(u64::MAX, self.clone());
        meter.calculate_gas_cost(category, amount)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gas_meter_basic() {
        let mut meter = GasMeter::new(100000);
        
        assert!(meter.consume_gas(GasCategory::Base, 1).is_ok());
        assert_eq!(meter.gas_used(), 2); // Base cost is 2
        assert_eq!(meter.gas_remaining(), 99998);
    }
    
    #[test]
    fn test_gas_meter_out_of_gas() {
        let mut meter = GasMeter::new(10);
        
        // This should fail due to insufficient gas
        assert!(meter.consume_gas(GasCategory::StorageSet, 1).is_err());
    }
    
    #[test]
    fn test_gas_price_oracle() {
        let mut oracle = GasPriceOracle::new(U256::new(1_000_000_000u64));
        
        let slow_price = oracle.get_gas_price(GasPriceTier::Slow);
        let fast_price = oracle.get_gas_price(GasPriceTier::Fast);
        
        assert!(fast_price > slow_price);
        
        // Test base fee update
        oracle.update_base_fee(15_000_000, 30_000_000); // 50% utilization
        oracle.update_base_fee(20_000_000, 30_000_000); // 66% utilization (should increase base fee)
        
        let new_slow_price = oracle.get_gas_price(GasPriceTier::Slow);
        assert!(new_slow_price >= slow_price); // Base fee should have increased
    }
    
    #[test]
    fn test_gas_estimation() {
        let estimator = GasEstimator::new(U256::new(1_000_000_000u64));
        
        let data = vec![0, 1, 2, 3, 0, 0]; // Mix of zero and non-zero bytes
        let estimate = estimator.estimate_transaction_gas(&data, Some(&[0; 20]));
        
        assert!(estimate.estimated_gas > 21000); // Should be more than base transaction cost
        assert!(estimate.confidence > 0.0);
        assert!(!estimate.gas_price_tiers.is_empty());
    }
    
    #[test]
    fn test_memory_expansion_cost() {
        let meter = GasMeter::new(1_000_000);
        
        let cost_32 = meter.calculate_memory_expansion_cost(32).unwrap();
        let cost_64 = meter.calculate_memory_expansion_cost(64).unwrap();
        
        assert!(cost_64 > cost_32); // Larger memory should cost more
    }
    
    #[test]
    fn test_gas_optimization_analysis() {
        let estimator = GasEstimator::new(U256::new(1_000_000_000u64));
        
        let operations = vec![
            (GasCategory::StorageSet, 1),
            (GasCategory::ContractCall, 1000),
            (GasCategory::Log, 2000),
        ];
        
        let optimization = estimator.analyze_gas_usage(&operations);
        
        assert!(optimization.current_gas > 0);
        assert!(optimization.optimized_gas <= optimization.current_gas);
        assert!(!optimization.suggestions.is_empty());
    }
}