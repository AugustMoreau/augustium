// Cross-chain bridge support
// Lets contracts talk to other blockchains like Ethereum, BSC, Polygon etc.

use crate::error::{CompilerError, SemanticError, SemanticErrorKind, SourceLocation};
use crate::evm_compat::{EvmTransaction};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Supported blockchain networks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u64)]
pub enum ChainId {
    Ethereum = 1,
    BinanceSmartChain = 56,
    Polygon = 137,
    Avalanche = 43114,
    Fantom = 250,
    Arbitrum = 42161,
    Optimism = 10,
    Custom(u64),
}

impl ChainId {
    /// Get the numeric chain ID
    #[allow(dead_code)]
    pub fn as_u64(&self) -> u64 {
        match self {
            ChainId::Ethereum => 1,
            ChainId::BinanceSmartChain => 56,
            ChainId::Polygon => 137,
            ChainId::Avalanche => 43114,
            ChainId::Fantom => 250,
            ChainId::Arbitrum => 42161,
            ChainId::Optimism => 10,
            ChainId::Custom(id) => *id,
        }
    }
    
    /// Get the chain name
    #[allow(dead_code)]
    pub fn name(&self) -> &'static str {
        match self {
            ChainId::Ethereum => "Ethereum",
            ChainId::BinanceSmartChain => "Binance Smart Chain",
            ChainId::Polygon => "Polygon",
            ChainId::Avalanche => "Avalanche",
            ChainId::Fantom => "Fantom",
            ChainId::Arbitrum => "Arbitrum",
            ChainId::Optimism => "Optimism",
            ChainId::Custom(_) => "Custom Chain",
        }
    }
    
    /// Get the native token symbol
    #[allow(dead_code)]
    pub fn native_token(&self) -> &'static str {
        match self {
            ChainId::Ethereum => "ETH",
            ChainId::BinanceSmartChain => "BNB",
            ChainId::Polygon => "MATIC",
            ChainId::Avalanche => "AVAX",
            ChainId::Fantom => "FTM",
            ChainId::Arbitrum => "ETH",
            ChainId::Optimism => "ETH",
            ChainId::Custom(_) => "UNKNOWN",
        }
    }
}

/// Cross-chain bridge configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// Source chain configuration
    pub source_chain: ChainConfig,
    /// Target chain configuration
    pub target_chain: ChainConfig,
    /// Bridge contract addresses
    pub bridge_contracts: HashMap<ChainId, String>,
    /// Supported tokens for bridging
    pub supported_tokens: Vec<TokenConfig>,
    /// Bridge fees configuration
    pub fees: FeeConfig,
    /// Security settings
    pub security: SecurityConfig,
}

/// Chain-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainConfig {
    pub chain_id: ChainId,
    pub rpc_url: String,
    pub explorer_url: String,
    pub gas_price: u64,
    pub gas_limit: u64,
    pub block_time: u64, // in seconds
    pub confirmations: u32,
}

/// Token configuration for cross-chain transfers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenConfig {
    pub symbol: String,
    pub name: String,
    pub decimals: u8,
    pub addresses: HashMap<ChainId, String>,
    pub min_transfer: u64,
    pub max_transfer: u64,
}

/// Bridge fee configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeConfig {
    pub base_fee: u64,
    pub percentage_fee: f64, // 0.1 = 0.1%
    pub gas_multiplier: f64,
    pub relayer_fee: u64,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub require_confirmations: u32,
    pub max_daily_volume: u64,
    pub enable_rate_limiting: bool,
    pub trusted_relayers: Vec<String>,
    pub emergency_pause: bool,
}

/// Cross-chain message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossChainMessage {
    TokenTransfer {
        token: String,
        amount: u64,
        recipient: String,
        source_chain: ChainId,
        target_chain: ChainId,
    },
    ContractCall {
        target_contract: String,
        function_data: Vec<u8>,
        gas_limit: u64,
        source_chain: ChainId,
        target_chain: ChainId,
    },
    StateSync {
        state_root: String,
        proof: Vec<u8>,
        source_chain: ChainId,
        target_chain: ChainId,
    },
}

/// Bridge transaction status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BridgeStatus {
    Pending,
    Confirmed,
    Executed,
    Failed,
    Cancelled,
}

/// Bridge transaction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeTransaction {
    pub id: String,
    pub message: CrossChainMessage,
    pub status: BridgeStatus,
    pub source_tx_hash: Option<String>,
    pub target_tx_hash: Option<String>,
    pub created_at: u64,
    pub confirmed_at: Option<u64>,
    pub executed_at: Option<u64>,
    pub fees_paid: u64,
    pub relayer: Option<String>,
}

/// Cross-chain bridge implementation
#[allow(dead_code)]
pub struct CrossChainBridge {
    config: BridgeConfig,
    pending_transactions: HashMap<String, BridgeTransaction>,
    validators: Vec<String>,
    relayers: Vec<String>,
}

impl CrossChainBridge {
    /// Create a new cross-chain bridge
    #[allow(dead_code)]
    pub fn new(config: BridgeConfig) -> Self {
        Self {
            config,
            pending_transactions: HashMap::new(),
            validators: Vec::new(),
            relayers: Vec::new(),
        }
    }
    
    /// Initialize bridge with validators and relayers
    #[allow(dead_code)]
    pub fn initialize(
        &mut self,
        validators: Vec<String>,
        relayers: Vec<String>,
    ) -> Result<(), CompilerError> {
        self.validators = validators;
        self.relayers = relayers;
        Ok(())
    }
    
    /// Submit a cross-chain message
    #[allow(dead_code)]
    pub fn submit_message(
        &mut self,
        message: CrossChainMessage,
        _sender: &str,
    ) -> Result<String, CompilerError> {
        // Validate message
        self.validate_message(&message)?;
        
        // Calculate fees
        let fees = self.calculate_fees(&message)?;
        
        // Create transaction record
        let tx_id = self.generate_transaction_id(&message);
        let bridge_tx = BridgeTransaction {
            id: tx_id.clone(),
            message,
            status: BridgeStatus::Pending,
            source_tx_hash: None,
            target_tx_hash: None,
            created_at: self.current_timestamp(),
            confirmed_at: None,
            executed_at: None,
            fees_paid: fees,
            relayer: None,
        };
        
        self.pending_transactions.insert(tx_id.clone(), bridge_tx);
        
        // Emit bridge event
        self.emit_bridge_event(&tx_id, "MessageSubmitted")?;
        
        Ok(tx_id)
    }
    
    /// Confirm a cross-chain transaction
    #[allow(dead_code)]
    pub fn confirm_transaction(
        &mut self,
        tx_id: &str,
        source_tx_hash: &str,
        validator: &str,
    ) -> Result<(), CompilerError> {
        // Verify validator
        if !self.validators.contains(&validator.to_string()) {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::UnauthorizedAccess,
                location: SourceLocation::default(),
                message: "Unauthorized validator".to_string(),
            }));
        }
        
        let ts_now = self.current_timestamp();
        if let Some(tx) = self.pending_transactions.get_mut(tx_id) {
            tx.status = BridgeStatus::Confirmed;
            tx.source_tx_hash = Some(source_tx_hash.to_string());
            tx.confirmed_at = Some(ts_now);
            
            self.emit_bridge_event(tx_id, "TransactionConfirmed")?;
        } else {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::default(),
                message: "Transaction not found".to_string(),
            }));
        }
        
        Ok(())
    }
    
    /// Execute a cross-chain transaction
    #[allow(dead_code)]
    pub fn execute_transaction(
        &mut self,
        tx_id: &str,
        relayer: &str,
    ) -> Result<EvmTransaction, CompilerError> {
        // Verify relayer
        if !self.relayers.contains(&relayer.to_string()) {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::UnauthorizedAccess,
                location: SourceLocation::default(),
                message: "Unauthorized relayer".to_string(),
            }));
        }
        
        // Clone message first so we can drop immutable borrow before mut borrow
        let msg = if let Some(tx) = self.pending_transactions.get(tx_id) {
            tx.message.clone()
        } else {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::default(),
                message: "Transaction not found".to_string(),
            }));
        };

        // Pre-compute timestamp and execution tx before mutable borrow
        let ts_now = self.current_timestamp();
        let execution_tx = self.generate_execution_transaction(&msg)?;

        let tx = self.pending_transactions.get_mut(tx_id)
            .ok_or_else(|| CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::default(),
                message: "Transaction not found".to_string(),
            }))?;
        
        // Check if transaction is confirmed
        if tx.status != BridgeStatus::Confirmed {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidStateTransition,
                location: SourceLocation::default(),
                message: "Transaction not confirmed".to_string(),
            }));
        }
        
        // Update transaction status
        tx.executed_at = Some(ts_now);
        tx.relayer = Some(relayer.to_string());
        tx.status = BridgeStatus::Executed;
        
        self.emit_bridge_event(tx_id, "TransactionExecuted")?;
        
        Ok(execution_tx)
    }
    
    /// Get transaction status
    #[allow(dead_code)]
    pub fn get_transaction_status(&self, tx_id: &str) -> Option<&BridgeTransaction> {
        self.pending_transactions.get(tx_id)
    }
    
    /// List pending transactions
    #[allow(dead_code)]
    pub fn list_pending_transactions(&self) -> Vec<&BridgeTransaction> {
        self.pending_transactions.values()
            .filter(|tx| tx.status == BridgeStatus::Pending || tx.status == BridgeStatus::Confirmed)
            .collect()
    }
    
    /// Validate cross-chain message
    #[allow(dead_code)]
    fn validate_message(&self, message: &CrossChainMessage) -> Result<(), CompilerError> {
        match message {
            CrossChainMessage::TokenTransfer { token, amount, source_chain, target_chain, .. } => {
                // Check if token is supported
                let token_config = self.config.supported_tokens.iter()
                    .find(|t| t.symbol == *token)
                    .ok_or_else(|| CompilerError::SemanticError(SemanticError {
                        kind: SemanticErrorKind::InvalidOperation,
                        location: SourceLocation::default(),
                        message: format!("Unsupported token: {}", token),
                    }))?;
                
                // Check transfer limits
                if *amount < token_config.min_transfer || *amount > token_config.max_transfer {
                    return Err(CompilerError::SemanticError(SemanticError {
                        kind: SemanticErrorKind::InvalidArguments,
                        location: SourceLocation::default(),
                        message: "Transfer amount outside allowed limits".to_string(),
                    }));
                }
                
                // Check if chains are supported
                if !token_config.addresses.contains_key(source_chain) ||
                   !token_config.addresses.contains_key(target_chain) {
                    return Err(CompilerError::SemanticError(SemanticError {
                        kind: SemanticErrorKind::InvalidOperation,
                        location: SourceLocation::default(),
                        message: "Token not available on specified chains".to_string(),
                    }));
                }
            }
            CrossChainMessage::ContractCall { gas_limit, .. } => {
                // Validate gas limit
                if *gas_limit > 10_000_000 {
                    return Err(CompilerError::SemanticError(SemanticError {
                        kind: SemanticErrorKind::InvalidArguments,
                        location: SourceLocation::default(),
                        message: "Gas limit too high".to_string(),
                    }));
                }
            }
            CrossChainMessage::StateSync { .. } => {
                // Validate state sync (simplified)
                // In real implementation, verify merkle proofs
            }
        }
        
        Ok(())
    }
    
    /// Calculate bridge fees
    #[allow(dead_code)]
    fn calculate_fees(&self, message: &CrossChainMessage) -> Result<u64, CompilerError> {
        let base_fee = self.config.fees.base_fee;
        let relayer_fee = self.config.fees.relayer_fee;
        
        let amount_fee = match message {
            CrossChainMessage::TokenTransfer { amount, .. } => {
                (*amount as f64 * self.config.fees.percentage_fee / 100.0) as u64
            }
            _ => 0,
        };
        
        Ok(base_fee + relayer_fee + amount_fee)
    }
    
    /// Generate execution transaction for target chain
    #[allow(dead_code)]
    fn generate_execution_transaction(
        &self,
        message: &CrossChainMessage,
    ) -> Result<EvmTransaction, CompilerError> {
        match message {
            CrossChainMessage::TokenTransfer { token, amount, recipient, target_chain, .. } => {
                let bridge_contract = self.config.bridge_contracts.get(target_chain)
                    .ok_or_else(|| CompilerError::SemanticError(SemanticError {
                        kind: SemanticErrorKind::InvalidOperation,
                        location: SourceLocation::default(),
                        message: "Bridge contract not found for target chain".to_string(),
                    }))?;
                
                // Generate mint/unlock transaction data
                let mut data = Vec::new();
                data.extend_from_slice(&self.encode_function_selector("mintTokens"));
                data.extend_from_slice(&self.encode_address(recipient)?); 
                data.extend_from_slice(&self.encode_uint256(*amount));
                data.extend_from_slice(&self.encode_string(token));
                
                Ok(EvmTransaction {
                    to: Some(bridge_contract.clone()),
                    value: 0,
                    gas_limit: 200_000,
                    gas_price: 20_000_000_000,
                    data,
                    chain_id: target_chain.as_u64(),
                })
            }
            CrossChainMessage::ContractCall { target_contract, function_data, gas_limit, target_chain, .. } => {
                Ok(EvmTransaction {
                    to: Some(target_contract.clone()),
                    value: 0,
                    gas_limit: *gas_limit,
                    gas_price: 20_000_000_000,
                    data: function_data.clone(),
                    chain_id: target_chain.as_u64(),
                })
            }
            CrossChainMessage::StateSync { .. } => {
                // Generate state sync transaction
                Ok(EvmTransaction {
                    to: None,
                    value: 0,
                    gas_limit: 100_000,
                    gas_price: 20_000_000_000,
                    data: vec![],
                    chain_id: 1,
                })
            }
        }
    }
    
    /// Generate unique transaction ID
    #[allow(dead_code)]
    fn generate_transaction_id(&self, _message: &CrossChainMessage) -> String {
        // Simple ID generation (in real implementation, use proper hash)
        format!("bridge_tx_{}", self.current_timestamp())
    }
    
    /// Get current timestamp
    #[allow(dead_code)]
    fn current_timestamp(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
    
    /// Emit bridge event
    #[allow(dead_code)]
    fn emit_bridge_event(&self, tx_id: &str, event_type: &str) -> Result<(), CompilerError> {
        // In real implementation, emit blockchain events
        println!("Bridge Event: {} - {}", event_type, tx_id);
        Ok(())
    }
    
    /// Encode function selector (first 4 bytes of keccak256)
    #[allow(dead_code)]
    fn encode_function_selector(&self, function_name: &str) -> [u8; 4] {
        // Simplified encoding
        let hash = self.simple_hash(function_name.as_bytes());
        [hash[0], hash[1], hash[2], hash[3]]
    }
    
    /// Encode Ethereum address
    #[allow(dead_code)]
    fn encode_address(&self, address: &str) -> Result<Vec<u8>, CompilerError> {
        if address.len() != 42 || !address.starts_with("0x") {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidArguments,
                location: SourceLocation::default(),
                message: "Invalid address format".to_string(),
            }));
        }
        
        // Convert hex string to bytes (simplified)
        let mut bytes = vec![0u8; 32]; // Padded to 32 bytes
        for (i, chunk) in address[2..].chars().collect::<Vec<_>>().chunks(2).enumerate() {
            if i < 20 {
                let hex_str: String = chunk.iter().collect();
                if let Ok(byte) = u8::from_str_radix(&hex_str, 16) {
                    bytes[12 + i] = byte; // Address starts at byte 12
                }
            }
        }
        
        Ok(bytes)
    }
    
    /// Encode uint256
    #[allow(dead_code)]
    fn encode_uint256(&self, value: u64) -> Vec<u8> {
        let mut bytes = vec![0u8; 32];
        let value_bytes = value.to_be_bytes();
        bytes[24..32].copy_from_slice(&value_bytes);
        bytes
    }
    
    /// Encode string
    #[allow(dead_code)]
    fn encode_string(&self, s: &str) -> Vec<u8> {
        let mut encoded = Vec::new();
        
        // String offset (simplified)
        encoded.extend_from_slice(&self.encode_uint256(32));
        
        // String length
        encoded.extend_from_slice(&self.encode_uint256(s.len() as u64));
        
        // String data (padded to 32-byte boundary)
        let mut data = s.as_bytes().to_vec();
        while data.len() % 32 != 0 {
            data.push(0);
        }
        encoded.extend_from_slice(&data);
        
        encoded
    }
    
    /// Simple hash function
    #[allow(dead_code)]
    fn simple_hash(&self, data: &[u8]) -> Vec<u8> {
        let mut hash = vec![0u8; 32];
        for (i, &byte) in data.iter().enumerate() {
            hash[i % 32] ^= byte;
        }
        hash
    }
}

/// Cross-chain bridge factory
#[allow(dead_code)]
pub struct BridgeFactory;

impl BridgeFactory {
    /// Create a bridge configuration for popular chain pairs
    #[allow(dead_code)]
    pub fn create_eth_bsc_bridge() -> BridgeConfig {
        let mut bridge_contracts = HashMap::new();
        bridge_contracts.insert(ChainId::Ethereum, "0x1234567890123456789012345678901234567890".to_string());
        bridge_contracts.insert(ChainId::BinanceSmartChain, "0x0987654321098765432109876543210987654321".to_string());
        
        let mut supported_tokens = Vec::new();
        
        // USDT configuration
        let mut usdt_addresses = HashMap::new();
        usdt_addresses.insert(ChainId::Ethereum, "0xdAC17F958D2ee523a2206206994597C13D831ec7".to_string());
        usdt_addresses.insert(ChainId::BinanceSmartChain, "0x55d398326f99059fF775485246999027B3197955".to_string());
        
        supported_tokens.push(TokenConfig {
            symbol: "USDT".to_string(),
            name: "Tether USD".to_string(),
            decimals: 6,
            addresses: usdt_addresses,
            min_transfer: 1_000_000, // 1 USDT
            max_transfer: 1_000_000_000_000, // 1M USDT
        });
        
        BridgeConfig {
            source_chain: ChainConfig {
                chain_id: ChainId::Ethereum,
                rpc_url: "https://mainnet.infura.io/v3/YOUR_PROJECT_ID".to_string(),
                explorer_url: "https://etherscan.io".to_string(),
                gas_price: 20_000_000_000,
                gas_limit: 21_000,
                block_time: 15,
                confirmations: 12,
            },
            target_chain: ChainConfig {
                chain_id: ChainId::BinanceSmartChain,
                rpc_url: "https://bsc-dataseed1.binance.org".to_string(),
                explorer_url: "https://bscscan.com".to_string(),
                gas_price: 5_000_000_000,
                gas_limit: 21_000,
                block_time: 3,
                confirmations: 15,
            },
            bridge_contracts,
            supported_tokens,
            fees: FeeConfig {
                base_fee: 1_000_000, // 1 USDT
                percentage_fee: 0.1, // 0.1%
                gas_multiplier: 1.2,
                relayer_fee: 500_000, // 0.5 USDT
            },
            security: SecurityConfig {
                require_confirmations: 12,
                max_daily_volume: 10_000_000_000_000, // 10M USDT
                enable_rate_limiting: true,
                trusted_relayers: vec![
                    "0xRelayer1".to_string(),
                    "0xRelayer2".to_string(),
                ],
                emergency_pause: false,
            },
        }
    }
    
    /// Create a bridge configuration for Ethereum-Polygon
    #[allow(dead_code)]
    pub fn create_eth_polygon_bridge() -> BridgeConfig {
        let mut bridge_contracts = HashMap::new();
        bridge_contracts.insert(ChainId::Ethereum, "0xEthereumBridge".to_string());
        bridge_contracts.insert(ChainId::Polygon, "0xPolygonBridge".to_string());
        
        BridgeConfig {
            source_chain: ChainConfig {
                chain_id: ChainId::Ethereum,
                rpc_url: "https://mainnet.infura.io/v3/YOUR_PROJECT_ID".to_string(),
                explorer_url: "https://etherscan.io".to_string(),
                gas_price: 20_000_000_000,
                gas_limit: 21_000,
                block_time: 15,
                confirmations: 12,
            },
            target_chain: ChainConfig {
                chain_id: ChainId::Polygon,
                rpc_url: "https://polygon-rpc.com".to_string(),
                explorer_url: "https://polygonscan.com".to_string(),
                gas_price: 30_000_000_000,
                gas_limit: 21_000,
                block_time: 2,
                confirmations: 20,
            },
            bridge_contracts,
            supported_tokens: vec![],
            fees: FeeConfig {
                base_fee: 500_000,
                percentage_fee: 0.05,
                gas_multiplier: 1.1,
                relayer_fee: 250_000,
            },
            security: SecurityConfig {
                require_confirmations: 20,
                max_daily_volume: 5_000_000_000_000,
                enable_rate_limiting: true,
                trusted_relayers: vec![],
                emergency_pause: false,
            },
        }
    }
}

/// Cross-chain utilities
pub mod utils {
    use super::*;
    
    /// Check if two chains are compatible for bridging
    #[allow(dead_code)]
    pub fn are_chains_compatible(chain1: &ChainId, chain2: &ChainId) -> bool {
        // All EVM chains are compatible
        matches!((chain1, chain2), 
            (ChainId::Ethereum, _) | 
            (ChainId::BinanceSmartChain, _) |
            (ChainId::Polygon, _) |
            (ChainId::Avalanche, _) |
            (ChainId::Fantom, _) |
            (ChainId::Arbitrum, _) |
            (ChainId::Optimism, _) |
            (ChainId::Custom(_), _)
        )
    }
    
    /// Estimate bridge time between chains
    #[allow(dead_code)]
    pub fn estimate_bridge_time(source: &ChainId, _target: &ChainId) -> u64 {
        let source_confirmations = match source {
            ChainId::Ethereum => 12 * 15, // 12 blocks * 15 seconds
            ChainId::BinanceSmartChain => 15 * 3, // 15 blocks * 3 seconds
            ChainId::Polygon => 20 * 2, // 20 blocks * 2 seconds
            _ => 10 * 15, // Default
        };
        
        let target_execution = 60; // 1 minute for execution
        
        source_confirmations + target_execution
    }
    
    /// Calculate optimal gas price for chain
    #[allow(dead_code)]
    pub fn calculate_optimal_gas_price(chain: &ChainId) -> u64 {
        match chain {
            ChainId::Ethereum => 20_000_000_000, // 20 gwei
            ChainId::BinanceSmartChain => 5_000_000_000, // 5 gwei
            ChainId::Polygon => 30_000_000_000, // 30 gwei
            ChainId::Avalanche => 25_000_000_000, // 25 gwei
            ChainId::Fantom => 1_000_000_000, // 1 gwei
            ChainId::Arbitrum => 1_000_000_000, // 1 gwei
            ChainId::Optimism => 1_000_000_000, // 1 gwei
            ChainId::Custom(_) => 10_000_000_000, // 10 gwei default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_chain_id_conversion() {
        assert_eq!(ChainId::Ethereum.as_u64(), 1);
        assert_eq!(ChainId::BinanceSmartChain.as_u64(), 56);
        assert_eq!(ChainId::Polygon.as_u64(), 137);
    }
    
    #[test]
    fn test_chain_names() {
        assert_eq!(ChainId::Ethereum.name(), "Ethereum");
        assert_eq!(ChainId::BinanceSmartChain.name(), "Binance Smart Chain");
        assert_eq!(ChainId::Polygon.name(), "Polygon");
    }
    
    #[test]
    fn test_native_tokens() {
        assert_eq!(ChainId::Ethereum.native_token(), "ETH");
        assert_eq!(ChainId::BinanceSmartChain.native_token(), "BNB");
        assert_eq!(ChainId::Polygon.native_token(), "MATIC");
    }
    
    #[test]
    fn test_bridge_creation() {
        let config = BridgeFactory::create_eth_bsc_bridge();
        assert_eq!(config.source_chain.chain_id, ChainId::Ethereum);
        assert_eq!(config.target_chain.chain_id, ChainId::BinanceSmartChain);
        assert!(!config.supported_tokens.is_empty());
    }
    
    #[test]
    fn test_cross_chain_message() {
        let message = CrossChainMessage::TokenTransfer {
            token: "USDT".to_string(),
            amount: 1000000,
            recipient: "0x1234567890123456789012345678901234567890".to_string(),
            source_chain: ChainId::Ethereum,
            target_chain: ChainId::BinanceSmartChain,
        };
        
        match message {
            CrossChainMessage::TokenTransfer { token, amount, .. } => {
                assert_eq!(token, "USDT");
                assert_eq!(amount, 1000000);
            }
            _ => panic!("Wrong message type"),
        }
    }
    
    #[test]
    fn test_bridge_transaction_lifecycle() {
        let config = BridgeFactory::create_eth_bsc_bridge();
        let mut bridge = CrossChainBridge::new(config);
        
        let validators = vec!["validator1".to_string()];
        let relayers = vec!["relayer1".to_string()];
        bridge.initialize(validators, relayers).unwrap();
        
        let message = CrossChainMessage::TokenTransfer {
            token: "USDT".to_string(),
            amount: 1000000,
            recipient: "0x1234567890123456789012345678901234567890".to_string(),
            source_chain: ChainId::Ethereum,
            target_chain: ChainId::BinanceSmartChain,
        };
        
        let tx_id = bridge.submit_message(message, "sender").unwrap();
        assert!(bridge.get_transaction_status(&tx_id).is_some());
        
        bridge.confirm_transaction(&tx_id, "0xsourcetx", "validator1").unwrap();
        let tx = bridge.get_transaction_status(&tx_id).unwrap();
        assert_eq!(tx.status, BridgeStatus::Confirmed);
    }
    
    #[test]
    fn test_chain_compatibility() {
        assert!(utils::are_chains_compatible(&ChainId::Ethereum, &ChainId::BinanceSmartChain));
        assert!(utils::are_chains_compatible(&ChainId::Polygon, &ChainId::Avalanche));
    }
    
    #[test]
    fn test_bridge_time_estimation() {
        let time = utils::estimate_bridge_time(&ChainId::Ethereum, &ChainId::BinanceSmartChain);
        assert!(time > 0);
    }
}