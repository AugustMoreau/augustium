//! Events Module
//!
//! This module provides utilities for defining, emitting, and handling events
//! in smart contracts, including event logging and filtering capabilities.

use crate::error::Result;
use crate::stdlib::core_types::{U256, AugustiumType};
use crate::stdlib::address::Address as ContractAddress;
use serde::{Serialize, Deserialize};

/// Event topic type (32 bytes)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventTopic {
    bytes: [u8; 32],
}

impl EventTopic {
    /// Create a new event topic from 32 bytes
    pub fn new(bytes: [u8; 32]) -> Self {
        Self { bytes }
    }
    
    /// Create an event topic from a string (hashed)
    pub fn from_string(topic: &str) -> Self {
        let hash = keccak256(topic.as_bytes());
        Self { bytes: hash }
    }
    
    /// Create an event topic from a U256
    pub fn from_u256(value: U256) -> Self {
        Self {
            bytes: {
                let bytes_vec = value.to_bytes();
                let mut bytes_array = [0u8; 32];
                bytes_array.copy_from_slice(&bytes_vec);
                bytes_array
            },
        }
    }
    
    /// Get the raw bytes of the topic
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }
    
    /// Convert to hex string
    pub fn to_hex(&self) -> String {
        format!("0x{}", hex::encode(&self.bytes))
    }
}

/// Event data that can be logged
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EventData {
    /// Raw bytes
    Bytes(Vec<u8>),
    /// String data
    String(String),
    /// 256-bit unsigned integer
    U256(U256),
    /// Boolean value
    Bool(bool),
    /// Address value
    Address(ContractAddress),
    /// Array of values
    Array(Vec<EventData>),
}

impl EventData {
    /// Convert to bytes for logging
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            EventData::Bytes(bytes) => bytes.clone(),
            EventData::String(s) => s.as_bytes().to_vec(),
            EventData::U256(val) => val.to_bytes(),
            EventData::Bool(val) => vec![if *val { 1 } else { 0 }],
            EventData::Address(addr) => addr.as_bytes().to_vec(),
            EventData::Array(arr) => {
                let mut result = Vec::new();
                for item in arr {
                    result.extend(item.to_bytes());
                }
                result
            }
        }
    }
    
    /// Get the size in bytes
    pub fn size(&self) -> usize {
        self.to_bytes().len()
    }
}

/// Event log entry
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventLog {
    /// Contract address that emitted the event
    pub address: ContractAddress,
    /// Event topics (indexed parameters)
    pub topics: Vec<EventTopic>,
    /// Event data (non-indexed parameters)
    pub data: EventData,
    /// Block number where the event was emitted
    pub block_number: u64,
    /// Transaction hash
    pub transaction_hash: [u8; 32],
    /// Log index within the transaction
    pub log_index: u32,
}

impl EventLog {
    /// Create a new event log
    pub fn new(
        address: ContractAddress,
        topics: Vec<EventTopic>,
        data: EventData,
        block_number: u64,
        transaction_hash: [u8; 32],
        log_index: u32,
    ) -> Self {
        Self {
            address,
            topics,
            data,
            block_number,
            transaction_hash,
            log_index,
        }
    }
    
    /// Get the event signature (first topic)
    pub fn event_signature(&self) -> Option<&EventTopic> {
        self.topics.first()
    }
    
    /// Get indexed parameters (topics excluding signature)
    pub fn indexed_params(&self) -> &[EventTopic] {
        if self.topics.is_empty() {
            &[]
        } else {
            &self.topics[1..]
        }
    }
}

/// Event definition for a smart contract event
#[derive(Debug, Clone, PartialEq)]
pub struct EventDefinition {
    /// Event name
    pub name: String,
    /// Event signature hash
    pub signature: EventTopic,
    /// Parameter definitions
    pub parameters: Vec<EventParameter>,
}

/// Event parameter definition
#[derive(Debug, Clone, PartialEq)]
pub struct EventParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: EventParameterType,
    /// Whether the parameter is indexed
    pub indexed: bool,
}

/// Event parameter types
#[derive(Debug, Clone, PartialEq)]
pub enum EventParameterType {
    U256,
    Bool,
    Address,
    String,
    Bytes,
    Array(Box<EventParameterType>),
}

impl EventDefinition {
    /// Create a new event definition
    pub fn new(name: String, parameters: Vec<EventParameter>) -> Self {
        let signature = EventTopic::from_string(&Self::compute_signature(&name, &parameters));
        Self {
            name,
            signature,
            parameters,
        }
    }
    
    /// Compute the event signature string
    fn compute_signature(name: &str, parameters: &[EventParameter]) -> String {
        let param_types: Vec<String> = parameters
            .iter()
            .map(|p| p.param_type.to_string())
            .collect();
        format!("{}({})", name, param_types.join(","))
    }
    
    /// Get indexed parameters
    pub fn indexed_parameters(&self) -> Vec<&EventParameter> {
        self.parameters.iter().filter(|p| p.indexed).collect()
    }
    
    /// Get non-indexed parameters
    pub fn non_indexed_parameters(&self) -> Vec<&EventParameter> {
        self.parameters.iter().filter(|p| !p.indexed).collect()
    }
}

impl fmt::Display for EventParameterType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EventParameterType::U256 => write!(f, "uint256"),
            EventParameterType::Bool => write!(f, "bool"),
            EventParameterType::Address => write!(f, "address"),
            EventParameterType::String => write!(f, "string"),
            EventParameterType::Bytes => write!(f, "bytes"),
            EventParameterType::Array(inner) => write!(f, "{}[]", inner),
        }
    }
}

/// Event filter for querying events
#[derive(Debug, Clone, Default)]
pub struct EventFilter {
    /// Contract addresses to filter by
    pub addresses: Vec<ContractAddress>,
    /// Topics to filter by (None means any topic)
    pub topics: Vec<Option<EventTopic>>,
    /// Starting block number
    pub from_block: Option<u64>,
    /// Ending block number
    pub to_block: Option<u64>,
}

impl EventFilter {
    /// Create a new empty event filter
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add an address filter
    pub fn address(mut self, address: ContractAddress) -> Self {
        self.addresses.push(address);
        self
    }
    
    /// Add a topic filter
    pub fn topic(mut self, index: usize, topic: EventTopic) -> Self {
        // Extend topics vector if necessary
        while self.topics.len() <= index {
            self.topics.push(None);
        }
        self.topics[index] = Some(topic);
        self
    }
    
    /// Set block range
    pub fn block_range(mut self, from: u64, to: u64) -> Self {
        self.from_block = Some(from);
        self.to_block = Some(to);
        self
    }
    
    /// Check if an event log matches this filter
    pub fn matches(&self, log: &EventLog) -> bool {
        // Check address filter
        if !self.addresses.is_empty() && !self.addresses.contains(&log.address) {
            return false;
        }
        
        // Check topic filters
        for (i, topic_filter) in self.topics.iter().enumerate() {
            if let Some(expected_topic) = topic_filter {
                if log.topics.get(i) != Some(expected_topic) {
                    return false;
                }
            }
        }
        
        // Check block range
        if let Some(from_block) = self.from_block {
            if log.block_number < from_block {
                return false;
            }
        }
        
        if let Some(to_block) = self.to_block {
            if log.block_number > to_block {
                return false;
            }
        }
        
        true
    }
}

/// Event emitter for contracts
pub trait EventEmitter {
    /// Emit an event
    fn emit_event(
        &mut self,
        definition: &EventDefinition,
        indexed_data: Vec<EventData>,
        non_indexed_data: EventData,
    ) -> Result<()>;
    
    /// Get all emitted events
    fn get_events(&self) -> &[EventLog];
    
    /// Get events matching a filter
    fn get_events_filtered(&self, filter: &EventFilter) -> Vec<&EventLog>;
}

/// In-memory event store for testing
#[derive(Debug, Clone, Default)]
pub struct MemoryEventStore {
    events: Vec<EventLog>,
    current_block: u64,
    current_tx_hash: [u8; 32],
    log_index: u32,
}

impl MemoryEventStore {
    /// Create a new memory event store
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the current block number
    pub fn set_block(&mut self, block_number: u64) {
        self.current_block = block_number;
    }
    
    /// Set the current transaction hash
    pub fn set_transaction(&mut self, tx_hash: [u8; 32]) {
        self.current_tx_hash = tx_hash;
        self.log_index = 0;
    }
}

impl EventEmitter for MemoryEventStore {
    fn emit_event(
        &mut self,
        definition: &EventDefinition,
        indexed_data: Vec<EventData>,
        non_indexed_data: EventData,
    ) -> Result<()> {
        // Create topics
        let mut topics = vec![definition.signature.clone()];
        
        // Add indexed parameters as topics
        for data in indexed_data {
            let topic_bytes = keccak256(&data.to_bytes());
            topics.push(EventTopic::new(topic_bytes));
        }
        
        // Create event log
        let log = EventLog::new(
            ContractAddress::ZERO, // Would be set by the contract
            topics,
            non_indexed_data,
            self.current_block,
            self.current_tx_hash,
            self.log_index,
        );
        
        self.events.push(log);
        self.log_index += 1;
        
        Ok(())
    }
    
    fn get_events(&self) -> &[EventLog] {
        &self.events
    }
    
    fn get_events_filtered(&self, filter: &EventFilter) -> Vec<&EventLog> {
        self.events.iter().filter(|log| filter.matches(log)).collect()
    }
}

/// Simple Keccak-256 implementation
fn keccak256(input: &[u8]) -> [u8; 32] {
    // Simplified hash function for demonstration
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

use std::fmt;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_event_topic_creation() {
        let topic1 = EventTopic::from_string("Transfer(address,address,uint256)");
        let topic2 = EventTopic::from_u256(U256::new(42));
        
        assert_ne!(topic1.as_bytes(), &[0u8; 32]);
        assert_ne!(topic2.as_bytes(), &[0u8; 32]);
        assert_ne!(topic1, topic2);
    }
    
    #[test]
    fn test_event_data() {
        let data1 = EventData::U256(U256::new(1000));
        let data2 = EventData::String("Hello".to_string());
        let data3 = EventData::Bool(true);
        
        assert_eq!(data1.to_bytes().len(), 32);
        assert_eq!(data2.to_bytes(), b"Hello");
        assert_eq!(data3.to_bytes(), vec![1]);
    }
    
    #[test]
    fn test_event_definition() {
        let params = vec![
            EventParameter {
                name: "from".to_string(),
                param_type: EventParameterType::Address,
                indexed: true,
            },
            EventParameter {
                name: "to".to_string(),
                param_type: EventParameterType::Address,
                indexed: true,
            },
            EventParameter {
                name: "value".to_string(),
                param_type: EventParameterType::U256,
                indexed: false,
            },
        ];
        
        let event_def = EventDefinition::new("Transfer".to_string(), params);
        
        assert_eq!(event_def.name, "Transfer");
        assert_eq!(event_def.indexed_parameters().len(), 2);
        assert_eq!(event_def.non_indexed_parameters().len(), 1);
    }
    
    #[test]
    fn test_event_filter() {
        let addr = ContractAddress::from_hex("0x742d35Cc6634C0532925a3b8D4C9db96590c4C87").unwrap();
        let topic = EventTopic::from_string("Transfer(address,address,uint256)");
        
        let filter = EventFilter::new()
            .address(addr)
            .topic(0, topic.clone())
            .block_range(100, 200);
        
        let log = EventLog::new(
            addr,
            vec![topic.clone()],
            EventData::U256(U256::new(1000)),
            150,
            [1u8; 32],
            0,
        );
        
        assert!(filter.matches(&log));
        
        // Test non-matching log
        let wrong_addr = ContractAddress::ZERO;
        let log2 = EventLog::new(
            wrong_addr,
            vec![topic],
            EventData::U256(U256::new(1000)),
            150,
            [1u8; 32],
            0,
        );
        
        assert!(!filter.matches(&log2));
    }
    
    #[test]
    fn test_memory_event_store() {
        let mut store = MemoryEventStore::new();
        store.set_block(100);
        store.set_transaction([1u8; 32]);
        
        let event_def = EventDefinition::new(
            "Transfer".to_string(),
            vec![
                EventParameter {
                    name: "from".to_string(),
                    param_type: EventParameterType::Address,
                    indexed: true,
                },
                EventParameter {
                    name: "value".to_string(),
                    param_type: EventParameterType::U256,
                    indexed: false,
                },
            ],
        );
        
        let indexed_data = vec![EventData::Address(ContractAddress::ZERO)];
        let non_indexed_data = EventData::U256(U256::new(1000));
        
        store.emit_event(&event_def, indexed_data, non_indexed_data).unwrap();
        
        let events = store.get_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].block_number, 100);
        assert_eq!(events[0].topics.len(), 2); // signature + 1 indexed param
    }
}