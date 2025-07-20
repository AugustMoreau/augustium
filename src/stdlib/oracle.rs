//! Oracle Module
//!
//! This module provides oracle functionality for accessing external data feeds,
//! price information, and other off-chain data sources in smart contracts.

use crate::error::{Result, VmError, VmErrorKind, CompilerError};
use crate::stdlib::core_types::{Address, U256, AugustiumType};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Price feed data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceFeed {
    pub asset: String,
    pub price: U256,
    pub decimals: u8,
    pub timestamp: U256,
    pub round_id: U256,
    pub confidence: U256, // Confidence level in basis points
}

impl PriceFeed {
    /// Create a new price feed
    pub fn new(
        asset: String,
        price: U256,
        decimals: u8,
        timestamp: U256,
        round_id: U256,
        confidence: U256,
    ) -> Self {
        Self {
            asset,
            price,
            decimals,
            timestamp,
            round_id,
            confidence,
        }
    }

    /// Check if the price feed is stale
    pub fn is_stale(&self, current_timestamp: U256, max_age: U256) -> bool {
        current_timestamp - self.timestamp > max_age
    }

    /// Check if the price feed has sufficient confidence
    pub fn has_sufficient_confidence(&self, min_confidence: U256) -> bool {
        self.confidence >= min_confidence
    }

    /// Get price adjusted for decimals
    pub fn get_price_with_decimals(&self, target_decimals: u8) -> U256 {
        if target_decimals == self.decimals {
            return self.price;
        }
        
        if target_decimals > self.decimals {
            let multiplier = U256::new(10).pow(U256::new((target_decimals - self.decimals) as u64));
            self.price * multiplier
        } else {
            let divisor = U256::new(10).pow(U256::new((self.decimals - target_decimals) as u64));
            self.price / divisor
        }
    }
}

/// Oracle aggregator for combining multiple price sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleAggregator {
    pub asset: String,
    pub feeds: Vec<PriceFeed>,
    pub min_feeds: usize,
    pub max_deviation: U256, // Maximum allowed deviation in basis points
    pub aggregation_method: AggregationMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationMethod {
    Mean,
    Median,
    WeightedAverage,
    TrimmedMean { trim_percent: u8 },
}

impl OracleAggregator {
    /// Create a new oracle aggregator
    pub fn new(
        asset: String,
        min_feeds: usize,
        max_deviation: U256,
        aggregation_method: AggregationMethod,
    ) -> Self {
        Self {
            asset,
            feeds: Vec::new(),
            min_feeds,
            max_deviation,
            aggregation_method,
        }
    }

    /// Add a price feed
    pub fn add_feed(&mut self, feed: PriceFeed) -> Result<()> {
        if feed.asset != self.asset {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InvalidInput,
                "Feed asset does not match aggregator asset".to_string(),
            )));
        }
        
        self.feeds.push(feed);
        Ok(())
    }

    /// Remove stale feeds
    pub fn remove_stale_feeds(&mut self, current_timestamp: U256, max_age: U256) {
        self.feeds.retain(|feed| !feed.is_stale(current_timestamp, max_age));
    }

    /// Get aggregated price
    pub fn get_aggregated_price(&self, min_confidence: U256) -> Result<PriceFeed> {
        // Filter feeds by confidence
        let valid_feeds: Vec<&PriceFeed> = self.feeds
            .iter()
            .filter(|feed| feed.has_sufficient_confidence(min_confidence))
            .collect();

        if valid_feeds.len() < self.min_feeds {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InsufficientData,
                format!("Insufficient valid feeds: {} < {}", valid_feeds.len(), self.min_feeds),
            )));
        }

        // Check for excessive deviation
        if let Err(e) = self.check_deviation(&valid_feeds) {
            return Err(e);
        }

        // Calculate aggregated price
        let aggregated_price = match self.aggregation_method {
            AggregationMethod::Mean => self.calculate_mean(&valid_feeds),
            AggregationMethod::Median => self.calculate_median(&valid_feeds),
            AggregationMethod::WeightedAverage => self.calculate_weighted_average(&valid_feeds),
            AggregationMethod::TrimmedMean { trim_percent } => {
                self.calculate_trimmed_mean(&valid_feeds, trim_percent)
            }
        };

        // Get latest timestamp and round ID
        let latest_timestamp = valid_feeds.iter().map(|f| f.timestamp).max().unwrap_or(U256::zero());
        let latest_round = valid_feeds.iter().map(|f| f.round_id).max().unwrap_or(U256::zero());
        
        // Calculate average confidence
        let avg_confidence = valid_feeds.iter()
            .map(|f| f.confidence)
            .fold(U256::zero(), |acc, conf| acc + conf) / U256::new(valid_feeds.len() as u64);

        Ok(PriceFeed::new(
            self.asset.clone(),
            aggregated_price,
            18, // Standard decimals
            latest_timestamp,
            latest_round,
            avg_confidence,
        ))
    }

    /// Check for excessive price deviation
    fn check_deviation(&self, feeds: &[&PriceFeed]) -> Result<()> {
        if feeds.len() < 2 {
            return Ok(());
        }

        let prices: Vec<U256> = feeds.iter().map(|f| f.price).collect();
        let min_price = prices.iter().min().unwrap();
        let max_price = prices.iter().max().unwrap();
        
        if min_price.is_zero() {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InvalidData,
                "Zero price detected in feeds".to_string(),
            )));
        }

        let deviation = ((*max_price - *min_price) * U256::new(10000)) / *min_price;
        
        if deviation > self.max_deviation {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::ExcessiveDeviation,
                format!("Price deviation {} exceeds maximum {}", deviation, self.max_deviation),
            )));
        }

        Ok(())
    }

    /// Calculate mean price
    fn calculate_mean(&self, feeds: &[&PriceFeed]) -> U256 {
        let sum = feeds.iter().map(|f| f.price).fold(U256::zero(), |acc, price| acc + price);
        sum / U256::new(feeds.len() as u64)
    }

    /// Calculate median price
    fn calculate_median(&self, feeds: &[&PriceFeed]) -> U256 {
        let mut prices: Vec<U256> = feeds.iter().map(|f| f.price).collect();
        prices.sort();
        
        let len = prices.len();
        if len % 2 == 0 {
            (prices[len / 2 - 1] + prices[len / 2]) / U256::new(2)
        } else {
            prices[len / 2]
        }
    }

    /// Calculate weighted average (by confidence)
    fn calculate_weighted_average(&self, feeds: &[&PriceFeed]) -> U256 {
        let mut weighted_sum = U256::zero();
        let mut total_weight = U256::zero();
        
        for feed in feeds {
            weighted_sum = weighted_sum + (feed.price * feed.confidence);
            total_weight = total_weight + feed.confidence;
        }
        
        if total_weight.is_zero() {
            return self.calculate_mean(feeds);
        }
        
        weighted_sum / total_weight
    }

    /// Calculate trimmed mean
    fn calculate_trimmed_mean(&self, feeds: &[&PriceFeed], trim_percent: u8) -> U256 {
        if trim_percent >= 50 {
            return self.calculate_median(feeds);
        }
        
        let mut prices: Vec<U256> = feeds.iter().map(|f| f.price).collect();
        prices.sort();
        
        let trim_count = (prices.len() * trim_percent as usize) / 100;
        let trimmed_prices = &prices[trim_count..prices.len() - trim_count];
        
        if trimmed_prices.is_empty() {
            return self.calculate_mean(feeds);
        }
        
        let sum = trimmed_prices.iter().fold(U256::zero(), |acc, price| acc + *price);
        sum / U256::new(trimmed_prices.len() as u64)
    }
}

/// Oracle registry for managing multiple oracle sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleRegistry {
    pub oracles: HashMap<String, Address>, // asset -> oracle address
    pub authorized_updaters: HashMap<Address, bool>,
    pub price_feeds: HashMap<String, PriceFeed>,
    pub aggregators: HashMap<String, OracleAggregator>,
}

impl OracleRegistry {
    /// Create a new oracle registry
    pub fn new() -> Self {
        Self {
            oracles: HashMap::new(),
            authorized_updaters: HashMap::new(),
            price_feeds: HashMap::new(),
            aggregators: HashMap::new(),
        }
    }

    /// Register an oracle for an asset
    pub fn register_oracle(&mut self, asset: String, oracle_address: Address) {
        self.oracles.insert(asset, oracle_address);
    }

    /// Authorize an updater
    pub fn authorize_updater(&mut self, updater: Address) {
        self.authorized_updaters.insert(updater, true);
    }

    /// Revoke updater authorization
    pub fn revoke_updater(&mut self, updater: Address) {
        self.authorized_updaters.remove(&updater);
    }

    /// Check if an address is authorized to update prices
    pub fn is_authorized_updater(&self, updater: Address) -> bool {
        self.authorized_updaters.get(&updater).copied().unwrap_or(false)
    }

    /// Update price feed
    pub fn update_price_feed(&mut self, updater: Address, feed: PriceFeed) -> Result<()> {
        if !self.is_authorized_updater(updater) {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::Unauthorized,
                "Updater not authorized".to_string(),
            )));
        }

        self.price_feeds.insert(feed.asset.clone(), feed);
        Ok(())
    }

    /// Get price feed for an asset
    pub fn get_price_feed(&self, asset: &str) -> Result<&PriceFeed> {
        self.price_feeds.get(asset).ok_or_else(|| {
            CompilerError::VmError(VmError::new(
                VmErrorKind::NotFound,
                format!("Price feed not found for asset: {}", asset),
            ))
        })
    }

    /// Get latest price for an asset
    pub fn get_latest_price(&self, asset: &str, max_age: U256, current_timestamp: U256) -> Result<U256> {
        let feed = self.get_price_feed(asset)?;
        
        if feed.is_stale(current_timestamp, max_age) {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::StaleData,
                format!("Price feed for {} is stale", asset),
            )));
        }
        
        Ok(feed.price)
    }

    /// Add aggregator for an asset
    pub fn add_aggregator(&mut self, aggregator: OracleAggregator) {
        self.aggregators.insert(aggregator.asset.clone(), aggregator);
    }

    /// Get aggregated price for an asset
    pub fn get_aggregated_price(
        &self,
        asset: &str,
        min_confidence: U256,
        current_timestamp: U256,
        max_age: U256,
    ) -> Result<PriceFeed> {
        let aggregator = self.aggregators.get(asset).ok_or_else(|| {
            CompilerError::VmError(VmError::new(
                VmErrorKind::NotFound,
                format!("Aggregator not found for asset: {}", asset),
            ))
        })?;

        // Create a mutable copy to remove stale feeds
        let mut aggregator_copy = aggregator.clone();
        aggregator_copy.remove_stale_feeds(current_timestamp, max_age);
        
        aggregator_copy.get_aggregated_price(min_confidence)
    }
}

/// TWAP (Time-Weighted Average Price) calculator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TWAPCalculator {
    pub asset: String,
    pub observations: Vec<PriceObservation>,
    pub window_size: U256, // Time window in seconds
    pub max_observations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceObservation {
    pub price: U256,
    pub timestamp: U256,
    pub cumulative_price: U256,
}

impl TWAPCalculator {
    /// Create a new TWAP calculator
    pub fn new(asset: String, window_size: U256, max_observations: usize) -> Self {
        Self {
            asset,
            observations: Vec::new(),
            window_size,
            max_observations,
        }
    }

    /// Add a price observation
    pub fn add_observation(&mut self, price: U256, timestamp: U256) {
        let cumulative_price = if let Some(last_obs) = self.observations.last() {
            let time_elapsed = timestamp - last_obs.timestamp;
            last_obs.cumulative_price + (last_obs.price * time_elapsed)
        } else {
            U256::zero()
        };

        let observation = PriceObservation {
            price,
            timestamp,
            cumulative_price,
        };

        self.observations.push(observation);

        // Remove old observations (but keep at least 2 for TWAP calculation)
        while self.observations.len() > self.max_observations {
            self.observations.remove(0);
        }

        // Remove observations outside the window (but keep at least 2 observations)
        if self.observations.len() > 2 {
            let cutoff_time = if timestamp > self.window_size {
                timestamp - self.window_size
            } else {
                U256::zero()
            };
            
            // Count observations that would remain after filter
            let valid_count = self.observations.iter().filter(|obs| obs.timestamp >= cutoff_time).count();
            
            // Only apply filter if we'd still have at least 2 observations
            if valid_count >= 2 {
                self.observations.retain(|obs| obs.timestamp >= cutoff_time);
            }
        }
    }

    /// Calculate TWAP for the specified period
    pub fn calculate_twap(&self, start_time: U256, end_time: U256) -> Result<U256> {
        if start_time >= end_time {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InvalidInput,
                "Start time must be before end time".to_string(),
            )));
        }

        if self.observations.len() < 2 {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InsufficientData,
                "Need at least 2 observations for TWAP calculation".to_string(),
            )));
        }

        // Find observations that bracket the time period
        let start_obs = self.find_observation_at_time(start_time)?;
        let end_obs = self.find_observation_at_time(end_time)?;

        let time_elapsed = end_time - start_time;
        if time_elapsed.is_zero() {
            return Ok(start_obs.price);
        }

        let price_time_sum = end_obs.cumulative_price - start_obs.cumulative_price;
        Ok(price_time_sum / time_elapsed)
    }

    /// Find or interpolate observation at a specific time
    fn find_observation_at_time(&self, target_time: U256) -> Result<PriceObservation> {
        // Find the observation at or before the target time
        let mut before_obs: Option<&PriceObservation> = None;
        let mut after_obs: Option<&PriceObservation> = None;

        for obs in &self.observations {
            if obs.timestamp <= target_time {
                before_obs = Some(obs);
            } else if after_obs.is_none() {
                after_obs = Some(obs);
                break;
            }
        }

        match (before_obs, after_obs) {
            (Some(before), Some(after)) => {
                // Interpolate between observations
                let time_diff = after.timestamp - before.timestamp;
                let target_diff = target_time - before.timestamp;
                
                if time_diff.is_zero() {
                    return Ok(before.clone());
                }
                
                let price_diff = after.price - before.price;
                let interpolated_price = before.price + (price_diff * target_diff) / time_diff;
                
                let cumulative_diff = target_diff * before.price;
                let interpolated_cumulative = before.cumulative_price + cumulative_diff;
                
                Ok(PriceObservation {
                    price: interpolated_price,
                    timestamp: target_time,
                    cumulative_price: interpolated_cumulative,
                })
            }
            (Some(before), None) => Ok(before.clone()),
            (None, Some(after)) => Ok(after.clone()),
            (None, None) => Err(CompilerError::VmError(VmError::new(
                VmErrorKind::NotFound,
                "No observations found for the specified time".to_string(),
            ))),
        }
    }

    /// Get current TWAP
    pub fn get_current_twap(&self) -> Result<U256> {
        if self.observations.is_empty() {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InsufficientData,
                "No observations available".to_string(),
            )));
        }

        let latest_time = self.observations.last().unwrap().timestamp;
        let start_time = latest_time - self.window_size;
        
        self.calculate_twap(start_time, latest_time)
    }
}

/// Oracle utilities
pub struct OracleUtils;

impl OracleUtils {
    /// Convert price between different decimal places
    pub fn convert_price_decimals(price: U256, from_decimals: u8, to_decimals: u8) -> U256 {
        if from_decimals == to_decimals {
            return price;
        }
        
        if to_decimals > from_decimals {
            let multiplier = U256::new(10).pow(U256::new((to_decimals - from_decimals) as u64));
            price * multiplier
        } else {
            let divisor = U256::new(10).pow(U256::new((from_decimals - to_decimals) as u64));
            price / divisor
        }
    }

    /// Calculate price impact for a trade
    pub fn calculate_price_impact(
        current_price: U256,
        new_price: U256,
    ) -> U256 {
        if current_price.is_zero() {
            return U256::zero();
        }
        
        let diff = if new_price > current_price {
            new_price - current_price
        } else {
            current_price - new_price
        };
        
        (diff * U256::new(10000)) / current_price
    }

    /// Validate price feed freshness
    pub fn is_price_fresh(
        timestamp: U256,
        current_time: U256,
        max_age: U256,
    ) -> bool {
        current_time - timestamp <= max_age
    }

    /// Calculate confidence score based on multiple factors
    pub fn calculate_confidence_score(
        price_deviation: U256,
        data_age: U256,
        source_count: usize,
        max_deviation: U256,
        max_age: U256,
    ) -> U256 {
        let mut score = U256::new(10000); // Start with 100%
        
        // Reduce score based on price deviation
        if price_deviation > U256::zero() {
            let deviation_penalty = (price_deviation * U256::new(5000)) / max_deviation;
            score = score - std::cmp::min(score, deviation_penalty);
        }
        
        // Reduce score based on data age
        if data_age > U256::zero() {
            let age_penalty = (data_age * U256::new(3000)) / max_age;
            score = score - std::cmp::min(score, age_penalty);
        }
        
        // Boost score based on source count
        if source_count > 1 {
            let source_bonus = U256::new((source_count - 1) as u64 * 500);
            score = score + std::cmp::min(U256::new(2000), source_bonus);
        }
        
        std::cmp::min(score, U256::new(10000))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_feed_creation() {
        let feed = PriceFeed::new(
            "ETH/USD".to_string(),
            U256::new(2_000000000000000000u64), // $2000 with 18 decimals
            18,
            U256::new(1000),
            U256::new(1),
            U256::new(9500), // 95% confidence
        );
        
        assert_eq!(feed.asset, "ETH/USD");
        assert_eq!(feed.decimals, 18);
        assert!(feed.has_sufficient_confidence(U256::new(9000)));
    }

    #[test]
    fn test_oracle_aggregator() {
        let mut aggregator = OracleAggregator::new(
            "ETH/USD".to_string(),
            2,
            U256::new(500), // 5% max deviation
            AggregationMethod::Mean,
        );
        
        let feed1 = PriceFeed::new(
            "ETH/USD".to_string(),
            U256::new(2_000000000000000000u64),
            18,
            U256::new(1000),
            U256::new(1),
            U256::new(9500),
        );
        
        let feed2 = PriceFeed::new(
            "ETH/USD".to_string(),
            U256::new(2_010000000000000000u64),
            18,
            U256::new(1001),
            U256::new(2),
            U256::new(9600),
        );
        
        aggregator.add_feed(feed1).unwrap();
        aggregator.add_feed(feed2).unwrap();
        
        let aggregated = aggregator.get_aggregated_price(U256::new(9000)).unwrap();
        assert_eq!(aggregated.asset, "ETH/USD");
    }

    #[test]
    fn test_oracle_registry() {
        let mut registry = OracleRegistry::new();
        let updater = Address::new([1u8; 20]);
        
        registry.authorize_updater(updater);
        assert!(registry.is_authorized_updater(updater));
        
        let feed = PriceFeed::new(
            "BTC/USD".to_string(),
            U256::new(5_000000000000000000u64),
            18,
            U256::new(1000),
            U256::new(1),
            U256::new(9800),
        );
        
        registry.update_price_feed(updater, feed).unwrap();
        let retrieved_feed = registry.get_price_feed("BTC/USD").unwrap();
        assert_eq!(retrieved_feed.asset, "BTC/USD");
    }

    #[test]
    fn test_twap_calculator() {
        let mut twap = TWAPCalculator::new(
            "ETH/USD".to_string(),
            U256::new(3600), // 1 hour window
            100,
        );
        
        twap.add_observation(U256::new(2000), U256::new(1000));
        twap.add_observation(U256::new(2100), U256::new(1100));
        twap.add_observation(U256::new(2050), U256::new(1200));
        
        assert_eq!(twap.observations.len(), 3);
        
        let twap_result = twap.calculate_twap(U256::new(1000), U256::new(1200));
        assert!(twap_result.is_ok());
    }

    #[test]
    fn test_oracle_utils() {
        let price = U256::new(2_000000000000000000u64);
        let converted = OracleUtils::convert_price_decimals(price, 18, 8);
        assert_eq!(converted, U256::new(200000000u64));
        
        let impact = OracleUtils::calculate_price_impact(
            U256::new(2000),
            U256::new(2100),
        );
        assert_eq!(impact, U256::new(500)); // 5% impact
        
        let confidence = OracleUtils::calculate_confidence_score(
            U256::new(100), // 1% deviation
            U256::new(300), // 5 minutes old
            3, // 3 sources
            U256::new(500), // 5% max deviation
            U256::new(3600), // 1 hour max age
        );
        assert!(confidence > U256::new(8000)); // Should be > 80%
    }
}