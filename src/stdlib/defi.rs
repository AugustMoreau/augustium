// DeFi utilities - AMM, lending, staking etc.
// Common patterns for building financial smart contracts

use crate::error::{Result, VmError, VmErrorKind, CompilerError};
use crate::stdlib::core_types::{Address, U256, AugustiumType};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

// Basic token info that we need to track
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    pub address: Address,
    pub symbol: String,
    pub decimals: u8,
    pub total_supply: U256,
}

// AMM liquidity pool (like Uniswap)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityPool {
    pub token_a: TokenInfo,
    pub token_b: TokenInfo,
    pub reserve_a: U256,
    pub reserve_b: U256,
    pub total_liquidity: U256,
    pub fee_rate: U256, // basis points, so 30 = 0.3%
    pub liquidity_providers: HashMap<Address, U256>,
}

impl LiquidityPool {
    /// Create a new liquidity pool
    pub fn new(token_a: TokenInfo, token_b: TokenInfo, fee_rate: U256) -> Self {
        Self {
            token_a,
            token_b,
            reserve_a: U256::zero(),
            reserve_b: U256::zero(),
            total_liquidity: U256::zero(),
            fee_rate,
            liquidity_providers: HashMap::new(),
        }
    }

    /// Add liquidity to the pool
    pub fn add_liquidity(
        &mut self,
        provider: Address,
        amount_a: U256,
        amount_b: U256,
    ) -> Result<U256> {
        if amount_a.is_zero() || amount_b.is_zero() {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InvalidInput,
                "Liquidity amounts must be greater than zero".to_string(),
            )));
        }

        let liquidity_minted = if self.total_liquidity.is_zero() {
            // First liquidity provision
            let liquidity = self.sqrt(amount_a * amount_b);
            self.reserve_a = amount_a;
            self.reserve_b = amount_b;
            liquidity
        } else {
            // Subsequent liquidity provision
            let liquidity_a = (amount_a * self.total_liquidity) / self.reserve_a;
            let liquidity_b = (amount_b * self.total_liquidity) / self.reserve_b;
            let liquidity = std::cmp::min(liquidity_a, liquidity_b);
            
            self.reserve_a = self.reserve_a + amount_a;
            self.reserve_b = self.reserve_b + amount_b;
            liquidity
        };

        self.total_liquidity = self.total_liquidity + liquidity_minted;
        *self.liquidity_providers.entry(provider).or_insert(U256::zero()) += liquidity_minted;

        Ok(liquidity_minted)
    }

    /// Remove liquidity from the pool
    pub fn remove_liquidity(
        &mut self,
        provider: Address,
        liquidity: U256,
    ) -> Result<(U256, U256)> {
        let provider_liquidity = self.liquidity_providers.get(&provider).copied().unwrap_or(U256::zero());
        
        if liquidity > provider_liquidity {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InsufficientBalance,
                "Insufficient liquidity balance".to_string(),
            )));
        }

        let amount_a = (liquidity * self.reserve_a) / self.total_liquidity;
        let amount_b = (liquidity * self.reserve_b) / self.total_liquidity;

        self.reserve_a = self.reserve_a - amount_a;
        self.reserve_b = self.reserve_b - amount_b;
        self.total_liquidity = self.total_liquidity - liquidity;
        
        *self.liquidity_providers.get_mut(&provider).unwrap() -= liquidity;
        if self.liquidity_providers[&provider].is_zero() {
            self.liquidity_providers.remove(&provider);
        }

        Ok((amount_a, amount_b))
    }

    /// Calculate output amount for a swap
    pub fn get_amount_out(&self, amount_in: U256, reserve_in: U256, reserve_out: U256) -> Result<U256> {
        if amount_in.is_zero() {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InvalidInput,
                "Input amount must be greater than zero".to_string(),
            )));
        }

        if reserve_in.is_zero() || reserve_out.is_zero() {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InsufficientLiquidity,
                "Insufficient liquidity".to_string(),
            )));
        }

        // Apply fee (fee_rate is in basis points)
        let amount_in_with_fee = amount_in * (U256::new(10000) - self.fee_rate);
        let numerator = amount_in_with_fee * reserve_out;
        let denominator = (reserve_in * U256::new(10000)) + amount_in_with_fee;
        
        Ok(numerator / denominator)
    }

    /// Perform a swap from token A to token B
    pub fn swap_a_to_b(&mut self, amount_in: U256) -> Result<U256> {
        let amount_out = self.get_amount_out(amount_in, self.reserve_a, self.reserve_b)?;
        
        self.reserve_a = self.reserve_a + amount_in;
        self.reserve_b = self.reserve_b - amount_out;
        
        Ok(amount_out)
    }

    /// Perform a swap from token B to token A
    pub fn swap_b_to_a(&mut self, amount_in: U256) -> Result<U256> {
        let amount_out = self.get_amount_out(amount_in, self.reserve_b, self.reserve_a)?;
        
        self.reserve_b = self.reserve_b + amount_in;
        self.reserve_a = self.reserve_a - amount_out;
        
        Ok(amount_out)
    }

    /// Get current price of token A in terms of token B
    pub fn get_price_a_in_b(&self) -> Result<U256> {
        if self.reserve_a.is_zero() {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InsufficientLiquidity,
                "No liquidity for token A".to_string(),
            )));
        }
        Ok(self.reserve_b / self.reserve_a)
    }

    /// Get current price of token B in terms of token A
    pub fn get_price_b_in_a(&self) -> Result<U256> {
        if self.reserve_b.is_zero() {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InsufficientLiquidity,
                "No liquidity for token B".to_string(),
            )));
        }
        Ok(self.reserve_a / self.reserve_b)
    }

    /// Simple square root implementation for liquidity calculation
    fn sqrt(&self, value: U256) -> U256 {
        if value.is_zero() {
            return U256::zero();
        }
        
        // Newton's method for square root
        let mut x = value;
        let mut y = (value + U256::new(1)) / U256::new(2);
        
        while y < x {
            x = y;
            y = (x + value / x) / U256::new(2);
        }
        
        x
    }
}

/// Lending pool for borrowing and lending
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LendingPool {
    pub asset: TokenInfo,
    pub total_deposits: U256,
    pub total_borrows: U256,
    pub interest_rate: U256, // Annual interest rate in basis points
    pub collateral_ratio: U256, // Required collateral ratio in basis points
    pub deposits: HashMap<Address, U256>,
    pub borrows: HashMap<Address, U256>,
    pub collateral: HashMap<Address, U256>,
}

impl LendingPool {
    /// Create a new lending pool
    pub fn new(asset: TokenInfo, interest_rate: U256, collateral_ratio: U256) -> Self {
        Self {
            asset,
            total_deposits: U256::zero(),
            total_borrows: U256::zero(),
            interest_rate,
            collateral_ratio,
            deposits: HashMap::new(),
            borrows: HashMap::new(),
            collateral: HashMap::new(),
        }
    }

    /// Deposit assets to earn interest
    pub fn deposit(&mut self, depositor: Address, amount: U256) -> Result<()> {
        if amount.is_zero() {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InvalidInput,
                "Deposit amount must be greater than zero".to_string(),
            )));
        }

        self.total_deposits = self.total_deposits + amount;
        *self.deposits.entry(depositor).or_insert(U256::zero()) += amount;
        
        Ok(())
    }

    /// Withdraw deposited assets
    pub fn withdraw(&mut self, depositor: Address, amount: U256) -> Result<()> {
        let deposited = self.deposits.get(&depositor).copied().unwrap_or(U256::zero());
        
        if amount > deposited {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InsufficientBalance,
                "Insufficient deposit balance".to_string(),
            )));
        }

        // Check if withdrawal would leave enough liquidity for borrows
        let available_liquidity = self.total_deposits - self.total_borrows;
        if amount > available_liquidity {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InsufficientLiquidity,
                "Insufficient liquidity for withdrawal".to_string(),
            )));
        }

        self.total_deposits = self.total_deposits - amount;
        *self.deposits.get_mut(&depositor).unwrap() -= amount;
        
        if self.deposits[&depositor].is_zero() {
            self.deposits.remove(&depositor);
        }
        
        Ok(())
    }

    /// Deposit collateral
    pub fn deposit_collateral(&mut self, borrower: Address, amount: U256) -> Result<()> {
        if amount.is_zero() {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InvalidInput,
                "Collateral amount must be greater than zero".to_string(),
            )));
        }

        *self.collateral.entry(borrower).or_insert(U256::zero()) += amount;
        Ok(())
    }

    /// Borrow assets against collateral
    pub fn borrow(&mut self, borrower: Address, amount: U256) -> Result<()> {
        if amount.is_zero() {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InvalidInput,
                "Borrow amount must be greater than zero".to_string(),
            )));
        }

        let collateral_value = self.collateral.get(&borrower).copied().unwrap_or(U256::zero());
        let current_borrows = self.borrows.get(&borrower).copied().unwrap_or(U256::zero());
        let total_borrows_after = current_borrows + amount;
        
        // Check collateral ratio
        let required_collateral = (total_borrows_after * self.collateral_ratio) / U256::new(10000);
        if collateral_value < required_collateral {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InsufficientCollateral,
                "Insufficient collateral for borrow".to_string(),
            )));
        }

        // Check available liquidity
        let available_liquidity = self.total_deposits - self.total_borrows;
        if amount > available_liquidity {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InsufficientLiquidity,
                "Insufficient liquidity for borrow".to_string(),
            )));
        }

        self.total_borrows = self.total_borrows + amount;
        *self.borrows.entry(borrower).or_insert(U256::zero()) += amount;
        
        Ok(())
    }

    /// Repay borrowed assets
    pub fn repay(&mut self, borrower: Address, amount: U256) -> Result<()> {
        let borrowed = self.borrows.get(&borrower).copied().unwrap_or(U256::zero());
        
        if amount > borrowed {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InvalidInput,
                "Repay amount exceeds borrowed amount".to_string(),
            )));
        }

        self.total_borrows = self.total_borrows - amount;
        *self.borrows.get_mut(&borrower).unwrap() -= amount;
        
        if self.borrows[&borrower].is_zero() {
            self.borrows.remove(&borrower);
        }
        
        Ok(())
    }

    /// Withdraw collateral
    pub fn withdraw_collateral(&mut self, borrower: Address, amount: U256) -> Result<()> {
        let collateral_value = self.collateral.get(&borrower).copied().unwrap_or(U256::zero());
        
        if amount > collateral_value {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InsufficientBalance,
                "Insufficient collateral balance".to_string(),
            )));
        }

        let current_borrows = self.borrows.get(&borrower).copied().unwrap_or(U256::zero());
        let remaining_collateral = collateral_value - amount;
        
        // Check if remaining collateral is sufficient
        if !current_borrows.is_zero() {
            let required_collateral = (current_borrows * self.collateral_ratio) / U256::new(10000);
            if remaining_collateral < required_collateral {
                return Err(CompilerError::VmError(VmError::new(
                    VmErrorKind::InsufficientCollateral,
                    "Withdrawal would leave insufficient collateral".to_string(),
                )));
            }
        }

        *self.collateral.get_mut(&borrower).unwrap() -= amount;
        
        if self.collateral[&borrower].is_zero() {
            self.collateral.remove(&borrower);
        }
        
        Ok(())
    }

    /// Calculate utilization rate
    pub fn utilization_rate(&self) -> U256 {
        if self.total_deposits.is_zero() {
            return U256::zero();
        }
        (self.total_borrows * U256::new(10000)) / self.total_deposits
    }
}

/// Staking pool for earning rewards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakingPool {
    pub staking_token: TokenInfo,
    pub reward_token: TokenInfo,
    pub total_staked: U256,
    pub reward_rate: U256, // Rewards per block
    pub last_update_block: U256,
    pub reward_per_token_stored: U256,
    pub stakes: HashMap<Address, U256>,
    pub user_reward_per_token_paid: HashMap<Address, U256>,
    pub rewards: HashMap<Address, U256>,
}

impl StakingPool {
    /// Create a new staking pool
    pub fn new(staking_token: TokenInfo, reward_token: TokenInfo, reward_rate: U256) -> Self {
        Self {
            staking_token,
            reward_token,
            total_staked: U256::zero(),
            reward_rate,
            last_update_block: U256::zero(),
            reward_per_token_stored: U256::zero(),
            stakes: HashMap::new(),
            user_reward_per_token_paid: HashMap::new(),
            rewards: HashMap::new(),
        }
    }

    /// Update reward calculations
    pub fn update_reward(&mut self, current_block: U256, account: Option<Address>) {
        self.reward_per_token_stored = self.reward_per_token(current_block);
        self.last_update_block = current_block;
        
        if let Some(account) = account {
            self.rewards.insert(account, self.earned(account, current_block));
            self.user_reward_per_token_paid.insert(account, self.reward_per_token_stored);
        }
    }

    /// Calculate reward per token
    pub fn reward_per_token(&self, current_block: U256) -> U256 {
        if self.total_staked.is_zero() {
            return self.reward_per_token_stored;
        }
        
        let blocks_passed = current_block - self.last_update_block;
        let reward_increment = (blocks_passed * self.reward_rate * U256::new(1e18 as u64)) / self.total_staked;
        
        self.reward_per_token_stored + reward_increment
    }

    /// Calculate earned rewards for an account
    pub fn earned(&self, account: Address, current_block: U256) -> U256 {
        let staked = self.stakes.get(&account).copied().unwrap_or(U256::zero());
        let user_reward_per_token_paid = self.user_reward_per_token_paid.get(&account).copied().unwrap_or(U256::zero());
        let current_rewards = self.rewards.get(&account).copied().unwrap_or(U256::zero());
        
        let reward_per_token_diff = self.reward_per_token(current_block) - user_reward_per_token_paid;
        let new_rewards = (staked * reward_per_token_diff) / U256::new(1e18 as u64);
        
        current_rewards + new_rewards
    }

    /// Stake tokens
    pub fn stake(&mut self, staker: Address, amount: U256, current_block: U256) -> Result<()> {
        if amount.is_zero() {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InvalidInput,
                "Stake amount must be greater than zero".to_string(),
            )));
        }

        self.update_reward(current_block, Some(staker));
        
        self.total_staked = self.total_staked + amount;
        *self.stakes.entry(staker).or_insert(U256::zero()) += amount;
        
        Ok(())
    }

    /// Withdraw staked tokens
    pub fn withdraw(&mut self, staker: Address, amount: U256, current_block: U256) -> Result<()> {
        let staked = self.stakes.get(&staker).copied().unwrap_or(U256::zero());
        
        if amount > staked {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InsufficientBalance,
                "Insufficient staked balance".to_string(),
            )));
        }

        self.update_reward(current_block, Some(staker));
        
        self.total_staked = self.total_staked - amount;
        *self.stakes.get_mut(&staker).unwrap() -= amount;
        
        if self.stakes[&staker].is_zero() {
            self.stakes.remove(&staker);
        }
        
        Ok(())
    }

    /// Claim rewards
    pub fn claim_reward(&mut self, staker: Address, current_block: U256) -> Result<U256> {
        self.update_reward(current_block, Some(staker));
        
        let reward = self.rewards.get(&staker).copied().unwrap_or(U256::zero());
        if !reward.is_zero() {
            self.rewards.insert(staker, U256::zero());
        }
        
        Ok(reward)
    }
}

/// DeFi utilities and helper functions
pub struct DeFiUtils;

impl DeFiUtils {
    /// Calculate compound interest
    pub fn compound_interest(principal: U256, rate: U256, periods: U256) -> U256 {
        let mut result = principal;
        for _ in 0..periods.as_u64() {
            result = result + (result * rate) / U256::new(10000);
        }
        result
    }

    /// Calculate simple interest
    pub fn simple_interest(principal: U256, rate: U256, periods: U256) -> U256 {
        principal + (principal * rate * periods) / U256::new(10000)
    }

    /// Calculate annual percentage yield (APY)
    pub fn calculate_apy(rate_per_period: U256, periods_per_year: U256) -> U256 {
        // APY = (1 + rate)^periods - 1
        // Simplified calculation for demonstration
        rate_per_period * periods_per_year
    }

    /// Calculate impermanent loss for liquidity providers
    pub fn impermanent_loss(price_ratio: U256) -> U256 {
        // Simplified calculation: IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
        // This is a simplified version for demonstration
        if price_ratio == U256::new(1) {
            return U256::zero();
        }
        
        // Return a simplified approximation
        if price_ratio > U256::new(1) {
            (price_ratio - U256::new(1)) * U256::new(100) / price_ratio
        } else {
            (U256::new(1) - price_ratio) * U256::new(100)
        }
    }

    /// Calculate liquidation price
    pub fn liquidation_price(
        collateral_amount: U256,
        borrowed_amount: U256,
        liquidation_ratio: U256,
    ) -> U256 {
        (borrowed_amount * liquidation_ratio) / (collateral_amount * U256::new(10000))
    }

    /// Calculate health factor for lending positions
    pub fn health_factor(
        collateral_value: U256,
        borrowed_value: U256,
        liquidation_threshold: U256,
    ) -> U256 {
        if borrowed_value.is_zero() {
            return U256::new(u64::MAX); // Infinite health factor
        }
        
        // Health factor = (collateral_value * liquidation_threshold) / (borrowed_value * 100)
        // Fixed: Use 100 instead of 10000 for proper percentage calculation
        (collateral_value * liquidation_threshold) / (borrowed_value * U256::new(100))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_token(symbol: &str) -> TokenInfo {
        TokenInfo {
            address: Address::new([1u8; 20]),
            symbol: symbol.to_string(),
            decimals: 18,
            total_supply: U256::new(1000000),
        }
    }

    #[test]
    fn test_liquidity_pool_creation() {
        let token_a = create_test_token("USDC");
        let token_b = create_test_token("ETH");
        let pool = LiquidityPool::new(token_a, token_b, U256::new(30)); // 0.3% fee
        
        assert_eq!(pool.fee_rate, U256::new(30));
        assert!(pool.total_liquidity.is_zero());
    }

    #[test]
    fn test_add_liquidity() {
        let token_a = create_test_token("USDC");
        let token_b = create_test_token("ETH");
        let mut pool = LiquidityPool::new(token_a, token_b, U256::new(30));
        let provider = Address::new([1u8; 20]);
        
        let liquidity = pool.add_liquidity(provider, U256::new(1000), U256::new(1000)).unwrap();
        assert!(!liquidity.is_zero());
        assert_eq!(pool.reserve_a, U256::new(1000));
        assert_eq!(pool.reserve_b, U256::new(1000));
    }

    #[test]
    fn test_lending_pool() {
        let asset = create_test_token("USDC");
        let mut pool = LendingPool::new(asset, U256::new(500), U256::new(15000)); // 5% APR, 150% collateral ratio
        let user = Address::new([1u8; 20]);
        
        // Deposit
        pool.deposit(user, U256::new(1000)).unwrap();
        assert_eq!(pool.total_deposits, U256::new(1000));
        
        // Deposit collateral and borrow
        pool.deposit_collateral(user, U256::new(2000)).unwrap();
        pool.borrow(user, U256::new(500)).unwrap();
        assert_eq!(pool.total_borrows, U256::new(500));
    }

    #[test]
    fn test_staking_pool() {
        let staking_token = create_test_token("STAKE");
        let reward_token = create_test_token("REWARD");
        let mut pool = StakingPool::new(staking_token, reward_token, U256::new(100));
        let staker = Address::new([1u8; 20]);
        
        pool.stake(staker, U256::new(1000), U256::new(1)).unwrap();
        assert_eq!(pool.total_staked, U256::new(1000));
        
        let earned = pool.earned(staker, U256::new(10));
        assert!(!earned.is_zero());
    }

    #[test]
    fn test_defi_utils() {
        let principal = U256::new(1000);
        let rate = U256::new(500); // 5%
        let periods = U256::new(12);
        
        let simple = DeFiUtils::simple_interest(principal, rate, periods);
        let compound = DeFiUtils::compound_interest(principal, rate, periods);
        
        assert!(compound > simple);
        
        let health = DeFiUtils::health_factor(U256::new(1500), U256::new(1000), U256::new(8000));
        assert!(health > U256::new(1));
    }
}