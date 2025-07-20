// Automated Market Maker (AMM) Liquidity Pool Example
// Demonstrates DeFi patterns, mathematical operations, and advanced safety features

use std::math::{sqrt, min, max};
use std::collections::HashMap;
use std::events::Event;
use std::access::ReentrancyGuard;
use std::tokens::{IERC20, SafeERC20};

// Events
#[derive(Event)]
struct LiquidityAdded {
    provider: address,
    token_a_amount: u256,
    token_b_amount: u256,
    liquidity_minted: u256
}

#[derive(Event)]
struct LiquidityRemoved {
    provider: address,
    token_a_amount: u256,
    token_b_amount: u256,
    liquidity_burned: u256
}

#[derive(Event)]
struct Swap {
    trader: address,
    token_in: address,
    token_out: address,
    amount_in: u256,
    amount_out: u256,
    fee_amount: u256
}

#[derive(Event)]
struct Sync {
    reserve_a: u256,
    reserve_b: u256
}

#[derive(Event)]
struct FeesCollected {
    token_a_fees: u256,
    token_b_fees: u256,
    collector: address
}

// Price oracle data
#[derive(Clone)]
struct PriceData {
    price_cumulative_last: u256,
    block_timestamp_last: u32,
    price_average: u256
}

// Liquidity pool implementing constant product formula (x * y = k)
contract LiquidityPool extends ReentrancyGuard {
    // Pool tokens
    token_a: IERC20;
    token_b: IERC20;
    
    // Pool state
    reserve_a: u256;
    reserve_b: u256;
    total_supply: u256;
    
    // Liquidity provider balances
    balances: HashMap<address, u256>;
    
    // Fee configuration (in basis points, 1 bp = 0.01%)
    fee_rate: u256; // Default: 30 bp = 0.3%
    protocol_fee_rate: u256; // Default: 5 bp = 0.05%
    
    // Fee collection
    protocol_fee_collector: address;
    accumulated_fees_a: u256;
    accumulated_fees_b: u256;
    
    // Price oracle
    price_data: PriceData;
    
    // Constants
    MINIMUM_LIQUIDITY: u256 = 1000; // Minimum liquidity locked forever
    BASIS_POINTS: u256 = 10000; // 100% in basis points
    
    // Slippage protection
    max_slippage_bp: u256; // Maximum allowed slippage in basis points
    
    constructor(
        token_a_addr: address,
        token_b_addr: address,
        fee_bp: u256,
        protocol_fee_bp: u256,
        fee_collector: address,
        max_slippage: u256
    ) {
        require!(token_a_addr != address::zero(), "Invalid token A address");
        require!(token_b_addr != address::zero(), "Invalid token B address");
        require!(token_a_addr != token_b_addr, "Tokens must be different");
        require!(fee_bp <= 1000, "Fee rate too high"); // Max 10%
        require!(protocol_fee_bp <= 500, "Protocol fee too high"); // Max 5%
        require!(fee_collector != address::zero(), "Invalid fee collector");
        require!(max_slippage <= 5000, "Max slippage too high"); // Max 50%
        
        self.token_a = IERC20::at(token_a_addr);
        self.token_b = IERC20::at(token_b_addr);
        self.fee_rate = fee_bp;
        self.protocol_fee_rate = protocol_fee_bp;
        self.protocol_fee_collector = fee_collector;
        self.max_slippage_bp = max_slippage;
        
        self.reserve_a = 0;
        self.reserve_b = 0;
        self.total_supply = 0;
        self.accumulated_fees_a = 0;
        self.accumulated_fees_b = 0;
        
        // Initialize price oracle
        self.price_data = PriceData {
            price_cumulative_last: 0,
            block_timestamp_last: block.timestamp as u32,
            price_average: 0
        };
    }
    
    // Add liquidity to the pool
    #[payable(false)]
    #[non_reentrant]
    fn add_liquidity(
        &mut self,
        amount_a_desired: u256,
        amount_b_desired: u256,
        amount_a_min: u256,
        amount_b_min: u256,
        deadline: u256
    ) -> (u256, u256, u256) {
        require!(block.timestamp <= deadline, "Transaction expired");
        require!(amount_a_desired > 0 && amount_b_desired > 0, "Amounts must be positive");
        
        let (amount_a, amount_b) = if self.total_supply == 0 {
            // First liquidity provision
            (amount_a_desired, amount_b_desired)
        } else {
            // Calculate optimal amounts based on current reserves
            self._calculate_optimal_amounts(
                amount_a_desired,
                amount_b_desired,
                amount_a_min,
                amount_b_min
            )?
        };
        
        require!(amount_a >= amount_a_min, "Insufficient token A amount");
        require!(amount_b >= amount_b_min, "Insufficient token B amount");
        
        // Transfer tokens from user
        self.token_a.transfer_from(msg.sender, address::this(), amount_a)?;
        self.token_b.transfer_from(msg.sender, address::this(), amount_b)?;
        
        // Calculate liquidity tokens to mint
        let liquidity = if self.total_supply == 0 {
            let initial_liquidity = sqrt(amount_a * amount_b);
            require!(initial_liquidity > self.MINIMUM_LIQUIDITY, "Insufficient initial liquidity");
            
            // Lock minimum liquidity forever
            self.balances.insert(address::zero(), self.MINIMUM_LIQUIDITY);
            initial_liquidity - self.MINIMUM_LIQUIDITY
        } else {
            min(
                (amount_a * self.total_supply) / self.reserve_a,
                (amount_b * self.total_supply) / self.reserve_b
            )
        };
        
        require!(liquidity > 0, "Insufficient liquidity minted");
        
        // Update state
        self.total_supply += liquidity;
        let current_balance = self.balances.get(&msg.sender).unwrap_or(&0).clone();
        self.balances.insert(msg.sender, current_balance + liquidity);
        
        self.reserve_a += amount_a;
        self.reserve_b += amount_b;
        
        self._update_price_oracle();
        
        emit LiquidityAdded {
            provider: msg.sender,
            token_a_amount: amount_a,
            token_b_amount: amount_b,
            liquidity_minted: liquidity
        };
        
        emit Sync {
            reserve_a: self.reserve_a,
            reserve_b: self.reserve_b
        };
        
        (amount_a, amount_b, liquidity)
    }
    
    // Remove liquidity from the pool
    #[payable(false)]
    #[non_reentrant]
    fn remove_liquidity(
        &mut self,
        liquidity: u256,
        amount_a_min: u256,
        amount_b_min: u256,
        deadline: u256
    ) -> (u256, u256) {
        require!(block.timestamp <= deadline, "Transaction expired");
        require!(liquidity > 0, "Liquidity must be positive");
        
        let user_balance = self.balances.get(&msg.sender).unwrap_or(&0).clone();
        require!(user_balance >= liquidity, "Insufficient liquidity balance");
        
        // Calculate token amounts to return
        let amount_a = (liquidity * self.reserve_a) / self.total_supply;
        let amount_b = (liquidity * self.reserve_b) / self.total_supply;
        
        require!(amount_a >= amount_a_min, "Insufficient token A amount");
        require!(amount_b >= amount_b_min, "Insufficient token B amount");
        require!(amount_a > 0 && amount_b > 0, "Insufficient liquidity burned");
        
        // Update state
        self.balances.insert(msg.sender, user_balance - liquidity);
        self.total_supply -= liquidity;
        self.reserve_a -= amount_a;
        self.reserve_b -= amount_b;
        
        // Transfer tokens to user
        self.token_a.transfer(msg.sender, amount_a)?;
        self.token_b.transfer(msg.sender, amount_b)?;
        
        self._update_price_oracle();
        
        emit LiquidityRemoved {
            provider: msg.sender,
            token_a_amount: amount_a,
            token_b_amount: amount_b,
            liquidity_burned: liquidity
        };
        
        emit Sync {
            reserve_a: self.reserve_a,
            reserve_b: self.reserve_b
        };
        
        (amount_a, amount_b)
    }
    
    // Swap tokens using constant product formula
    #[payable(false)]
    #[non_reentrant]
    fn swap(
        &mut self,
        token_in: address,
        amount_in: u256,
        amount_out_min: u256,
        deadline: u256
    ) -> u256 {
        require!(block.timestamp <= deadline, "Transaction expired");
        require!(amount_in > 0, "Amount in must be positive");
        require!(
            token_in == self.token_a.address() || token_in == self.token_b.address(),
            "Invalid token"
        );
        
        let (reserve_in, reserve_out, token_out) = if token_in == self.token_a.address() {
            (self.reserve_a, self.reserve_b, self.token_b.address())
        } else {
            (self.reserve_b, self.reserve_a, self.token_a.address())
        };
        
        require!(reserve_in > 0 && reserve_out > 0, "Insufficient liquidity");
        
        // Calculate output amount with fees
        let amount_out = self._get_amount_out(amount_in, reserve_in, reserve_out)?;
        require!(amount_out >= amount_out_min, "Insufficient output amount");
        
        // Check slippage protection
        let price_impact = self._calculate_price_impact(amount_in, amount_out, reserve_in, reserve_out);
        require!(price_impact <= self.max_slippage_bp, "Slippage too high");
        
        // Calculate fees
        let fee_amount = (amount_in * self.fee_rate) / self.BASIS_POINTS;
        let protocol_fee = (fee_amount * self.protocol_fee_rate) / self.fee_rate;
        
        // Transfer input token from user
        if token_in == self.token_a.address() {
            self.token_a.transfer_from(msg.sender, address::this(), amount_in)?;
            self.accumulated_fees_a += protocol_fee;
        } else {
            self.token_b.transfer_from(msg.sender, address::this(), amount_in)?;
            self.accumulated_fees_b += protocol_fee;
        }
        
        // Transfer output token to user
        if token_out == self.token_a.address() {
            self.token_a.transfer(msg.sender, amount_out)?;
        } else {
            self.token_b.transfer(msg.sender, amount_out)?;
        }
        
        // Update reserves
        if token_in == self.token_a.address() {
            self.reserve_a += amount_in;
            self.reserve_b -= amount_out;
        } else {
            self.reserve_b += amount_in;
            self.reserve_a -= amount_out;
        }
        
        // Verify constant product formula (with fee adjustment)
        let balance_a = self.token_a.balance_of(address::this());
        let balance_b = self.token_b.balance_of(address::this());
        
        let adjusted_balance_a = balance_a - self.accumulated_fees_a;
        let adjusted_balance_b = balance_b - self.accumulated_fees_b;
        
        require!(
            adjusted_balance_a * adjusted_balance_b >= self.reserve_a * self.reserve_b,
            "Constant product violated"
        );
        
        self._update_price_oracle();
        
        emit Swap {
            trader: msg.sender,
            token_in,
            token_out,
            amount_in,
            amount_out,
            fee_amount
        };
        
        emit Sync {
            reserve_a: self.reserve_a,
            reserve_b: self.reserve_b
        };
        
        amount_out
    }
    
    // Flash loan functionality
    #[payable(false)]
    #[non_reentrant]
    fn flash_loan(
        &mut self,
        recipient: address,
        amount_a: u256,
        amount_b: u256,
        data: bytes
    ) {
        require!(recipient != address::zero(), "Invalid recipient");
        require!(amount_a > 0 || amount_b > 0, "At least one amount must be positive");
        require!(amount_a <= self.reserve_a, "Insufficient token A reserves");
        require!(amount_b <= self.reserve_b, "Insufficient token B reserves");
        
        let balance_a_before = self.token_a.balance_of(address::this());
        let balance_b_before = self.token_b.balance_of(address::this());
        
        // Calculate fees (0.1% flash loan fee)
        let fee_a = (amount_a * 10) / self.BASIS_POINTS;
        let fee_b = (amount_b * 10) / self.BASIS_POINTS;
        
        // Send tokens to recipient
        if amount_a > 0 {
            self.token_a.transfer(recipient, amount_a)?;
        }
        if amount_b > 0 {
            self.token_b.transfer(recipient, amount_b)?;
        }
        
        // Call recipient's flash loan callback
        let callback_result = self._call_flash_loan_callback(recipient, amount_a, amount_b, fee_a, fee_b, data);
        require!(callback_result.is_ok(), "Flash loan callback failed");
        
        // Verify repayment
        let balance_a_after = self.token_a.balance_of(address::this());
        let balance_b_after = self.token_b.balance_of(address::this());
        
        require!(
            balance_a_after >= balance_a_before + fee_a,
            "Insufficient token A repayment"
        );
        require!(
            balance_b_after >= balance_b_before + fee_b,
            "Insufficient token B repayment"
        );
        
        // Update accumulated fees
        self.accumulated_fees_a += fee_a;
        self.accumulated_fees_b += fee_b;
    }
    
    // Collect protocol fees
    #[payable(false)]
    fn collect_protocol_fees(&mut self) {
        require!(msg.sender == self.protocol_fee_collector, "Only fee collector");
        
        let fees_a = self.accumulated_fees_a;
        let fees_b = self.accumulated_fees_b;
        
        if fees_a > 0 {
            self.accumulated_fees_a = 0;
            self.token_a.transfer(self.protocol_fee_collector, fees_a)?;
        }
        
        if fees_b > 0 {
            self.accumulated_fees_b = 0;
            self.token_b.transfer(self.protocol_fee_collector, fees_b)?;
        }
        
        emit FeesCollected {
            token_a_fees: fees_a,
            token_b_fees: fees_b,
            collector: self.protocol_fee_collector
        };
    }
    
    // View functions
    #[view]
    fn get_reserves(&self) -> (u256, u256, u32) {
        (self.reserve_a, self.reserve_b, self.price_data.block_timestamp_last)
    }
    
    #[view]
    fn get_amount_out(&self, amount_in: u256, token_in: address) -> Result<u256, string> {
        require!(amount_in > 0, "Amount in must be positive");
        
        let (reserve_in, reserve_out) = if token_in == self.token_a.address() {
            (self.reserve_a, self.reserve_b)
        } else if token_in == self.token_b.address() {
            (self.reserve_b, self.reserve_a)
        } else {
            return Err("Invalid token".to_string());
        };
        
        self._get_amount_out(amount_in, reserve_in, reserve_out)
    }
    
    #[view]
    fn get_amount_in(&self, amount_out: u256, token_out: address) -> Result<u256, string> {
        require!(amount_out > 0, "Amount out must be positive");
        
        let (reserve_in, reserve_out) = if token_out == self.token_a.address() {
            (self.reserve_b, self.reserve_a)
        } else if token_out == self.token_b.address() {
            (self.reserve_a, self.reserve_b)
        } else {
            return Err("Invalid token".to_string());
        };
        
        require!(amount_out < reserve_out, "Insufficient liquidity");
        
        let numerator = reserve_in * amount_out * self.BASIS_POINTS;
        let denominator = (reserve_out - amount_out) * (self.BASIS_POINTS - self.fee_rate);
        
        Ok((numerator / denominator) + 1) // Add 1 for rounding
    }
    
    #[view]
    fn get_liquidity_balance(&self, provider: address) -> u256 {
        self.balances.get(&provider).unwrap_or(&0).clone()
    }
    
    #[view]
    fn get_total_supply(&self) -> u256 {
        self.total_supply
    }
    
    #[view]
    fn get_price(&self) -> (u256, u256) {
        if self.reserve_a > 0 && self.reserve_b > 0 {
            // Price of token A in terms of token B, and vice versa
            // Scaled by 1e18 for precision
            let price_a_in_b = (self.reserve_b * 1e18) / self.reserve_a;
            let price_b_in_a = (self.reserve_a * 1e18) / self.reserve_b;
            (price_a_in_b, price_b_in_a)
        } else {
            (0, 0)
        }
    }
    
    #[view]
    fn get_pool_info(&self) -> (address, address, u256, u256, u256, u256) {
        (
            self.token_a.address(),
            self.token_b.address(),
            self.reserve_a,
            self.reserve_b,
            self.total_supply,
            self.fee_rate
        )
    }
    
    #[view]
    fn get_accumulated_fees(&self) -> (u256, u256) {
        (self.accumulated_fees_a, self.accumulated_fees_b)
    }
    
    // Internal functions
    fn _get_amount_out(&self, amount_in: u256, reserve_in: u256, reserve_out: u256) -> Result<u256, string> {
        require!(amount_in > 0, "Insufficient input amount");
        require!(reserve_in > 0 && reserve_out > 0, "Insufficient liquidity");
        
        let amount_in_with_fee = amount_in * (self.BASIS_POINTS - self.fee_rate);
        let numerator = amount_in_with_fee * reserve_out;
        let denominator = (reserve_in * self.BASIS_POINTS) + amount_in_with_fee;
        
        Ok(numerator / denominator)
    }
    
    fn _calculate_optimal_amounts(
        &self,
        amount_a_desired: u256,
        amount_b_desired: u256,
        amount_a_min: u256,
        amount_b_min: u256
    ) -> Result<(u256, u256), string> {
        let amount_b_optimal = (amount_a_desired * self.reserve_b) / self.reserve_a;
        
        if amount_b_optimal <= amount_b_desired {
            require!(amount_b_optimal >= amount_b_min, "Insufficient token B amount");
            Ok((amount_a_desired, amount_b_optimal))
        } else {
            let amount_a_optimal = (amount_b_desired * self.reserve_a) / self.reserve_b;
            require!(amount_a_optimal <= amount_a_desired, "Optimal amount A exceeds desired");
            require!(amount_a_optimal >= amount_a_min, "Insufficient token A amount");
            Ok((amount_a_optimal, amount_b_desired))
        }
    }
    
    fn _calculate_price_impact(
        &self,
        amount_in: u256,
        amount_out: u256,
        reserve_in: u256,
        reserve_out: u256
    ) -> u256 {
        // Calculate price impact as percentage in basis points
        let price_before = (reserve_out * self.BASIS_POINTS) / reserve_in;
        let price_after = ((reserve_out - amount_out) * self.BASIS_POINTS) / (reserve_in + amount_in);
        
        if price_before > price_after {
            ((price_before - price_after) * self.BASIS_POINTS) / price_before
        } else {
            0
        }
    }
    
    fn _update_price_oracle(&mut self) {
        let block_timestamp = block.timestamp as u32;
        let time_elapsed = block_timestamp - self.price_data.block_timestamp_last;
        
        if time_elapsed > 0 && self.reserve_a > 0 && self.reserve_b > 0 {
            // Update cumulative price
            let price_a = (self.reserve_b * 1e18) / self.reserve_a;
            self.price_data.price_cumulative_last += price_a * (time_elapsed as u256);
            
            // Update average price (simple moving average)
            if time_elapsed >= 3600 { // Update average every hour
                self.price_data.price_average = price_a;
            }
            
            self.price_data.block_timestamp_last = block_timestamp;
        }
    }
    
    fn _call_flash_loan_callback(
        &self,
        recipient: address,
        amount_a: u256,
        amount_b: u256,
        fee_a: u256,
        fee_b: u256,
        data: bytes
    ) -> Result<(), string> {
        // This would call the recipient's flash loan callback function
        // For now, we'll simulate success
        if recipient != address::zero() {
            Ok(())
        } else {
            Err("Invalid recipient".to_string())
        }
    }
    
    // Emergency functions
    #[only_owner]
    fn emergency_pause(&mut self) {
        // Implementation would pause all pool operations
        // This is a placeholder for emergency functionality
    }
    
    #[only_owner]
    fn update_fee_collector(&mut self, new_collector: address) {
        require!(new_collector != address::zero(), "Invalid collector address");
        self.protocol_fee_collector = new_collector;
    }
    
    #[only_owner]
    fn update_fee_rates(&mut self, new_fee_rate: u256, new_protocol_fee_rate: u256) {
        require!(new_fee_rate <= 1000, "Fee rate too high"); // Max 10%
        require!(new_protocol_fee_rate <= 500, "Protocol fee too high"); // Max 5%
        
        self.fee_rate = new_fee_rate;
        self.protocol_fee_rate = new_protocol_fee_rate;
    }
}

// Example usage and comprehensive tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::testing::*;
    
    // Mock ERC20 token for testing
    contract MockToken {
        name: string;
        symbol: string;
        decimals: u8;
        total_supply: u256;
        balances: HashMap<address, u256>;
        allowances: HashMap<address, HashMap<address, u256>>;
        
        constructor(name: string, symbol: string, initial_supply: u256) {
            self.name = name;
            self.symbol = symbol;
            self.decimals = 18;
            self.total_supply = initial_supply;
            self.balances.insert(msg.sender, initial_supply);
        }
        
        #[view]
        fn balance_of(&self, account: address) -> u256 {
            self.balances.get(&account).unwrap_or(&0).clone()
        }
        
        fn transfer(&mut self, to: address, amount: u256) -> Result<bool, string> {
            let from_balance = self.balance_of(msg.sender);
            require!(from_balance >= amount, "Insufficient balance");
            
            self.balances.insert(msg.sender, from_balance - amount);
            let to_balance = self.balance_of(to);
            self.balances.insert(to, to_balance + amount);
            
            Ok(true)
        }
        
        fn transfer_from(&mut self, from: address, to: address, amount: u256) -> Result<bool, string> {
            let from_balance = self.balance_of(from);
            require!(from_balance >= amount, "Insufficient balance");
            
            self.balances.insert(from, from_balance - amount);
            let to_balance = self.balance_of(to);
            self.balances.insert(to, to_balance + amount);
            
            Ok(true)
        }
        
        fn approve(&mut self, spender: address, amount: u256) -> Result<bool, string> {
            if !self.allowances.contains_key(&msg.sender) {
                self.allowances.insert(msg.sender, HashMap::new());
            }
            self.allowances.get_mut(&msg.sender).unwrap().insert(spender, amount);
            Ok(true)
        }
    }
    
    fn setup_pool() -> (LiquidityPool, MockToken, MockToken) {
        let token_a = MockToken::new("Token A".to_string(), "TKNA".to_string(), 1000000 * 1e18);
        let token_b = MockToken::new("Token B".to_string(), "TKNB".to_string(), 1000000 * 1e18);
        
        let pool = LiquidityPool::new(
            token_a.address(),
            token_b.address(),
            30,  // 0.3% fee
            5,   // 0.05% protocol fee
            test::caller(),
            500  // 5% max slippage
        );
        
        (pool, token_a, token_b)
    }
    
    #[test]
    fn test_pool_deployment() {
        let (pool, token_a, token_b) = setup_pool();
        
        let (reserve_a, reserve_b, _) = pool.get_reserves();
        assert_eq!(reserve_a, 0);
        assert_eq!(reserve_b, 0);
        assert_eq!(pool.get_total_supply(), 0);
        
        let (token_a_addr, token_b_addr, _, _, _, fee_rate) = pool.get_pool_info();
        assert_eq!(token_a_addr, token_a.address());
        assert_eq!(token_b_addr, token_b.address());
        assert_eq!(fee_rate, 30);
    }
    
    #[test]
    fn test_add_initial_liquidity() {
        let (mut pool, mut token_a, mut token_b) = setup_pool();
        
        // Approve tokens
        token_a.approve(pool.address(), 1000 * 1e18).unwrap();
        token_b.approve(pool.address(), 1000 * 1e18).unwrap();
        
        // Add initial liquidity
        let deadline = block.timestamp + 3600;
        let (amount_a, amount_b, liquidity) = pool.add_liquidity(
            1000 * 1e18,
            1000 * 1e18,
            900 * 1e18,
            900 * 1e18,
            deadline
        );
        
        assert_eq!(amount_a, 1000 * 1e18);
        assert_eq!(amount_b, 1000 * 1e18);
        assert!(liquidity > 0);
        
        let (reserve_a, reserve_b, _) = pool.get_reserves();
        assert_eq!(reserve_a, 1000 * 1e18);
        assert_eq!(reserve_b, 1000 * 1e18);
        
        assert_eq!(pool.get_liquidity_balance(test::caller()), liquidity);
    }
    
    #[test]
    fn test_swap() {
        let (mut pool, mut token_a, mut token_b) = setup_pool();
        
        // Add initial liquidity
        token_a.approve(pool.address(), 1000 * 1e18).unwrap();
        token_b.approve(pool.address(), 1000 * 1e18).unwrap();
        
        let deadline = block.timestamp + 3600;
        pool.add_liquidity(1000 * 1e18, 1000 * 1e18, 0, 0, deadline);
        
        // Perform swap
        let swap_amount = 100 * 1e18;
        token_a.approve(pool.address(), swap_amount).unwrap();
        
        let amount_out = pool.get_amount_out(swap_amount, token_a.address()).unwrap();
        let actual_out = pool.swap(token_a.address(), swap_amount, 0, deadline);
        
        assert_eq!(amount_out, actual_out);
        assert!(actual_out > 0);
        assert!(actual_out < swap_amount); // Due to fees and slippage
    }
    
    #[test]
    fn test_remove_liquidity() {
        let (mut pool, mut token_a, mut token_b) = setup_pool();
        
        // Add initial liquidity
        token_a.approve(pool.address(), 1000 * 1e18).unwrap();
        token_b.approve(pool.address(), 1000 * 1e18).unwrap();
        
        let deadline = block.timestamp + 3600;
        let (_, _, liquidity) = pool.add_liquidity(1000 * 1e18, 1000 * 1e18, 0, 0, deadline);
        
        // Remove half of liquidity
        let remove_amount = liquidity / 2;
        let (amount_a, amount_b) = pool.remove_liquidity(remove_amount, 0, 0, deadline);
        
        assert!(amount_a > 0);
        assert!(amount_b > 0);
        assert_eq!(pool.get_liquidity_balance(test::caller()), liquidity - remove_amount);
    }
    
    #[test]
    fn test_price_calculation() {
        let (mut pool, mut token_a, mut token_b) = setup_pool();
        
        // Add liquidity with 2:1 ratio (2000 A : 1000 B)
        token_a.approve(pool.address(), 2000 * 1e18).unwrap();
        token_b.approve(pool.address(), 1000 * 1e18).unwrap();
        
        let deadline = block.timestamp + 3600;
        pool.add_liquidity(2000 * 1e18, 1000 * 1e18, 0, 0, deadline);
        
        let (price_a_in_b, price_b_in_a) = pool.get_price();
        
        // Price of A in terms of B should be 0.5 (1000/2000)
        // Price of B in terms of A should be 2.0 (2000/1000)
        assert_eq!(price_a_in_b, (1000 * 1e18) / 2000); // 0.5 * 1e18
        assert_eq!(price_b_in_a, (2000 * 1e18) / 1000); // 2.0 * 1e18
    }
    
    #[test]
    fn test_slippage_protection() {
        let (mut pool, mut token_a, mut token_b) = setup_pool();
        
        // Add initial liquidity
        token_a.approve(pool.address(), 1000 * 1e18).unwrap();
        token_b.approve(pool.address(), 1000 * 1e18).unwrap();
        
        let deadline = block.timestamp + 3600;
        pool.add_liquidity(1000 * 1e18, 1000 * 1e18, 0, 0, deadline);
        
        // Try to swap a large amount that would cause high slippage
        let large_swap = 800 * 1e18; // 80% of reserves
        token_a.approve(pool.address(), large_swap).unwrap();
        
        let result = test::try_call(|| {
            pool.swap(token_a.address(), large_swap, 0, deadline);
        });
        
        // Should fail due to slippage protection
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Slippage too high"));
    }
    
    #[test]
    fn test_fee_collection() {
        let (mut pool, mut token_a, mut token_b) = setup_pool();
        
        // Add initial liquidity
        token_a.approve(pool.address(), 1000 * 1e18).unwrap();
        token_b.approve(pool.address(), 1000 * 1e18).unwrap();
        
        let deadline = block.timestamp + 3600;
        pool.add_liquidity(1000 * 1e18, 1000 * 1e18, 0, 0, deadline);
        
        // Perform several swaps to accumulate fees
        for _ in 0..5 {
            let swap_amount = 10 * 1e18;
            token_a.approve(pool.address(), swap_amount).unwrap();
            pool.swap(token_a.address(), swap_amount, 0, deadline);
        }
        
        let (fees_a, fees_b) = pool.get_accumulated_fees();
        assert!(fees_a > 0 || fees_b > 0);
        
        // Collect fees
        pool.collect_protocol_fees();
        
        let (fees_a_after, fees_b_after) = pool.get_accumulated_fees();
        assert_eq!(fees_a_after, 0);
        assert_eq!(fees_b_after, 0);
    }
}