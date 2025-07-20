// Basic token contract - like ERC-20 but with our safety features
// Good starting point for building tokens

use std::collections::HashMap;
use std::events::Event;
use std::access::Ownable;

// Events
#[derive(Event)]
struct Transfer {
    from: address,
    to: address,
    value: u256
}

#[derive(Event)]
struct Approval {
    owner: address,
    spender: address,
    value: u256
}

// Token contract with some safety checks built in
contract SimpleToken extends Ownable {
    // Token metadata
    name: string;
    symbol: string;
    decimals: u8;
    total_supply: u256;
    
    // Balances and allowances
    balances: HashMap<address, u256>;
    allowances: HashMap<address, HashMap<address, u256>>;
    
    // Constructor
    constructor(
        token_name: string,
        token_symbol: string,
        token_decimals: u8,
        initial_supply: u256
    ) {
        require!(!token_name.is_empty(), "Name cannot be empty");
        require!(!token_symbol.is_empty(), "Symbol cannot be empty");
        require!(initial_supply > 0, "Initial supply must be positive");
        
        self.name = token_name;
        self.symbol = token_symbol;
        self.decimals = token_decimals;
        self.total_supply = initial_supply;
        
        // Mint initial supply to deployer
        self.balances.insert(msg.sender, initial_supply);
        
        emit Transfer {
            from: address::zero(),
            to: msg.sender,
            value: initial_supply
        };
    }
    
    // View functions
    #[view]
    fn get_name(&self) -> string {
        self.name.clone()
    }
    
    #[view]
    fn get_symbol(&self) -> string {
        self.symbol.clone()
    }
    
    #[view]
    fn get_decimals(&self) -> u8 {
        self.decimals
    }
    
    #[view]
    fn get_total_supply(&self) -> u256 {
        self.total_supply
    }
    
    #[view]
    fn balance_of(&self, account: address) -> u256 {
        self.balances.get(&account).unwrap_or(&0).clone()
    }
    
    #[view]
    fn allowance(&self, owner: address, spender: address) -> u256 {
        self.allowances
            .get(&owner)
            .and_then(|owner_allowances| owner_allowances.get(&spender))
            .unwrap_or(&0)
            .clone()
    }
    
    // Transfer functions
    #[payable(false)]
    fn transfer(&mut self, to: address, amount: u256) -> bool {
        self._transfer(msg.sender, to, amount)
    }
    
    #[payable(false)]
    fn transfer_from(&mut self, from: address, to: address, amount: u256) -> bool {
        let current_allowance = self.allowance(from, msg.sender);
        require!(current_allowance >= amount, "Transfer amount exceeds allowance");
        
        // Update allowance (automatic overflow protection)
        self._approve(from, msg.sender, current_allowance - amount);
        self._transfer(from, to, amount)
    }
    
    #[payable(false)]
    fn approve(&mut self, spender: address, amount: u256) -> bool {
        self._approve(msg.sender, spender, amount)
    }
    
    // Convenience functions
    #[payable(false)]
    fn increase_allowance(&mut self, spender: address, added_value: u256) -> bool {
        let current_allowance = self.allowance(msg.sender, spender);
        // Automatic overflow protection
        self._approve(msg.sender, spender, current_allowance + added_value)
    }
    
    #[payable(false)]
    fn decrease_allowance(&mut self, spender: address, subtracted_value: u256) -> bool {
        let current_allowance = self.allowance(msg.sender, spender);
        require!(current_allowance >= subtracted_value, "Decreased allowance below zero");
        self._approve(msg.sender, spender, current_allowance - subtracted_value)
    }
    
    // Owner-only functions
    #[only_owner]
    fn mint(&mut self, to: address, amount: u256) {
        require!(to != address::zero(), "Cannot mint to zero address");
        require!(amount > 0, "Amount must be positive");
        
        // Automatic overflow protection
        self.total_supply += amount;
        let to_balance = self.balance_of(to);
        self.balances.insert(to, to_balance + amount);
        
        emit Transfer {
            from: address::zero(),
            to,
            value: amount
        };
    }
    
    #[payable(false)]
    fn burn(&mut self, amount: u256) {
        self._burn(msg.sender, amount);
    }
    
    #[payable(false)]
    fn burn_from(&mut self, account: address, amount: u256) {
        let current_allowance = self.allowance(account, msg.sender);
        require!(current_allowance >= amount, "Burn amount exceeds allowance");
        
        self._approve(account, msg.sender, current_allowance - amount);
        self._burn(account, amount);
    }
    
    // Internal functions
    fn _transfer(&mut self, from: address, to: address, amount: u256) -> bool {
        require!(from != address::zero(), "Transfer from zero address");
        require!(to != address::zero(), "Transfer to zero address");
        require!(amount > 0, "Transfer amount must be positive");
        
        let from_balance = self.balance_of(from);
        require!(from_balance >= amount, "Transfer amount exceeds balance");
        
        // Update balances (automatic overflow protection)
        self.balances.insert(from, from_balance - amount);
        let to_balance = self.balance_of(to);
        self.balances.insert(to, to_balance + amount);
        
        emit Transfer { from, to, value: amount };
        true
    }
    
    fn _approve(&mut self, owner: address, spender: address, amount: u256) -> bool {
        require!(owner != address::zero(), "Approve from zero address");
        require!(spender != address::zero(), "Approve to zero address");
        
        // Initialize nested HashMap if needed
        if !self.allowances.contains_key(&owner) {
            self.allowances.insert(owner, HashMap::new());
        }
        
        self.allowances.get_mut(&owner).unwrap().insert(spender, amount);
        
        emit Approval { owner, spender, value: amount };
        true
    }
    
    fn _burn(&mut self, account: address, amount: u256) {
        require!(account != address::zero(), "Burn from zero address");
        require!(amount > 0, "Burn amount must be positive");
        
        let account_balance = self.balance_of(account);
        require!(account_balance >= amount, "Burn amount exceeds balance");
        
        // Update balances and total supply (automatic overflow protection)
        self.balances.insert(account, account_balance - amount);
        self.total_supply -= amount;
        
        emit Transfer {
            from: account,
            to: address::zero(),
            value: amount
        };
    }
    
    // Utility functions
    #[view]
    fn get_token_info(&self) -> (string, string, u8, u256) {
        (self.name.clone(), self.symbol.clone(), self.decimals, self.total_supply)
    }
    
    #[view]
    fn get_account_info(&self, account: address) -> (u256, u256) {
        let balance = self.balance_of(account);
        let total_allowance = self._get_total_allowance_given(account);
        (balance, total_allowance)
    }
    
    fn _get_total_allowance_given(&self, owner: address) -> u256 {
        if let Some(owner_allowances) = self.allowances.get(&owner) {
            let mut total = 0u256;
            for (_, &allowance) in owner_allowances.iter() {
                total += allowance; // Automatic overflow protection
            }
            total
        } else {
            0
        }
    }
}

// Example usage and deployment script
#[cfg(test)]
mod tests {
    use super::*;
    use std::testing::*;
    
    #[test]
    fn test_deployment() {
        let token = SimpleToken::new(
            "Test Token",
            "TEST",
            18,
            1000000 * 10u256.pow(18) // 1 million tokens
        );
        
        assert_eq!(token.get_name(), "Test Token");
        assert_eq!(token.get_symbol(), "TEST");
        assert_eq!(token.get_decimals(), 18);
        assert_eq!(token.get_total_supply(), 1000000 * 10u256.pow(18));
        assert_eq!(token.balance_of(test::caller()), 1000000 * 10u256.pow(18));
    }
    
    #[test]
    fn test_transfer() {
        let mut token = SimpleToken::new("Test", "TEST", 18, 1000);
        let recipient = address::from_hex("0x1234567890123456789012345678901234567890");
        
        // Transfer tokens
        assert!(token.transfer(recipient, 100));
        
        // Check balances
        assert_eq!(token.balance_of(test::caller()), 900);
        assert_eq!(token.balance_of(recipient), 100);
    }
    
    #[test]
    fn test_approval_and_transfer_from() {
        let mut token = SimpleToken::new("Test", "TEST", 18, 1000);
        let spender = address::from_hex("0x1234567890123456789012345678901234567890");
        let recipient = address::from_hex("0x0987654321098765432109876543210987654321");
        
        // Approve spender
        assert!(token.approve(spender, 200));
        assert_eq!(token.allowance(test::caller(), spender), 200);
        
        // Simulate spender calling transfer_from
        test::set_caller(spender);
        assert!(token.transfer_from(test::original_caller(), recipient, 150));
        
        // Check balances and allowance
        assert_eq!(token.balance_of(test::original_caller()), 850);
        assert_eq!(token.balance_of(recipient), 150);
        assert_eq!(token.allowance(test::original_caller(), spender), 50);
    }
    
    #[test]
    fn test_minting() {
        let mut token = SimpleToken::new("Test", "TEST", 18, 1000);
        let recipient = address::from_hex("0x1234567890123456789012345678901234567890");
        
        // Mint new tokens
        token.mint(recipient, 500);
        
        // Check updated balances and total supply
        assert_eq!(token.balance_of(recipient), 500);
        assert_eq!(token.get_total_supply(), 1500);
    }
    
    #[test]
    fn test_burning() {
        let mut token = SimpleToken::new("Test", "TEST", 18, 1000);
        
        // Burn tokens
        token.burn(200);
        
        // Check updated balance and total supply
        assert_eq!(token.balance_of(test::caller()), 800);
        assert_eq!(token.get_total_supply(), 800);
    }
    
    #[test]
    fn test_overflow_protection() {
        let mut token = SimpleToken::new("Test", "TEST", 18, u256::MAX - 100);
        
        // This should fail due to overflow protection
        let result = test::try_call(|| {
            token.mint(test::caller(), 200);
        });
        
        assert!(result.is_err());
    }
    
    #[test]
    fn test_access_control() {
        let mut token = SimpleToken::new("Test", "TEST", 18, 1000);
        let non_owner = address::from_hex("0x1234567890123456789012345678901234567890");
        
        // Try to mint from non-owner account
        test::set_caller(non_owner);
        let result = test::try_call(|| {
            token.mint(non_owner, 100);
        });
        
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Only owner"));
    }
}