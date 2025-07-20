// Simple ERC-20 style token contract in Augustium
// Demonstrates token functionality with safety features

contract SimpleToken {
    // Token metadata
    let name: string = "Augustium Token";
    let symbol: string = "AUG";
    let decimals: u8 = 18;
    let total_supply: u256;
    
    // State variables
    let mut balances: mapping(address => u256);
    let mut allowances: mapping(address => mapping(address => u256));
    let owner: address;
    let mut paused: bool = false;
    
    // Events
    event Transfer(from: address, to: address, value: u256);
    event Approval(owner: address, spender: address, value: u256);
    event Mint(to: address, value: u256);
    event Burn(from: address, value: u256);
    event Pause();
    event Unpause();
    
    // Modifiers
    modifier only_owner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier when_not_paused() {
        require(!paused, "Contract is paused");
        _;
    }
    
    modifier valid_address(addr: address) {
        require(addr != address(0), "Invalid address");
        _;
    }
    
    // Constructor
    constructor(initial_supply: u256) {
        require(initial_supply > 0, "Initial supply must be positive");
        
        owner = msg.sender;
        total_supply = initial_supply * 10**decimals;
        balances[owner] = total_supply;
        
        emit Transfer(address(0), owner, total_supply);
    }
    
    // View functions
    pub fn get_name() -> string {
        return name;
    }
    
    pub fn get_symbol() -> string {
        return symbol;
    }
    
    pub fn get_decimals() -> u8 {
        return decimals;
    }
    
    pub fn get_total_supply() -> u256 {
        return total_supply;
    }
    
    pub fn balance_of(account: address) -> u256 {
        return balances[account];
    }
    
    pub fn allowance(owner_addr: address, spender: address) -> u256 {
        return allowances[owner_addr][spender];
    }
    
    pub fn is_paused() -> bool {
        return paused;
    }
    
    // Transfer functions
    pub fn transfer(to: address, amount: u256) -> bool 
        when_not_paused 
        valid_address(to) 
    {
        return _transfer(msg.sender, to, amount);
    }
    
    pub fn transfer_from(from: address, to: address, amount: u256) -> bool 
        when_not_paused 
        valid_address(from) 
        valid_address(to) 
    {
        let current_allowance = allowances[from][msg.sender];
        require(current_allowance >= amount, "Transfer amount exceeds allowance");
        
        allowances[from][msg.sender] = current_allowance - amount;
        
        return _transfer(from, to, amount);
    }
    
    // Approval functions
    pub fn approve(spender: address, amount: u256) -> bool 
        when_not_paused 
        valid_address(spender) 
    {
        allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }
    
    pub fn increase_allowance(spender: address, added_value: u256) -> bool 
        when_not_paused 
        valid_address(spender) 
    {
        let new_allowance = allowances[msg.sender][spender] + added_value;
        allowances[msg.sender][spender] = new_allowance;
        emit Approval(msg.sender, spender, new_allowance);
        return true;
    }
    
    pub fn decrease_allowance(spender: address, subtracted_value: u256) -> bool 
        when_not_paused 
        valid_address(spender) 
    {
        let current_allowance = allowances[msg.sender][spender];
        require(current_allowance >= subtracted_value, "Decreased allowance below zero");
        
        let new_allowance = current_allowance - subtracted_value;
        allowances[msg.sender][spender] = new_allowance;
        emit Approval(msg.sender, spender, new_allowance);
        return true;
    }
    
    // Owner functions
    pub fn mint(to: address, amount: u256) 
        only_owner 
        when_not_paused 
        valid_address(to) 
    {
        require(amount > 0, "Mint amount must be positive");
        
        total_supply += amount;
        balances[to] += amount;
        
        emit Transfer(address(0), to, amount);
        emit Mint(to, amount);
    }
    
    pub fn burn(amount: u256) when_not_paused {
        require(amount > 0, "Burn amount must be positive");
        require(balances[msg.sender] >= amount, "Burn amount exceeds balance");
        
        balances[msg.sender] -= amount;
        total_supply -= amount;
        
        emit Transfer(msg.sender, address(0), amount);
        emit Burn(msg.sender, amount);
    }
    
    pub fn burn_from(from: address, amount: u256) 
        when_not_paused 
        valid_address(from) 
    {
        require(amount > 0, "Burn amount must be positive");
        
        let current_allowance = allowances[from][msg.sender];
        require(current_allowance >= amount, "Burn amount exceeds allowance");
        require(balances[from] >= amount, "Burn amount exceeds balance");
        
        allowances[from][msg.sender] = current_allowance - amount;
        balances[from] -= amount;
        total_supply -= amount;
        
        emit Transfer(from, address(0), amount);
        emit Burn(from, amount);
    }
    
    pub fn pause() only_owner {
        require(!paused, "Contract is already paused");
        paused = true;
        emit Pause();
    }
    
    pub fn unpause() only_owner {
        require(paused, "Contract is not paused");
        paused = false;
        emit Unpause();
    }
    
    // Internal functions
    fn _transfer(from: address, to: address, amount: u256) -> bool {
        require(amount > 0, "Transfer amount must be positive");
        require(balances[from] >= amount, "Transfer amount exceeds balance");
        
        balances[from] -= amount;
        balances[to] += amount;
        
        emit Transfer(from, to, amount);
        return true;
    }
    
    // Emergency functions
    pub fn emergency_withdraw() only_owner {
        // In case of emergency, owner can withdraw all ETH
        let balance = address(this).balance;
        if (balance > 0) {
            payable(owner).transfer(balance);
        }
    }
    
    // Batch operations for gas efficiency
    pub fn batch_transfer(recipients: address[], amounts: u256[]) -> bool 
        when_not_paused 
    {
        require(recipients.length == amounts.length, "Arrays length mismatch");
        require(recipients.length > 0, "Empty arrays");
        
        for (let i = 0; i < recipients.length; i++) {
            require(_transfer(msg.sender, recipients[i], amounts[i]), "Transfer failed");
        }
        
        return true;
    }
}