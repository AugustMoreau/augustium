# Examples Guide

Comprehensive examples showing Augustium features and best practices.

## Table of Contents

- [Basic Examples](#basic-examples)
- [Token Contracts](#token-contracts)
- [DeFi Examples](#defi-examples)
- [Governance](#governance)
- [Advanced Patterns](#advanced-patterns)
- [Security Examples](#security-examples)

## Basic Examples

### Hello World Contract

```augustium
// examples/hello_world.aug

contract HelloWorld {
    owner: address;
    message: string;
    
    constructor(initial_message: string) {
        owner = msg.sender;
        message = initial_message;
    }
    
    pub fn get_message() -> string {
        return message;
    }
    
    pub fn set_message(new_message: string) {
        require(msg.sender == owner, "Only owner can update message");
        message = new_message;
        emit MessageUpdated(new_message);
    }
    
    event MessageUpdated(message: string);
}
```

### Simple Storage

```augustium
// examples/simple_storage.aug

contract SimpleStorage {
    stored_data: u256;
    owner: address;
    
    constructor() {
        owner = msg.sender;
        stored_data = 0;
    }
    
    pub fn set(value: u256) {
        require(msg.sender == owner, "Not authorized");
        stored_data = value;
        emit DataStored(value);
    }
    
    pub fn get() -> u256 {
        return stored_data;
    }
    
    pub fn increment() {
        require(msg.sender == owner, "Not authorized");
        stored_data += 1;  // Safe - no overflow
        emit DataStored(stored_data);
    }
    
    event DataStored(value: u256);
}
```

## Token Contracts

### Basic ERC20 Token

```augustium
// examples/basic_token.aug

contract MyToken {
    total_supply: u256;
    balances: mapping<address, u256>;
    allowances: mapping<address, mapping<address, u256>>;
    
    name: string;
    symbol: string;
    decimals: u8;
    
    constructor(_name: string, _symbol: string, _total_supply: u256) {
        name = _name;
        symbol = _symbol;
        decimals = 18;
        total_supply = _total_supply;
        balances[msg.sender] = _total_supply;
        
        emit Transfer(address(0), msg.sender, _total_supply);
    }
    
    pub fn balance_of(account: address) -> u256 {
        return balances[account];
    }
    
    pub fn transfer(to: address, amount: u256) -> bool {
        require(to != address(0), "Transfer to zero address");
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        balances[msg.sender] -= amount;
        balances[to] += amount;
        
        emit Transfer(msg.sender, to, amount);
        return true;
    }
    
    pub fn approve(spender: address, amount: u256) -> bool {
        require(spender != address(0), "Approve to zero address");
        allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }
    
    pub fn transfer_from(from: address, to: address, amount: u256) -> bool {
        require(from != address(0), "Transfer from zero address");
        require(to != address(0), "Transfer to zero address");
        require(balances[from] >= amount, "Insufficient balance");
        require(allowances[from][msg.sender] >= amount, "Insufficient allowance");
        
        balances[from] -= amount;
        balances[to] += amount;
        allowances[from][msg.sender] -= amount;
        
        emit Transfer(from, to, amount);
        return true;
    }
    
    event Transfer(from: address, to: address, value: u256);
    event Approval(owner: address, spender: address, value: u256);
}
```

### NFT Contract

```augustium
// examples/simple_nft.aug

contract SimpleNFT {
    // Token tracking
    owners: mapping<u256, address>;
    balances: mapping<address, u256>;
    token_approvals: mapping<u256, address>;
    operator_approvals: mapping<address, mapping<address, bool>>;
    
    // Metadata
    name: string;
    symbol: string;
    base_uri: string;
    
    // Token counter
    current_token_id: u256 = 0;
    
    // Contract owner (for minting)
    contract_owner: address;
    
    constructor(_name: string, _symbol: string, _base_uri: string) {
        name = _name;
        symbol = _symbol;
        base_uri = _base_uri;
        contract_owner = msg.sender;
    }
    
    pub fn mint(to: address) -> u256 {
        require(msg.sender == contract_owner, "Only owner can mint");
        require(to != address(0), "Mint to zero address");
        
        current_token_id += 1;
        let token_id = current_token_id;
        
        owners[token_id] = to;
        balances[to] += 1;
        
        emit Transfer(address(0), to, token_id);
        return token_id;
    }
    
    pub fn owner_of(token_id: u256) -> address {
        let owner = owners[token_id];
        require(owner != address(0), "Token doesn't exist");
        return owner;
    }
    
    pub fn balance_of(owner: address) -> u256 {
        require(owner != address(0), "Balance of zero address");
        return balances[owner];
    }
    
    pub fn approve(to: address, token_id: u256) {
        let owner = owner_of(token_id);
        require(msg.sender == owner, "Not token owner");
        require(to != owner, "Approve to current owner");
        
        token_approvals[token_id] = to;
        emit Approval(owner, to, token_id);
    }
    
    pub fn transfer_from(from: address, to: address, token_id: u256) {
        require(is_approved_or_owner(msg.sender, token_id), "Not approved or owner");
        require(to != address(0), "Transfer to zero address");
        
        // Clear approval
        delete token_approvals[token_id];
        
        // Update balances and owner
        balances[from] -= 1;
        balances[to] += 1;
        owners[token_id] = to;
        
        emit Transfer(from, to, token_id);
    }
    
    fn is_approved_or_owner(spender: address, token_id: u256) -> bool {
        let owner = owner_of(token_id);
        return spender == owner || 
               token_approvals[token_id] == spender ||
               operator_approvals[owner][spender];
    }
    
    event Transfer(from: address, to: address, token_id: u256);
    event Approval(owner: address, approved: address, token_id: u256);
}
```

## DeFi Examples

### Simple DEX

```augustium
// examples/simple_dex.aug

contract SimpleDEX {
    // Token pair
    token_a: address;
    token_b: address;
    
    // Reserves
    reserve_a: u256;
    reserve_b: u256;
    
    // LP tracking
    total_lp_tokens: u256;
    lp_balances: mapping<address, u256>;
    
    constructor(token_a_addr: address, token_b_addr: address) {
        token_a = token_a_addr;
        token_b = token_b_addr;
    }
    
    // Add liquidity
    pub fn add_liquidity(amount_a: u256, amount_b: u256) -> u256 {
        require(amount_a > 0 && amount_b > 0, "Amounts must be positive");
        
        // Transfer tokens
        IERC20(token_a).transfer_from(msg.sender, address(this), amount_a);
        IERC20(token_b).transfer_from(msg.sender, address(this), amount_b);
        
        // Calculate LP tokens
        let lp_tokens: u256;
        if (total_lp_tokens == 0) {
            lp_tokens = sqrt(amount_a * amount_b);
        } else {
            lp_tokens = min(
                (amount_a * total_lp_tokens) / reserve_a,
                (amount_b * total_lp_tokens) / reserve_b
            );
        }
        
        lp_balances[msg.sender] += lp_tokens;
        total_lp_tokens += lp_tokens;
        
        reserve_a += amount_a;
        reserve_b += amount_b;
        
        emit LiquidityAdded(msg.sender, amount_a, amount_b, lp_tokens);
        return lp_tokens;
    }
    
    // Swap A for B
    pub fn swap_a_for_b(amount_in: u256) -> u256 {
        require(amount_in > 0, "Amount must be positive");
        
        let amount_out = (amount_in * reserve_b) / (reserve_a + amount_in);
        require(amount_out > 0, "Insufficient output");
        
        IERC20(token_a).transfer_from(msg.sender, address(this), amount_in);
        IERC20(token_b).transfer(msg.sender, amount_out);
        
        reserve_a += amount_in;
        reserve_b -= amount_out;
        
        emit Swap(msg.sender, amount_in, amount_out);
        return amount_out;
    }
    
    fn min(a: u256, b: u256) -> u256 {
        return if a < b { a } else { b };
    }
    
    fn sqrt(x: u256) -> u256 {
        // Simplified square root
        if (x == 0) return 0;
        let z = (x + 1) / 2;
        let y = x;
        while (z < y) {
            y = z;
            z = (x / z + z) / 2;
        }
        return y;
    }
    
    event LiquidityAdded(user: address, amount_a: u256, amount_b: u256, lp_tokens: u256);
    event Swap(user: address, amount_in: u256, amount_out: u256);
}
```

## Governance

### Simple DAO

```augustium
// examples/simple_dao.aug

contract SimpleDAO {
    struct Proposal {
        id: u256;
        title: string;
        description: string;
        proposer: address;
        target: address;
        value: u256;
        data: bytes;
        start_time: u256;
        end_time: u256;
        for_votes: u256;
        against_votes: u256;
        executed: bool;
    }
    
    // State
    proposals: mapping<u256, Proposal>;
    votes: mapping<u256, mapping<address, bool>>;
    proposal_count: u256 = 0;
    
    // Governance token
    governance_token: address;
    
    // Parameters
    voting_period: u256 = 3 days;
    proposal_threshold: u256 = 1000 * 10**18;  // 1000 tokens
    
    constructor(token_address: address) {
        governance_token = token_address;
    }
    
    pub fn propose(
        title: string,
        description: string,
        target: address,
        value: u256,
        data: bytes
    ) -> u256 {
        let balance = IERC20(governance_token).balance_of(msg.sender);
        require(balance >= proposal_threshold, "Insufficient tokens");
        
        proposal_count += 1;
        let id = proposal_count;
        
        proposals[id] = Proposal({
            id: id,
            title: title,
            description: description,
            proposer: msg.sender,
            target: target,
            value: value,
            data: data,
            start_time: block.timestamp,
            end_time: block.timestamp + voting_period,
            for_votes: 0,
            against_votes: 0,
            executed: false
        });
        
        emit ProposalCreated(id, title, msg.sender);
        return id;
    }
    
    pub fn vote(proposal_id: u256, support: bool) {
        require(proposals[proposal_id].id != 0, "Proposal doesn't exist");
        require(!votes[proposal_id][msg.sender], "Already voted");
        require(block.timestamp <= proposals[proposal_id].end_time, "Voting ended");
        
        let voting_power = IERC20(governance_token).balance_of(msg.sender);
        require(voting_power > 0, "No voting power");
        
        votes[proposal_id][msg.sender] = true;
        
        if (support) {
            proposals[proposal_id].for_votes += voting_power;
        } else {
            proposals[proposal_id].against_votes += voting_power;
        }
        
        emit VoteCast(msg.sender, proposal_id, support, voting_power);
    }
    
    pub fn execute(proposal_id: u256) {
        let proposal = proposals[proposal_id];
        require(proposal.id != 0, "Proposal doesn't exist");
        require(!proposal.executed, "Already executed");
        require(block.timestamp > proposal.end_time, "Voting not ended");
        require(proposal.for_votes > proposal.against_votes, "Proposal failed");
        
        proposals[proposal_id].executed = true;
        
        // Execute the proposal
        let success = proposal.target.call{value: proposal.value}(proposal.data);
        require(success, "Execution failed");
        
        emit ProposalExecuted(proposal_id);
    }
    
    event ProposalCreated(id: u256, title: string, proposer: address);
    event VoteCast(voter: address, proposal_id: u256, support: bool, weight: u256);
    event ProposalExecuted(id: u256);
}
```

## Advanced Patterns

### Proxy Pattern

```augustium
// examples/proxy_pattern.aug

// Upgradeable proxy contract
contract UpgradeableProxy {
    // Storage slot for implementation address
    implementation: address;
    admin: address;
    
    constructor(initial_implementation: address) {
        implementation = initial_implementation;
        admin = msg.sender;
    }
    
    modifier only_admin() {
        require(msg.sender == admin, "Not admin");
        _;
    }
    
    pub fn upgrade(new_implementation: address) {
        require(msg.sender == admin, "Not admin");
        require(new_implementation != address(0), "Invalid implementation");
        
        implementation = new_implementation;
        emit Upgraded(new_implementation);
    }
    
    // Fallback function to delegate calls
    fallback() {
        let impl = implementation;
        assembly {
            calldatacopy(0, 0, calldatasize())
            let result := delegatecall(gas(), impl, 0, calldatasize(), 0, 0)
            returndatacopy(0, 0, returndatasize())
            
            match result
            case 0 { revert(0, returndatasize()) }
            default { return(0, returndatasize()) }
        }
    }
    
    event Upgraded(new_implementation: address);
}
```

### MultiSig Wallet

```augustium
// examples/multisig_wallet.aug

contract MultiSigWallet {
    struct Transaction {
        to: address;
        value: u256;
        data: bytes;
        executed: bool;
        confirmations: u256;
    }
    
    // State
    owners: Vec<address>;
    is_owner: mapping<address, bool>;
    required_confirmations: u256;
    
    transactions: mapping<u256, Transaction>;
    confirmations: mapping<u256, mapping<address, bool>>;
    transaction_count: u256 = 0;
    
    constructor(owners_list: Vec<address>, required: u256) {
        require(owners_list.len() > 0, "Need at least one owner");
        require(required > 0 && required <= owners_list.len(), "Invalid required confirmations");
        
        for owner in owners_list.iter() {
            require(*owner != address(0), "Invalid owner");
            require(!is_owner[*owner], "Duplicate owner");
            
            is_owner[*owner] = true;
            owners.push(*owner);
        }
        
        required_confirmations = required;
    }
    
    modifier only_owner() {
        require(is_owner[msg.sender], "Not owner");
        _;
    }
    
    pub fn submit_transaction(to: address, value: u256, data: bytes) -> u256 {
        require(msg.sender == is_owner[msg.sender], "Not owner");
        
        let tx_id = add_transaction(to, value, data);
        confirm_transaction(tx_id);
        return tx_id;
    }
    
    fn add_transaction(to: address, value: u256, data: bytes) -> u256 {
        transaction_count += 1;
        let tx_id = transaction_count;
        
        transactions[tx_id] = Transaction({
            to: to,
            value: value,
            data: data,
            executed: false,
            confirmations: 0
        });
        
        emit TransactionSubmitted(tx_id, to, value);
        return tx_id;
    }
    
    #[only_owner]
    pub fn confirm_transaction(tx_id: u256) {
        require(transactions[tx_id].to != address(0), "Transaction doesn't exist");
        require(!confirmations[tx_id][msg.sender], "Already confirmed");
        require(!transactions[tx_id].executed, "Already executed");
        
        confirmations[tx_id][msg.sender] = true;
        transactions[tx_id].confirmations += 1;
        
        emit TransactionConfirmed(tx_id, msg.sender);
        
        if (transactions[tx_id].confirmations >= required_confirmations) {
            execute_transaction(tx_id);
        }
    }
    
    fn execute_transaction(tx_id: u256) {
        require(!transactions[tx_id].executed, "Already executed");
        require(transactions[tx_id].confirmations >= required_confirmations, "Not enough confirmations");
        
        transactions[tx_id].executed = true;
        
        let success = transactions[tx_id].to.call{value: transactions[tx_id].value}(transactions[tx_id].data);
        require(success, "Transaction execution failed");
        
        emit TransactionExecuted(tx_id);
    }
    
    event TransactionSubmitted(tx_id: u256, to: address, value: u256);
    event TransactionConfirmed(tx_id: u256, owner: address);
    event TransactionExecuted(tx_id: u256);
}
```

## Security Examples

### Reentrancy Protection

```augustium
// examples/reentrancy_protection.aug

contract BankWithProtection {
    balances: mapping<address, u256>;
    
    // Built-in reentrancy protection
    #[nonreentrant]
    pub fn withdraw(amount: u256) {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        balances[msg.sender] -= amount;
        
        // Safe to make external call after state update
        msg.sender.call{value: amount}("");
        
        emit Withdrawal(msg.sender, amount);
    }
    
    pub fn deposit() {
        balances[msg.sender] += msg.value;
        emit Deposit(msg.sender, msg.value);
    }
    
    event Deposit(user: address, amount: u256);
    event Withdrawal(user: address, amount: u256);
}
```

### Access Control

```augustium
// examples/access_control.aug

contract AccessControlledContract {
    // Role-based access control
    roles: mapping<bytes32, mapping<address, bool>>;
    
    // Role constants
    ADMIN_ROLE: bytes32 = keccak256("ADMIN_ROLE");
    MINTER_ROLE: bytes32 = keccak256("MINTER_ROLE");
    PAUSER_ROLE: bytes32 = keccak256("PAUSER_ROLE");
    
    constructor() {
        // Grant admin role to deployer
        roles[ADMIN_ROLE][msg.sender] = true;
        emit RoleGranted(ADMIN_ROLE, msg.sender, msg.sender);
    }
    
    modifier has_role(role: bytes32) {
        require(roles[role][msg.sender], "Missing required role");
        _;
    }
    
    #[has_role(ADMIN_ROLE)]
    pub fn grant_role(role: bytes32, account: address) {
        require(!roles[role][account], "Account already has role");
        
        roles[role][account] = true;
        emit RoleGranted(role, account, msg.sender);
    }
    
    #[has_role(ADMIN_ROLE)]  
    pub fn revoke_role(role: bytes32, account: address) {
        require(roles[role][account], "Account doesn't have role");
        
        roles[role][account] = false;
        emit RoleRevoked(role, account, msg.sender);
    }
    
    #[has_role(MINTER_ROLE)]
    pub fn mint(to: address, amount: u256) {
        // Only accounts with MINTER_ROLE can call this
        // Minting logic here
        emit Mint(to, amount);
    }
    
    pub fn has_role_check(role: bytes32, account: address) -> bool {
        return roles[role][account];
    }
    
    event RoleGranted(role: bytes32, account: address, sender: address);
    event RoleRevoked(role: bytes32, account: address, sender: address);
    event Mint(to: address, amount: u256);
}
```

---

For more examples, check the `examples/` directory in the repository. Each example includes tests showing how to use the contracts safely and effectively.
