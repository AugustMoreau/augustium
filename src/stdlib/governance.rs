// Governance and DAO functionality
// Voting, proposals, and democratic decision making

use crate::error::{Result, VmError, VmErrorKind, CompilerError};
use crate::stdlib::core_types::{Address, U256, AugustiumType};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

// Different states a governance proposal can be in
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProposalState {
    Pending,
    Active,
    Succeeded,
    Defeated,
    Queued,
    Executed,
    Canceled,
    Expired,
}

/// Vote types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VoteType {
    For,
    Against,
    Abstain,
}

/// Proposal structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    pub id: U256,
    pub proposer: Address,
    pub title: String,
    pub description: String,
    pub targets: Vec<Address>,
    pub values: Vec<U256>,
    pub signatures: Vec<String>,
    pub calldatas: Vec<Vec<u8>>,
    pub start_block: U256,
    pub end_block: U256,
    pub for_votes: U256,
    pub against_votes: U256,
    pub abstain_votes: U256,
    pub state: ProposalState,
    pub eta: U256, // Execution time for queued proposals
    pub quorum_votes: U256,
    pub votes: HashMap<Address, Vote>,
}

/// Individual vote
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    pub voter: Address,
    pub vote_type: VoteType,
    pub weight: U256,
    pub reason: String,
    pub block_number: U256,
}

impl Proposal {
    /// Create a new proposal
    pub fn new(
        id: U256,
        proposer: Address,
        title: String,
        description: String,
        targets: Vec<Address>,
        values: Vec<U256>,
        signatures: Vec<String>,
        calldatas: Vec<Vec<u8>>,
        start_block: U256,
        voting_period: U256,
        quorum_votes: U256,
    ) -> Result<Self> {
        if targets.len() != values.len() || 
           values.len() != signatures.len() || 
           signatures.len() != calldatas.len() {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InvalidInput,
                "Proposal arrays must have equal length".to_string(),
            )));
        }

        Ok(Self {
            id,
            proposer,
            title,
            description,
            targets,
            values,
            signatures,
            calldatas,
            start_block,
            end_block: start_block + voting_period,
            for_votes: U256::zero(),
            against_votes: U256::zero(),
            abstain_votes: U256::zero(),
            state: ProposalState::Pending,
            eta: U256::zero(),
            quorum_votes,
            votes: HashMap::new(),
        })
    }

    /// Update proposal state based on current block
    pub fn update_state(&mut self, current_block: U256, timelock_delay: U256) {
        match self.state {
            ProposalState::Pending => {
                if current_block >= self.start_block {
                    self.state = ProposalState::Active;
                }
            }
            ProposalState::Active => {
                if current_block > self.end_block {
                    if self.for_votes > self.against_votes && self.for_votes >= self.quorum_votes {
                        self.state = ProposalState::Succeeded;
                    } else {
                        self.state = ProposalState::Defeated;
                    }
                }
            }
            ProposalState::Succeeded => {
                // Proposals can be queued after succeeding
            }
            ProposalState::Queued => {
                if current_block >= self.eta {
                    // Ready for execution
                } else if current_block > self.eta + timelock_delay * U256::new(2) {
                    self.state = ProposalState::Expired;
                }
            }
            _ => {} // Terminal states
        }
    }

    /// Cast a vote on the proposal
    pub fn cast_vote(
        &mut self,
        voter: Address,
        vote_type: VoteType,
        weight: U256,
        reason: String,
        current_block: U256,
    ) -> Result<()> {
        if self.state != ProposalState::Active {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InvalidState,
                "Proposal is not active for voting".to_string(),
            )));
        }

        if current_block > self.end_block {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::VotingEnded,
                "Voting period has ended".to_string(),
            )));
        }

        // Check if voter has already voted
        if self.votes.contains_key(&voter) {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::AlreadyVoted,
                "Voter has already cast a vote".to_string(),
            )));
        }

        // Record the vote
        let vote = Vote {
            voter,
            vote_type: vote_type.clone(),
            weight,
            reason,
            block_number: current_block,
        };

        // Update vote tallies
        match vote_type {
            VoteType::For => self.for_votes = self.for_votes + weight,
            VoteType::Against => self.against_votes = self.against_votes + weight,
            VoteType::Abstain => self.abstain_votes = self.abstain_votes + weight,
        }

        self.votes.insert(voter, vote);
        Ok(())
    }

    /// Queue the proposal for execution
    pub fn queue(&mut self, current_block: U256, timelock_delay: U256) -> Result<()> {
        if self.state != ProposalState::Succeeded {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InvalidState,
                "Proposal must be succeeded to queue".to_string(),
            )));
        }

        self.state = ProposalState::Queued;
        self.eta = current_block + timelock_delay;
        Ok(())
    }

    /// Execute the proposal
    pub fn execute(&mut self, current_block: U256) -> Result<()> {
        if self.state != ProposalState::Queued {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InvalidState,
                "Proposal must be queued to execute".to_string(),
            )));
        }

        if current_block < self.eta {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::TimelockNotReady,
                "Timelock delay has not passed".to_string(),
            )));
        }

        self.state = ProposalState::Executed;
        Ok(())
    }

    /// Cancel the proposal
    pub fn cancel(&mut self, canceler: Address) -> Result<()> {
        // Only proposer or governance contract can cancel
        if canceler != self.proposer {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::Unauthorized,
                "Only proposer can cancel proposal".to_string(),
            )));
        }

        match self.state {
            ProposalState::Executed | ProposalState::Canceled => {
                return Err(CompilerError::VmError(VmError::new(
                    VmErrorKind::InvalidState,
                    "Cannot cancel executed or already canceled proposal".to_string(),
                )));
            }
            _ => {
                self.state = ProposalState::Canceled;
                Ok(())
            }
        }
    }

    /// Get vote participation rate
    pub fn participation_rate(&self, total_supply: U256) -> U256 {
        if total_supply.is_zero() {
            return U256::zero();
        }
        
        let total_votes = self.for_votes + self.against_votes + self.abstain_votes;
        (total_votes * U256::new(10000)) / total_supply
    }

    /// Check if proposal has reached quorum
    pub fn has_quorum(&self) -> bool {
        self.for_votes + self.against_votes + self.abstain_votes >= self.quorum_votes
    }
}

/// Governance token for voting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceToken {
    pub name: String,
    pub symbol: String,
    pub total_supply: U256,
    pub balances: HashMap<Address, U256>,
    pub delegates: HashMap<Address, Address>, // delegator -> delegate
    pub voting_power: HashMap<Address, U256>, // delegate -> voting power
    pub checkpoints: HashMap<Address, Vec<Checkpoint>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub block_number: U256,
    pub votes: U256,
}

impl GovernanceToken {
    /// Create a new governance token
    pub fn new(name: String, symbol: String, total_supply: U256, initial_holder: Address) -> Self {
        let mut balances = HashMap::new();
        balances.insert(initial_holder, total_supply);
        
        // Initialize voting power to zero - users need to self-delegate to vote
        let voting_power = HashMap::new();
        
        Self {
            name,
            symbol,
            total_supply,
            balances,
            delegates: HashMap::new(),
            voting_power,
            checkpoints: HashMap::new(),
        }
    }

    /// Transfer tokens
    pub fn transfer(&mut self, from: Address, to: Address, amount: U256) -> Result<()> {
        let from_balance = self.balances.get(&from).copied().unwrap_or(U256::zero());
        
        if from_balance < amount {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InsufficientBalance,
                "Insufficient token balance".to_string(),
            )));
        }

        *self.balances.entry(from).or_insert(U256::zero()) -= amount;
        *self.balances.entry(to).or_insert(U256::zero()) += amount;
        
        // Update voting power if tokens are delegated
        self.update_voting_power_on_transfer(from, to, amount);
        
        Ok(())
    }

    /// Delegate voting power
    pub fn delegate(&mut self, delegator: Address, delegate: Address, current_block: U256) -> Result<()> {
        let old_delegate = self.delegates.get(&delegator).copied();
        let delegator_balance = self.balances.get(&delegator).copied().unwrap_or(U256::zero());
        
        // Remove voting power from old delegate
        if let Some(old_delegate) = old_delegate {
            let old_power = self.voting_power.get(&old_delegate).copied().unwrap_or(U256::zero());
            self.voting_power.insert(old_delegate, old_power - delegator_balance);
            self.write_checkpoint(old_delegate, old_power - delegator_balance, current_block);
        }
        
        // Add voting power to new delegate
        let new_power = self.voting_power.get(&delegate).copied().unwrap_or(U256::zero());
        self.voting_power.insert(delegate, new_power + delegator_balance);
        self.write_checkpoint(delegate, new_power + delegator_balance, current_block);
        
        // Update delegation
        self.delegates.insert(delegator, delegate);
        
        Ok(())
    }

    /// Get voting power at a specific block
    pub fn get_votes_at_block(&self, account: Address, block_number: U256) -> U256 {
        if let Some(checkpoints) = self.checkpoints.get(&account) {
            // Binary search for the checkpoint at or before the block
            let mut left = 0;
            let mut right = checkpoints.len();
            
            while left < right {
                let mid = (left + right) / 2;
                if checkpoints[mid].block_number <= block_number {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            
            if left > 0 {
                checkpoints[left - 1].votes
            } else {
                U256::zero()
            }
        } else {
            U256::zero()
        }
    }

    /// Get current voting power
    pub fn get_current_votes(&self, account: Address) -> U256 {
        self.voting_power.get(&account).copied().unwrap_or(U256::zero())
    }

    /// Write a checkpoint
    fn write_checkpoint(&mut self, account: Address, votes: U256, block_number: U256) {
        let checkpoints = self.checkpoints.entry(account).or_insert_with(Vec::new);
        
        // If this is the same block as the last checkpoint, update it
        if let Some(last_checkpoint) = checkpoints.last_mut() {
            if last_checkpoint.block_number == block_number {
                last_checkpoint.votes = votes;
                return;
            }
        }
        
        // Add new checkpoint
        checkpoints.push(Checkpoint {
            block_number,
            votes,
        });
    }

    /// Update voting power when tokens are transferred
    fn update_voting_power_on_transfer(&mut self, from: Address, to: Address, amount: U256) {
        // Update voting power for delegated tokens
        if let Some(from_delegate) = self.delegates.get(&from).copied() {
            let current_power = self.voting_power.get(&from_delegate).copied().unwrap_or(U256::zero());
            self.voting_power.insert(from_delegate, current_power - amount);
        }
        
        if let Some(to_delegate) = self.delegates.get(&to).copied() {
            let current_power = self.voting_power.get(&to_delegate).copied().unwrap_or(U256::zero());
            self.voting_power.insert(to_delegate, current_power + amount);
        }
    }
}

/// Governor contract for managing proposals and voting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Governor {
    pub name: String,
    pub governance_token: Address,
    pub timelock: Address,
    pub voting_delay: U256, // Blocks between proposal creation and voting start
    pub voting_period: U256, // Blocks for voting duration
    pub proposal_threshold: U256, // Minimum tokens needed to create proposal
    pub quorum_numerator: U256, // Numerator for quorum calculation
    pub quorum_denominator: U256, // Denominator for quorum calculation
    pub timelock_delay: U256, // Blocks to wait before execution
    pub proposals: HashMap<U256, Proposal>,
    pub proposal_count: U256,
    pub guardian: Option<Address>, // Can cancel proposals in emergency
}

impl Governor {
    /// Create a new governor
    pub fn new(
        name: String,
        governance_token: Address,
        timelock: Address,
        voting_delay: U256,
        voting_period: U256,
        proposal_threshold: U256,
        quorum_numerator: U256,
        quorum_denominator: U256,
        timelock_delay: U256,
    ) -> Self {
        Self {
            name,
            governance_token,
            timelock,
            voting_delay,
            voting_period,
            proposal_threshold,
            quorum_numerator,
            quorum_denominator,
            timelock_delay,
            proposals: HashMap::new(),
            proposal_count: U256::zero(),
            guardian: None,
        }
    }

    /// Set guardian address
    pub fn set_guardian(&mut self, guardian: Address) {
        self.guardian = Some(guardian);
    }

    /// Create a new proposal
    pub fn propose(
        &mut self,
        proposer: Address,
        proposer_votes: U256,
        title: String,
        description: String,
        targets: Vec<Address>,
        values: Vec<U256>,
        signatures: Vec<String>,
        calldatas: Vec<Vec<u8>>,
        current_block: U256,
    ) -> Result<U256> {
        // Check proposal threshold
        if proposer_votes < self.proposal_threshold {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InsufficientVotingPower,
                "Proposer does not meet threshold".to_string(),
            )));
        }

        self.proposal_count = self.proposal_count + U256::new(1);
        let proposal_id = self.proposal_count;
        
        let start_block = current_block + self.voting_delay;
        let quorum_votes = self.calculate_quorum();
        
        let proposal = Proposal::new(
            proposal_id,
            proposer,
            title,
            description,
            targets,
            values,
            signatures,
            calldatas,
            start_block,
            self.voting_period,
            quorum_votes,
        )?;
        
        self.proposals.insert(proposal_id, proposal);
        Ok(proposal_id)
    }

    /// Cast a vote on a proposal
    pub fn cast_vote(
        &mut self,
        proposal_id: U256,
        voter: Address,
        vote_type: VoteType,
        voting_power: U256,
        reason: String,
        current_block: U256,
    ) -> Result<()> {
        let proposal = self.proposals.get_mut(&proposal_id).ok_or_else(|| {
            CompilerError::VmError(VmError::new(
                VmErrorKind::NotFound,
                "Proposal not found".to_string(),
            ))
        })?;

        proposal.cast_vote(voter, vote_type, voting_power, reason, current_block)
    }

    /// Queue a proposal for execution
    pub fn queue_proposal(&mut self, proposal_id: U256, current_block: U256) -> Result<()> {
        let proposal = self.proposals.get_mut(&proposal_id).ok_or_else(|| {
            CompilerError::VmError(VmError::new(
                VmErrorKind::NotFound,
                "Proposal not found".to_string(),
            ))
        })?;

        proposal.queue(current_block, self.timelock_delay)
    }

    /// Execute a proposal
    pub fn execute_proposal(&mut self, proposal_id: U256, current_block: U256) -> Result<()> {
        let proposal = self.proposals.get_mut(&proposal_id).ok_or_else(|| {
            CompilerError::VmError(VmError::new(
                VmErrorKind::NotFound,
                "Proposal not found".to_string(),
            ))
        })?;

        proposal.execute(current_block)
    }

    /// Cancel a proposal
    pub fn cancel_proposal(
        &mut self,
        proposal_id: U256,
        canceler: Address,
    ) -> Result<()> {
        // Check if canceler is guardian or proposer
        let proposal = self.proposals.get_mut(&proposal_id).ok_or_else(|| {
            CompilerError::VmError(VmError::new(
                VmErrorKind::NotFound,
                "Proposal not found".to_string(),
            ))
        })?;

        let is_guardian = self.guardian.map_or(false, |g| g == canceler);
        let is_proposer = proposal.proposer == canceler;
        
        if !is_guardian && !is_proposer {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::Unauthorized,
                "Only guardian or proposer can cancel".to_string(),
            )));
        }

        proposal.cancel(canceler)
    }

    /// Update all proposal states
    pub fn update_proposal_states(&mut self, current_block: U256) {
        for proposal in self.proposals.values_mut() {
            proposal.update_state(current_block, self.timelock_delay);
        }
    }

    /// Calculate quorum based on total supply
    fn calculate_quorum(&self) -> U256 {
        // This would typically query the governance token for total supply
        // For now, return a placeholder
        U256::new(100000) // 100k tokens minimum quorum
    }

    /// Get proposal by ID
    pub fn get_proposal(&self, proposal_id: U256) -> Option<&Proposal> {
        self.proposals.get(&proposal_id)
    }

    /// Get all active proposals
    pub fn get_active_proposals(&self) -> Vec<&Proposal> {
        self.proposals
            .values()
            .filter(|p| p.state == ProposalState::Active)
            .collect()
    }

    /// Get proposals by state
    pub fn get_proposals_by_state(&self, state: ProposalState) -> Vec<&Proposal> {
        self.proposals
            .values()
            .filter(|p| p.state == state)
            .collect()
    }
}

/// Timelock contract for delayed execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timelock {
    pub admin: Address,
    pub delay: U256,
    pub queued_transactions: HashMap<Vec<u8>, U256>, // transaction hash -> eta
    pub grace_period: U256,
}

impl Timelock {
    /// Create a new timelock
    pub fn new(admin: Address, delay: U256, grace_period: U256) -> Self {
        Self {
            admin,
            delay,
            queued_transactions: HashMap::new(),
            grace_period,
        }
    }

    /// Queue a transaction
    pub fn queue_transaction(
        &mut self,
        target: Address,
        value: U256,
        signature: String,
        data: Vec<u8>,
        eta: U256,
        current_block: U256,
    ) -> Result<Vec<u8>> {
        if eta < current_block + self.delay {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InvalidInput,
                "ETA must be at least delay from now".to_string(),
            )));
        }

        let tx_hash = self.get_transaction_hash(target, value, signature, data, eta);
        self.queued_transactions.insert(tx_hash.clone(), eta);
        
        Ok(tx_hash)
    }

    /// Execute a queued transaction
    pub fn execute_transaction(
        &mut self,
        target: Address,
        value: U256,
        signature: String,
        data: Vec<u8>,
        eta: U256,
        current_block: U256,
    ) -> Result<()> {
        let tx_hash = self.get_transaction_hash(target, value, signature.clone(), data.clone(), eta);
        
        let queued_eta = self.queued_transactions.get(&tx_hash).copied().ok_or_else(|| {
            CompilerError::VmError(VmError::new(
                VmErrorKind::NotFound,
                "Transaction not queued".to_string(),
            ))
        })?;

        if current_block < queued_eta {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::TimelockNotReady,
                "Transaction is not ready for execution".to_string(),
            )));
        }

        if current_block > queued_eta + self.grace_period {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::TransactionExpired,
                "Transaction has expired".to_string(),
            )));
        }

        // Remove from queue
        self.queued_transactions.remove(&tx_hash);
        
        // Execute transaction (implementation would call the target)
        Ok(())
    }

    /// Cancel a queued transaction
    pub fn cancel_transaction(
        &mut self,
        target: Address,
        value: U256,
        signature: String,
        data: Vec<u8>,
        eta: U256,
    ) -> Result<()> {
        let tx_hash = self.get_transaction_hash(target, value, signature, data, eta);
        
        if !self.queued_transactions.contains_key(&tx_hash) {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::NotFound,
                "Transaction not queued".to_string(),
            )));
        }

        self.queued_transactions.remove(&tx_hash);
        Ok(())
    }

    /// Generate transaction hash
    fn get_transaction_hash(
        &self,
        target: Address,
        value: U256,
        signature: String,
        data: Vec<u8>,
        eta: U256,
    ) -> Vec<u8> {
        // Simple hash implementation for demonstration
        let mut hash_input = Vec::new();
        hash_input.extend_from_slice(&target.to_bytes());
        hash_input.extend_from_slice(&value.to_bytes());
        hash_input.extend_from_slice(signature.as_bytes());
        hash_input.extend_from_slice(&data);
        hash_input.extend_from_slice(&eta.to_bytes());
        
        // Return first 32 bytes as hash (simplified)
        hash_input.into_iter().take(32).collect()
    }
}

/// Governance utilities
pub struct GovernanceUtils;

impl GovernanceUtils {
    /// Calculate voting power based on token balance and delegation
    pub fn calculate_voting_power(
        token_balance: U256,
        delegated_power: U256,
        is_delegated: bool,
    ) -> U256 {
        if is_delegated {
            delegated_power
        } else {
            token_balance
        }
    }

    /// Calculate quorum percentage
    pub fn calculate_quorum_percentage(
        total_votes: U256,
        total_supply: U256,
    ) -> U256 {
        if total_supply.is_zero() {
            return U256::zero();
        }
        (total_votes * U256::new(10000)) / total_supply
    }

    /// Check if proposal meets execution criteria
    pub fn meets_execution_criteria(
        for_votes: U256,
        against_votes: U256,
        quorum: U256,
        total_votes: U256,
    ) -> bool {
        for_votes > against_votes && total_votes >= quorum
    }

    /// Calculate vote weight with time decay
    pub fn calculate_time_weighted_vote(
        base_weight: U256,
        vote_block: U256,
        proposal_start: U256,
        proposal_end: U256,
    ) -> U256 {
        if vote_block < proposal_start || vote_block > proposal_end {
            return U256::zero();
        }
        
        let voting_period = proposal_end - proposal_start;
        let time_since_start = vote_block - proposal_start;
        
        // Linear decay: earlier votes have more weight
        let decay_factor = (voting_period - time_since_start) * U256::new(10000) / voting_period;
        (base_weight * decay_factor) / U256::new(10000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proposal_creation() {
        let proposer = Address::new([1u8; 20]);
        let targets = vec![Address::new([2u8; 20])];
        let values = vec![U256::zero()];
        let signatures = vec!["transfer(address,uint256)".to_string()];
        let calldatas = vec![vec![0u8; 32]];
        
        let proposal = Proposal::new(
            U256::new(1),
            proposer,
            "Test Proposal".to_string(),
            "A test proposal".to_string(),
            targets,
            values,
            signatures,
            calldatas,
            U256::new(100),
            U256::new(1000),
            U256::new(10000),
        ).unwrap();
        
        assert_eq!(proposal.id, U256::new(1));
        assert_eq!(proposal.state, ProposalState::Pending);
        assert_eq!(proposal.end_block, U256::new(1100));
    }

    #[test]
    fn test_voting() {
        let proposer = Address::new([1u8; 20]);
        let voter = Address::new([2u8; 20]);
        let targets = vec![Address::new([3u8; 20])];
        let values = vec![U256::zero()];
        let signatures = vec!["test()".to_string()];
        let calldatas = vec![vec![]];
        
        let mut proposal = Proposal::new(
            U256::new(1),
            proposer,
            "Test".to_string(),
            "Test".to_string(),
            targets,
            values,
            signatures,
            calldatas,
            U256::new(100),
            U256::new(1000),
            U256::new(5000),
        ).unwrap();
        
        // Activate proposal
        proposal.update_state(U256::new(100), U256::new(100));
        assert_eq!(proposal.state, ProposalState::Active);
        
        // Cast vote
        proposal.cast_vote(
            voter,
            VoteType::For,
            U256::new(10000),
            "Support this proposal".to_string(),
            U256::new(500),
        ).unwrap();
        
        assert_eq!(proposal.for_votes, U256::new(10000));
        assert!(proposal.has_quorum());
    }

    #[test]
    fn test_governance_token() {
        let initial_holder = Address::new([1u8; 20]);
        let mut token = GovernanceToken::new(
            "Governance Token".to_string(),
            "GOV".to_string(),
            U256::new(1000000),
            initial_holder,
        );
        
        let delegate = Address::new([2u8; 20]);
        token.delegate(initial_holder, delegate, U256::new(1)).unwrap();
        
        assert_eq!(token.get_current_votes(delegate), U256::new(1000000));
        assert_eq!(token.get_current_votes(initial_holder), U256::zero());
    }

    #[test]
    fn test_governor() {
        let governance_token = Address::new([1u8; 20]);
        let timelock = Address::new([2u8; 20]);
        let proposer = Address::new([3u8; 20]);
        
        let mut governor = Governor::new(
            "Test Governor".to_string(),
            governance_token,
            timelock,
            U256::new(1), // 1 block delay
            U256::new(100), // 100 block voting period
            U256::new(1000), // 1000 token threshold
            U256::new(4), // 4% quorum
            U256::new(100),
            U256::new(172800), // 2 day timelock
        );
        
        let proposal_id = governor.propose(
            proposer,
            U256::new(5000), // Proposer has 5000 tokens
            "Test Proposal".to_string(),
            "Description".to_string(),
            vec![Address::new([4u8; 20])],
            vec![U256::zero()],
            vec!["test()".to_string()],
            vec![vec![]],
            U256::new(1000),
        ).unwrap();
        
        assert_eq!(proposal_id, U256::new(1));
        assert!(governor.get_proposal(proposal_id).is_some());
    }

    #[test]
    fn test_timelock() {
        let admin = Address::new([1u8; 20]);
        let mut timelock = Timelock::new(
            admin,
            U256::new(172800), // 2 day delay
            U256::new(1209600), // 2 week grace period
        );
        
        let target = Address::new([2u8; 20]);
        let tx_hash = timelock.queue_transaction(
            target,
            U256::zero(),
            "test()".to_string(),
            vec![],
            U256::new(172801),
            U256::new(1),
        ).unwrap();
        
        assert!(!tx_hash.is_empty());
        assert!(timelock.queued_transactions.contains_key(&tx_hash));
    }

    #[test]
    fn test_governance_utils() {
        let voting_power = GovernanceUtils::calculate_voting_power(
            U256::new(1000),
            U256::new(5000),
            true,
        );
        assert_eq!(voting_power, U256::new(5000));
        
        let quorum_pct = GovernanceUtils::calculate_quorum_percentage(
            U256::new(25000),
            U256::new(100000),
        );
        assert_eq!(quorum_pct, U256::new(2500)); // 25%
        
        assert!(GovernanceUtils::meets_execution_criteria(
            U256::new(6000),
            U256::new(4000),
            U256::new(5000),
            U256::new(10000),
        ));
    }
}