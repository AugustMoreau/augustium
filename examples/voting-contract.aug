// Decentralized Voting Contract Example
// Demonstrates governance patterns, time-based access control, and advanced safety features

use std::collections::{HashMap, HashSet};
use std::time::{Timestamp, Duration};
use std::events::Event;
use std::access::Ownable;
use std::crypto::Hash;

// Events
#[derive(Event)]
struct ProposalCreated {
    proposal_id: u256,
    proposer: address,
    description: string,
    voting_start: Timestamp,
    voting_end: Timestamp
}

#[derive(Event)]
struct VoteCast {
    proposal_id: u256,
    voter: address,
    support: bool,
    weight: u256,
    reason: string
}

#[derive(Event)]
struct ProposalExecuted {
    proposal_id: u256,
    success: bool
}

#[derive(Event)]
struct VoterRegistered {
    voter: address,
    weight: u256
}

// Enums
#[derive(Clone, PartialEq)]
enum ProposalState {
    Pending,
    Active,
    Succeeded,
    Defeated,
    Executed,
    Cancelled
}

#[derive(Clone)]
struct Vote {
    support: bool,
    weight: u256,
    reason: string,
    timestamp: Timestamp
}

#[derive(Clone)]
struct Proposal {
    id: u256,
    proposer: address,
    description: string,
    voting_start: Timestamp,
    voting_end: Timestamp,
    for_votes: u256,
    against_votes: u256,
    executed: bool,
    cancelled: bool,
    // Optional execution data
    target: Option<address>,
    call_data: Option<bytes>,
    value: u256
}

// Voting contract with comprehensive governance features
contract VotingContract extends Ownable {
    // Configuration
    voting_duration: Duration;
    min_voting_weight: u256;
    quorum_threshold: u256; // Percentage (0-100)
    approval_threshold: u256; // Percentage (0-100)
    
    // State
    proposal_count: u256;
    proposals: HashMap<u256, Proposal>;
    votes: HashMap<u256, HashMap<address, Vote>>; // proposal_id -> voter -> vote
    voter_weights: HashMap<address, u256>;
    registered_voters: HashSet<address>;
    
    // Proposal execution queue
    execution_delay: Duration;
    queued_proposals: HashMap<u256, Timestamp>; // proposal_id -> execution_time
    
    // Constructor
    constructor(
        voting_duration_hours: u64,
        min_weight: u256,
        quorum_pct: u256,
        approval_pct: u256,
        exec_delay_hours: u64
    ) {
        require!(voting_duration_hours > 0, "Voting duration must be positive");
        require!(quorum_pct <= 100, "Quorum threshold cannot exceed 100%");
        require!(approval_pct <= 100, "Approval threshold cannot exceed 100%");
        require!(approval_pct > 50, "Approval threshold must be greater than 50%");
        
        self.voting_duration = Duration::hours(voting_duration_hours);
        self.min_voting_weight = min_weight;
        self.quorum_threshold = quorum_pct;
        self.approval_threshold = approval_pct;
        self.execution_delay = Duration::hours(exec_delay_hours);
        self.proposal_count = 0;
        
        // Register owner as initial voter with weight 1
        self._register_voter(msg.sender, 1);
    }
    
    // Voter management
    #[only_owner]
    fn register_voter(&mut self, voter: address, weight: u256) {
        require!(voter != address::zero(), "Cannot register zero address");
        require!(weight >= self.min_voting_weight, "Weight below minimum");
        
        self._register_voter(voter, weight);
    }
    
    #[only_owner]
    fn update_voter_weight(&mut self, voter: address, new_weight: u256) {
        require!(self.registered_voters.contains(&voter), "Voter not registered");
        require!(new_weight >= self.min_voting_weight, "Weight below minimum");
        
        self.voter_weights.insert(voter, new_weight);
    }
    
    #[only_owner]
    fn remove_voter(&mut self, voter: address) {
        require!(self.registered_voters.contains(&voter), "Voter not registered");
        require!(voter != self.owner(), "Cannot remove owner");
        
        self.registered_voters.remove(&voter);
        self.voter_weights.remove(&voter);
    }
    
    // Proposal creation
    #[payable(false)]
    fn create_proposal(
        &mut self,
        description: string,
        target: Option<address>,
        call_data: Option<bytes>,
        value: u256
    ) -> u256 {
        require!(self._is_registered_voter(msg.sender), "Only registered voters can propose");
        require!(!description.is_empty(), "Description cannot be empty");
        require!(description.len() <= 1000, "Description too long");
        
        // If target is specified, validate call data
        if target.is_some() {
            require!(call_data.is_some(), "Call data required for target execution");
        }
        
        self.proposal_count += 1;
        let proposal_id = self.proposal_count;
        
        let voting_start = block.timestamp;
        let voting_end = voting_start + self.voting_duration;
        
        let proposal = Proposal {
            id: proposal_id,
            proposer: msg.sender,
            description: description.clone(),
            voting_start,
            voting_end,
            for_votes: 0,
            against_votes: 0,
            executed: false,
            cancelled: false,
            target,
            call_data,
            value
        };
        
        self.proposals.insert(proposal_id, proposal);
        
        emit ProposalCreated {
            proposal_id,
            proposer: msg.sender,
            description,
            voting_start,
            voting_end
        };
        
        proposal_id
    }
    
    // Voting
    #[payable(false)]
    fn cast_vote(&mut self, proposal_id: u256, support: bool, reason: string) {
        require!(self._is_registered_voter(msg.sender), "Only registered voters can vote");
        require!(self.proposals.contains_key(&proposal_id), "Proposal does not exist");
        
        let proposal = self.proposals.get(&proposal_id).unwrap();
        require!(!proposal.cancelled, "Proposal is cancelled");
        require!(block.timestamp >= proposal.voting_start, "Voting has not started");
        require!(block.timestamp <= proposal.voting_end, "Voting has ended");
        
        // Check if voter has already voted
        if let Some(proposal_votes) = self.votes.get(&proposal_id) {
            require!(!proposal_votes.contains_key(&msg.sender), "Already voted on this proposal");
        }
        
        let voter_weight = self.voter_weights.get(&msg.sender).unwrap().clone();
        
        let vote = Vote {
            support,
            weight: voter_weight,
            reason: reason.clone(),
            timestamp: block.timestamp
        };
        
        // Initialize votes HashMap for proposal if needed
        if !self.votes.contains_key(&proposal_id) {
            self.votes.insert(proposal_id, HashMap::new());
        }
        
        self.votes.get_mut(&proposal_id).unwrap().insert(msg.sender, vote);
        
        // Update proposal vote counts
        let mut proposal = self.proposals.get_mut(&proposal_id).unwrap();
        if support {
            proposal.for_votes += voter_weight;
        } else {
            proposal.against_votes += voter_weight;
        }
        
        emit VoteCast {
            proposal_id,
            voter: msg.sender,
            support,
            weight: voter_weight,
            reason
        };
    }
    
    // Proposal execution
    #[payable(false)]
    fn queue_proposal(&mut self, proposal_id: u256) {
        require!(self.proposals.contains_key(&proposal_id), "Proposal does not exist");
        
        let state = self.get_proposal_state(proposal_id);
        require!(state == ProposalState::Succeeded, "Proposal must be succeeded to queue");
        require!(!self.queued_proposals.contains_key(&proposal_id), "Proposal already queued");
        
        let execution_time = block.timestamp + self.execution_delay;
        self.queued_proposals.insert(proposal_id, execution_time);
    }
    
    #[payable(true)]
    fn execute_proposal(&mut self, proposal_id: u256) {
        require!(self.proposals.contains_key(&proposal_id), "Proposal does not exist");
        require!(self.queued_proposals.contains_key(&proposal_id), "Proposal not queued");
        
        let execution_time = self.queued_proposals.get(&proposal_id).unwrap();
        require!(block.timestamp >= *execution_time, "Execution delay not met");
        
        let mut proposal = self.proposals.get_mut(&proposal_id).unwrap();
        require!(!proposal.executed, "Proposal already executed");
        require!(!proposal.cancelled, "Proposal is cancelled");
        
        let state = self.get_proposal_state(proposal_id);
        require!(state == ProposalState::Succeeded, "Proposal not in succeeded state");
        
        proposal.executed = true;
        self.queued_proposals.remove(&proposal_id);
        
        let mut success = true;
        
        // Execute the proposal if it has a target
        if let Some(target) = proposal.target {
            if let Some(call_data) = &proposal.call_data {
                let result = self._execute_call(target, call_data.clone(), proposal.value);
                success = result.is_ok();
            }
        }
        
        emit ProposalExecuted { proposal_id, success };
    }
    
    #[only_owner]
    fn cancel_proposal(&mut self, proposal_id: u256) {
        require!(self.proposals.contains_key(&proposal_id), "Proposal does not exist");
        
        let mut proposal = self.proposals.get_mut(&proposal_id).unwrap();
        require!(!proposal.executed, "Cannot cancel executed proposal");
        require!(!proposal.cancelled, "Proposal already cancelled");
        
        proposal.cancelled = true;
        
        // Remove from queue if queued
        self.queued_proposals.remove(&proposal_id);
    }
    
    // View functions
    #[view]
    fn get_proposal(&self, proposal_id: u256) -> Option<Proposal> {
        self.proposals.get(&proposal_id).cloned()
    }
    
    #[view]
    fn get_proposal_state(&self, proposal_id: u256) -> ProposalState {
        if let Some(proposal) = self.proposals.get(&proposal_id) {
            if proposal.cancelled {
                return ProposalState::Cancelled;
            }
            
            if proposal.executed {
                return ProposalState::Executed;
            }
            
            if block.timestamp < proposal.voting_start {
                return ProposalState::Pending;
            }
            
            if block.timestamp <= proposal.voting_end {
                return ProposalState::Active;
            }
            
            // Voting has ended, check results
            let total_votes = proposal.for_votes + proposal.against_votes;
            let total_weight = self._get_total_voting_weight();
            
            // Check quorum
            let quorum_met = (total_votes * 100) >= (total_weight * self.quorum_threshold);
            
            if !quorum_met {
                return ProposalState::Defeated;
            }
            
            // Check approval threshold
            let approval_met = (proposal.for_votes * 100) >= (total_votes * self.approval_threshold);
            
            if approval_met {
                ProposalState::Succeeded
            } else {
                ProposalState::Defeated
            }
        } else {
            ProposalState::Defeated // Invalid proposal ID
        }
    }
    
    #[view]
    fn get_vote(&self, proposal_id: u256, voter: address) -> Option<Vote> {
        self.votes
            .get(&proposal_id)
            .and_then(|proposal_votes| proposal_votes.get(&voter))
            .cloned()
    }
    
    #[view]
    fn get_voter_weight(&self, voter: address) -> u256 {
        self.voter_weights.get(&voter).unwrap_or(&0).clone()
    }
    
    #[view]
    fn is_registered_voter(&self, voter: address) -> bool {
        self._is_registered_voter(voter)
    }
    
    #[view]
    fn get_proposal_votes(&self, proposal_id: u256) -> (u256, u256, u256) {
        if let Some(proposal) = self.proposals.get(&proposal_id) {
            let total_votes = proposal.for_votes + proposal.against_votes;
            (proposal.for_votes, proposal.against_votes, total_votes)
        } else {
            (0, 0, 0)
        }
    }
    
    #[view]
    fn get_voting_config(&self) -> (Duration, u256, u256, u256, Duration) {
        (
            self.voting_duration,
            self.min_voting_weight,
            self.quorum_threshold,
            self.approval_threshold,
            self.execution_delay
        )
    }
    
    #[view]
    fn get_total_voting_weight(&self) -> u256 {
        self._get_total_voting_weight()
    }
    
    #[view]
    fn get_registered_voters_count(&self) -> u256 {
        self.registered_voters.len() as u256
    }
    
    // Internal functions
    fn _register_voter(&mut self, voter: address, weight: u256) {
        self.registered_voters.insert(voter);
        self.voter_weights.insert(voter, weight);
        
        emit VoterRegistered { voter, weight };
    }
    
    fn _is_registered_voter(&self, voter: address) -> bool {
        self.registered_voters.contains(&voter)
    }
    
    fn _get_total_voting_weight(&self) -> u256 {
        let mut total = 0u256;
        for (_, &weight) in self.voter_weights.iter() {
            total += weight; // Automatic overflow protection
        }
        total
    }
    
    fn _execute_call(&self, target: address, call_data: bytes, value: u256) -> Result<bytes, string> {
        // This would be implemented by the runtime
        // For now, we'll simulate success
        if target != address::zero() && !call_data.is_empty() {
            Ok(bytes::new())
        } else {
            Err("Invalid call parameters".to_string())
        }
    }
    
    // Emergency functions
    #[only_owner]
    fn emergency_pause(&mut self) {
        // Implementation would pause all voting activities
        // This is a placeholder for emergency functionality
    }
    
    #[only_owner]
    fn update_voting_config(
        &mut self,
        new_duration_hours: Option<u64>,
        new_quorum: Option<u256>,
        new_approval: Option<u256>
    ) {
        if let Some(duration) = new_duration_hours {
            require!(duration > 0, "Duration must be positive");
            self.voting_duration = Duration::hours(duration);
        }
        
        if let Some(quorum) = new_quorum {
            require!(quorum <= 100, "Quorum cannot exceed 100%");
            self.quorum_threshold = quorum;
        }
        
        if let Some(approval) = new_approval {
            require!(approval <= 100, "Approval cannot exceed 100%");
            require!(approval > 50, "Approval must be greater than 50%");
            self.approval_threshold = approval;
        }
    }
}

// Example usage and comprehensive tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::testing::*;
    
    fn setup_voting_contract() -> VotingContract {
        VotingContract::new(
            24, // 24 hours voting duration
            1,  // minimum weight of 1
            20, // 20% quorum
            60, // 60% approval threshold
            48  // 48 hours execution delay
        )
    }
    
    #[test]
    fn test_deployment() {
        let contract = setup_voting_contract();
        
        assert_eq!(contract.get_registered_voters_count(), 1);
        assert_eq!(contract.get_voter_weight(test::caller()), 1);
        assert!(contract.is_registered_voter(test::caller()));
    }
    
    #[test]
    fn test_voter_registration() {
        let mut contract = setup_voting_contract();
        let voter = address::from_hex("0x1234567890123456789012345678901234567890");
        
        contract.register_voter(voter, 5);
        
        assert!(contract.is_registered_voter(voter));
        assert_eq!(contract.get_voter_weight(voter), 5);
        assert_eq!(contract.get_registered_voters_count(), 2);
    }
    
    #[test]
    fn test_proposal_creation() {
        let mut contract = setup_voting_contract();
        
        let proposal_id = contract.create_proposal(
            "Test proposal".to_string(),
            None,
            None,
            0
        );
        
        assert_eq!(proposal_id, 1);
        
        let proposal = contract.get_proposal(proposal_id).unwrap();
        assert_eq!(proposal.description, "Test proposal");
        assert_eq!(proposal.proposer, test::caller());
        assert_eq!(contract.get_proposal_state(proposal_id), ProposalState::Active);
    }
    
    #[test]
    fn test_voting() {
        let mut contract = setup_voting_contract();
        
        // Create proposal
        let proposal_id = contract.create_proposal(
            "Test proposal".to_string(),
            None,
            None,
            0
        );
        
        // Vote on proposal
        contract.cast_vote(proposal_id, true, "I support this".to_string());
        
        let vote = contract.get_vote(proposal_id, test::caller()).unwrap();
        assert!(vote.support);
        assert_eq!(vote.weight, 1);
        assert_eq!(vote.reason, "I support this");
        
        let (for_votes, against_votes, total_votes) = contract.get_proposal_votes(proposal_id);
        assert_eq!(for_votes, 1);
        assert_eq!(against_votes, 0);
        assert_eq!(total_votes, 1);
    }
    
    #[test]
    fn test_proposal_success() {
        let mut contract = setup_voting_contract();
        
        // Register additional voters to meet quorum
        for i in 1..=5 {
            let voter = address::from_u256(i);
            contract.register_voter(voter, 1);
        }
        
        // Create proposal
        let proposal_id = contract.create_proposal(
            "Test proposal".to_string(),
            None,
            None,
            0
        );
        
        // Vote with majority support
        contract.cast_vote(proposal_id, true, "Support".to_string());
        
        for i in 1..=4 {
            let voter = address::from_u256(i);
            test::set_caller(voter);
            contract.cast_vote(proposal_id, true, "Support".to_string());
        }
        
        // Advance time past voting period
        test::advance_time(Duration::hours(25));
        
        assert_eq!(contract.get_proposal_state(proposal_id), ProposalState::Succeeded);
    }
    
    #[test]
    fn test_proposal_defeat_quorum() {
        let mut contract = setup_voting_contract();
        
        // Create proposal
        let proposal_id = contract.create_proposal(
            "Test proposal".to_string(),
            None,
            None,
            0
        );
        
        // Vote but don't meet quorum (need 20% of total weight)
        contract.cast_vote(proposal_id, true, "Support".to_string());
        
        // Advance time past voting period
        test::advance_time(Duration::hours(25));
        
        // Should be defeated due to lack of quorum
        assert_eq!(contract.get_proposal_state(proposal_id), ProposalState::Defeated);
    }
    
    #[test]
    fn test_proposal_defeat_approval() {
        let mut contract = setup_voting_contract();
        
        // Register additional voters
        for i in 1..=5 {
            let voter = address::from_u256(i);
            contract.register_voter(voter, 1);
        }
        
        // Create proposal
        let proposal_id = contract.create_proposal(
            "Test proposal".to_string(),
            None,
            None,
            0
        );
        
        // Vote with majority against (meets quorum but not approval threshold)
        contract.cast_vote(proposal_id, false, "Against".to_string());
        
        for i in 1..=3 {
            let voter = address::from_u256(i);
            test::set_caller(voter);
            contract.cast_vote(proposal_id, false, "Against".to_string());
        }
        
        // Advance time past voting period
        test::advance_time(Duration::hours(25));
        
        assert_eq!(contract.get_proposal_state(proposal_id), ProposalState::Defeated);
    }
    
    #[test]
    fn test_access_control() {
        let mut contract = setup_voting_contract();
        let non_voter = address::from_hex("0x1234567890123456789012345678901234567890");
        
        // Try to create proposal from non-registered voter
        test::set_caller(non_voter);
        let result = test::try_call(|| {
            contract.create_proposal(
                "Test proposal".to_string(),
                None,
                None,
                0
            );
        });
        
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Only registered voters"));
    }
}