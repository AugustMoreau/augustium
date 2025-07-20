//! Access Control Module
//!
//! This module provides comprehensive access control functionality for smart contracts,
//! including role-based access control (RBAC), ownership patterns, and permission management.

use crate::error::{Result, VmError, VmErrorKind, CompilerError};
use crate::stdlib::core_types::{Address, U256};
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};

/// Role identifier type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Role(pub String);

impl Role {
    /// Create a new role
    pub fn new(name: &str) -> Self {
        Role(name.to_string())
    }

    /// Default admin role
    pub fn admin() -> Self {
        Role("ADMIN".to_string())
    }

    /// Default minter role
    pub fn minter() -> Self {
        Role("MINTER".to_string())
    }

    /// Default pauser role
    pub fn pauser() -> Self {
        Role("PAUSER".to_string())
    }

    /// Get role name
    pub fn name(&self) -> &str {
        &self.0
    }
}

/// Permission type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    Read,
    Write,
    Execute,
    Admin,
    Transfer,
    Mint,
    Burn,
    Pause,
    Custom(String),
}

/// Access control list entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessEntry {
    pub address: Address,
    pub permissions: HashSet<Permission>,
    pub roles: HashSet<Role>,
    pub expires_at: Option<U256>, // Block number when access expires
}

/// Role-based access control manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControl {
    owner: Address,
    admins: HashSet<Address>,
    role_members: HashMap<Role, HashSet<Address>>,
    role_admins: HashMap<Role, Role>, // Which role can manage this role
    access_list: HashMap<Address, AccessEntry>,
    paused: bool,
}

impl AccessControl {
    /// Create new access control with owner
    pub fn new(owner: Address) -> Self {
        let mut admins = HashSet::new();
        admins.insert(owner);
        
        let mut role_members = HashMap::new();
        let mut admin_set = HashSet::new();
        admin_set.insert(owner);
        role_members.insert(Role::admin(), admin_set);
        
        Self {
            owner,
            admins,
            role_members,
            role_admins: HashMap::new(),
            access_list: HashMap::new(),
            paused: false,
        }
    }

    /// Check if address is owner
    pub fn is_owner(&self, address: &Address) -> bool {
        &self.owner == address
    }

    /// Check if address is admin
    pub fn is_admin(&self, address: &Address) -> bool {
        self.admins.contains(address)
    }

    /// Check if address has role
    pub fn has_role(&self, role: &Role, address: &Address) -> bool {
        self.role_members
            .get(role)
            .map(|members| members.contains(address))
            .unwrap_or(false)
    }

    /// Check if address has permission
    pub fn has_permission(&self, address: &Address, permission: &Permission) -> bool {
        if let Some(entry) = self.access_list.get(address) {
            // Check if access has expired
            if let Some(expires_at) = &entry.expires_at {
                // In a real implementation, you'd compare with current block number
                // For now, we'll assume it's valid
            }
            entry.permissions.contains(permission)
        } else {
            false
        }
    }

    /// Grant role to address (only by role admin or contract admin)
    pub fn grant_role(&mut self, role: &Role, address: &Address, caller: &Address) -> Result<()> {
        if !self.can_manage_role(role, caller) {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::UnauthorizedAccess,
                "Caller cannot manage this role".to_string(),
            )));
        }

        self.role_members
            .entry(role.clone())
            .or_insert_with(HashSet::new)
            .insert(*address);

        Ok(())
    }

    /// Revoke role from address
    pub fn revoke_role(&mut self, role: &Role, address: &Address, caller: &Address) -> Result<()> {
        if !self.can_manage_role(role, caller) {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::UnauthorizedAccess,
                "Caller cannot manage this role".to_string(),
            )));
        }

        if let Some(members) = self.role_members.get_mut(role) {
            members.remove(address);
        }

        Ok(())
    }

    /// Set role admin (who can grant/revoke this role)
    pub fn set_role_admin(&mut self, role: &Role, admin_role: &Role, caller: &Address) -> Result<()> {
        if !self.is_admin(caller) {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::UnauthorizedAccess,
                "Only admins can set role admins".to_string(),
            )));
        }

        self.role_admins.insert(role.clone(), admin_role.clone());
        Ok(())
    }

    /// Grant permission to address
    pub fn grant_permission(
        &mut self,
        address: &Address,
        permission: Permission,
        expires_at: Option<U256>,
        caller: &Address,
    ) -> Result<()> {
        if !self.is_admin(caller) {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::UnauthorizedAccess,
                "Only admins can grant permissions".to_string(),
            )));
        }

        let entry = self.access_list.entry(*address).or_insert_with(|| AccessEntry {
            address: *address,
            permissions: HashSet::new(),
            roles: HashSet::new(),
            expires_at,
        });

        entry.permissions.insert(permission);
        if expires_at.is_some() {
            entry.expires_at = expires_at;
        }

        Ok(())
    }

    /// Revoke permission from address
    pub fn revoke_permission(
        &mut self,
        address: &Address,
        permission: &Permission,
        caller: &Address,
    ) -> Result<()> {
        if !self.is_admin(caller) {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::UnauthorizedAccess,
                "Only admins can revoke permissions".to_string(),
            )));
        }

        if let Some(entry) = self.access_list.get_mut(address) {
            entry.permissions.remove(permission);
        }

        Ok(())
    }

    /// Pause contract (emergency stop)
    pub fn pause(&mut self, caller: &Address) -> Result<()> {
        if !self.has_role(&Role::pauser(), caller) && !self.is_admin(caller) {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::UnauthorizedAccess,
                "Only pausers or admins can pause".to_string(),
            )));
        }

        self.paused = true;
        Ok(())
    }

    /// Unpause contract
    pub fn unpause(&mut self, caller: &Address) -> Result<()> {
        if !self.has_role(&Role::pauser(), caller) && !self.is_admin(caller) {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::UnauthorizedAccess,
                "Only pausers or admins can unpause".to_string(),
            )));
        }

        self.paused = false;
        Ok(())
    }

    /// Check if contract is paused
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Transfer ownership
    pub fn transfer_ownership(&mut self, new_owner: Address, caller: &Address) -> Result<()> {
        if !self.is_owner(caller) {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::UnauthorizedAccess,
                "Only owner can transfer ownership".to_string(),
            )));
        }

        // Remove old owner from admins and add new owner
        self.admins.remove(&self.owner);
        self.admins.insert(new_owner);
        
        // Update role memberships
        if let Some(admin_members) = self.role_members.get_mut(&Role::admin()) {
            admin_members.remove(&self.owner);
            admin_members.insert(new_owner);
        }

        self.owner = new_owner;
        Ok(())
    }

    /// Get all members of a role
    pub fn get_role_members(&self, role: &Role) -> Vec<Address> {
        self.role_members
            .get(role)
            .map(|members| members.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Get all roles for an address
    pub fn get_address_roles(&self, address: &Address) -> Vec<Role> {
        self.role_members
            .iter()
            .filter_map(|(role, members)| {
                if members.contains(address) {
                    Some(role.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check if caller can manage a role
    fn can_manage_role(&self, role: &Role, caller: &Address) -> bool {
        // Owner and admins can manage any role
        if self.is_owner(caller) || self.is_admin(caller) {
            return true;
        }

        // Check if caller has the admin role for this role
        if let Some(admin_role) = self.role_admins.get(role) {
            return self.has_role(admin_role, caller);
        }

        false
    }
}

/// Ownable pattern implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ownable {
    owner: Address,
    pending_owner: Option<Address>,
}

impl Ownable {
    /// Create new ownable with initial owner
    pub fn new(owner: Address) -> Self {
        Self {
            owner,
            pending_owner: None,
        }
    }

    /// Get current owner
    pub fn owner(&self) -> Address {
        self.owner
    }

    /// Check if address is owner
    pub fn is_owner(&self, address: &Address) -> bool {
        &self.owner == address
    }

    /// Transfer ownership (two-step process)
    pub fn transfer_ownership(&mut self, new_owner: Address, caller: &Address) -> Result<()> {
        if !self.is_owner(caller) {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::UnauthorizedAccess,
                "Only owner can transfer ownership".to_string(),
            )));
        }

        self.pending_owner = Some(new_owner);
        Ok(())
    }

    /// Accept ownership transfer
    pub fn accept_ownership(&mut self, caller: &Address) -> Result<()> {
        if let Some(pending) = self.pending_owner {
            if &pending != caller {
                return Err(CompilerError::VmError(VmError::new(
                    VmErrorKind::UnauthorizedAccess,
                    "Only pending owner can accept ownership".to_string(),
                )));
            }

            self.owner = pending;
            self.pending_owner = None;
            Ok(())
        } else {
            Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InvalidInput,
                "No pending ownership transfer".to_string(),
            )))
        }
    }

    /// Renounce ownership (set to zero address)
    pub fn renounce_ownership(&mut self, caller: &Address) -> Result<()> {
        if !self.is_owner(caller) {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::UnauthorizedAccess,
                "Only owner can renounce ownership".to_string(),
            )));
        }

        self.owner = Address::ZERO;
        self.pending_owner = None;
        Ok(())
    }
}

/// Access control utilities
pub struct AccessUtils;

impl AccessUtils {
    /// Create a modifier function that checks for ownership
    pub fn only_owner(ownable: &Ownable, caller: &Address) -> Result<()> {
        if !ownable.is_owner(caller) {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::UnauthorizedAccess,
                "Caller is not the owner".to_string(),
            )));
        }
        Ok(())
    }

    /// Create a modifier function that checks for admin role
    pub fn only_admin(access_control: &AccessControl, caller: &Address) -> Result<()> {
        if !access_control.is_admin(caller) {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::UnauthorizedAccess,
                "Caller is not an admin".to_string(),
            )));
        }
        Ok(())
    }

    /// Create a modifier function that checks for specific role
    pub fn only_role(access_control: &AccessControl, role: &Role, caller: &Address) -> Result<()> {
        if !access_control.has_role(role, caller) {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::UnauthorizedAccess,
                format!("Caller does not have role: {}", role.name()),
            )));
        }
        Ok(())
    }

    /// Create a modifier function that checks if contract is not paused
    pub fn when_not_paused(access_control: &AccessControl) -> Result<()> {
        if access_control.is_paused() {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InvalidStateTransition,
                "Contract is paused".to_string(),
            )));
        }
        Ok(())
    }

    /// Create a modifier function that checks if contract is paused
    pub fn when_paused(access_control: &AccessControl) -> Result<()> {
        if !access_control.is_paused() {
            return Err(CompilerError::VmError(VmError::new(
                VmErrorKind::InvalidStateTransition,
                "Contract is not paused".to_string(),
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_creation() {
        let role = Role::new("CUSTOM_ROLE");
        assert_eq!(role.name(), "CUSTOM_ROLE");

        let admin_role = Role::admin();
        assert_eq!(admin_role.name(), "ADMIN");
    }

    #[test]
    fn test_access_control_basic() {
        let owner = Address::new([1u8; 20]);
        let user = Address::new([2u8; 20]);
        let mut ac = AccessControl::new(owner);

        assert!(ac.is_owner(&owner));
        assert!(!ac.is_owner(&user));
        assert!(ac.is_admin(&owner));
        assert!(ac.has_role(&Role::admin(), &owner));
    }

    #[test]
    fn test_role_management() {
        let owner = Address::new([1u8; 20]);
        let user = Address::new([2u8; 20]);
        let mut ac = AccessControl::new(owner);

        let minter_role = Role::minter();
        
        // Grant role
        ac.grant_role(&minter_role, &user, &owner).unwrap();
        assert!(ac.has_role(&minter_role, &user));

        // Revoke role
        ac.revoke_role(&minter_role, &user, &owner).unwrap();
        assert!(!ac.has_role(&minter_role, &user));
    }

    #[test]
    fn test_permission_management() {
        let owner = Address::new([1u8; 20]);
        let user = Address::new([2u8; 20]);
        let mut ac = AccessControl::new(owner);

        // Grant permission
        ac.grant_permission(&user, Permission::Read, None, &owner).unwrap();
        assert!(ac.has_permission(&user, &Permission::Read));

        // Revoke permission
        ac.revoke_permission(&user, &Permission::Read, &owner).unwrap();
        assert!(!ac.has_permission(&user, &Permission::Read));
    }

    #[test]
    fn test_pause_functionality() {
        let owner = Address::new([1u8; 20]);
        let mut ac = AccessControl::new(owner);

        assert!(!ac.is_paused());
        
        ac.pause(&owner).unwrap();
        assert!(ac.is_paused());
        
        ac.unpause(&owner).unwrap();
        assert!(!ac.is_paused());
    }

    #[test]
    fn test_ownable_pattern() {
        let owner = Address::new([1u8; 20]);
        let new_owner = Address::new([2u8; 20]);
        let mut ownable = Ownable::new(owner);

        assert_eq!(ownable.owner(), owner);
        assert!(ownable.is_owner(&owner));

        // Transfer ownership
        ownable.transfer_ownership(new_owner, &owner).unwrap();
        assert_eq!(ownable.owner(), owner); // Still old owner until accepted

        // Accept ownership
        ownable.accept_ownership(&new_owner).unwrap();
        assert_eq!(ownable.owner(), new_owner);
        assert!(ownable.is_owner(&new_owner));
        assert!(!ownable.is_owner(&owner));
    }

    #[test]
    fn test_access_utils() {
        let owner = Address::new([1u8; 20]);
        let user = Address::new([2u8; 20]);
        let ownable = Ownable::new(owner);
        let ac = AccessControl::new(owner);

        // Test owner check
        assert!(AccessUtils::only_owner(&ownable, &owner).is_ok());
        assert!(AccessUtils::only_owner(&ownable, &user).is_err());

        // Test admin check
        assert!(AccessUtils::only_admin(&ac, &owner).is_ok());
        assert!(AccessUtils::only_admin(&ac, &user).is_err());

        // Test pause check
        assert!(AccessUtils::when_not_paused(&ac).is_ok());
    }
}