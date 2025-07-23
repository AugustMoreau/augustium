//! Augustium Standard Library
//!
//! This module contains the standard library for the Augustium programming language,
//! providing essential types, functions, and utilities for smart contract development.

pub mod access_control;
pub mod address;
pub mod collections;
pub mod core_types;
pub mod crypto;
pub mod defi;
pub mod events;
pub mod governance;
pub mod math;
pub mod ml;
pub mod oracle;
pub mod storage;
pub mod string;
pub mod time;

#[cfg(test)]
mod tests;

// Re-export commonly used types
pub use access_control::*;
pub use core_types::*;
pub use collections::*;
pub use crypto::*;
pub use defi::*;
pub use governance::*;
pub use math::*;
pub use ml::*;
pub use oracle::*;
pub use string::*;

/// Standard library version
pub const VERSION: &str = "0.1.0";

// Set up the standard library when compiler starts
pub fn init() {
    // Global initialization code here
    log::info!("Augustium Standard Library v{} initialized", VERSION);
}