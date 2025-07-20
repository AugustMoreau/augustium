//! Time module for AugustC standard library
//! Provides timestamp and duration handling functionality

use crate::error::{Result, CompilerError, SemanticError, SemanticErrorKind, SourceLocation};
use crate::stdlib::core_types::AugustiumType;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

/// Timestamp type representing Unix timestamp in seconds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Timestamp(pub u64);

impl Timestamp {
    /// Create a new timestamp from Unix seconds
    pub fn new(seconds: u64) -> Self {
        Timestamp(seconds)
    }
    
    /// Get current timestamp
    pub fn now() -> Result<Self> {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| Timestamp(d.as_secs()))
            .map_err(|_| CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Failed to get current time".to_string(),
            }))
    }
    
    /// Create timestamp from milliseconds
    pub fn from_millis(millis: u64) -> Self {
        Timestamp(millis / 1000)
    }
    
    /// Convert to milliseconds
    pub fn to_millis(&self) -> u64 {
        self.0 * 1000
    }
    
    /// Add duration to timestamp
    pub fn add_duration(&self, duration: Duration) -> Option<Self> {
        self.0.checked_add(duration.0).map(Timestamp)
    }
    
    /// Subtract duration from timestamp
    pub fn sub_duration(&self, duration: Duration) -> Option<Self> {
        self.0.checked_sub(duration.0).map(Timestamp)
    }
    
    /// Calculate duration between two timestamps
    pub fn duration_since(&self, other: Timestamp) -> Option<Duration> {
        if self.0 >= other.0 {
            Some(Duration::new(self.0 - other.0))
        } else {
            None
        }
    }
    
    /// Check if timestamp is before another
    pub fn is_before(&self, other: Timestamp) -> bool {
        self.0 < other.0
    }
    
    /// Check if timestamp is after another
    pub fn is_after(&self, other: Timestamp) -> bool {
        self.0 > other.0
    }
}

impl AugustiumType for Timestamp {
    fn type_name() -> &'static str { "timestamp" }
    fn size_bytes() -> usize { 8 }
    
    fn to_bytes(&self) -> Vec<u8> {
        self.0.to_be_bytes().to_vec()
    }
    
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 8 {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::TypeMismatch {
                    expected: "8 bytes".to_string(),
                    found: format!("{} bytes", bytes.len()),
                },
                location: SourceLocation::unknown(),
                message: "Invalid byte length for timestamp".to_string(),
            }));
        }
        let mut array = [0u8; 8];
        array.copy_from_slice(bytes);
        Ok(Timestamp(u64::from_be_bytes(array)))
    }
    
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Duration type representing time duration in seconds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Duration(pub u64);

impl Duration {
    /// Create a new duration from seconds
    pub fn new(seconds: u64) -> Self {
        Duration(seconds)
    }
    
    /// Create duration from milliseconds
    pub fn from_millis(millis: u64) -> Self {
        Duration(millis / 1000)
    }
    
    /// Create duration from minutes
    pub fn from_minutes(minutes: u64) -> Self {
        Duration(minutes * 60)
    }
    
    /// Create duration from hours
    pub fn from_hours(hours: u64) -> Self {
        Duration(hours * 3600)
    }
    
    /// Create duration from days
    pub fn from_days(days: u64) -> Self {
        Duration(days * 86400)
    }
    
    /// Convert to milliseconds
    pub fn to_millis(&self) -> u64 {
        self.0 * 1000
    }
    
    /// Convert to minutes
    pub fn to_minutes(&self) -> u64 {
        self.0 / 60
    }
    
    /// Convert to hours
    pub fn to_hours(&self) -> u64 {
        self.0 / 3600
    }
    
    /// Convert to days
    pub fn to_days(&self) -> u64 {
        self.0 / 86400
    }
    
    /// Add two durations
    pub fn add(&self, other: Duration) -> Option<Duration> {
        self.0.checked_add(other.0).map(Duration)
    }
    
    /// Subtract duration
    pub fn sub(&self, other: Duration) -> Option<Duration> {
        self.0.checked_sub(other.0).map(Duration)
    }
    
    /// Multiply duration by scalar
    pub fn mul(&self, scalar: u64) -> Option<Duration> {
        self.0.checked_mul(scalar).map(Duration)
    }
    
    /// Divide duration by scalar
    pub fn div(&self, scalar: u64) -> Option<Duration> {
        if scalar == 0 {
            None
        } else {
            Some(Duration(self.0 / scalar))
        }
    }
    
    /// Check if duration is zero
    pub fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl AugustiumType for Duration {
    fn type_name() -> &'static str { "duration" }
    fn size_bytes() -> usize { 8 }
    
    fn to_bytes(&self) -> Vec<u8> {
        self.0.to_be_bytes().to_vec()
    }
    
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 8 {
            return Err(CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::TypeMismatch {
                    expected: "8 bytes".to_string(),
                    found: format!("{} bytes", bytes.len()),
                },
                location: SourceLocation::unknown(),
                message: "Invalid byte length for duration".to_string(),
            }));
        }
        let mut array = [0u8; 8];
        array.copy_from_slice(bytes);
        Ok(Duration(u64::from_be_bytes(array)))
    }
    
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 >= 86400 {
            write!(f, "{}d {}h {}m {}s", 
                   self.to_days(), 
                   (self.0 % 86400) / 3600,
                   (self.0 % 3600) / 60,
                   self.0 % 60)
        } else if self.0 >= 3600 {
            write!(f, "{}h {}m {}s", 
                   self.to_hours(), 
                   (self.0 % 3600) / 60,
                   self.0 % 60)
        } else if self.0 >= 60 {
            write!(f, "{}m {}s", self.to_minutes(), self.0 % 60)
        } else {
            write!(f, "{}s", self.0)
        }
    }
}

/// Time utilities
pub struct Time;

impl Time {
    /// Sleep for a duration (simulation for blockchain context)
    pub fn sleep(_duration: Duration) -> Result<()> {
        // In a real blockchain context, this would be handled by the runtime
        // For now, we just return success
        Ok(())
    }
    
    /// Get elapsed time since a timestamp
    pub fn elapsed_since(timestamp: Timestamp) -> Result<Duration> {
        let now = Timestamp::now()?;
        now.duration_since(timestamp)
            .ok_or_else(|| CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Timestamp is in the future".to_string(),
            }))
    }
    
    /// Check if a timeout has occurred
    pub fn is_timeout(start: Timestamp, timeout: Duration) -> Result<bool> {
        let now = Timestamp::now()?;
        let deadline = start.add_duration(timeout)
            .ok_or_else(|| CompilerError::SemanticError(SemanticError {
                kind: SemanticErrorKind::InvalidOperation,
                location: SourceLocation::unknown(),
                message: "Timestamp overflow".to_string(),
            }))?;
        Ok(now.is_after(deadline))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_creation() {
        let ts = Timestamp::new(1234567890);
        assert_eq!(ts.0, 1234567890);
        
        let ts_millis = Timestamp::from_millis(1234567890000);
        assert_eq!(ts_millis.0, 1234567890);
    }
    
    #[test]
    fn test_timestamp_operations() {
        let ts1 = Timestamp::new(1000);
        let ts2 = Timestamp::new(2000);
        let duration = Duration::new(500);
        
        assert!(ts1.is_before(ts2));
        assert!(ts2.is_after(ts1));
        
        let ts3 = ts1.add_duration(duration).unwrap();
        assert_eq!(ts3.0, 1500);
        
        let ts4 = ts2.sub_duration(duration).unwrap();
        assert_eq!(ts4.0, 1500);
        
        let diff = ts2.duration_since(ts1).unwrap();
        assert_eq!(diff.0, 1000);
    }
    
    #[test]
    fn test_duration_creation() {
        let d1 = Duration::new(3600);
        assert_eq!(d1.to_hours(), 1);
        
        let d2 = Duration::from_hours(2);
        assert_eq!(d2.0, 7200);
        
        let d3 = Duration::from_days(1);
        assert_eq!(d3.0, 86400);
    }
    
    #[test]
    fn test_duration_operations() {
        let d1 = Duration::new(1000);
        let d2 = Duration::new(500);
        
        let sum = d1.add(d2).unwrap();
        assert_eq!(sum.0, 1500);
        
        let diff = d1.sub(d2).unwrap();
        assert_eq!(diff.0, 500);
        
        let product = d1.mul(2).unwrap();
        assert_eq!(product.0, 2000);
        
        let quotient = d1.div(2).unwrap();
        assert_eq!(quotient.0, 500);
    }
    
    #[test]
    fn test_serialization() {
        let ts = Timestamp::new(1234567890);
        let bytes = ts.to_bytes();
        let restored = Timestamp::from_bytes(&bytes).unwrap();
        assert_eq!(ts, restored);
        
        let duration = Duration::new(3600);
        let bytes = duration.to_bytes();
        let restored = Duration::from_bytes(&bytes).unwrap();
        assert_eq!(duration, restored);
    }
    
    #[test]
    fn test_display() {
        let ts = Timestamp::new(1234567890);
        assert_eq!(ts.to_string(), "1234567890");
        
        let d1 = Duration::new(3661); // 1h 1m 1s
        assert!(d1.to_string().contains("1h"));
        assert!(d1.to_string().contains("1m"));
        assert!(d1.to_string().contains("1s"));
        
        let d2 = Duration::new(90061); // 1d 1h 1m 1s
        assert!(d2.to_string().contains("1d"));
    }
    
    #[test]
    fn test_error_handling() {
        let ts = Timestamp::new(100);
        let duration = Duration::new(200);
        
        // Underflow should return None
        assert!(ts.sub_duration(duration).is_none());
        
        // Division by zero should return None
        assert!(duration.div(0).is_none());
        
        // Invalid byte length should return error
        let invalid_bytes = vec![1, 2, 3]; // Wrong length
        assert!(Timestamp::from_bytes(&invalid_bytes).is_err());
        assert!(Duration::from_bytes(&invalid_bytes).is_err());
    }
}