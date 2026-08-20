//! Hardware-Locked Model IP Protection — bind packages to specific hardware serials.

use serde::{Deserialize, Serialize};
use sha2::Digest;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum IpLockError {
    #[error("lock verification failed: package is bound to different hardware")]
    Mismatch,
    #[error("lock has no allowed hardware IDs")]
    EmptyLock,
}

/// Hardware lock that binds a `.tptpkg` to specific hardware serial numbers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HardwareLock {
    pub fingerprint_sha256: String,
    pub lock_type: String,
    pub locked_at: f64,
    pub issuer: String,
    pub allowed_ids: Vec<String>,
}

impl HardwareLock {
    pub fn create(hardware_ids: &[String]) -> Self {
        let mut sorted = hardware_ids.to_vec();
        sorted.sort();
        let combined: Vec<&str> = sorted.iter().map(|s| s.as_str()).collect();
        let joined = combined.join("|");
        let mut hasher = sha2::Sha256::new();
        hasher.update(joined.as_bytes());
        let fingerprint = format!("{:x}", hasher.finalize());

        Self {
            fingerprint_sha256: fingerprint,
            lock_type: "hardware_bound".into(),
            locked_at: 0.0,
            issuer: "tpt-crucible".into(),
            allowed_ids: sorted,
        }
    }

    pub fn verify(&self, present_ids: &[String]) -> Result<(), IpLockError> {
        if self.allowed_ids.is_empty() {
            return Ok(());
        }
        let allowed_set: std::collections::HashSet<&str> =
            self.allowed_ids.iter().map(|s| s.as_str()).collect();
        for id in present_ids {
            if !allowed_set.contains(id.as_str()) {
                return Err(IpLockError::Mismatch);
            }
        }
        Ok(())
    }
}
