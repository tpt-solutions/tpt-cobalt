//! # tpt-alloy-deploy — the Alloy deploy layer, native Rust (Phase 5)
//!
//! Closes the gap the spec flags for the Alloy (MCU swarm) backend: the
//! forked `tpt-alloy` crate already partitions models and generates per-node
//! firmware sources (`FirmwareBundle`), but upstream had *"no deployment
//! mechanism"* (`docs/phase5-exotic-backends.md`). This crate provides the
//! missing deploy half natively — **no port of tpt-basestation's Python**,
//! only its protocol knowledge re-implemented:
//!
//! - **RP2040**: raw firmware bytes → [UF2](uf2) blocks (the USB
//!   mass-storage drop format the RP2040 bootloader consumes),
//! - **ESP32**: ROM-UART **SLIP** framing + command packets + a flash-plan
//!   driver (`flash_begin` → `flash_data`… → `flash_end`) over a [`Transport`]
//!   trait, so the real serial backend is one impl away and tests run on an
//!   in-memory transport,
//! - **Fleet OTA**: a staged rollout manifest (sha256-digested artifacts keyed
//!   by node id + [`tpt_alloy::FirmwareTarget`]) with an abort-on-failure
//!   state machine.
//!
//! Everything here is offline-testable byte-level protocol work; hardware
//! bring-up (real `serialport` transport, board bring-up) is deliberately
//! the only part left behind a trait.

pub mod esp;
pub mod ota;
pub mod uf2;

pub use esp::{slip_decode, slip_encode, EspError, EspFlasher, Transport};
pub use ota::{NodeArtifact, Rollout, RolloutStatus, UpdateManifest};
pub use uf2::{to_uf2, UF2_FAMILY_RP2040};

/// SHA-256 hex digest of an artifact (manifest integrity).
pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_is_well_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
