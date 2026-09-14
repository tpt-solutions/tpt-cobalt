//! Fleet OTA rollout: sha256-digested artifacts keyed by node id, staged
//! two-phase (stage every node, then commit), aborting on the first failure.

use serde::{Deserialize, Serialize};
use tpt_alloy::firmware::FirmwareTarget;

/// One node's deployable artifact: the forked partitioner's node id plus its
/// firmware target and image bytes (UF2 for RP2040, raw binary for the ESP32
/// serial path).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeArtifact {
    pub node_id: usize,
    pub target: FirmwareTarget,
    pub release_id: String,
    pub image: Vec<u8>,
    pub sha256: String,
}

impl NodeArtifact {
    /// Build an artifact, computing the image digest.
    pub fn new(node_id: usize, target: FirmwareTarget, release_id: &str, image: Vec<u8>) -> Self {
        let sha256 = crate::sha256_hex(&image);
        NodeArtifact {
            node_id,
            target,
            release_id: release_id.to_string(),
            image,
            sha256,
        }
    }

    /// Whether the stored digest still matches the image (tamper/corruption
    /// check before staging).
    pub fn verify(&self) -> bool {
        crate::sha256_hex(&self.image) == self.sha256
    }
}

/// The full fleet update plan (JSON-serializable; `image` fields are
/// typically stripped before serializing — see [`UpdateManifest::to_json`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateManifest {
    pub release_id: String,
    pub artifacts: Vec<NodeArtifact>,
    /// Abort the whole rollout after the first failed node (true) or skip
    /// and continue (false). Default true.
    pub abort_on_failure: bool,
}

impl UpdateManifest {
    pub fn new(release_id: &str) -> Self {
        UpdateManifest {
            release_id: release_id.to_string(),
            artifacts: Vec::new(),
            abort_on_failure: true,
        }
    }

    /// Add an artifact; duplicate node ids are rejected (a rollout stages
    /// every node exactly once).
    pub fn push(&mut self, artifact: NodeArtifact) -> Result<(), String> {
        if self.artifacts.iter().any(|a| a.node_id == artifact.node_id) {
            return Err(format!("duplicate node id {}", artifact.node_id));
        }
        self.artifacts.push(artifact);
        Ok(())
    }

    /// Every artifact must carry this release's id and verify its digest.
    pub fn validate(&self) -> Result<(), String> {
        for a in &self.artifacts {
            if a.release_id != self.release_id {
                return Err(format!(
                    "node {} carries release '{}' but manifest is '{}'",
                    a.node_id, a.release_id, self.release_id
                ));
            }
            if !a.verify() {
                return Err(format!("node {} digest mismatch", a.node_id));
            }
        }
        Ok(())
    }

    /// JSON export with image bytes stripped (digests remain).
    pub fn to_json(&self) -> serde_json::Result<String> {
        let mut meta = self.clone();
        for a in &mut meta.artifacts {
            a.image.clear();
        }
        serde_json::to_string_pretty(&meta)
    }
}

/// Progress through the two-phase rollout: stage every node, then commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RolloutStatus {
    /// Some nodes still need staging.
    Staging,
    /// All nodes staged OK; ready to commit.
    ReadyToCommit,
    /// A node failed and the rollout aborted (nodes staged after the
    /// failure were never touched).
    Aborted,
    /// All nodes staged and committed.
    Committed,
}

/// The staged rollout state machine over an [`UpdateManifest`].
#[derive(Debug)]
pub struct Rollout {
    manifest: UpdateManifest,
    next: usize,
    staged_ok: Vec<usize>,
    failed: Option<(usize, String)>,
    committed: bool,
}

impl Rollout {
    /// Validate the manifest, then start a fresh rollout. Staging visits
    /// nodes in ascending node-id order regardless of manifest order.
    pub fn begin(mut manifest: UpdateManifest) -> Result<Self, String> {
        manifest.validate()?;
        manifest.artifacts.sort_by_key(|a| a.node_id);
        Ok(Rollout {
            manifest,
            next: 0,
            staged_ok: Vec::new(),
            failed: None,
            committed: false,
        })
    }

    /// The next node to stage, in node-id order (`None` once staged/aborted).
    pub fn next_node(&self) -> Option<&NodeArtifact> {
        if self.failed.is_some() || self.committed {
            return None;
        }
        self.manifest.artifacts.get(self.next)
    }

    /// Mark the current node staged successfully.
    pub fn mark_staged(&mut self) {
        if let Some(a) = self.manifest.artifacts.get(self.next) {
            if self.failed.is_none() && !self.committed {
                self.staged_ok.push(a.node_id);
                self.next += 1;
            }
        }
    }

    /// Mark the current node failed (aborts the rollout when
    /// `abort_on_failure`).
    pub fn mark_failed(&mut self, reason: &str) {
        if let Some(a) = self.manifest.artifacts.get(self.next) {
            if self.manifest.abort_on_failure {
                self.failed = Some((a.node_id, reason.to_string()));
            } else {
                self.next += 1;
            }
        }
    }

    /// Commit after every node staged OK.
    pub fn commit(&mut self) -> Result<(), String> {
        if self.failed.is_some() {
            return Err("rollout aborted; cannot commit".into());
        }
        if self.next < self.manifest.artifacts.len() {
            return Err("not all nodes staged".into());
        }
        self.committed = true;
        Ok(())
    }

    pub fn status(&self) -> RolloutStatus {
        if self.failed.is_some() {
            RolloutStatus::Aborted
        } else if self.committed {
            RolloutStatus::Committed
        } else if self.next >= self.manifest.artifacts.len() && !self.manifest.artifacts.is_empty()
        {
            RolloutStatus::ReadyToCommit
        } else {
            RolloutStatus::Staging
        }
    }

    /// `(staged, total)` node counts.
    pub fn progress(&self) -> (usize, usize) {
        (self.staged_ok.len(), self.manifest.artifacts.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> UpdateManifest {
        let mut m = UpdateManifest::new("release-7");
        m.push(NodeArtifact::new(
            2,
            FirmwareTarget::Rp2040,
            "release-7",
            b"firmware-node2".to_vec(),
        ))
        .unwrap();
        m.push(NodeArtifact::new(
            0,
            FirmwareTarget::Esp32,
            "release-7",
            b"firmware-node0".to_vec(),
        ))
        .unwrap();
        m
    }

    #[test]
    fn manifest_rejects_duplicates_digest_mismatch_and_foreign_release() {
        let mut m = UpdateManifest::new("release-7");
        m.push(NodeArtifact::new(0, FirmwareTarget::Esp32, "release-7", b"x".to_vec()))
            .unwrap();
        assert!(m.push(NodeArtifact::new(0, FirmwareTarget::Rp2040, "release-7", b"y".to_vec()))
            .is_err());

        let mut bad = manifest();
        bad.artifacts[0].image = b"tampered".to_vec();
        assert_eq!(
            bad.validate().unwrap_err(),
            format!("node {} digest mismatch", bad.artifacts[0].node_id)
        );

        let mut foreign = manifest();
        foreign.artifacts[0].release_id = "release-6".into();
        assert!(foreign.validate().unwrap_err().contains("release-6"));
    }

    #[test]
    fn json_export_strips_images_but_keeps_digests() {
        let m = manifest();
        let json = m.to_json().unwrap();
        assert!(!json.contains("firmware-node2"));
        assert!(json.contains(&m.artifacts[0].sha256));
        assert!(json.contains("release-7"));
    }

    #[test]
    fn rollout_stages_every_node_then_commits() {
        let mut r = Rollout::begin(manifest()).unwrap();
        assert_eq!(r.status(), RolloutStatus::Staging);
        // node-id order: 0 (esp32), 2 (rp2040)
        assert_eq!(r.next_node().unwrap().node_id, 0);
        r.mark_staged();
        assert_eq!(r.next_node().unwrap().node_id, 2);
        r.mark_staged();
        assert_eq!(r.status(), RolloutStatus::ReadyToCommit);
        assert_eq!(r.progress(), (2, 2));
        assert!(r.next_node().is_none());
        r.commit().unwrap();
        assert_eq!(r.status(), RolloutStatus::Committed);
    }

    #[test]
    fn rollout_aborts_on_failure_and_commit_refuses() {
        let mut r = Rollout::begin(manifest()).unwrap();
        r.mark_staged(); // node 0 ok
        r.mark_failed("serial port gone"); // node 2 fails
        assert_eq!(r.status(), RolloutStatus::Aborted);
        assert!(r.next_node().is_none(), "no further staging after abort");
        assert_eq!(r.commit().unwrap_err(), "rollout aborted; cannot commit");
    }

    #[test]
    fn rollout_skip_mode_continues_past_failures() {
        let mut m = manifest();
        m.abort_on_failure = false;
        let mut r = Rollout::begin(m).unwrap();
        r.mark_failed("node 0 offline");
        assert_eq!(r.next_node().unwrap().node_id, 2);
        r.mark_staged();
        assert_eq!(r.status(), RolloutStatus::ReadyToCommit);
        assert_eq!(r.progress(), (1, 2));
    }

    #[test]
    fn commit_refuses_before_everything_is_staged() {
        let mut r = Rollout::begin(manifest()).unwrap();
        r.mark_staged();
        assert_eq!(r.commit().unwrap_err(), "not all nodes staged");
        assert_eq!(r.status(), RolloutStatus::Staging);
    }
}
