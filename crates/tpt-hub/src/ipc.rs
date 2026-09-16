//! Cross-process tensor sharing (Phase 4 System Layer: IPC).
//!
//! A dependency-free, cross-platform mailbox built on the [`crate::serialize`
//! `TPTB`] container and atomic filesystem renames:
//!
//! 1. `publish(dir, name, t)` writes `dir/<name>.tptb.tmp` and renames it to
//!    `dir/<name>.tptb`. POSIX `rename` and Windows `MoveFileEx` replacement
//!    are both atomic within a volume, so a reader never observes a partial
//!    tensor — it either sees the previous complete tensor or the new one.
//! 2. `wait_for(dir, name, timeout)` polls until the tensor appears (or times
//!    out), then loads it.
//! 3. `available(dir)` lists every published name.
//!
//! This gives two processes (or a daemon and a client) shared *tensor* state
//! without any unsafe mmap code: the workspace forbids `unsafe`, so instead of
//! shared memory we rely on the page cache making the published file cheap to
//! read.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::safetensors::HubError;
use crate::serialize::{load_tptb, save_tptb};
use tpt_tensor::Tensor;

fn tptb_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.tptb"))
}

/// Publish `tensor` under `name` into `dir` (creating `dir` if needed).
/// Readers see either the old or the new complete tensor — never a partial one.
pub fn publish(dir: &Path, name: &str, tensor: &Tensor) -> Result<PathBuf, HubError> {
    std::fs::create_dir_all(dir).map_err(|_| HubError::TooSmall)?;
    let final_path = tptb_path(dir, name);
    let tmp_path = dir.join(format!("{name}.tptb.tmp"));
    std::fs::write(&tmp_path, save_tptb(tensor)).map_err(|_| HubError::OffsetOutOfRange)?;
    // Atomic within a volume: readers of `.tptb` never see the temp file.
    std::fs::rename(&tmp_path, &final_path).map_err(|_| HubError::BadHeaderLen)?;
    Ok(final_path)
}

/// Load a previously published tensor. `Err(HubError::TooSmall)` if absent.
pub fn load(dir: &Path, name: &str) -> Result<Tensor, HubError> {
    let bytes = std::fs::read(tptb_path(dir, name)).map_err(|_| HubError::TooSmall)?;
    load_tptb(&bytes)
}

/// Poll until the named tensor appears (or the deadline passes).
pub fn wait_for(dir: &Path, name: &str, timeout: Duration) -> Result<Tensor, HubError> {
    let deadline = Instant::now() + timeout;
    loop {
        match load(dir, name) {
            Ok(t) => return Ok(t),
            Err(HubError::TooSmall) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(e) => return Err(e),
        }
    }
}

/// Every tensor name currently published in `dir`.
pub fn available(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let fname = e.file_name().to_string_lossy().to_string();
            if let Some(stem) = fname.strip_suffix(".tptb") {
                out.push(stem.to_string());
            }
        }
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_tensor::DType;

    #[test]
    fn publish_load_roundtrip() {
        let dir = std::env::temp_dir().join("tpt_ipc_test_pub");
        let _ = std::fs::remove_dir_all(&dir);
        let t = Tensor::from_typed(vec![1.5_f64, -2.5, 3.0])
            .reshape(&[3])
            .unwrap();
        publish(&dir, "weights", &t).unwrap();

        assert_eq!(available(&dir), vec!["weights".to_string()]);
        let back = load(&dir, "weights").unwrap();
        assert_eq!(back.dtype(), DType::F64);
        assert_eq!(back.to_vec::<f64>().unwrap(), vec![1.5, -2.5, 3.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn republish_is_atomic_replacement() {
        let dir = std::env::temp_dir().join("tpt_ipc_test_rep");
        let _ = std::fs::remove_dir_all(&dir);
        let v1 = Tensor::from_typed(vec![1.0_f64]);
        let v2 = Tensor::from_typed(vec![2.0_f64]);
        publish(&dir, "slot", &v1).unwrap();
        publish(&dir, "slot", &v2).unwrap();
        // exactly one published tensor, holding the new value
        assert_eq!(available(&dir), vec!["slot".to_string()]);
        assert_eq!(
            load(&dir, "slot").unwrap().to_vec::<f64>().unwrap(),
            vec![2.0]
        );
        // no leftover temp files
        let entries: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(entries.len(), 1, "leftover files: {entries:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wait_for_times_out_when_absent() {
        let dir = std::env::temp_dir().join("tpt_ipc_test_wait");
        let _ = std::fs::remove_dir_all(&dir);
        let r = wait_for(&dir, "never", Duration::from_millis(20));
        assert!(matches!(r, Err(HubError::TooSmall)));
    }

    #[test]
    fn wait_for_returns_once_published() {
        let dir = std::env::temp_dir().join("tpt_ipc_test_wait_ok");
        let _ = std::fs::remove_dir_all(&dir);
        let writer_dir = dir.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(40));
            let t = Tensor::from_typed(vec![7_i32, 8]);
            let _ = publish(&writer_dir, "signal", &t);
        });
        let got = wait_for(&dir, "signal", Duration::from_secs(2)).unwrap();
        assert_eq!(got.to_vec::<i32>().unwrap(), vec![7, 8]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
