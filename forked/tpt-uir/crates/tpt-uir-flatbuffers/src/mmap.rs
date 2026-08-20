//! Zero-copy memory-mapped access to a TPT-UIR FlatBuffers file.
//!
//! Requires the `mmap` feature (which implies `std`). The buffer is never
//! copied or decoded: [`root_as_region`] reads structs directly out of the
//! mapped pages.

use memmap2::Mmap;

/// Open `path` and memory-map it read-only.
///
/// Example:
///
/// ```no_run
/// # #[cfg(feature = "mmap")]
/// # fn run() -> std::io::Result<()> {
/// use tpt_uir_flatbuffers::mmap::open_mmap;
/// use tpt_uir_flatbuffers::root_as_region;
///
/// let mmap = open_mmap("model.tptuir")?;
/// let region = root_as_region(&mmap[..]).expect("valid region");
/// assert!(region.blocks().is_some());
/// # Ok(())
/// # }
/// ```
pub fn open_mmap<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Mmap> {
    let file = std::fs::File::open(path)?;
    // SAFETY: we only ever read the mapping, and the underlying file is not
    // mutated while mapped (it is opened read-only by us).
    unsafe { Mmap::map(&file) }
}
