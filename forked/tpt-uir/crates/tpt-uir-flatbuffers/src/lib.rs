#![no_std]
#![doc = include_str!("../README.md")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod convert;

#[allow(clippy::all)]
#[allow(warnings)]
pub mod generated {
    include!("./generated/tpt_uir_generated.rs");
}

#[cfg(feature = "mmap")]
pub mod mmap;

pub use convert::{from_flatbuffer, to_flatbuffer};
pub use generated::tpt_uir::root_as_region;
