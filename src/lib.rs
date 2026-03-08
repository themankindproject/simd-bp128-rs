//! BP128 compression library for integer arrays.
//!
//! Provides compression and decompression of `u32` arrays using the BP128
//! variable-bit-width algorithm, with SIMD acceleration support.

pub use compress::compress;
pub use decompress::decompress;
pub use error::{CompressionError, DecompressionError, Error};

pub(crate) mod bitwidth;
pub(crate) mod compress;
pub(crate) mod decompress;
pub(crate) mod dispatch;
pub(crate) mod error;
pub(crate) mod simd;

pub(crate) const BLOCK_SIZE: usize = 128;
pub(crate) const FORMAT_VERSION: u8 = 1;

/// Internal types exposed for benchmarks and integration tests.
///
/// **Not part of the public API.** These items may change or be removed at
/// any point. Do not depend on them from outside this crate.
#[doc(hidden)]
pub mod internal {
    pub use crate::simd::scalar::ScalarBackend;
    pub use crate::simd::SimdBackend;
}
