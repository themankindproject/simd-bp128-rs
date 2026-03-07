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
