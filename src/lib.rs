//! BP128 compression library for integer arrays.
//!
//! Provides compression and decompression of `u32` arrays using the BP128
//! variable-bit-width algorithm, with SIMD acceleration support.
//!
//! # Binary Format
//!
//! The compressed format is designed for fast random access and is
//! structured as follows:
//!
//! ```text
//! [version: u8][input_len: u32 LE][num_blocks: u32 LE][bit_widths: u8 × N][packed_data: u8[]]
//! ```
//!
//! | Field       | Type     | Description                                      |
//! |-------------|----------|--------------------------------------------------|
//! | version     | [u8]     | Format version (currently 1)                   |
//! | input_len   | [u32] LE | Original number of u32 values                   |
//! | num_blocks  | [u32] LE | Number of blocks (ceiling of input_len / 128)    |
//! | bit_widths  | [u8]     | One byte per block (0 = all zeros)               |
//! | packed_data | [u8]     | Bit-packed values, blocks concatenated           |
//!
//! # Algorithm
//!
//! BP128 divides the input into blocks of 128 values. For each block,
//! it calculates the minimum number of bits required to represent the
//! maximum value in that block, then stores all values using that bit width.
//! This achieves variable-bit-width compression optimized for integer arrays
//! with moderate value ranges.
//!
//! # Performance
//!
//! - **Compression**: O(n) time complexity where n = input.len()
//! - **Decompression**: O(n) time complexity
//! - **SIMD Support**: Automatic detection and use of SSE4.1, AVX2, or AVX512
//! - **Throughput**: Typically 3-10 GB/s depending on bit width and CPU

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
    pub use crate::simd::sse::SseBackend;
    pub use crate::simd::SimdBackend;
}
