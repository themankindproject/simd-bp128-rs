//! BP128 compression library for integer arrays.
//!
//! Provides compression and decompression of `u32` arrays using the BP128
//! variable-bit-width algorithm, with SIMD acceleration support.
//!
//! # Quick Start
//!
//! ```
//! use packsimd::{compress, decompress};
//!
//! let data: Vec<u32> = (0..256).map(|i| i % 1000).collect();
//! let compressed = compress(&data).unwrap();
//! let decompressed = decompress(&compressed).unwrap();
//! assert_eq!(data, decompressed);
//! ```
//!
//! # Pre-allocating Buffers
//!
//! For zero-allocation hot paths, use [`max_compressed_size`] / [`decompressed_len`]
//! with [`compress_into`] / [`decompress_into`]:
//!
//! ```
//! use packsimd::{compress_into, max_compressed_size};
//!
//! let data: Vec<u32> = (0..256).map(|i| i % 1000).collect();
//! let mut buffer = vec![0u8; max_compressed_size(data.len())];
//! let bytes_written = compress_into(&data, &mut buffer).unwrap();
//! buffer.truncate(bytes_written);
//! ```
//!
//! ```
//! use packsimd::{compress, decompressed_len, decompress_into};
//!
//! let data: Vec<u32> = (0..256).map(|i| i % 1000).collect();
//! let compressed = compress(&data).unwrap();
//! let len = decompressed_len(&compressed).unwrap();
//! let mut output = vec![0u32; len];
//! decompress_into(&compressed, &mut output).unwrap();
//! ```
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
//! - **SIMD Support**: Automatic detection and use of SSE4.1 on x86_64, with scalar fallback
//! - **Throughput**: Typically 3-10 GB/s depending on bit width and CPU
//!
//! # Safety
//!
//! This crate uses `unsafe` code in performance-critical SIMD kernels and
//! for zero-copy reinterpretation of u32 slices as byte slices. All unsafe
//! blocks are documented with safety invariants and are covered by extensive
//! property-based testing (proptest) and fuzz testing.

#[cfg(target_endian = "big")]
compile_error!(
    "packsimd requires a little-endian target. \
     The bit-packing format and zero-copy u32↔u8 reinternals assume little-endian byte order."
);

pub use compress::{compress, compress_into, max_compressed_size};
pub use decompress::{decompress, decompress_into, decompressed_len};
pub use error::{CompressionError, DecompressionError, Error};

pub(crate) mod bitwidth;
pub(crate) mod compress;
pub(crate) mod decompress;
pub(crate) mod dispatch;
pub(crate) mod error;
pub(crate) mod simd;

/// Number of u32 values in each compressed block.
pub(crate) const BLOCK_SIZE: usize = 128;

/// Binary format version written to every compressed header.
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
