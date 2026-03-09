use crate::error::Error;

#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512;
pub mod scalar;
#[cfg(target_arch = "x86_64")]
pub mod sse;

/// Trait for implementing BP128 compression backends.
///
/// Defines the interface for packing/unpacking 128-value blocks using
/// variable bit widths. Implementations may use scalar or SIMD algorithms.
pub trait SimdBackend: Send + Sync {
    /// Packs 128 `u32` values into bit-packed output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidBitWidth`] if `bit_width > 32`.
    /// Returns [`Error::OutputTooSmall`] if output buffer is too small.
    fn pack_block(input: &[u32; 128], bit_width: u8, output: &mut [u8]) -> Result<(), Error>;

    /// Unpacks 128 values from bit-packed input.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidBitWidth`] if `bit_width > 32`.
    /// Returns [`Error::InputTooShort`] if input buffer is too small.
    fn unpack_block(input: &[u8], bit_width: u8, output: &mut [u32; 128]) -> Result<(), Error>;
}
