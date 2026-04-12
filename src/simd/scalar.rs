//! Optimized scalar implementation of BP128 bit-packing kernels.
//!
//! # Safety
//!
//! All kernels validate input/output buffer bounds before access.
//! The bit layout is little-endian: value 0 occupies the least significant bits.

#![allow(clippy::needless_range_loop)]

use crate::error::Error;
use crate::simd::SimdBackend;

/// Maximum valid bit width.
const MAX_BIT_WIDTH: u8 = 32;

/// Scalar (non-SIMD) implementation of the BP128 bit-packing backend.
///
/// This implementation provides correct bit-packing operations without
/// requiring SIMD instruction sets. It serves as:
/// - A fallback when SIMD is unavailable
/// - A reference implementation for correctness verification
/// - The implementation used for partial blocks (< 128 values)
///
/// # Performance
///
/// Time complexity: O(n) where n is the number of values.
/// Throughput: Approximately 3-5 GB/s depending on bit width and CPU.
///
/// For higher throughput on x86_64, the dispatch module automatically
/// selects SSE4.1, AVX2, or AVX512 implementations when available.
///
/// # Bit Layout
///
/// Values are packed in little-endian order within each block:
/// - Value 0 occupies bits 0..(b-1)
/// - Value 1 occupies bits b..(2b-1)
/// - etc.
///
/// When values cross byte boundaries, they are split appropriately.
#[must_use]
pub struct ScalarBackend;

impl SimdBackend for ScalarBackend {
    fn pack_block(input: &[u32; 128], bit_width: u8, output: &mut [u8]) -> Result<(), Error> {
        pack_n(input, bit_width, output)
    }

    fn unpack_block(input: &[u8], bit_width: u8, output: &mut [u32; 128]) -> Result<(), Error> {
        unpack_n(input, bit_width, 128, output)
    }
}

impl ScalarBackend {
    /// Packs a partial block (fewer than 128 values).
    ///
    /// Handles the final block in a compressed array.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidBitWidth`] if `bit_width > 32`.
    /// Returns [`Error::OutputTooSmall`] if output buffer is too small.
    pub fn pack_partial_block(
        input: &[u32],
        bit_width: u8,
        output: &mut [u8],
    ) -> Result<(), Error> {
        pack_n(input, bit_width, output)
    }

    /// Unpacks a partial block (fewer than 128 values).
    ///
    /// Handles the final block in a compressed array.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidBitWidth`] if `bit_width > 32`.
    /// Returns [`Error::InputTooShort`] if input buffer is too small.
    /// Returns [`Error::OutputTooSmall`] if output buffer is too small.
    pub fn unpack_partial_block(
        input: &[u8],
        bit_width: u8,
        num_values: usize,
        output: &mut [u32],
    ) -> Result<(), Error> {
        if output.len() < num_values {
            return Err(Error::OutputTooSmall {
                need: num_values,
                got: output.len(),
            });
        }
        unpack_n(input, bit_width, num_values, output)
    }
}

/// Pack `num_values` integers using `bit_width` bits each.
#[inline]
fn pack_n(input: &[u32], bit_width: u8, output: &mut [u8]) -> Result<(), Error> {
    if bit_width > MAX_BIT_WIDTH {
        return Err(Error::InvalidBitWidth(bit_width));
    }

    if bit_width == 0 {
        return Ok(());
    }

    let num_values = input.len();
    let required_bytes = packed_size_bytes(num_values, bit_width);

    if output.len() < required_bytes {
        return Err(Error::OutputTooSmall {
            need: required_bytes,
            got: output.len(),
        });
    }

    match bit_width {
        1 => pack_1bit(input, output),
        2 => pack_2bit(input, output),
        3 => pack_3bit(input, output),
        4 => pack_4bit(input, output),
        5 => pack_5bit(input, output),
        6 => pack_6bit(input, output),
        7 => pack_7bit(input, output),
        8 => pack_8bit(input, output),
        9 => pack_9bit(input, output),
        10 => pack_10bit(input, output),
        11 => pack_11bit(input, output),
        12 => pack_12bit(input, output),
        13 => pack_13bit(input, output),
        14 => pack_14bit(input, output),
        15 => pack_15bit(input, output),
        16 => pack_16bit(input, output),
        17 => pack_17bit(input, output),
        18 => pack_18bit(input, output),
        19 => pack_19bit(input, output),
        20 => pack_20bit(input, output),
        21 => pack_21bit(input, output),
        22 => pack_22bit(input, output),
        23 => pack_23bit(input, output),
        24 => pack_24bit(input, output),
        25 => pack_25bit(input, output),
        26 => pack_26bit(input, output),
        27 => pack_27bit(input, output),
        28 => pack_28bit(input, output),
        29 => pack_29bit(input, output),
        30 => pack_30bit(input, output),
        31 => pack_31bit(input, output),
        32 => pack_32bit(input, output),
        _ => unreachable!("bit_width validated above, must be 0-32"),
    }
}

/// Calculate required bytes to pack `num_values` integers with `bit_width` bits each.
#[inline(always)]
const fn packed_size_bytes(num_values: usize, bit_width: u8) -> usize {
    if bit_width == 0 {
        0
    } else {
        (num_values * bit_width as usize + 7) / 8
    }
}

/// Pack 1-bit values by extracting the LSB from each input value.
///
/// Each group of 8 input values becomes one output byte, with value `i`'s
/// bit occupying bit position `i` in the output byte.
#[inline]
fn pack_1bit(input: &[u32], output: &mut [u8]) -> Result<(), Error> {
    let num_values = input.len();
    let full_chunks = num_values / 8;

    for i in 0..full_chunks {
        let base = i * 8;
        let byte = (input[base] & 1)
            | ((input[base + 1] & 1) << 1)
            | ((input[base + 2] & 1) << 2)
            | ((input[base + 3] & 1) << 3)
            | ((input[base + 4] & 1) << 4)
            | ((input[base + 5] & 1) << 5)
            | ((input[base + 6] & 1) << 6)
            | ((input[base + 7] & 1) << 7);
        output[i] = byte as u8;
    }

    let remainder_start = full_chunks * 8;
    if remainder_start < num_values {
        let mut byte = 0u8;
        let remaining = num_values - remainder_start;
        for j in 0..remaining {
            byte |= ((input[remainder_start + j] & 1) as u8) << j;
        }
        output[full_chunks] = byte;
    }

    Ok(())
}

/// Pack 2-bit values using optimized bit manipulation.
#[inline]
fn pack_2bit(input: &[u32], output: &mut [u8]) -> Result<(), Error> {
    let num_values = input.len();

    let full_chunks = num_values / 4;
    for i in 0..full_chunks {
        output[i] = ((input[i * 4] & 3) as u8)
            | (((input[i * 4 + 1] & 3) as u8) << 2)
            | (((input[i * 4 + 2] & 3) as u8) << 4)
            | (((input[i * 4 + 3] & 3) as u8) << 6);
    }

    let remainder_start = full_chunks * 4;
    if remainder_start < num_values {
        let mut byte = 0u8;
        let remaining = num_values - remainder_start;
        for j in 0..remaining {
            byte |= ((input[remainder_start + j] & 3) as u8) << (j * 2);
        }
        output[full_chunks] = byte;
    }

    Ok(())
}

/// Macro to generate accumulator-based pack kernels.
///
/// This macro reduces code duplication for bit widths 3-7, 9-15, 17-23, 25-31.
macro_rules! define_pack_accumulator {
    ($name:ident, $bits:expr, $mask:expr) => {
        #[inline]
        fn $name(input: &[u32], output: &mut [u8]) -> Result<(), Error> {
            const BITS: usize = $bits;
            const MASK: u64 = $mask;

            let required = (input.len() * BITS + 7) / 8;
            let output = &mut output[..required];

            let mut acc: u64 = 0;
            let mut acc_bits: usize = 0;
            let mut out_idx: usize = 0;

            for &val in input {
                acc |= ((val as u64) & MASK) << acc_bits;
                acc_bits += BITS;

                while acc_bits >= 8 {
                    output[out_idx] = acc as u8;
                    out_idx += 1;
                    acc >>= 8;
                    acc_bits -= 8;
                }
            }

            if acc_bits > 0 {
                output[out_idx] = acc as u8;
            }

            Ok(())
        }
    };
}

define_pack_accumulator!(pack_3bit, 3, 0x7);
define_pack_accumulator!(pack_5bit, 5, 0x1F);
define_pack_accumulator!(pack_6bit, 6, 0x3F);
define_pack_accumulator!(pack_7bit, 7, 0x7F);

#[inline]
fn pack_4bit(input: &[u32], output: &mut [u8]) -> Result<(), Error> {
    let num_values = input.len();
    let full_pairs = num_values / 2;

    for i in 0..full_pairs {
        output[i] = ((input[i * 2] & 0x0F) as u8) | (((input[i * 2 + 1] & 0x0F) as u8) << 4);
    }

    if num_values % 2 == 1 {
        output[full_pairs] = (input[num_values - 1] & 0x0F) as u8;
    }

    Ok(())
}

#[inline]
fn pack_8bit(input: &[u32], output: &mut [u8]) -> Result<(), Error> {
    for (i, &value) in input.iter().enumerate() {
        // Truncation to u8 implicitly masks to the low 8 bits
        output[i] = value as u8;
    }
    Ok(())
}

#[inline]
fn pack_16bit(input: &[u32], output: &mut [u8]) -> Result<(), Error> {
    for (i, &value) in input.iter().enumerate() {
        // Only store the lower 16 bits as little-endian bytes
        let lo = value as u8;
        let hi = (value >> 8) as u8;
        output[i * 2] = lo;
        output[i * 2 + 1] = hi;
    }
    Ok(())
}

#[inline]
fn pack_24bit(input: &[u32], output: &mut [u8]) -> Result<(), Error> {
    for (i, &value) in input.iter().enumerate() {
        output[i * 3] = value as u8;
        output[i * 3 + 1] = (value >> 8) as u8;
        output[i * 3 + 2] = (value >> 16) as u8;
    }
    Ok(())
}

#[inline]
fn pack_32bit(input: &[u32], output: &mut [u8]) -> Result<(), Error> {
    // SAFETY: `input.as_ptr()` is valid for `input.len() * 4` bytes because it
    // originates from a `&[u32]`.  The output buffer was validated by `pack_n`
    // to have at least `packed_size_bytes(input.len(), 32) == input.len() * 4`
    // bytes.  Reinterpreting `&[u32]` as `&[u8]` with the same element count is
    // sound on little-endian targets (enforced at compile time via
    // `compile_error!` in `lib.rs`).
    let src: &[u8] =
        unsafe { core::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    output[..src.len()].copy_from_slice(src);
    Ok(())
}

define_pack_accumulator!(pack_9bit, 9, 0x1FF);
define_pack_accumulator!(pack_10bit, 10, 0x3FF);
define_pack_accumulator!(pack_11bit, 11, 0x7FF);
define_pack_accumulator!(pack_12bit, 12, 0xFFF);
define_pack_accumulator!(pack_13bit, 13, 0x1FFF);
define_pack_accumulator!(pack_14bit, 14, 0x3FFF);
define_pack_accumulator!(pack_15bit, 15, 0x7FFF);
define_pack_accumulator!(pack_17bit, 17, 0x1FFFF);
define_pack_accumulator!(pack_18bit, 18, 0x3FFFF);
define_pack_accumulator!(pack_19bit, 19, 0x7FFFF);
define_pack_accumulator!(pack_20bit, 20, 0xFFFFF);
define_pack_accumulator!(pack_21bit, 21, 0x1FFFFF);
define_pack_accumulator!(pack_22bit, 22, 0x3FFFFF);
define_pack_accumulator!(pack_23bit, 23, 0x7FFFFF);
define_pack_accumulator!(pack_25bit, 25, 0x1FFFFFF);
define_pack_accumulator!(pack_26bit, 26, 0x3FFFFFF);
define_pack_accumulator!(pack_27bit, 27, 0x7FFFFFF);
define_pack_accumulator!(pack_28bit, 28, 0xFFFFFFF);
define_pack_accumulator!(pack_29bit, 29, 0x1FFFFFFF);
define_pack_accumulator!(pack_30bit, 30, 0x3FFFFFFF);
define_pack_accumulator!(pack_31bit, 31, 0x7FFFFFFF);

/// Unpack `num_values` integers using `bit_width` bits each.
#[inline]
fn unpack_n(
    input: &[u8],
    bit_width: u8,
    num_values: usize,
    output: &mut [u32],
) -> Result<(), Error> {
    if bit_width > MAX_BIT_WIDTH {
        return Err(Error::InvalidBitWidth(bit_width));
    }

    if bit_width == 0 {
        output[..num_values].fill(0);
        return Ok(());
    }

    let required_bytes = packed_size_bytes(num_values, bit_width);
    if input.len() < required_bytes {
        return Err(Error::InputTooShort {
            need: required_bytes,
            got: input.len(),
        });
    }

    match bit_width {
        1 => unpack_1bit(input, num_values, output),
        2 => unpack_2bit(input, num_values, output),
        3 => unpack_3bit(input, num_values, output),
        4 => unpack_4bit(input, num_values, output),
        5 => unpack_5bit(input, num_values, output),
        6 => unpack_6bit(input, num_values, output),
        7 => unpack_7bit(input, num_values, output),
        8 => unpack_8bit(input, num_values, output),
        9 => unpack_9bit(input, num_values, output),
        10 => unpack_10bit(input, num_values, output),
        11 => unpack_11bit(input, num_values, output),
        12 => unpack_12bit(input, num_values, output),
        13 => unpack_13bit(input, num_values, output),
        14 => unpack_14bit(input, num_values, output),
        15 => unpack_15bit(input, num_values, output),
        16 => unpack_16bit(input, num_values, output),
        17 => unpack_17bit(input, num_values, output),
        18 => unpack_18bit(input, num_values, output),
        19 => unpack_19bit(input, num_values, output),
        20 => unpack_20bit(input, num_values, output),
        21 => unpack_21bit(input, num_values, output),
        22 => unpack_22bit(input, num_values, output),
        23 => unpack_23bit(input, num_values, output),
        24 => unpack_24bit(input, num_values, output),
        25 => unpack_25bit(input, num_values, output),
        26 => unpack_26bit(input, num_values, output),
        27 => unpack_27bit(input, num_values, output),
        28 => unpack_28bit(input, num_values, output),
        29 => unpack_29bit(input, num_values, output),
        30 => unpack_30bit(input, num_values, output),
        31 => unpack_31bit(input, num_values, output),
        32 => unpack_32bit(input, num_values, output),
        _ => unreachable!("bit_width validated above, must be 0-32"),
    }
}

/// Unpack 1-bit values.
#[inline]
fn unpack_1bit(input: &[u8], num_values: usize, output: &mut [u32]) -> Result<(), Error> {
    for i in 0..num_values {
        output[i] = ((input[i >> 3] >> (i & 7)) & 1) as u32;
    }
    Ok(())
}

/// Unpack 2-bit values.
#[inline]
fn unpack_2bit(input: &[u8], num_values: usize, output: &mut [u32]) -> Result<(), Error> {
    for i in 0..num_values {
        let byte_idx = i >> 2;
        let shift = (i & 3) << 1;
        output[i] = ((input[byte_idx] >> shift) & 3) as u32;
    }
    Ok(())
}

/// Macro to generate accumulator-based unpack kernels.
macro_rules! define_unpack_accumulator {
    ($name:ident, $bits:expr, $mask:expr) => {
        #[inline]
        fn $name(input: &[u8], num_values: usize, output: &mut [u32]) -> Result<(), Error> {
            const BITS: usize = $bits;
            const MASK: u64 = $mask;

            let required = (num_values * BITS + 7) / 8;
            let input = &input[..required];
            let output = &mut output[..num_values];

            let mut acc: u64 = 0;
            let mut acc_bits: usize = 0;
            let mut in_idx: usize = 0;

            for out in output {
                while acc_bits < BITS {
                    acc |= (input[in_idx] as u64) << acc_bits;
                    acc_bits += 8;
                    in_idx += 1;
                }

                *out = (acc & MASK) as u32;
                acc >>= BITS;
                acc_bits -= BITS;
            }

            Ok(())
        }
    };
}

#[inline]
fn unpack_4bit(input: &[u8], num_values: usize, output: &mut [u32]) -> Result<(), Error> {
    for i in 0..num_values {
        let byte = input[i / 2];
        output[i] = if i % 2 == 0 {
            (byte & 0x0F) as u32
        } else {
            (byte >> 4) as u32
        };
    }
    Ok(())
}

#[inline]
fn unpack_8bit(input: &[u8], num_values: usize, output: &mut [u32]) -> Result<(), Error> {
    for (i, out) in output.iter_mut().take(num_values).enumerate() {
        *out = input[i] as u32;
    }
    Ok(())
}

#[inline]
fn unpack_16bit(input: &[u8], num_values: usize, output: &mut [u32]) -> Result<(), Error> {
    for (i, out) in output.iter_mut().take(num_values).enumerate() {
        let lo = input[i * 2] as u32;
        let hi = input[i * 2 + 1] as u32;
        *out = lo | (hi << 8);
    }
    Ok(())
}

#[inline]
fn unpack_24bit(input: &[u8], num_values: usize, output: &mut [u32]) -> Result<(), Error> {
    for (i, out) in output.iter_mut().take(num_values).enumerate() {
        let lo = input[i * 3] as u32;
        let mid = input[i * 3 + 1] as u32;
        let hi = input[i * 3 + 2] as u32;
        *out = lo | (mid << 8) | (hi << 16);
    }
    Ok(())
}

#[inline]
fn unpack_32bit(input: &[u8], num_values: usize, output: &mut [u32]) -> Result<(), Error> {
    // SAFETY: `output.as_mut_ptr()` is valid for `num_values * 4` bytes because
    // it comes from a `&mut [u32]` of at least `num_values` elements (guaranteed
    // by `unpack_n`).  The input buffer was validated to have at least
    // `packed_size_bytes(num_values, 32) == num_values * 4` bytes.
    // Reinterpreting `&mut [u32]` as `&mut [u8]` with the same element count is
    // sound on little-endian targets (enforced at compile time via
    // `compile_error!` in `lib.rs`).
    let dst: &mut [u8] =
        unsafe { core::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut u8, num_values * 4) };
    dst.copy_from_slice(&input[..num_values * 4]);
    Ok(())
}

define_unpack_accumulator!(unpack_3bit, 3, 0x7);
define_unpack_accumulator!(unpack_5bit, 5, 0x1F);
define_unpack_accumulator!(unpack_6bit, 6, 0x3F);
define_unpack_accumulator!(unpack_7bit, 7, 0x7F);
define_unpack_accumulator!(unpack_9bit, 9, 0x1FF);
define_unpack_accumulator!(unpack_10bit, 10, 0x3FF);
define_unpack_accumulator!(unpack_11bit, 11, 0x7FF);
define_unpack_accumulator!(unpack_12bit, 12, 0xFFF);
define_unpack_accumulator!(unpack_13bit, 13, 0x1FFF);
define_unpack_accumulator!(unpack_14bit, 14, 0x3FFF);
define_unpack_accumulator!(unpack_15bit, 15, 0x7FFF);
define_unpack_accumulator!(unpack_17bit, 17, 0x1FFFF);
define_unpack_accumulator!(unpack_18bit, 18, 0x3FFFF);
define_unpack_accumulator!(unpack_19bit, 19, 0x7FFFF);
define_unpack_accumulator!(unpack_20bit, 20, 0xFFFFF);
define_unpack_accumulator!(unpack_21bit, 21, 0x1FFFFF);
define_unpack_accumulator!(unpack_22bit, 22, 0x3FFFFF);
define_unpack_accumulator!(unpack_23bit, 23, 0x7FFFFF);
define_unpack_accumulator!(unpack_25bit, 25, 0x1FFFFFF);
define_unpack_accumulator!(unpack_26bit, 26, 0x3FFFFFF);
define_unpack_accumulator!(unpack_27bit, 27, 0x7FFFFFF);
define_unpack_accumulator!(unpack_28bit, 28, 0xFFFFFFF);
define_unpack_accumulator!(unpack_29bit, 29, 0x1FFFFFFF);
define_unpack_accumulator!(unpack_30bit, 30, 0x3FFFFFFF);
define_unpack_accumulator!(unpack_31bit, 31, 0x7FFFFFFF);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_1bit() {
        let input: [u32; 128] = [0, 1]
            .iter()
            .cycle()
            .take(128)
            .copied()
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let mut packed = vec![0u8; 16];
        let mut output = [0u32; 128];

        ScalarBackend::pack_block(&input, 1, &mut packed).unwrap();
        ScalarBackend::unpack_block(&packed, 1, &mut output).unwrap();

        assert_eq!(input, output);
    }

    #[test]
    fn test_roundtrip_8bit() {
        let input: [u32; 128] = (0..128)
            .map(|i| (i % 256) as u32)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let mut packed = vec![0u8; 128];
        let mut output = [0u32; 128];

        ScalarBackend::pack_block(&input, 8, &mut packed).unwrap();
        ScalarBackend::unpack_block(&packed, 8, &mut output).unwrap();

        assert_eq!(input, output);
    }

    #[test]
    fn test_roundtrip_32bit() {
        let input: [u32; 128] = [u32::MAX, 0x80000000, 0xDEADBEEF, 0x12345678]
            .iter()
            .cycle()
            .take(128)
            .copied()
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let mut packed = vec![0u8; 512];
        let mut output = [0u32; 128];

        ScalarBackend::pack_block(&input, 32, &mut packed).unwrap();
        ScalarBackend::unpack_block(&packed, 32, &mut output).unwrap();

        assert_eq!(input, output);
    }

    #[test]
    fn test_bit_width_zero_roundtrip() {
        let input = [0u32; 128];
        let mut packed = vec![0u8; 16];
        let mut output = [0u32; 128];

        ScalarBackend::pack_block(&input, 0, &mut packed).unwrap();
        ScalarBackend::unpack_block(&packed, 0, &mut output).unwrap();

        assert_eq!(input, output);
        assert!(output.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_bit_width_zero_partial_block() {
        let input = vec![0u32; 50];
        let mut packed = vec![0u8; 16];
        let mut output = vec![0u32; 50];

        ScalarBackend::pack_partial_block(&input, 0, &mut packed).unwrap();
        ScalarBackend::unpack_partial_block(&packed, 0, 50, &mut output).unwrap();

        assert_eq!(input, output);
    }

    #[test]
    fn test_invalid_bit_width_too_large() {
        let input = [0u32; 128];
        let mut packed = vec![0u8; 544];

        assert!(matches!(
            ScalarBackend::pack_block(&input, 33, &mut packed),
            Err(Error::InvalidBitWidth(33))
        ));

        let mut output = [0u32; 128];
        assert!(matches!(
            ScalarBackend::unpack_block(&packed, 33, &mut output),
            Err(Error::InvalidBitWidth(33))
        ));
    }

    #[test]
    fn test_output_buffer_too_small() {
        let input: [u32; 128] = (0..128)
            .map(|i| i as u32)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let mut packed = vec![0u8; 10];

        assert!(matches!(
            ScalarBackend::pack_block(&input, 8, &mut packed),
            Err(Error::OutputTooSmall { need: 128, got: 10 })
        ));
    }

    #[test]
    fn test_input_buffer_too_small() {
        let packed = vec![0xFFu8; 10];
        let mut output = [0u32; 128];

        assert!(matches!(
            ScalarBackend::unpack_block(&packed, 8, &mut output),
            Err(Error::InputTooShort { need: 128, got: 10 })
        ));
    }

    #[test]
    fn test_truncated_input_detected() {
        let packed = vec![0xFFu8; 64];
        let mut output = [0u32; 128];

        let result = ScalarBackend::unpack_block(&packed, 8, &mut output);
        assert!(result.is_err());

        if let Err(Error::InputTooShort { need, got }) = result {
            assert_eq!(need, 128);
            assert_eq!(got, 64);
        } else {
            panic!("Expected InputTooShort error");
        }
    }

    #[test]
    fn test_partial_block_roundtrip() {
        let input: Vec<u32> = (0..50).map(|i| (i % 100) as u32).collect();
        let mut packed = vec![0u8; 50];
        let mut output = vec![0u32; 50];

        ScalarBackend::pack_partial_block(&input, 7, &mut packed).unwrap();
        ScalarBackend::unpack_partial_block(&packed, 7, 50, &mut output).unwrap();

        assert_eq!(input, output);
    }

    #[test]
    fn test_randomized_roundtrip() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for bit_width in [1, 2, 4, 7, 8, 12, 16, 24, 31, 32] {
            let max_val = if bit_width == 32 {
                u32::MAX
            } else {
                (1u32 << bit_width) - 1
            };

            let input: Vec<u32> = (0..128).map(|_| rng.gen_range(0..=max_val)).collect();
            let required_bytes = (128 * bit_width as usize + 7) / 8;
            let mut packed = vec![0u8; required_bytes];
            let mut output = [0u32; 128];

            let input_array: [u32; 128] = input.try_into().unwrap();
            ScalarBackend::pack_block(&input_array, bit_width as u8, &mut packed).unwrap();
            ScalarBackend::unpack_block(&packed, bit_width as u8, &mut output).unwrap();

            assert_eq!(input_array, output, "Failed for bit_width {}", bit_width);
        }
    }

    #[test]
    fn test_all_ones_pattern() {
        for bit_width in [1, 2, 4, 7, 8, 12, 16, 24, 31, 32] {
            let max_val = if bit_width == 32 {
                u32::MAX
            } else {
                (1u32 << bit_width) - 1
            };

            let input: [u32; 128] = [max_val; 128];
            let required_bytes = (128 * bit_width as usize + 7) / 8;
            let mut packed = vec![0u8; required_bytes];
            let mut output = [0u32; 128];

            ScalarBackend::pack_block(&input, bit_width as u8, &mut packed).unwrap();
            ScalarBackend::unpack_block(&packed, bit_width as u8, &mut output).unwrap();

            assert_eq!(
                input, output,
                "All-ones pattern failed for bit_width {}",
                bit_width
            );
        }
    }

    #[test]
    fn test_no_value_leakage_between_blocks() {
        let block1: [u32; 128] = [255; 128];
        let block2: [u32; 128] = [1; 128];

        let mut packed1 = vec![0u8; 128];
        let mut packed2 = vec![0u8; 16];
        let mut output1 = [0u32; 128];
        let mut output2 = [0u32; 128];

        ScalarBackend::pack_block(&block1, 8, &mut packed1).unwrap();
        ScalarBackend::pack_block(&block2, 1, &mut packed2).unwrap();

        ScalarBackend::unpack_block(&packed1, 8, &mut output1).unwrap();
        ScalarBackend::unpack_block(&packed2, 1, &mut output2).unwrap();

        assert_eq!(block1, output1, "Block 1 value leakage");
        assert_eq!(block2, output2, "Block 2 value leakage");
        assert!(
            output2.iter().all(|&v| v == 1),
            "Block 2 should only contain 1s"
        );
    }

    #[test]
    fn test_values_exceeding_bit_width_are_masked() {
        let input: [u32; 128] = [0xFF; 128];
        let mut packed = vec![0u8; 64];
        let mut output = [0u32; 128];

        ScalarBackend::pack_block(&input, 4, &mut packed).unwrap();
        ScalarBackend::unpack_block(&packed, 4, &mut output).unwrap();

        assert!(
            output.iter().all(|&v| v == 0x0F),
            "Values exceeding bit_width should be masked, expected 0x0F, got 0x{:02X}",
            output[0]
        );
    }

    #[test]
    fn test_bit_width_32_max_values() {
        let input: [u32; 128] = [u32::MAX; 128];
        let mut packed = vec![0u8; 512];
        let mut output = [0u32; 128];

        ScalarBackend::pack_block(&input, 32, &mut packed).unwrap();
        ScalarBackend::unpack_block(&packed, 32, &mut output).unwrap();

        assert_eq!(input, output);
        assert!(output.iter().all(|&v| v == u32::MAX));
    }

    #[test]
    fn test_bit_width_1_masking() {
        let input: [u32; 128] = [0xFFFFFFFF; 128];
        let mut packed = vec![0u8; 16];
        let mut output = [0u32; 128];

        ScalarBackend::pack_block(&input, 1, &mut packed).unwrap();
        ScalarBackend::unpack_block(&packed, 1, &mut output).unwrap();

        assert!(
            output.iter().all(|&v| v == 1),
            "bit_width=1 should mask to 0x1, got {:?}",
            &output[..5]
        );
    }

    #[test]
    fn test_bit_width_31_masking() {
        let input: [u32; 128] = [u32::MAX; 128];
        let mut packed = vec![0u8; 496];
        let mut output = [0u32; 128];

        ScalarBackend::pack_block(&input, 31, &mut packed).unwrap();
        ScalarBackend::unpack_block(&packed, 31, &mut output).unwrap();

        let expected_mask = (1u32 << 31) - 1;
        assert!(
            output.iter().all(|&v| v == expected_mask),
            "bit_width=31 should mask to 0x7FFFFFFF, got 0x{:08X}",
            output[0]
        );
    }

    #[test]
    fn test_exact_output_byte_count() {
        let test_cases = [
            (0, 0),    // 128 * 0 / 8 = 0
            (1, 16),   // ceil(128 * 1 / 8) = 16
            (2, 32),   // ceil(128 * 2 / 8) = 32
            (3, 48),   // ceil(128 * 3 / 8) = 48
            (4, 64),   // ceil(128 * 4 / 8) = 64
            (5, 80),   // ceil(128 * 5 / 8) = 80
            (6, 96),   // ceil(128 * 6 / 8) = 96
            (7, 112),  // ceil(128 * 7 / 8) = 112
            (8, 128),  // ceil(128 * 8 / 8) = 128
            (9, 144),  // ceil(128 * 9 / 8) = 144
            (15, 240), // ceil(128 * 15 / 8) = 240
            (16, 256), // ceil(128 * 16 / 8) = 256
            (17, 272), // ceil(128 * 17 / 8) = 272
            (31, 496), // ceil(128 * 31 / 8) = 496
            (32, 512), // ceil(128 * 32 / 8) = 512
        ];

        let input: [u32; 128] = [0xAAAAAAAA; 128];

        for (bit_width, expected_bytes) in test_cases {
            let packed_size = (128 * bit_width + 7) / 8;
            assert_eq!(
                packed_size, expected_bytes,
                "Byte count mismatch for bit_width={}",
                bit_width
            );

            let mut packed = vec![0u8; packed_size];
            let result = ScalarBackend::pack_block(&input, bit_width as u8, &mut packed);
            assert!(result.is_ok(), "Packing failed for bit_width={}", bit_width);
        }
    }

    #[test]
    fn test_multi_block_consistency() {
        let input1: [u32; 128] = [0x12345678; 128];
        let input2: [u32; 128] = [0x9ABCDEF0; 128];

        let mut packed1_a = vec![0u8; 512];
        let mut packed1_b = vec![0u8; 512];
        let mut packed2 = vec![0u8; 512];

        ScalarBackend::pack_block(&input1, 32, &mut packed1_a).unwrap();
        ScalarBackend::pack_block(&input1, 32, &mut packed1_b).unwrap();
        ScalarBackend::pack_block(&input2, 32, &mut packed2).unwrap();

        assert_eq!(
            packed1_a, packed1_b,
            "Same input must produce deterministic output"
        );
        assert_ne!(
            packed1_a, packed2,
            "Different inputs should produce different outputs"
        );
    }

    #[test]
    fn test_full_roundtrip_compress_decompress() {
        use crate::{compress, decompress};

        let test_cases = [
            vec![0u32; 256],
            vec![u32::MAX; 128],
            (0..256).map(|i| i as u32).collect::<Vec<_>>(),
            (0..256)
                .map(|i| ((i * 9301 + 49297) % 1000) as u32)
                .collect::<Vec<_>>(),
            {
                let mut v = vec![0u32; 256];
                for (i, vi) in v.iter_mut().enumerate().take(256) {
                    *vi = if i % 2 == 0 { i as u32 } else { u32::MAX };
                }
                v
            },
        ];

        for (idx, input) in test_cases.iter().enumerate() {
            let compressed =
                compress(input).unwrap_or_else(|_| panic!("Compression failed for case {}", idx));
            let decompressed = decompress(&compressed)
                .unwrap_or_else(|_| panic!("Decompression failed for case {}", idx));

            assert_eq!(
                input, &decompressed,
                "Full roundtrip failed for test case {}",
                idx
            );
        }
    }

    #[test]
    fn test_decode_symmetry_all_bit_widths() {
        for bit_width in 0..=32u8 {
            let mask = if bit_width >= 32 {
                u32::MAX
            } else {
                (1u32 << bit_width) - 1
            };

            let patterns: Vec<[u32; 128]> = vec![
                [0u32; 128],
                [mask; 128],
                std::array::from_fn(|i| if i % 2 == 0 { mask } else { 0 }),
                std::array::from_fn(|i| ((i as u32) * 7) & mask),
            ];

            for (pattern_idx, input) in patterns.iter().enumerate() {
                let packed_size = (128 * bit_width as usize + 7) / 8;
                let mut packed = vec![0u8; packed_size.max(1)];
                let mut output = [0u32; 128];

                ScalarBackend::pack_block(input, bit_width, &mut packed).unwrap();
                ScalarBackend::unpack_block(&packed, bit_width, &mut output).unwrap();

                for i in 0..128 {
                    assert_eq!(
                        output[i], input[i],
                        "Decode symmetry failed for bit_width={}, pattern={}, index={}",
                        bit_width, pattern_idx, i
                    );
                }
            }
        }
    }

    #[test]
    fn test_all_zeros_roundtrip() {
        use crate::{compress, decompress};

        for size in [1, 10, 100, 128, 129, 200, 256, 1000] {
            let input = vec![0u32; size];
            let compressed = compress(&input).unwrap();
            let decompressed = decompress(&compressed).unwrap();

            assert_eq!(
                input, decompressed,
                "All zeros roundtrip failed for size={}",
                size
            );
            assert!(decompressed.iter().all(|&v| v == 0));
        }
    }
}
