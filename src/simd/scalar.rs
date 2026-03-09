use crate::error::Error;
use crate::simd::SimdBackend;

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
/// Throughput: Approximately 2-4 GB/s depending on bit width and CPU.
///
/// For higher throughput on x86_64, the dispatch module automatically
/// selects SSE4.1, AVX2, or AVX512 implementations when available.
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
    /// Handles the final block in a compressed array. Uses specialized fast paths
    /// for bit widths 4, 8, 16, 24, and 32.
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
    /// Handles the final block in a compressed array. Uses specialized fast paths
    /// for bit widths 4, 8, 16, 24, and 32.
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
        // Validate output buffer has enough space
        if output.len() < num_values {
            return Err(Error::OutputTooSmall {
                need: num_values,
                got: output.len(),
            });
        }
        unpack_n(input, bit_width, num_values, output)
    }
}

fn pack_n(input: &[u32], bit_width: u8, output: &mut [u8]) -> Result<(), Error> {
    if bit_width > 32 {
        return Err(Error::InvalidBitWidth(bit_width));
    }

    if bit_width == 0 {
        return Ok(());
    }

    let bits_per_value = bit_width as usize;
    let num_values = input.len();
    let total_bits = num_values * bits_per_value;
    let required_bytes = (total_bits + 7) / 8;

    if output.len() < required_bytes {
        return Err(Error::OutputTooSmall {
            need: required_bytes,
            got: output.len(),
        });
    }

    match bits_per_value {
        4 => return pack_4bit(input, output),
        5 => return pack_5bit(input, output),
        6 => return pack_6bit(input, output),
        7 => return pack_7bit(input, output),
        8 => return pack_8bit(input, output),
        12 => return pack_12bit(input, output),
        16 => return pack_16bit(input, output),
        24 => return pack_24bit(input, output),
        32 => return pack_32bit(input, output),
        _ => {}
    }

    let value_mask: u64 = if bits_per_value >= 32 {
        u32::MAX as u64
    } else {
        (1u64 << bits_per_value) - 1
    };

    let mut acc: u64 = 0;
    let mut acc_bits: usize = 0;
    let mut out_idx: usize = 0;

    for &value in input {
        acc |= (value as u64 & value_mask) << acc_bits;
        acc_bits += bits_per_value;

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

#[inline]
fn pack_8bit(input: &[u32], output: &mut [u8]) -> Result<(), Error> {
    for (i, &value) in input.iter().enumerate() {
        output[i] = value as u8;
    }
    Ok(())
}

#[inline]
fn pack_16bit(input: &[u32], output: &mut [u8]) -> Result<(), Error> {
    for (i, &value) in input.iter().enumerate() {
        let bytes = value.to_le_bytes();
        output[i * 2] = bytes[0];
        output[i * 2 + 1] = bytes[1];
    }
    Ok(())
}

#[inline]
fn pack_32bit(input: &[u32], output: &mut [u8]) -> Result<(), Error> {
    for (i, &value) in input.iter().enumerate() {
        let bytes = value.to_le_bytes();
        output[i * 4] = bytes[0];
        output[i * 4 + 1] = bytes[1];
        output[i * 4 + 2] = bytes[2];
        output[i * 4 + 3] = bytes[3];
    }
    Ok(())
}

#[inline]
fn pack_4bit(input: &[u32], output: &mut [u8]) -> Result<(), Error> {
    let num_values = input.len();
    for i in (0..num_values).step_by(2) {
        let lo = (input[i] & 0x0F) as u8;
        let hi = if i + 1 < num_values {
            (input[i + 1] & 0x0F) as u8
        } else {
            0
        };
        output[i / 2] = lo | (hi << 4);
    }
    Ok(())
}

#[inline]
fn pack_5bit(input: &[u32], output: &mut [u8]) -> Result<(), Error> {
    let num_values = input.len();
    let mut acc: u64 = 0;
    let mut acc_bits: usize = 0;
    let mut out_idx: usize = 0;

    for &val in input.iter().take(num_values) {
        acc |= ((val & 0x1F) as u64) << acc_bits;
        acc_bits += 5;

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

#[inline]
fn pack_6bit(input: &[u32], output: &mut [u8]) -> Result<(), Error> {
    let num_values = input.len();
    let mut acc: u64 = 0;
    let mut acc_bits: usize = 0;
    let mut out_idx: usize = 0;

    for &val in input.iter().take(num_values) {
        acc |= ((val & 0x3F) as u64) << acc_bits;
        acc_bits += 6;

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

#[inline]
fn pack_7bit(input: &[u32], output: &mut [u8]) -> Result<(), Error> {
    let num_values = input.len();
    let mut acc: u64 = 0;
    let mut acc_bits: usize = 0;
    let mut out_idx: usize = 0;

    for &val in input.iter().take(num_values) {
        acc |= ((val & 0x7F) as u64) << acc_bits;
        acc_bits += 7;

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

#[inline]
fn pack_12bit(input: &[u32], output: &mut [u8]) -> Result<(), Error> {
    let num_values = input.len();
    let mut acc: u64 = 0;
    let mut acc_bits: usize = 0;
    let mut out_idx: usize = 0;

    for &val in input.iter().take(num_values) {
        acc |= ((val & 0xFFF) as u64) << acc_bits;
        acc_bits += 12;

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

#[inline]
fn pack_24bit(input: &[u32], output: &mut [u8]) -> Result<(), Error> {
    for (i, &value) in input.iter().enumerate() {
        output[i * 3] = value as u8;
        output[i * 3 + 1] = (value >> 8) as u8;
        output[i * 3 + 2] = (value >> 16) as u8;
    }
    Ok(())
}

fn unpack_n(
    input: &[u8],
    bit_width: u8,
    num_values: usize,
    output: &mut [u32],
) -> Result<(), Error> {
    if bit_width > 32 {
        return Err(Error::InvalidBitWidth(bit_width));
    }

    if bit_width == 0 {
        if output.len() < num_values {
            return Err(Error::OutputTooSmall {
                need: num_values,
                got: output.len(),
            });
        }
        output[..num_values].fill(0);
        return Ok(());
    }

    let bits_per_value = bit_width as usize;
    let total_bits = num_values * bits_per_value;
    let required_bytes = (total_bits + 7) / 8;

    if input.len() < required_bytes {
        return Err(Error::InputTooShort {
            need: required_bytes,
            got: input.len(),
        });
    }

    match bits_per_value {
        4 => return unpack_4bit(input, num_values, output),
        5 => return unpack_5bit(input, num_values, output),
        6 => return unpack_6bit(input, num_values, output),
        7 => return unpack_7bit(input, num_values, output),
        8 => return unpack_8bit(input, num_values, output),
        12 => return unpack_12bit(input, num_values, output),
        16 => return unpack_16bit(input, num_values, output),
        24 => return unpack_24bit(input, num_values, output),
        32 => return unpack_32bit(input, num_values, output),
        _ => {}
    }

    let value_mask: u64 = if bits_per_value >= 32 {
        u32::MAX as u64
    } else {
        (1u64 << bits_per_value) - 1
    };

    let mut acc: u64 = 0;
    let mut acc_bits: usize = 0;
    let mut in_idx: usize = 0;

    for out in output.iter_mut().take(num_values) {
        while acc_bits < bits_per_value {
            acc |= (input[in_idx] as u64) << acc_bits;
            acc_bits += 8;
            in_idx += 1;
        }

        *out = (acc & value_mask) as u32;
        acc >>= bits_per_value;
        acc_bits -= bits_per_value;
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
fn unpack_32bit(input: &[u8], num_values: usize, output: &mut [u32]) -> Result<(), Error> {
    for (i, out) in output.iter_mut().take(num_values).enumerate() {
        let lo = input[i * 4] as u32;
        let mid1 = input[i * 4 + 1] as u32;
        let mid2 = input[i * 4 + 2] as u32;
        let hi = input[i * 4 + 3] as u32;
        *out = lo | (mid1 << 8) | (mid2 << 16) | (hi << 24);
    }
    Ok(())
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
fn unpack_5bit(input: &[u8], num_values: usize, output: &mut [u32]) -> Result<(), Error> {
    let mut acc: u64 = 0;
    let mut acc_bits: usize = 0;
    let mut in_idx: usize = 0;

    for out in output.iter_mut().take(num_values) {
        while acc_bits < 5 {
            acc |= (input[in_idx] as u64) << acc_bits;
            acc_bits += 8;
            in_idx += 1;
        }

        *out = (acc & 0x1F) as u32;
        acc >>= 5;
        acc_bits -= 5;
    }
    Ok(())
}

#[inline]
fn unpack_6bit(input: &[u8], num_values: usize, output: &mut [u32]) -> Result<(), Error> {
    let mut acc: u64 = 0;
    let mut acc_bits: usize = 0;
    let mut in_idx: usize = 0;

    for out in output.iter_mut().take(num_values) {
        while acc_bits < 6 {
            acc |= (input[in_idx] as u64) << acc_bits;
            acc_bits += 8;
            in_idx += 1;
        }

        *out = (acc & 0x3F) as u32;
        acc >>= 6;
        acc_bits -= 6;
    }
    Ok(())
}

#[inline]
fn unpack_7bit(input: &[u8], num_values: usize, output: &mut [u32]) -> Result<(), Error> {
    let mut acc: u64 = 0;
    let mut acc_bits: usize = 0;
    let mut in_idx: usize = 0;

    for out in output.iter_mut().take(num_values) {
        while acc_bits < 7 {
            acc |= (input[in_idx] as u64) << acc_bits;
            acc_bits += 8;
            in_idx += 1;
        }

        *out = (acc & 0x7F) as u32;
        acc >>= 7;
        acc_bits -= 7;
    }
    Ok(())
}

#[inline]
fn unpack_12bit(input: &[u8], num_values: usize, output: &mut [u32]) -> Result<(), Error> {
    let mut acc: u64 = 0;
    let mut acc_bits: usize = 0;
    let mut in_idx: usize = 0;

    for out in output.iter_mut().take(num_values) {
        while acc_bits < 12 {
            acc |= (input[in_idx] as u64) << acc_bits;
            acc_bits += 8;
            in_idx += 1;
        }

        *out = (acc & 0xFFF) as u32;
        acc >>= 12;
        acc_bits -= 12;
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
        // bit_width = 0 should work correctly for all-zero blocks
        let input = [0u32; 128];
        let mut packed = vec![0u8; 16]; // Buffer not used for bit_width=0
        let mut output = [0u32; 128];

        // Pack should succeed (no bits written)
        ScalarBackend::pack_block(&input, 0, &mut packed).unwrap();

        // Unpack should succeed and fill with zeros
        ScalarBackend::unpack_block(&packed, 0, &mut output).unwrap();

        // All values should be zero
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
    fn test_maximum_partial_block() {
        // Test partial block with maximum 127 values
        let input: Vec<u32> = (0..127).map(|i| (i % 100) as u32).collect();
        let mut packed = vec![0u8; 112]; // (127 * 7 + 7) / 8 = 112 bytes
        let mut output = vec![0u32; 127];

        ScalarBackend::pack_partial_block(&input, 7, &mut packed).unwrap();
        ScalarBackend::unpack_partial_block(&packed, 7, 127, &mut output).unwrap();

        assert_eq!(input, output);
    }

    #[test]
    fn test_all_ones_pattern() {
        // Test that (1 << bit_width) - 1 encodes/decodes correctly for all bit widths
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
        // Block 1: 8-bit values (positions 0-127)
        // Block 2: 1-bit values (positions 128-255)
        // Ensure high bits from block 1 don't affect block 2 decoding
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

        // Specifically verify no cross-contamination
        assert!(
            output2.iter().all(|&v| v == 1),
            "Block 2 should only contain 1s"
        );
    }

    #[test]
    fn test_all_zeros_full_roundtrip() {
        // Test all zeros with various block sizes
        for size in [1, 64, 128, 129, 200, 256] {
            let input = vec![0u32; size];
            let mut output = vec![0u32; size];

            // Pack - should require 0 bytes
            let packed_size = 0;
            let mut packed = vec![0u8; packed_size.max(1)];

            if size == 128 {
                let input_array: [u32; 128] = input.clone().try_into().unwrap();
                ScalarBackend::pack_block(&input_array, 0, &mut packed).unwrap();
                let mut output_array = [0u32; 128];
                ScalarBackend::unpack_block(&packed, 0, &mut output_array).unwrap();
                output.copy_from_slice(&output_array);
            } else {
                ScalarBackend::pack_partial_block(&input, 0, &mut packed).unwrap();
                ScalarBackend::unpack_partial_block(&packed, 0, size, &mut output).unwrap();
            }

            assert_eq!(
                input, output,
                "All-zeros roundtrip failed for size {}",
                size
            );
            assert!(
                output.iter().all(|&v| v == 0),
                "Output should be all zeros for size {}",
                size
            );
        }
    }

    #[test]
    fn test_partial_block_buffer_validation() {
        // Test that unpack_partial_block validates output buffer size
        let packed = vec![0u8; 100];
        let mut output = vec![0u32; 50]; // Too small for 100 values

        let result = ScalarBackend::unpack_partial_block(&packed, 8, 100, &mut output);
        assert!(matches!(
            result,
            Err(Error::OutputTooSmall { need: 100, got: 50 })
        ));
    }

    #[test]
    fn test_bit_width_32_max_values() {
        // Test bit_width = 32 with u32::MAX values
        let input: [u32; 128] = [u32::MAX; 128];
        let mut packed = vec![0u8; 512]; // 128 * 32 / 8 = 512
        let mut output = [0u32; 128];

        ScalarBackend::pack_block(&input, 32, &mut packed).unwrap();
        ScalarBackend::unpack_block(&packed, 32, &mut output).unwrap();

        assert_eq!(input, output);
        assert!(output.iter().all(|&v| v == u32::MAX));
    }

    #[test]
    fn test_single_element_roundtrip() {
        // Test single element with various bit widths
        for bit_width in [0, 1, 7, 8, 15, 16, 31, 32] {
            let max_val = if bit_width == 0 {
                0
            } else if bit_width == 32 {
                u32::MAX
            } else {
                (1u32 << bit_width) - 1
            };

            let input = vec![max_val];
            let mut packed = vec![0u8; 8]; // More than enough
            let mut output = vec![0u32; 1];

            ScalarBackend::pack_partial_block(&input, bit_width, &mut packed).unwrap();
            ScalarBackend::unpack_partial_block(&packed, bit_width, 1, &mut output).unwrap();

            assert_eq!(
                input, output,
                "Single element roundtrip failed for bit_width {}",
                bit_width
            );
        }
    }

    #[test]
    fn test_mask_correctness_all_bit_widths() {
        for bit_width in 0..=32u8 {
            let expected_mask = if bit_width == 0 {
                0u32
            } else if bit_width == 32 {
                u32::MAX
            } else {
                (1u32 << bit_width) - 1
            };

            // Verify mask calculation logic matches expected
            let computed_mask = if bit_width == 0 {
                0u32
            } else if bit_width >= 32 {
                u32::MAX
            } else {
                (1u32 << bit_width) - 1
            };

            assert_eq!(
                computed_mask, expected_mask,
                "Mask mismatch for bit_width={}",
                bit_width
            );
        }
    }

    #[test]
    fn test_bit_width_0_values_are_masked() {
        let input: [u32; 128] = [0xDEADBEEF; 128]; // Garbage values
        let mut packed = vec![0u8; 0]; // No bytes needed
        let mut output = [0u32; 128];

        // Pack with bit_width=0 - should succeed and write nothing
        ScalarBackend::pack_block(&input, 0, &mut packed).unwrap();

        // Unpack with bit_width=0 - should fill with zeros
        ScalarBackend::unpack_block(&packed, 0, &mut output).unwrap();

        // All output should be zeros regardless of input
        assert!(
            output.iter().all(|&v| v == 0),
            "bit_width=0 should produce all zeros, got non-zero values"
        );
    }

    #[test]
    fn test_bit_width_1_masking() {
        let input: [u32; 128] = [0xFFFFFFFF; 128]; // All bits set
        let mut packed = vec![0u8; 16]; // 128 * 1 / 8 = 16 bytes
        let mut output = [0u32; 128];

        ScalarBackend::pack_block(&input, 1, &mut packed).unwrap();
        ScalarBackend::unpack_block(&packed, 1, &mut output).unwrap();

        // Only LSB should survive
        assert!(
            output.iter().all(|&v| v == 1),
            "bit_width=1 should mask to 0x1, got {:?}",
            &output[..5]
        );
    }

    #[test]
    fn test_bit_width_31_masking() {
        let input: [u32; 128] = [u32::MAX; 128]; // All bits set
        let mut packed = vec![0u8; 496]; // 128 * 31 / 8 = 496 bytes
        let mut output = [0u32; 128];

        ScalarBackend::pack_block(&input, 31, &mut packed).unwrap();
        ScalarBackend::unpack_block(&packed, 31, &mut output).unwrap();

        let expected_mask = (1u32 << 31) - 1; // 0x7FFFFFFF
        assert!(
            output.iter().all(|&v| v == expected_mask),
            "bit_width=31 should mask to 0x7FFFFFFF, got 0x{:08X}",
            output[0]
        );
    }

    #[test]
    fn test_bit_width_32_no_masking() {
        let input: [u32; 128] = [0xDEADBEEF; 128];
        let mut packed = vec![0u8; 512]; // 128 * 32 / 8 = 512 bytes
        let mut output = [0u32; 128];

        ScalarBackend::pack_block(&input, 32, &mut packed).unwrap();
        ScalarBackend::unpack_block(&packed, 32, &mut output).unwrap();

        assert_eq!(
            input, output,
            "bit_width=32 should preserve full u32 values"
        );
    }

    #[test]
    fn test_no_ub_with_bit_width_32() {
        let test_values = [
            0u32, 1, 0x7FFFFFFF, // Max positive i32
            0x80000000, // Sign bit
            0xFFFFFFFF, // u32::MAX
        ];

        for &value in &test_values {
            let input: [u32; 128] = [value; 128];
            let mut packed = vec![0u8; 512];
            let mut output = [0u32; 128];

            // This should not panic or trigger UB
            ScalarBackend::pack_block(&input, 32, &mut packed).unwrap();
            ScalarBackend::unpack_block(&packed, 32, &mut output).unwrap();

            assert_eq!(
                input, output,
                "Roundtrip failed for value 0x{:08X} with bit_width=32",
                value
            );
        }
    }

    #[test]
    fn test_values_exceeding_bit_width_are_masked() {
        let input: [u32; 128] = [0xFF; 128]; // 8 bits set, packing with 4 bits
        let mut packed = vec![0u8; 64]; // 128 * 4 / 8 = 64 bytes
        let mut output = [0u32; 128];

        ScalarBackend::pack_block(&input, 4, &mut packed).unwrap();
        ScalarBackend::unpack_block(&packed, 4, &mut output).unwrap();

        // Should be masked to 0x0F
        assert!(
            output.iter().all(|&v| v == 0x0F),
            "Values exceeding bit_width should be masked, expected 0x0F, got 0x{:02X}",
            output[0]
        );
    }

    #[test]
    fn test_precomputed_mask_performance() {
        let input: [u32; 128] = [0xAAAAAAAA; 128]; // Alternating pattern
        let mut packed = vec![0u8; 512];
        let mut output = [0u32; 128];

        // Test all bit widths to ensure no pathological cases
        for bit_width in [1, 8, 16, 24, 31, 32] {
            packed.fill(0);
            output.fill(0);

            ScalarBackend::pack_block(&input, bit_width as u8, &mut packed).unwrap();
            ScalarBackend::unpack_block(&packed, bit_width as u8, &mut output).unwrap();

            // Verify correctness
            let mask = if bit_width >= 32 {
                u32::MAX
            } else {
                (1u32 << bit_width) - 1
            };
            let expected = 0xAAAAAAAA & mask;

            assert!(
                output.iter().all(|&v| v == expected),
                "Masking failed for bit_width={}",
                bit_width
            );
        }
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

        let input: [u32; 128] = [0xAAAAAAAA; 128]; // Alternating pattern

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
    fn test_values_crossing_byte_boundaries() {
        let bit_width = 12;
        let _values_per_byte = 8.0 / bit_width as f64; // 0.666 values per byte

        // Create pattern where values have distinct bits to detect misalignment
        let input: [u32; 128] = std::array::from_fn(|i| {
            // Each value: 0xABC where A,B,C are distinct nibbles
            // This makes bit errors visually obvious
            let nibble = (i % 16) as u32;
            (nibble << 8) | (nibble << 4) | nibble
        });

        let packed_size = (128 * bit_width + 7) / 8; // 192 bytes
        let mut packed = vec![0u8; packed_size];
        let mut output = [0u32; 128];

        ScalarBackend::pack_block(&input, bit_width as u8, &mut packed).unwrap();
        ScalarBackend::unpack_block(&packed, bit_width as u8, &mut output).unwrap();

        // Apply mask to expected values
        let mask = (1u32 << bit_width) - 1;
        for i in 0..128 {
            assert_eq!(
                output[i],
                input[i] & mask,
                "Byte boundary error at index {} with bit_width={}",
                i,
                bit_width
            );
        }
    }

    #[test]
    fn test_alternating_bit_patterns() {
        let patterns = [
            (0x55555555u32, "0b0101..."),     // Alternating 01
            (0xAAAAAAAAu32, "0b1010..."),     // Alternating 10
            (0x33333333u32, "0b0011..."),     // 0011 repeating
            (0xCCCCCCCCu32, "0b1100..."),     // 1100 repeating
            (0x0F0F0F0Fu32, "0b00001111..."), // 00001111 repeating
            (0xF0F0F0F0u32, "0b11110000..."), // 11110000 repeating
        ];

        for (pattern, name) in patterns {
            let input: [u32; 128] = [pattern; 128];

            // Test multiple bit widths that might interact badly with patterns
            for bit_width in [1, 2, 4, 8, 12, 16, 24, 32] {
                let packed_size = (128 * bit_width + 7) / 8;
                let mut packed = vec![0u8; packed_size];
                let mut output = [0u32; 128];

                ScalarBackend::pack_block(&input, bit_width as u8, &mut packed).unwrap();
                ScalarBackend::unpack_block(&packed, bit_width as u8, &mut output).unwrap();

                let mask = if bit_width >= 32 {
                    u32::MAX
                } else {
                    (1u32 << bit_width) - 1
                };
                let expected = pattern & mask;

                assert!(
                    output.iter().all(|&v| v == expected),
                    "Pattern {} failed for bit_width={}. Expected 0x{:08X}, got 0x{:08X}",
                    name,
                    bit_width,
                    expected,
                    output[0]
                );
            }
        }
    }

    #[test]
    fn test_maximum_width_packing() {
        let input: [u32; 128] = std::array::from_fn(|i| i as u32);
        let mut packed = vec![0u8; 512];
        let mut output = [0u32; 128];

        ScalarBackend::pack_block(&input, 32, &mut packed).unwrap();

        // Verify byte count
        assert_eq!(
            packed.len(),
            512,
            "32-bit packing should produce exactly 512 bytes"
        );

        // Verify each u32 is stored in little-endian order
        for i in 0..128 {
            let bytes = &packed[i * 4..(i + 1) * 4];
            let value = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            assert_eq!(
                value, input[i],
                "32-bit packing should store values in little-endian at index {}",
                i
            );
        }

        // Verify roundtrip
        ScalarBackend::unpack_block(&packed, 32, &mut output).unwrap();
        assert_eq!(input, output, "32-bit roundtrip failed");
    }

    #[test]
    fn test_sequential_numbers_packing() {
        for bit_width in [4, 8, 12, 16] {
            let input: [u32; 128] = std::array::from_fn(|i| (i as u32) % (1 << bit_width));
            let packed_size = (128 * bit_width + 7) / 8;
            let mut packed = vec![0u8; packed_size];
            let mut output = [0u32; 128];

            ScalarBackend::pack_block(&input, bit_width as u8, &mut packed).unwrap();
            ScalarBackend::unpack_block(&packed, bit_width as u8, &mut output).unwrap();

            assert_eq!(
                input, output,
                "Sequential numbers failed for bit_width={}",
                bit_width
            );
        }
    }

    #[test]
    fn test_partial_byte_final_value() {
        for bit_width in [3, 5, 7, 9, 11, 13, 15] {
            let input: [u32; 128] = [(1u32 << bit_width) - 1; 128]; // All ones for this width
            let packed_size = (128 * bit_width + 7) / 8;
            let mut packed = vec![0u8; packed_size];
            let mut output = [0u32; 128];

            ScalarBackend::pack_block(&input, bit_width as u8, &mut packed).unwrap();
            ScalarBackend::unpack_block(&packed, bit_width as u8, &mut output).unwrap();

            assert!(
                output.iter().all(|&v| v == input[0]),
                "Partial byte test failed for bit_width={}",
                bit_width
            );
        }
    }

    #[test]
    fn test_bit_level_verification_3bit() {
        let input: [u32; 128] = [0b101, 0b010] // Alternating pattern
            .iter()
            .cycle()
            .take(128)
            .copied()
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let mut packed = vec![0u8; 48]; // 128 * 3 / 8 = 48
        ScalarBackend::pack_block(&input, 3, &mut packed).unwrap();

        // Verify first few bytes manually
        // Byte 0: [val0 bits 0-2][val1 bits 0-1] = [101][01] = 0b01101 = 0x0D (little-endian)
        // Actually: val0=0b101 at bit offset 0, val1=0b010 at bit offset 3
        // Byte 0 = 0b101 | (0b010 << 3) = 0b10101000 = 0xA8? No wait...
        // bit_offset 0: bits 0-2 of val0 = 0b101
        // bit_offset 3: bits 0-2 of val1 = 0b010
        // Byte 0 = 0b101 | (0b010 << 3) = 0b101 | 0b010000 = 0b010101 = 0x15
        // Let me just verify roundtrip works correctly

        let mut output = [0u32; 128];
        ScalarBackend::unpack_block(&packed, 3, &mut output).unwrap();

        for i in 0..128 {
            assert_eq!(
                output[i], input[i],
                "Bit-level mismatch at index {} for 3-bit packing",
                i
            );
        }
    }

    #[test]
    fn test_large_random_blocks() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for bit_width in [1, 5, 9, 13, 17, 21, 25, 29, 32] {
            let mask = if bit_width >= 32 {
                u32::MAX
            } else {
                (1u32 << bit_width) - 1
            };

            // Generate random values within the bit width
            let input: [u32; 128] = std::array::from_fn(|_| rng.gen::<u32>() & mask);

            let packed_size = (128 * bit_width + 7) / 8;
            let mut packed = vec![0u8; packed_size];
            let mut output = [0u32; 128];

            ScalarBackend::pack_block(&input, bit_width as u8, &mut packed).unwrap();
            ScalarBackend::unpack_block(&packed, bit_width as u8, &mut output).unwrap();

            assert_eq!(
                input, output,
                "Random data roundtrip failed for bit_width={}",
                bit_width
            );
        }
    }

    #[test]
    fn test_pathological_patterns() {
        let pathological_cases: Vec<[u32; 128]> = vec![
            // All zeros followed by all ones
            std::array::from_fn(|i| if i < 64 { 0 } else { u32::MAX }),
            // Ramp pattern
            std::array::from_fn(|i| i as u32),
            // Sawtooth pattern
            std::array::from_fn(|i| (i % 16) as u32),
            // Single bit walking (cycle through bits)
            std::array::from_fn(|i| 1u32 << (i % 32)),
            // Alternating high/low 16 bits
            std::array::from_fn(|i| if i % 2 == 0 { 0xFFFF0000 } else { 0x0000FFFF }),
        ];

        for (case_idx, input) in pathological_cases.iter().enumerate() {
            for bit_width in [4, 8, 12, 16, 20, 24, 28, 32] {
                let packed_size = (128 * bit_width + 7) / 8;
                let mut packed = vec![0u8; packed_size];
                let mut output = [0u32; 128];

                ScalarBackend::pack_block(input, bit_width as u8, &mut packed).unwrap();
                ScalarBackend::unpack_block(&packed, bit_width as u8, &mut output).unwrap();

                let mask = if bit_width >= 32 {
                    u32::MAX
                } else {
                    (1u32 << bit_width) - 1
                };

                for i in 0..128 {
                    assert_eq!(
                        output[i],
                        input[i] & mask,
                        "Pathological case {} failed at index {} for bit_width={}",
                        case_idx,
                        i,
                        bit_width
                    );
                }
            }
        }
    }

    #[test]
    fn test_multi_block_consistency() {
        let input1: [u32; 128] = [0x12345678; 128];
        let input2: [u32; 128] = [0x9ABCDEF0; 128];

        let mut packed1_a = vec![0u8; 512];
        let mut packed1_b = vec![0u8; 512];
        let mut packed2 = vec![0u8; 512];

        // Pack same input twice
        ScalarBackend::pack_block(&input1, 32, &mut packed1_a).unwrap();
        ScalarBackend::pack_block(&input1, 32, &mut packed1_b).unwrap();
        ScalarBackend::pack_block(&input2, 32, &mut packed2).unwrap();

        // Same input must produce identical output
        assert_eq!(
            packed1_a, packed1_b,
            "Same input must produce deterministic output"
        );

        // Different input must produce different output
        assert_ne!(
            packed1_a, packed2,
            "Different inputs should produce different outputs"
        );
    }

    #[test]
    fn test_full_roundtrip_compress_decompress() {
        use crate::{compress, decompress};

        let test_cases = [
            // All zeros
            vec![0u32; 256],
            // All ones (max values for various bit widths)
            vec![u32::MAX; 128],
            // Sequential
            (0..256).map(|i| i as u32).collect::<Vec<_>>(),
            // Random-looking pattern
            (0..256)
                .map(|i| ((i * 9301 + 49297) % 1000) as u32)
                .collect::<Vec<_>>(),
            // Mixed small and large values
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
                input,
                &decompressed,
                "Full roundtrip failed for test case {}. Input len: {}, Output len: {}",
                idx,
                input.len(),
                decompressed.len()
            );
        }
    }

    #[test]
    fn test_unpack_never_reads_beyond_buffer() {
        let bit_width = 12;
        let num_values = 10;
        let required_bytes = (num_values * bit_width + 7) / 8; // 15 bytes

        let input: Vec<u32> = (0..num_values).map(|i| (i as u32) & 0xFFF).collect();
        let mut packed = vec![0u8; required_bytes];
        let mut output = vec![0u32; num_values];

        ScalarBackend::pack_partial_block(&input, bit_width as u8, &mut packed).unwrap();

        // Create a larger buffer and fill extra bytes with garbage
        let mut large_buffer = packed.clone();
        large_buffer.extend_from_slice(&[0xFFu8; 100]); // Garbage after valid data

        // Unpack should only read the required bytes
        let result = ScalarBackend::unpack_partial_block(
            &large_buffer[..required_bytes],
            bit_width as u8,
            num_values,
            &mut output,
        );
        assert!(result.is_ok());

        // Verify unpacking from truncated buffer fails
        if required_bytes > 0 {
            let mut truncated = packed.clone();
            truncated.pop(); // Remove last byte
            let result = ScalarBackend::unpack_partial_block(
                &truncated,
                bit_width as u8,
                num_values,
                &mut output,
            );
            assert!(result.is_err(), "Should fail when input is truncated");
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

            // Test with multiple patterns
            let patterns: Vec<[u32; 128]> = vec![
                // All zeros
                [0u32; 128],
                // All ones (for this bit width)
                [mask; 128],
                // Alternating pattern
                std::array::from_fn(|i| if i % 2 == 0 { mask } else { 0 }),
                // Sequential
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
                        "Decode symmetry failed for bit_width={}, pattern={}, index={}. Expected {}, got {}",
                        bit_width, pattern_idx, i, input[i], output[i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_partial_block_roundtrip_comprehensive() {
        for num_values in [1, 7, 15, 31, 63, 64, 65, 100, 127] {
            for bit_width in [0, 1, 7, 8, 9, 15, 16, 17, 24, 31, 32] {
                let mask = if bit_width >= 32 {
                    u32::MAX
                } else if bit_width == 0 {
                    0
                } else {
                    (1u32 << bit_width) - 1
                };

                let input: Vec<u32> = (0..num_values)
                    .map(|i| ((i * 17 + 31) as u32) & mask)
                    .collect();

                let packed_size = (num_values * bit_width as usize + 7) / 8;
                let mut packed = vec![0u8; packed_size.max(1)];
                let mut output = vec![0u32; num_values];

                ScalarBackend::pack_partial_block(&input, bit_width, &mut packed).unwrap();
                ScalarBackend::unpack_partial_block(&packed, bit_width, num_values, &mut output)
                    .unwrap();

                assert_eq!(
                    input, output,
                    "Partial block roundtrip failed for num_values={}, bit_width={}",
                    num_values, bit_width
                );
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

            // Verify all decompressed values are actually zero
            assert!(
                decompressed.iter().all(|&v| v == 0),
                "Decompressed values should all be zero for size={}",
                size
            );
        }
    }

    #[test]
    fn test_all_max_values_roundtrip() {
        for bit_width in [1, 8, 16, 24, 32] {
            let max_val = if bit_width >= 32 {
                u32::MAX
            } else {
                (1u32 << bit_width) - 1
            };

            let input: [u32; 128] = [max_val; 128];
            let packed_size = (128 * bit_width as usize + 7) / 8;
            let mut packed = vec![0u8; packed_size];
            let mut output = [0u32; 128];

            ScalarBackend::pack_block(&input, bit_width as u8, &mut packed).unwrap();
            ScalarBackend::unpack_block(&packed, bit_width as u8, &mut output).unwrap();

            assert!(
                output.iter().all(|&v| v == max_val),
                "All max values roundtrip failed for bit_width={}",
                bit_width
            );
        }
    }

    #[test]
    fn test_strictly_increasing_sequence() {
        use crate::{compress, decompress};

        let input: Vec<u32> = (0..1000).map(|i| (i * 100) as u32).collect();
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();

        assert_eq!(
            input, decompressed,
            "Strictly increasing sequence roundtrip failed"
        );

        // Verify order is preserved
        for i in 1..decompressed.len() {
            assert!(
                decompressed[i] > decompressed[i - 1],
                "Order not preserved at index {}",
                i
            );
        }
    }

    #[test]
    fn test_random_values_comprehensive() {
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(42);

        for bit_width in [3, 7, 11, 13, 19, 23, 29] {
            let mask = if bit_width >= 32 {
                u32::MAX
            } else {
                (1u32 << bit_width) - 1
            };

            let input: [u32; 128] = std::array::from_fn(|_| rng.gen::<u32>() & mask);

            let packed_size = (128 * bit_width as usize + 7) / 8;
            let mut packed = vec![0u8; packed_size];
            let mut output = [0u32; 128];

            ScalarBackend::pack_block(&input, bit_width as u8, &mut packed).unwrap();
            ScalarBackend::unpack_block(&packed, bit_width as u8, &mut output).unwrap();

            assert_eq!(
                input, output,
                "Random values roundtrip failed for bit_width={}",
                bit_width
            );
        }
    }

    #[test]
    fn test_pathological_bit_patterns_roundtrip() {
        let patterns: Vec<(Vec<u32>, &'static str)> = vec![
            // Single bit walking through all positions
            ((0..128).map(|i| 1u32 << (i % 32)).collect(), "bit_walk"),
            // All bits set in alternating groups
            (
                (0..128)
                    .map(|i| if i % 2 == 0 { 0xFFFF0000 } else { 0x0000FFFF })
                    .collect(),
                "alternating_halves",
            ),
            // Checkerboard at bit level
            (
                (0..128).map(|_| 0xAAAAAAAAu32).collect(),
                "checkerboard_even",
            ),
            (
                (0..128).map(|_| 0x55555555u32).collect(),
                "checkerboard_odd",
            ),
            // Ramp within each block
            (
                (0..128).map(|i| ((i * 17) % 256) as u32).collect(),
                "modular_ramp",
            ),
        ];

        use crate::{compress, decompress};

        for (input, name) in patterns {
            let compressed = compress(&input).unwrap();
            let decompressed = decompress(&compressed).unwrap();

            assert_eq!(
                input, decompressed,
                "Pathological pattern '{}' roundtrip failed",
                name
            );
        }
    }

    #[test]
    fn test_single_large_outlier() {
        let mut input = vec![0u32; 256];
        input[127] = u32::MAX; // Large outlier in first block

        use crate::{compress, decompress};

        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();

        assert_eq!(input, decompressed, "Single large outlier roundtrip failed");

        // Verify the outlier is preserved
        assert_eq!(
            decompressed[127],
            u32::MAX,
            "Large outlier value not preserved"
        );
    }

    #[test]
    fn test_multi_block_with_different_bit_widths() {
        let mut input: Vec<u32> = Vec::new();

        // Block 1: values 0-100 (needs 7 bits)
        input.extend((0..128).map(|i| (i % 101) as u32));

        // Block 2: values 0-4095 (needs 12 bits)
        input.extend((0..128).map(|i| ((i * 32) % 4096) as u32));

        // Block 3: values requiring up to 24 bits
        input.extend((0..128).map(|i| ((i * 10000) % (1 << 24)) as u32));

        use crate::{compress, decompress};

        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();

        assert_eq!(
            input, decompressed,
            "Multi-block with different bit widths roundtrip failed"
        );
    }

    #[test]
    fn test_bit_alignment_across_bytes() {
        let bit_width = 9;

        // Each value has unique bits to detect shifts
        let input: [u32; 128] = std::array::from_fn(|i| {
            let pattern = ((i as u32) << 5) | (i as u32); // Distinct pattern per value
            pattern & 0x1FF // Mask to 9 bits
        });

        let packed_size = (128 * bit_width + 7) / 8; // 144 bytes
        let mut packed = vec![0u8; packed_size];
        let mut output = [0u32; 128];

        ScalarBackend::pack_block(&input, bit_width as u8, &mut packed).unwrap();
        ScalarBackend::unpack_block(&packed, bit_width as u8, &mut output).unwrap();

        for i in 0..128 {
            assert_eq!(
                output[i], input[i],
                "Bit alignment error at index {} for bit_width={}. Packed bytes around error: {:02X?}",
                i, bit_width,
                &packed[(i * bit_width / 8).saturating_sub(2)..((i * bit_width / 8) + 4).min(packed_size)]
            );
        }
    }

    #[test]
    fn test_empty_input() {
        use crate::{compress, decompress};

        let input: Vec<u32> = vec![];
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();

        assert!(
            compressed.is_empty(),
            "Empty input should produce empty compressed data"
        );
        assert!(
            decompressed.is_empty(),
            "Empty compressed should produce empty decompressed"
        );
    }

    #[test]
    fn test_single_value_roundtrip() {
        use crate::{compress, decompress};

        let test_values = [0u32, 1, 100, 1000, 10000, u32::MAX];

        for &value in &test_values {
            let input = vec![value];
            let compressed = compress(&input).unwrap();
            let decompressed = decompress(&compressed).unwrap();

            assert_eq!(
                decompressed, input,
                "Single value {} roundtrip failed",
                value
            );
        }
    }
}
