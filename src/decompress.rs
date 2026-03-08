use crate::bitwidth::{packed_block_size, packed_partial_block_size};
use crate::dispatch::unpack_block_dispatch;
use crate::error::DecompressionError;
use crate::simd::scalar::ScalarBackend;
use crate::{BLOCK_SIZE, FORMAT_VERSION};

const MAX_DECOMPRESSED_VALUES: usize = 1_000_000_000;
const MAX_BLOCKS: usize = MAX_DECOMPRESSED_VALUES / 128 + 1;

/// Decompresses BP128-compressed data back to an array of `u32` integers.
///
/// The decompression process reads the binary format produced by [`compress`]:
/// 1. Parses header (version, original length, block count)
/// 2. Reads per-block bit widths
/// 3. Unpacks each block using the stored bit width
///
/// # Errors
///
/// Returns `Err(DecompressionError)` if the input is malformed:
/// - [`DecompressionError::HeaderTooSmall`] - Input shorter than 9 bytes
/// - [`DecompressionError::UnsupportedVersion`] - Version byte not equal to 1
/// - [`DecompressionError::InvalidBitWidth`] - Bit width byte > 32
/// - [`DecompressionError::TruncatedData`] - Insufficient bytes for packed data
/// - [`DecompressionError::BlockCountMismatch`] - Block count doesn't match length
///
/// # Panics
///
/// Does not panic. All errors are returned as [`DecompressionError`].
///
/// # Example
///
/// ```
/// use simd_bp128::{compress, decompress};
///
/// let data: Vec<u32> = (0..256).map(|i| i % 1000).collect();
/// let compressed = compress(&data).unwrap();
/// let decompressed = decompress(&compressed).unwrap();
/// assert_eq!(data, decompressed);
/// ```
pub fn decompress(input: &[u8]) -> Result<Vec<u32>, DecompressionError> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    const HEADER_SIZE: usize = 9;

    if input.len() < HEADER_SIZE {
        return Err(DecompressionError::HeaderTooSmall {
            needed: HEADER_SIZE,
            have: input.len(),
        });
    }

    let version = input[0];
    if version != FORMAT_VERSION {
        return Err(DecompressionError::UnsupportedVersion { version });
    }

    let total_count = u32::from_le_bytes([input[1], input[2], input[3], input[4]]) as usize;
    let num_blocks = u32::from_le_bytes([input[5], input[6], input[7], input[8]]) as usize;

    if total_count > MAX_DECOMPRESSED_VALUES {
        return Err(DecompressionError::InputTooLarge {
            max: MAX_DECOMPRESSED_VALUES,
            got: total_count,
        });
    }

    if num_blocks > MAX_BLOCKS {
        return Err(DecompressionError::ExcessiveBlockCount {
            max: MAX_BLOCKS,
            got: num_blocks,
        });
    }

    let expected_blocks = (total_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if total_count > 0 && num_blocks != expected_blocks {
        return Err(DecompressionError::BlockCountMismatch {
            expected: expected_blocks,
            found: num_blocks,
        });
    }

    let min_size = HEADER_SIZE + num_blocks;
    if input.len() < min_size {
        return Err(DecompressionError::TruncatedData {
            position: HEADER_SIZE,
            needed: num_blocks,
            have: input.len() - HEADER_SIZE,
        });
    }

    let bit_widths = &input[HEADER_SIZE..HEADER_SIZE + num_blocks];

    // Validate all bit widths up front in a single cheap pass — no size
    // computation here, just reject obviously bad input before allocating.
    for &bw in bit_widths {
        if bw > 32 {
            return Err(DecompressionError::InvalidBitWidth { bit_width: bw });
        }
    }

    // Allocate the full output buffer once. All blocks unpack directly into
    // their target slice — no intermediate [u32; 128] stack buffer, no
    // extend_from_slice copy, and zero-width blocks need no work at all
    // because the allocation is already zeroed.
    let mut output = vec![0u32; total_count];
    let mut data_pos = HEADER_SIZE + num_blocks;
    let num_full_blocks = total_count / BLOCK_SIZE;
    let remaining = total_count % BLOCK_SIZE;

    // --- Hot path: full blocks of exactly BLOCK_SIZE values ---
    // Splitting full and partial blocks into separate loops lets the compiler
    // see a fixed 128-element stride, eliminating the .min(BLOCK_SIZE) branch
    // on every iteration.
    for (block_idx, &bit_width) in bit_widths.iter().enumerate().take(num_full_blocks) {
        let write_pos = block_idx * BLOCK_SIZE;

        if bit_width == 0 {
            // Output is pre-zeroed; nothing to read from input.
            continue;
        }

        let packed_size = packed_block_size(bit_width);
        let data_end =
            data_pos
                .checked_add(packed_size)
                .ok_or(DecompressionError::TruncatedData {
                    position: data_pos,
                    needed: packed_size,
                    have: 0,
                })?;

        if data_end > input.len() {
            return Err(DecompressionError::TruncatedData {
                position: data_pos,
                needed: packed_size,
                have: input.len() - data_pos,
            });
        }

        let packed_data = &input[data_pos..data_end];

        // Unpack directly into the output slice — no stack buffer or memcpy.
        let dest: &mut [u32; BLOCK_SIZE] = (&mut output[write_pos..write_pos + BLOCK_SIZE])
            .try_into()
            .expect("slice length equals BLOCK_SIZE");

        unpack_block_dispatch(packed_data, bit_width, dest).map_err(|e| match e {
            crate::error::Error::InvalidBitWidth(bw) => {
                DecompressionError::InvalidBitWidth { bit_width: bw }
            }
            crate::error::Error::InputTooShort { need, got } => DecompressionError::TruncatedData {
                position: data_pos,
                needed: need,
                have: got,
            },
            crate::error::Error::OutputTooSmall { need, got } => {
                DecompressionError::TruncatedData {
                    position: data_pos,
                    needed: need,
                    have: got,
                }
            }
            crate::error::Error::CompressionError(_) => DecompressionError::TruncatedData {
                position: data_pos,
                needed: packed_size,
                have: 0,
            },
            crate::error::Error::DecompressionError(inner) => inner,
        })?;

        // Reuse data_end rather than checked_add a second time.
        data_pos = data_end;
    }

    // --- Partial block (last block only, 1-127 values) ---
    if remaining > 0 {
        let block_idx = num_full_blocks;
        let bit_width = bit_widths[block_idx];
        let write_pos = num_full_blocks * BLOCK_SIZE;

        if bit_width != 0 {
            let packed_size = packed_partial_block_size(remaining, bit_width);
            let data_end =
                data_pos
                    .checked_add(packed_size)
                    .ok_or(DecompressionError::TruncatedData {
                        position: data_pos,
                        needed: packed_size,
                        have: 0,
                    })?;

            if data_end > input.len() {
                return Err(DecompressionError::TruncatedData {
                    position: data_pos,
                    needed: packed_size,
                    have: input.len() - data_pos,
                });
            }

            let packed_data = &input[data_pos..data_end];

            ScalarBackend::unpack_partial_block(
                packed_data,
                bit_width,
                remaining,
                &mut output[write_pos..write_pos + remaining],
            )
            .map_err(|e| match e {
                crate::error::Error::InvalidBitWidth(bw) => {
                    DecompressionError::InvalidBitWidth { bit_width: bw }
                }
                crate::error::Error::InputTooShort { need, got } => {
                    DecompressionError::TruncatedData {
                        position: data_pos,
                        needed: need,
                        have: got,
                    }
                }
                crate::error::Error::OutputTooSmall { need, got } => {
                    DecompressionError::TruncatedData {
                        position: data_pos,
                        needed: need,
                        have: got,
                    }
                }
                crate::error::Error::CompressionError(_) => DecompressionError::TruncatedData {
                    position: data_pos,
                    needed: packed_size,
                    have: 0,
                },
                crate::error::Error::DecompressionError(inner) => inner,
            })?;
        }
        // bit_width == 0: output already zeroed, nothing to do.
    }

    if output.len() != total_count {
        return Err(DecompressionError::BlockCountMismatch {
            expected: total_count,
            found: output.len(),
        });
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress;

    #[test]
    fn test_decompress_empty() {
        let input: Vec<u8> = vec![];
        let decompressed = decompress(&input).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_decompress_zeros() {
        // Create compressed data manually for all zeros
        // Header: version=1, input_len=128, num_blocks=1
        let mut compressed = vec![];
        compressed.push(1); // version
        compressed.extend_from_slice(&128u32.to_le_bytes());
        compressed.extend_from_slice(&1u32.to_le_bytes());
        // Bit widths: 1 byte (bit_width=0)
        compressed.push(0);
        // No packed data for bit_width=0

        let decompressed = decompress(&compressed).unwrap();

        assert_eq!(decompressed.len(), 128);
        assert!(decompressed.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_roundtrip_simple() {
        let input: Vec<u32> = (0..128).map(|i| (i % 256) as u32).collect();
        let compressed = compress::compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();

        assert_eq!(input, decompressed);
    }

    #[test]
    fn test_roundtrip_multiple_blocks() {
        let input: Vec<u32> = (0..256).map(|i| (i % 100) as u32).collect();
        let compressed = compress::compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();

        assert_eq!(input, decompressed);
    }

    #[test]
    fn test_decompress_truncated_data() {
        // Header claims 1 value but no packed data
        let mut compressed = vec![];
        compressed.push(1); // version
        compressed.extend_from_slice(&1u32.to_le_bytes()); // input_len=1
        compressed.extend_from_slice(&1u32.to_le_bytes()); // num_blocks=1
        compressed.push(32); // bit_width=32 requires 4 bytes but we provide none

        let result = decompress(&compressed);
        assert!(result.is_err());
    }

    #[test]
    fn test_decompress_invalid_bit_width() {
        // bit_width = 33 (invalid)
        let mut compressed = vec![];
        compressed.push(1); // version
        compressed.extend_from_slice(&1u32.to_le_bytes());
        compressed.extend_from_slice(&1u32.to_le_bytes());
        compressed.push(33); // Invalid bit width

        let result = decompress(&compressed);
        assert!(matches!(
            result,
            Err(DecompressionError::InvalidBitWidth { bit_width: 33 })
        ));
    }

    #[test]
    fn test_decompress_header_too_small() {
        // Only 4 bytes, need 9 (version + input_len + num_blocks)
        let compressed = vec![0x01, 0x00, 0x00, 0x00];
        let result = decompress(&compressed);
        assert!(matches!(
            result,
            Err(DecompressionError::HeaderTooSmall { needed: 9, have: 4 })
        ));
    }

    #[test]
    fn test_decompress_excessive_total_count() {
        // Attempt to allocate 2 billion values (OOM attack)
        let mut compressed = vec![];
        compressed.push(1); // version
        compressed.extend_from_slice(&(2_000_000_000u32).to_le_bytes()); // input_len too large
        compressed.extend_from_slice(&1u32.to_le_bytes()); // num_blocks
        compressed.push(0); // bit_width=0

        let result = decompress(&compressed);
        assert!(matches!(
            result,
            Err(DecompressionError::InputTooLarge { .. })
        ));
    }

    #[test]
    fn test_decompress_excessive_block_count() {
        // More blocks than reasonable for the value count
        let mut compressed = vec![];
        compressed.push(1); // version
        compressed.extend_from_slice(&100u32.to_le_bytes()); // input_len=100
        compressed.extend_from_slice(&10_000_000u32.to_le_bytes()); // way too many blocks

        let result = decompress(&compressed);
        assert!(result.is_err());
    }

    #[test]
    fn test_decompress_insufficient_blocks() {
        // Claim 1000 values but only provide 1 block
        let mut compressed = vec![];
        compressed.push(1); // version
        compressed.extend_from_slice(&1000u32.to_le_bytes()); // input_len=1000 (needs 8 blocks)
        compressed.extend_from_slice(&1u32.to_le_bytes()); // num_blocks=1 (should be 8)
        compressed.push(8); // bit_width=8

        let result = decompress(&compressed);
        assert!(result.is_err());
    }

    #[test]
    fn test_decompress_malformed_data_various() {
        // Test various malformed inputs don't cause panics
        let test_cases = vec![
            // Empty header fields
            vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            // Negative-like values (interpreted as large positive)
            vec![0xFF, 0xFF, 0xFF, 0x7F, 0x01, 0x00, 0x00, 0x00, 0x08],
        ];

        for case in test_cases {
            let result = decompress(&case);
            // Should either succeed with empty/zero result or return error
            // Should NEVER panic
            match result {
                Ok(data) => {
                    // If it succeeds, verify it's reasonable
                    assert!(data.len() <= MAX_DECOMPRESSED_VALUES);
                }
                Err(_) => {
                    // Error is fine for malformed input
                }
            }
        }
    }

    #[test]
    fn test_decompress_zero_values_with_blocks() {
        // Edge case: 0 values but claims blocks exist
        let mut compressed = vec![];
        compressed.push(1); // version
        compressed.extend_from_slice(&0u32.to_le_bytes()); // input_len=0
        compressed.extend_from_slice(&1u32.to_le_bytes()); // num_blocks=1
        compressed.push(0); // bit_width=0

        let result = decompress(&compressed);
        // Should succeed with empty result (no values to decompress)
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_decompress_extra_data_after_blocks() {
        // Extra data after expected blocks should be ignored
        let input: Vec<u32> = (0..128).map(|i| (i % 100) as u32).collect();
        let mut compressed = compress::compress(&input).unwrap();

        // Append extra garbage data
        compressed.extend_from_slice(&[0xFFu8; 100]);

        // Should still decompress correctly (ignoring extra data)
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(input, decompressed);
    }
}
