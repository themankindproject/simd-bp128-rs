use crate::bitwidth::{packed_block_size, packed_partial_block_size};
use crate::dispatch::unpack_block_dispatch;
use crate::error::DecompressionError;
use crate::simd::scalar::ScalarBackend;
use crate::BLOCK_SIZE;

const MAX_DECOMPRESSED_VALUES: usize = 1_000_000_000;
const MAX_BLOCKS: usize = MAX_DECOMPRESSED_VALUES / 128 + 1;
const FORMAT_VERSION: u8 = 1;

/// Decompresss data previously compressed with BP128.
///
/// # Errors
///
/// Returns `Err(DecompressionError)` for malformed input including:
/// - Header too small
/// - Truncated data  
/// - Invalid bit width
/// - Block count mismatch
///
/// # Example
///
/// ```
/// use simd_bp128::{compress, decompress};
///
/// let data: Vec<u32> = (0..256).map(|i| i % 100).collect();
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

    for &_bw in bit_widths.iter() {
        if _bw > 32 {
            return Err(DecompressionError::InvalidBitWidth { bit_width: _bw });
        }
    }

    let mut min_packed_size: usize = 0;
    let mut values_to_process: usize = total_count;
    for &bit_width in bit_widths.iter() {
        if values_to_process == 0 {
            break;
        }
        let block_values = values_to_process.min(BLOCK_SIZE);

        if bit_width > 32 {
            return Err(DecompressionError::InvalidBitWidth { bit_width });
        }

        let block_size = if block_values == BLOCK_SIZE {
            packed_block_size(bit_width)
        } else {
            packed_partial_block_size(block_values, bit_width)
        };

        min_packed_size = min_packed_size.checked_add(block_size).ok_or_else(|| {
            DecompressionError::TruncatedData {
                position: HEADER_SIZE,
                needed: usize::MAX,
                have: input.len(),
            }
        })?;

        values_to_process = values_to_process.saturating_sub(block_values);
    }

    let min_total_size = HEADER_SIZE + num_blocks + min_packed_size;
    if input.len() < min_total_size {
        return Err(DecompressionError::TruncatedData {
            position: input.len(),
            needed: min_total_size,
            have: input.len(),
        });
    }

    let mut output = Vec::with_capacity(total_count);
    let mut data_pos = HEADER_SIZE + num_blocks;
    let mut values_remaining = total_count;

    for (_block_idx, &bit_width) in bit_widths.iter().enumerate() {
        if values_remaining == 0 {
            break;
        }

        let block_values = values_remaining.min(BLOCK_SIZE);
        let is_partial = block_values < BLOCK_SIZE;

        let packed_size = if is_partial {
            packed_partial_block_size(block_values, bit_width)
        } else {
            packed_block_size(bit_width)
        };

        if bit_width > 32 {
            return Err(DecompressionError::InvalidBitWidth { bit_width });
        }

        let data_end =
            data_pos
                .checked_add(packed_size)
                .ok_or_else(|| DecompressionError::TruncatedData {
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

        if packed_size > 0 {
            let packed_data = &input[data_pos..data_end];

            if is_partial {
                let mut block = [0u32; BLOCK_SIZE];
                ScalarBackend::unpack_partial_block(
                    packed_data,
                    bit_width,
                    block_values,
                    &mut block,
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
                output.extend_from_slice(&block[..block_values]);
            } else {
                let mut block = [0u32; BLOCK_SIZE];
                unpack_block_dispatch(packed_data, bit_width, &mut block).map_err(|e| match e {
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
                output.extend_from_slice(&block);
            }

            data_pos = data_pos.checked_add(packed_size).ok_or_else(|| {
                DecompressionError::TruncatedData {
                    position: data_pos,
                    needed: packed_size,
                    have: 0,
                }
            })?;
        } else {
            output.extend(std::iter::repeat(0u32).take(block_values));
        }

        values_remaining -= block_values;
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
