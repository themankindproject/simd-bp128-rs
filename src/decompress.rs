use crate::bitwidth::{packed_block_size, packed_partial_block_size};
use crate::dispatch::unpack_block_dispatch;
use crate::error::{DecompressionError, Error};
use crate::simd::scalar::ScalarBackend;
use crate::{BLOCK_SIZE, FORMAT_VERSION};

const MAX_DECOMPRESSED_VALUES: usize = 1_000_000_000;
const MAX_BLOCKS: usize = MAX_DECOMPRESSED_VALUES / 128 + 1;
const HEADER_SIZE: usize = 9;

/// Maps a generic `Error` to a `DecompressionError` with position context.
fn map_error(e: Error, data_pos: usize, packed_size: usize) -> DecompressionError {
    match e {
        Error::InvalidBitWidth(bw) => DecompressionError::InvalidBitWidth { bit_width: bw },
        Error::InputTooShort { need, got } => DecompressionError::TruncatedData {
            position: data_pos,
            needed: need,
            have: got,
        },
        Error::OutputTooSmall { need, got } => DecompressionError::TruncatedData {
            position: data_pos,
            needed: need,
            have: got,
        },
        Error::CompressionError(_) => DecompressionError::TruncatedData {
            position: data_pos,
            needed: packed_size,
            have: 0,
        },
        Error::DecompressionError(inner) => inner,
    }
}

/// Returns the number of `u32` values in compressed data without decompressing.
///
/// Parses the header to extract the original input length. Useful for
/// pre-allocating output buffers for [`decompress_into`].
///
/// Returns `Ok(0)` for empty input.
///
/// # Errors
///
/// Returns [`DecompressionError`] if the header is malformed:
/// - [`DecompressionError::HeaderTooSmall`] — Input shorter than 9 bytes
/// - [`DecompressionError::UnsupportedVersion`] — Version byte not equal to 1
/// - [`DecompressionError::InvalidBitWidth`] — Bit width byte > 32
/// - [`DecompressionError::BlockCountMismatch`] — Block count doesn't match length
/// - [`DecompressionError::InputTooLarge`] — Decompressed size exceeds safe limits
/// - [`DecompressionError::ExcessiveBlockCount`] — Block count exceeds safe limits
///
/// # Example
///
/// ```
/// use simd_bp128::{compress, decompressed_len, decompress_into};
///
/// let data: Vec<u32> = (0..256).map(|i| i % 1000).collect();
/// let compressed = compress(&data).unwrap();
///
/// let len = decompressed_len(&compressed).unwrap();
/// let mut output = vec![0u32; len];
/// let count = decompress_into(&compressed, &mut output).unwrap();
/// assert_eq!(&data[..], &output[..count]);
/// ```
pub fn decompressed_len(input: &[u8]) -> Result<usize, DecompressionError> {
    if input.is_empty() {
        return Ok(0);
    }

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

    for &bw in bit_widths {
        if bw > 32 {
            return Err(DecompressionError::InvalidBitWidth { bit_width: bw });
        }
    }

    Ok(total_count)
}

/// Decompresses `input` into the provided `output` buffer.
///
/// Returns the number of values written on success.
/// Use [`decompressed_len`] to determine the output buffer size.
///
/// # Errors
///
/// Returns an error if:
/// - The input header is malformed
/// - `output.len()` is less than the decompressed size
///
/// # Example
///
/// ```
/// use simd_bp128::{compress, decompressed_len, decompress_into};
///
/// let data: Vec<u32> = (0..256).map(|i| i % 1000).collect();
/// let compressed = compress(&data).unwrap();
/// let mut output = vec![0u32; decompressed_len(&compressed).unwrap()];
/// let count = decompress_into(&compressed, &mut output).unwrap();
/// assert_eq!(&data[..], &output[..count]);
/// ```
pub fn decompress_into(input: &[u8], output: &mut [u32]) -> Result<usize, DecompressionError> {
    let total_count = decompressed_len(input)?;

    if output.len() < total_count {
        return Err(DecompressionError::TruncatedData {
            position: 0,
            needed: total_count,
            have: output.len(),
        });
    }

    let num_blocks = u32::from_le_bytes([input[5], input[6], input[7], input[8]]) as usize;
    let bit_widths = &input[HEADER_SIZE..HEADER_SIZE + num_blocks];

    let mut data_pos = HEADER_SIZE + num_blocks;
    let num_full_blocks = total_count / BLOCK_SIZE;
    let remaining = total_count % BLOCK_SIZE;

    for (block_idx, &bit_width) in bit_widths.iter().enumerate().take(num_full_blocks) {
        let write_pos = block_idx * BLOCK_SIZE;

        if bit_width == 0 {
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

        // SAFETY: write_pos + BLOCK_SIZE <= output.len() is guaranteed by:
        // - output.len() >= total_count (checked above)
        // - write_pos = block_idx * BLOCK_SIZE where block_idx < num_full_blocks
        // - num_full_blocks * BLOCK_SIZE <= total_count <= output.len()
        let output_block: &mut [u32; BLOCK_SIZE] = (&mut output[write_pos..write_pos + BLOCK_SIZE])
            .try_into()
            .expect("output slice has exactly BLOCK_SIZE elements");

        unpack_block_dispatch(packed_data, bit_width, output_block)
            .map_err(|e| map_error(e, data_pos, packed_size))?;

        data_pos = data_end;
    }

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
            .map_err(|e| map_error(e, data_pos, packed_size))?;
        }
    }

    Ok(total_count)
}

/// Decompresses BP128-compressed data back to an array of `u32` integers.
///
/// The decompression process reads the binary format produced by [`compress`](crate::compress):
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

    let total_count: usize = decompressed_len(input)?;

    let mut output: Vec<u32> = vec![0u32; total_count];
    decompress_into(input, &mut output)?;

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
