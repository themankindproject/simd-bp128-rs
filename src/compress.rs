use crate::bitwidth::{packed_block_size, packed_partial_block_size, required_bit_width};
use crate::dispatch::pack_block_dispatch;
use crate::error::{CompressionError, Error};
use crate::simd::scalar::ScalarBackend;
use crate::{BLOCK_SIZE, FORMAT_VERSION};

/// Returns the maximum buffer size needed to compress `input_len` values.
///
/// This is the worst-case size assuming all blocks use 32-bit width.
/// The actual compressed size will be equal to or less than this value.
/// Use this to pre-allocate output buffers for [`compress`] or [`compress_into`].
///
/// # Example
///
/// ```
/// use simd_bp128::{compress_into, max_compressed_size};
///
/// let data: Vec<u32> = (0..256).map(|i| i % 1000).collect();
/// let mut buffer = vec![0u8; max_compressed_size(data.len())];
/// let bytes_written = compress_into(&data, &mut buffer).unwrap();
/// buffer.truncate(bytes_written);
/// assert!(bytes_written < max_compressed_size(data.len()));
/// ```
#[inline]
pub fn max_compressed_size(input_len: usize) -> usize {
    if input_len == 0 {
        return 0;
    }
    let num_full_blocks: usize = input_len / BLOCK_SIZE;
    let remaining: usize = input_len % BLOCK_SIZE;
    let num_blocks: usize = num_full_blocks + usize::from(remaining > 0);
    let packed = packed_block_size(32);
    // Checked arithmetic prevents silent overflow on 32-bit targets.
    // Returns usize::MAX if the result doesn't fit, so callers get a
    // clearly-wrong-but-safe value instead of a wrapped-too-small one.
    num_blocks
        .checked_mul(packed)
        .and_then(|v| v.checked_add(num_blocks))
        .and_then(|v| v.checked_add(9))
        .unwrap_or(usize::MAX)
}

/// Compresses `input` into the provided `output` buffer.
///
/// Returns the number of bytes written on success.
/// Use [`max_compressed_size`] to determine the minimum buffer size.
///
/// # Errors
///
/// Returns an error if:
/// - `input.len() > u32::MAX`
/// - `output.len() < max_compressed_size(input.len())`
///
/// # Example
///
/// ```
/// use simd_bp128::{compress_into, max_compressed_size};
///
/// let data: Vec<u32> = (0..256).map(|i| i % 1000).collect();
/// let mut buffer = vec![0u8; max_compressed_size(data.len())];
/// let bytes_written = compress_into(&data, &mut buffer).unwrap();
/// buffer.truncate(bytes_written);
/// ```
pub fn compress_into(input: &[u32], output: &mut [u8]) -> Result<usize, Error> {
    if input.is_empty() {
        return Ok(0);
    }

    if input.len() > u32::MAX as usize {
        return Err(CompressionError::InputTooLarge {
            max: u32::MAX as usize,
            got: input.len(),
        }
        .into());
    }

    let num_full_blocks: usize = input.len() / BLOCK_SIZE;
    let remaining: usize = input.len() % BLOCK_SIZE;
    let num_blocks: usize = num_full_blocks + usize::from(remaining > 0);

    let required_size = max_compressed_size(input.len());
    if output.len() < required_size {
        return Err(CompressionError::OutputTooSmall {
            need: required_size,
            got: output.len(),
        }
        .into());
    }

    let mut offset: usize = 0;

    output[offset] = FORMAT_VERSION;
    offset += 1;
    output[offset..offset + 4].copy_from_slice(&(input.len() as u32).to_le_bytes());
    offset += 4;
    output[offset..offset + 4].copy_from_slice(&(num_blocks as u32).to_le_bytes());
    offset += 4;

    let bit_widths_offset: usize = offset;
    offset += num_blocks;

    for block_idx in 0..num_full_blocks {
        let start: usize = block_idx * BLOCK_SIZE;
        let block: &[u32; BLOCK_SIZE] = input[start..start + BLOCK_SIZE]
            .try_into()
            .expect("block slice length is exactly BLOCK_SIZE");

        let acc = block.iter().fold(0u32, |acc, &v| acc | v);
        let bit_width = required_bit_width(acc);

        output[bit_widths_offset + block_idx] = bit_width;

        let packed_size: usize = packed_block_size(bit_width);
        if packed_size == 0 {
            continue;
        }

        pack_block_dispatch(block, bit_width, &mut output[offset..])?;
        offset += packed_size;
    }

    if remaining > 0 {
        let start: usize = num_full_blocks * BLOCK_SIZE;
        let block: &[u32] = &input[start..];

        let acc = block.iter().fold(0u32, |acc, &v| acc | v);
        let bit_width: u8 = required_bit_width(acc);

        output[bit_widths_offset + num_full_blocks] = bit_width;

        let packed_size: usize = packed_partial_block_size(remaining, bit_width);
        if packed_size > 0 {
            ScalarBackend::pack_partial_block(block, bit_width, &mut output[offset..])?;
            offset += packed_size;
        }
    }

    Ok(offset)
}

/// Compresses an array of `u32` integers using the BP128 algorithm.
///
/// BP128 divides input into blocks of 128 values and stores each block using the
/// minimum number of bits required to represent the maximum value in that block.
/// This achieves variable-bit-width compression optimized for integer arrays.
///
/// # Binary Format
///
/// ```text
/// [version: u8][input_len: u32 LE][num_blocks: u32 LE][bit_widths: u8 × N][packed_data: u8[]]
/// ```
///
/// | Field | Type | Description |
/// |-------|------|-------------|
/// | version | [u8] | Format version (currently 1) |
/// | input_len | [u32] LE | Original number of u32 values |
/// | num_blocks | [u32] LE | Number of blocks (last may be partial) |
/// | bit_widths | [u8] | One byte per block (0 = all zeros) |
/// | packed_data | [u8] | Bit-packed values, blocks concatenated |
///
/// # Performance
///
/// - Time complexity: O(n) where n = input.len()
/// - Space complexity: O(n) for output buffer
/// - Allocates once: output buffer sized to worst case, then truncated
///
/// # Errors
///
/// Returns [`Error::CompressionError`] if:
/// - [`CompressionError::InputTooLarge`] - input.len() > u32::MAX
///
/// # Example
///
/// ```
/// use simd_bp128::compress;
///
/// let data: Vec<u32> = (0..256).map(|i| i % 1000).collect();
/// let compressed = compress(&data).unwrap();
/// assert!(compressed.len() < data.len() * 4);
/// ```
pub fn compress(input: &[u32]) -> Result<Vec<u8>, Error> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let max_size = max_compressed_size(input.len());
    let mut output = vec![0; max_size];

    let bytes_written = compress_into(input, &mut output)?;
    output.truncate(bytes_written);

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_empty() {
        let input: Vec<u32> = vec![];
        let compressed = compress(&input).unwrap();
        assert!(compressed.is_empty());
    }

    #[test]
    fn test_compress_single_value() {
        let input: Vec<u32> = vec![42];
        let compressed = compress(&input).unwrap();

        // Layout:
        //   1  byte  – version
        //   4  bytes – input_len (= 1)
        //   4  bytes – num_blocks (= 1)
        //   1  byte  – bit_width (= 6, since 42 = 0b101010)
        //   1  byte  – packed data: ceil(1 × 6 / 8) = 1
        assert_eq!(compressed.len(), 11);

        assert_eq!(compressed[0], FORMAT_VERSION);

        let input_len =
            u32::from_le_bytes([compressed[1], compressed[2], compressed[3], compressed[4]]);
        assert_eq!(input_len, 1);

        let num_blocks =
            u32::from_le_bytes([compressed[5], compressed[6], compressed[7], compressed[8]]);
        assert_eq!(num_blocks, 1);

        assert_eq!(compressed[9], 6);
    }

    #[test]
    fn test_compress_full_block() {
        let input: Vec<u32> = (0..128).map(|i| (i % 128) as u32).collect();
        let compressed = compress(&input).unwrap();

        // Layout:
        //   1   byte  – version
        //   4   bytes – input_len (= 128)
        //   4   bytes – num_blocks (= 1)
        //   1   byte  – bit_width (= 7, max value 127 = 0b111_1111)
        //   112 bytes – packed data: 128 × 7 / 8
        assert_eq!(compressed.len(), 122);

        assert_eq!(compressed[0], FORMAT_VERSION);

        let input_len =
            u32::from_le_bytes([compressed[1], compressed[2], compressed[3], compressed[4]]);
        assert_eq!(input_len, 128);

        let num_blocks =
            u32::from_le_bytes([compressed[5], compressed[6], compressed[7], compressed[8]]);
        assert_eq!(num_blocks, 1);

        assert_eq!(compressed[9], 7);
    }

    #[test]
    fn test_compress_multiple_blocks() {
        let input: Vec<u32> = (0..256).map(|i| (i % 100) as u32).collect();
        let compressed = compress(&input).unwrap();

        let num_blocks =
            u32::from_le_bytes([compressed[5], compressed[6], compressed[7], compressed[8]]);
        assert_eq!(num_blocks, 2);

        // max value per block is 99; ceil(log2(100)) = 7 bits
        assert_eq!(compressed[9], 7);
        assert_eq!(compressed[10], 7);
    }

    #[test]
    fn test_compress_zeros() {
        let input = [0u32; 128];
        let compressed = compress(&input).unwrap();

        // Layout:
        //   1 byte – version
        //   4 bytes – input_len (= 128)
        //   4 bytes – num_blocks (= 1)
        //   1 byte – bit_width (= 0, all values are zero)
        //   0 bytes – packed data (skipped when bit_width == 0)
        assert_eq!(compressed.len(), 10);

        assert_eq!(compressed[0], FORMAT_VERSION);
        assert_eq!(compressed[9], 0);
    }

    #[test]
    fn test_compress_partial_block() {
        // 200 values = 1 full block (128) + 1 partial block (72)
        let input: Vec<u32> = (0..200).map(|i| (i % 100) as u32).collect();
        let compressed = compress(&input).unwrap();

        let input_len =
            u32::from_le_bytes([compressed[1], compressed[2], compressed[3], compressed[4]]);
        assert_eq!(input_len, 200);

        let num_blocks =
            u32::from_le_bytes([compressed[5], compressed[6], compressed[7], compressed[8]]);
        assert_eq!(num_blocks, 2);

        // Both blocks: max value 99, needs 7 bits
        assert_eq!(compressed[9], 7);
        assert_eq!(compressed[10], 7);
    }

    #[test]
    fn test_compress_input_too_large() {
        // We cannot allocate u32::MAX elements in a test, but we can verify
        // that a legitimately-sized input succeeds, and confirm the error
        // variant is correctly constructed when the limit is exceeded.
        let ok = compress(&[0u32; 100]);
        assert!(ok.is_ok());

        // Construct the error directly to confirm the variant compiles and
        // is wired through the public Error type.
        let err = CompressionError::InputTooLarge {
            max: u32::MAX as usize,
            got: u32::MAX as usize + 1,
        };
        let _: Error = err.into();
    }

    #[test]
    fn test_compress_different_bit_widths() {
        // First block:  values 0-99 (7 bits)
        // Second block: values 0, 1000, 2000, … 127000 (17 bits; max = 127000)
        let mut input: Vec<u32> = (0..128).map(|i| (i % 100) as u32).collect();
        input.extend((0..128u32).map(|i| i * 1000));

        let compressed = compress(&input).unwrap();

        assert_eq!(compressed[9], 7); // first block
        assert_eq!(compressed[10], 17); // second block (127000 < 2^17 = 131072)
    }

    #[test]
    fn test_compress_format_layout() {
        let input: Vec<u32> = vec![1, 2, 3, 4, 5];
        let compressed = compress(&input).unwrap();

        // Byte 0: version
        assert_eq!(compressed[0], FORMAT_VERSION);

        // Bytes 1-4: input_len = 5 (little-endian)
        assert_eq!(&compressed[1..5], &[5, 0, 0, 0]);

        // Bytes 5-8: num_blocks = 1 (little-endian)
        assert_eq!(&compressed[5..9], &[1, 0, 0, 0]);

        // Byte 9: bit_width = 3  (OR of {1,2,3,4,5} = 7 = 0b111; needs 3 bits)
        assert_eq!(compressed[9], 3);

        // Bytes 10-11: packed data – ceil(5 × 3 / 8) = 2 bytes
        // Total = 1 + 4 + 4 + 1 + 2 = 12
        assert_eq!(compressed.len(), 12);
    }

    #[test]
    fn test_compress_zeros_partial_block() {
        // Regression: a partial all-zero block must not invoke pack_partial_block
        // with packed_size == 0 (a missing `if packed_size > 0` guard would
        // pass a zero-length output slice to the backend, causing a panic or
        // undefined behaviour).
        let input = vec![0u32; 50]; // single partial block, all zeros
        let compressed = compress(&input).unwrap();

        // 1 (version) + 4 (input_len) + 4 (num_blocks) + 1 (bit_width=0) + 0 (data)
        assert_eq!(compressed.len(), 10);
        assert_eq!(compressed[9], 0);
    }

    #[test]
    fn test_compress_precomputed_output_size() {
        // Verify that the input_len header is correct across a range of sizes
        // and that no size triggers a panic or allocation failure.
        for &n in &[1usize, 127, 128, 129, 255, 256, 257, 512] {
            let input: Vec<u32> = (0..n as u32).collect();
            let compressed = compress(&input).unwrap();

            let input_len =
                u32::from_le_bytes([compressed[1], compressed[2], compressed[3], compressed[4]]);
            assert_eq!(input_len as usize, n, "input_len header mismatch for n={n}");
        }
    }

    #[test]
    fn test_max_compressed_size_overflow_safe() {
        // On 64-bit this is a normal value. On 32-bit, the old code would
        // silently wrap; the checked arithmetic returns usize::MAX instead.
        let size = max_compressed_size(u32::MAX as usize);
        assert!(size > 0, "must not wrap to zero");
        // Must be at least the header (9 bytes) + bit_widths directory.
        let expected_blocks = (u32::MAX as usize + 127) / 128;
        assert!(size >= 9 + expected_blocks);
    }
}
