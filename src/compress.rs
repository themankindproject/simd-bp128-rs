use crate::bitwidth::{packed_block_size, packed_partial_block_size, required_bit_width};
use crate::dispatch::pack_block_dispatch;
use crate::error::{CompressionError, Error};
use crate::simd::scalar::ScalarBackend;
use crate::BLOCK_SIZE;

/// Current format version
const FORMAT_VERSION: u8 = 1;

/// Compresses an array of u32 integers using the BP128 algorithm.
///
/// The BP128 algorithm divides input into blocks of 128 values and stores each
/// block using the minimum number of bits required to represent the maximum value
/// in that block.
///
/// # Format
///
/// ```text
/// [version: u8][input_len: u32][num_blocks: u32][bit_widths: u8 × num_blocks][packed_data: u8[]]
/// ```
///
/// - `version`    – format version byte (currently `1`)
/// - `input_len`  – original number of u32 values (little-endian)
/// - `num_blocks` – number of blocks (little-endian); the last block may be partial
/// - `bit_widths` – one byte per block giving the bit-width used to pack that block;
///                  `0` means every value in the block is zero (no packed bytes emitted)
/// - `packed_data`– all blocks concatenated; full blocks occupy `bit_width × 128 / 8` bytes,
///                  partial blocks occupy `ceil(bit_width × remaining / 8)` bytes
///
/// # Errors
///
/// Returns [`CompressionError::InputTooLarge`] if `input.len()` exceeds `u32::MAX`.
///
/// # Example
///
/// ```
/// use simd_bp128::compress;
///
/// let data: Vec<u32> = (0..256).map(|i| i % 100).collect();
/// let compressed = compress(&data).unwrap();
/// ```
pub fn compress(input: &[u32]) -> Result<Vec<u8>, Error> {
    if input.is_empty() {
        return Ok(Vec::new());
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
    // Full blocks + optional single trailing partial block.
    let num_blocks: usize = num_full_blocks + usize::from(remaining > 0);

    // num_blocks ≤ ceil(u32::MAX / 128), well within u32 range.
    debug_assert!(num_blocks <= u32::MAX as usize, "num_blocks overflows u32");

    // Upper-bound capacity: assume every block needs the full 32-bit width
    // (128 values × 32 bits / 8 = 512 bytes per block). Actual output is
    // usually far smaller; `data_offset` tracks the exact write cursor.
    let max_size: usize = 1 + 8 + num_blocks + num_blocks * packed_block_size(32);
    let mut output: Vec<u8> = Vec::with_capacity(max_size);

    // --- Write header -------------------------------------------------------
    output.push(FORMAT_VERSION);
    output.extend_from_slice(&(input.len() as u32).to_le_bytes());
    output.extend_from_slice(&(num_blocks as u32).to_le_bytes());

    // Reserve space for bit_widths; filled in per-block below.
    let bit_widths_offset: usize = output.len();
    output.resize(bit_widths_offset + num_blocks, 0);

    // `data_offset` tracks the write cursor within the packed-data region.
    let mut data_offset: usize = output.len();

    // --- Process full blocks ------------------------------------------------
    //
    // We use a bitwise-OR accumulator rather than tracking the running maximum.
    // This is valid because the highest set bit of (a | b | …) equals that of
    // max(a, b, …): OR can only *set* bits, so the MSB position of the OR
    // equals the MSB position of the largest value in the block.
    for block_idx in 0..num_full_blocks {
        let start: usize = block_idx * BLOCK_SIZE;
        let block: &[u32] = &input[start..start + BLOCK_SIZE];

        debug_assert_eq!(block.len(), BLOCK_SIZE);

        let acc = block.iter().fold(0u32, |a, &v| a | v);
        let bit_width = required_bit_width(acc);

        // Write bit_width unconditionally; hoisted above the packed_size check
        // to avoid the duplicate write that existed when it appeared in both
        // branches separately.
        output[bit_widths_offset + block_idx] = bit_width;

        let packed_size: usize = packed_block_size(bit_width);

        // bit_width == 0 means every value is zero; no bytes need to be written.
        if packed_size == 0 {
            continue;
        }

        output.resize(data_offset + packed_size, 0);

        // SAFETY: block.len() == BLOCK_SIZE is guaranteed by the slice bounds
        // above, so try_into() is infallible here.
        let block_array: [u32; BLOCK_SIZE] = block
            .try_into()
            .expect("block slice length equals BLOCK_SIZE; conversion is infallible");

        pack_block_dispatch(&block_array, bit_width, &mut output[data_offset..])?;
        data_offset += packed_size;
    }

    // --- Process trailing partial block (if any) ----------------------------
    if remaining > 0 {
        let start: usize = num_full_blocks * BLOCK_SIZE;
        let block: &[u32] = &input[start..];

        let acc: u32 = block.iter().fold(0u32, |a, &v| a | v);
        let bit_width: u8 = required_bit_width(acc);

        // Record the bit_width for the partial block.
        output[bit_widths_offset + num_full_blocks] = bit_width;

        let packed_size: usize = packed_partial_block_size(remaining, bit_width);

        // bit_width == 0 means every value is zero; skip the pack call entirely.
        if packed_size > 0 {
            output.resize(data_offset + packed_size, 0);
            ScalarBackend::pack_partial_block(block, bit_width, &mut output[data_offset..])?;
            // Keep data_offset consistent so the debug_assert below holds and
            // any future code appended here sees a correct write cursor.
            data_offset += packed_size;
        }
    }

    debug_assert_eq!(
        output.len(),
        data_offset,
        "data_offset must equal output length; a missing `data_offset +=` would be caught here"
    );

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
}