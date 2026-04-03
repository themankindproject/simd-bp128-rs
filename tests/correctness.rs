//! Correctness tests for SIMD-BP128 compression
//!
//! These tests verify that compression and decompression are inverse operations.

use packsimd::{compress, decompress};
use proptest::prelude::*;

/// Test that decompression correctly reverses compression
fn roundtrip_test(input: &[u32]) {
    let compressed = compress(input).expect("Compression should succeed");
    let decompressed = decompress(&compressed).expect("Decompression should succeed");
    assert_eq!(
        input.len(),
        decompressed.len(),
        "Length mismatch: expected {}, got {}",
        input.len(),
        decompressed.len()
    );
    assert_eq!(input, &decompressed, "Data mismatch after roundtrip");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property test: decompress(compress(x)) == x for any array
    #[test]
    fn prop_roundtrip_any_values(values in prop::collection::vec(any::<u32>(), 0..=1024)) {
        roundtrip_test(&values);
    }

    /// Property test: all zeros compress efficiently
    #[test]
    fn prop_zeros_compress_small(count in 1usize..=512) {
        let zeros = vec![0u32; count];
        let compressed = compress(&zeros).expect("Compression should succeed");
        // Format: 1 byte version + 8 byte header + bit_widths + data
        let expected_blocks = (count + 127) / 128;
        prop_assert!(compressed.len() <= 1 + 8 + expected_blocks + expected_blocks * 128);
    }

    /// Property test: small values use few bits
    #[test]
    fn prop_small_values_small_compression(values in prop::collection::vec(0u32..256u32, 128..=128)) {
        let compressed = compress(&values).expect("Compression should succeed");
        // Format: 1 byte version + 8 byte header + 1 bit_width byte + packed data (128*8/8=128)
        const EXPECTED_SIZE: usize = 1 + 8 + 1 + 128;
        prop_assert_eq!(compressed.len(), EXPECTED_SIZE);
    }

    /// Property test: single value arrays
    #[test]
    fn prop_single_value(value in any::<u32>()) {
        let input = vec![value];
        roundtrip_test(&input);
    }

    /// Property test: maximum values use 32 bits
    #[test]
    fn prop_max_values_32bit(count in 128usize..=256) {
        let max_vals = vec![u32::MAX; count];
        let compressed = compress(&max_vals).expect("Compression should succeed");
        let num_full_blocks = count / 128;
        let remaining = count % 128;
        // Format: 1 byte version + 8 byte header + bit_widths + packed data
        let partial_size = if remaining == 0 { 0 } else { ((remaining * 32) + 7) / 8 };
        let expected_size = 1 + 8 + num_full_blocks + if remaining > 0 { 1 } else { 0 } +
                           num_full_blocks * 512 + partial_size;
        prop_assert_eq!(compressed.len(), expected_size);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Property test: roundtrip invariant with random data (1-5000 elements)
    /// Tests multi-block coverage with at least 1000 iterations
    #[test]
    fn prop_random_roundtrip_large(data in prop::collection::vec(any::<u32>(), 1..5000)) {
        let compressed = compress(&data).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        prop_assert_eq!(data, decompressed);
    }

    /// Property test: roundtrip with random data, varying bit widths
    /// Generates values that span different bit width ranges
    #[test]
    fn prop_random_roundtrip_mixed_widths(
        data in prop::collection::vec(
            prop::sample::select(vec![
                0u32, 1, 255, 256, 65535, 65536, 16777215, 16777216, u32::MAX
            ]),
            1..5000
        )
    ) {
        let compressed = compress(&data).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        prop_assert_eq!(data, decompressed);
    }

    /// Property test: large arrays with many blocks (up to ~40 blocks)
    #[test]
    fn prop_many_blocks_roundtrip(data in prop::collection::vec(any::<u32>(), 3000..5000)) {
        let compressed = compress(&data).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        prop_assert_eq!(data, decompressed);
    }

    /// Property test: verify compressed size is consistent
    #[test]
    fn prop_compression_size_consistent(data in prop::collection::vec(any::<u32>(), 1..5000)) {
        let compressed1 = compress(&data).unwrap();
        let compressed2 = compress(&data).unwrap();
        prop_assert_eq!(compressed1.len(), compressed2.len());
        prop_assert_eq!(compressed1, compressed2);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_empty_array() {
        let input: Vec<u32> = vec![];
        roundtrip_test(&input);
    }

    #[test]
    fn test_single_element() {
        let input = vec![42u32];
        roundtrip_test(&input);
    }

    #[test]
    fn test_exactly_one_block() {
        let input: Vec<u32> = (0..128).map(|i| i as u32).collect();
        roundtrip_test(&input);
    }

    #[test]
    fn test_exactly_two_blocks() {
        let input: Vec<u32> = (0..256).map(|i| i as u32).collect();
        roundtrip_test(&input);
    }

    #[test]
    fn test_partial_block() {
        let input: Vec<u32> = (0..200).map(|i| i as u32).collect();
        roundtrip_test(&input);
    }

    #[test]
    fn test_all_zeros() {
        let input = vec![0u32; 500];
        roundtrip_test(&input);
    }

    #[test]
    fn test_all_ones() {
        let input = vec![1u32; 500];
        roundtrip_test(&input);
    }

    #[test]
    fn test_powers_of_two() {
        let input: Vec<u32> = (0..256).map(|i| 1u32 << (i % 32)).collect();
        roundtrip_test(&input);
    }

    #[test]
    fn test_random_patterns() {
        let input: Vec<u32> = (0u32..1000).map(|i| i.wrapping_mul(2654435761)).collect();
        roundtrip_test(&input);
    }

    #[test]
    fn test_alternating_patterns() {
        let input: Vec<u32> = (0..256)
            .map(|i| if i % 2 == 0 { 0 } else { u32::MAX })
            .collect();
        roundtrip_test(&input);
    }

    #[test]
    fn test_consecutive_blocks_different_widths() {
        // First block: small values (8 bits)
        // Second block: large values (32 bits)
        let mut input: Vec<u32> = (0..128).map(|i| (i % 256) as u32).collect();
        input.extend((0..128).map(|i| (i as u32) << 24));

        roundtrip_test(&input);

        // Verify compression uses different bit widths
        // Format: 1 byte version + 8 byte header + bit_widths array + packed data
        let compressed = compress(&input).expect("Compression should succeed");
        // Bytes 9+: bit_widths array (version at 0, header at 1-8)
        // First block bit_width at position 9, second at position 10
        assert_eq!(compressed[9], 7); // First block: 7 bits (values 0-127)
        assert_eq!(compressed[10], 31); // Second block: 31 bits (0x7F000000 needs 31)
    }

    #[test]
    fn test_boundary_values() {
        // Values at bit width boundaries
        let input = vec![
            0u32,     // 0 bits
            1,        // 1 bit
            2,        // 2 bits
            4,        // 3 bits
            8,        // 4 bits
            16,       // 5 bits
            32,       // 6 bits
            64,       // 7 bits
            128,      // 8 bits
            256,      // 9 bits
            65536,    // 17 bits
            16777216, // 25 bits
            u32::MAX, // 32 bits
        ];

        // Pad to block size
        let mut padded = input.clone();
        padded.resize(128, 0);

        roundtrip_test(&padded);
    }
}
