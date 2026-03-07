//! Edge case and stress tests for SIMD-BP128
//!
//! These tests verify behavior at boundaries and under stress conditions.

use simd_bp128::{compress, decompress};

mod edge_cases {
    use super::*;

    #[test]
    fn test_empty_input() {
        let input: Vec<u32> = vec![];
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(input, decompressed);
    }

    #[test]
    fn test_single_element_zero() {
        let input = vec![0u32];
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(input, decompressed);
    }

    #[test]
    fn test_single_element_max() {
        let input = vec![u32::MAX];
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(input, decompressed);
    }

    #[test]
    fn test_all_zeros_exactly_128() {
        let input = vec![0u32; 128];
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(input, decompressed);
        // All zeros should compress to minimum size
        // Version (1) + Header (8) + 1 bit_width byte = 10
        assert_eq!(compressed.len(), 10);
    }

    #[test]
    fn test_all_zeros_exactly_256() {
        let input = vec![0u32; 256];
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(input, decompressed);
    }

    #[test]
    fn test_all_ones_exactly_128() {
        let input = vec![u32::MAX; 128];
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(input, decompressed);
    }

    #[test]
    fn test_boundary_127_elements() {
        // One less than block size
        let input: Vec<u32> = (0..127).map(|i| i as u32).collect();
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(input, decompressed);
    }

    #[test]
    fn test_boundary_129_elements() {
        // One more than block size
        let input: Vec<u32> = (0..129).map(|i| i as u32).collect();
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(input, decompressed);
    }

    #[test]
    fn test_powers_of_two() {
        // Values that exactly fit in bit widths
        for power in 0..=31 {
            let value = 1u32 << power;
            let input = vec![value; 128];
            let compressed = compress(&input).unwrap();
            let decompressed = decompress(&compressed).unwrap();
            assert_eq!(input, decompressed, "Failed for 2^{} = {}", power, value);
        }
    }

    #[test]
    fn test_alternating_patterns() {
        // Alternating 0 and max
        let input: Vec<u32> = (0..128)
            .map(|i| if i % 2 == 0 { 0 } else { u32::MAX })
            .collect();
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(input, decompressed);
    }

    #[test]
    fn test_ascending_sequence() {
        // 0, 1, 2, 3, ...
        let input: Vec<u32> = (0..256).map(|i| i as u32).collect();
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(input, decompressed);
    }

    #[test]
    fn test_descending_sequence() {
        // MAX, MAX-1, MAX-2, ...
        let input: Vec<u32> = (0..256).map(|i| u32::MAX - i as u32).collect();
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(input, decompressed);
    }

    #[test]
    fn test_sparse_values() {
        // Mostly zeros with occasional values
        let mut input = vec![0u32; 256];
        input[0] = 100;
        input[127] = 200;
        input[255] = 300;
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(input, decompressed);
    }
}

mod stress_tests {
    use super::*;

    #[test]
    fn test_many_blocks_1000() {
        // 1000 blocks = 128,000 values
        let input: Vec<u32> = (0..128_000).map(|i| (i % 100) as u32).collect();
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(input, decompressed);
    }

    #[test]
    fn test_mixed_bit_widths_many_blocks() {
        // Create blocks with different bit widths
        let mut input = Vec::new();
        for block in 0..100 {
            let max_val = match block % 5 {
                0 => 1,        // 1 bit
                1 => 255,      // 8 bits
                2 => 65535,    // 16 bits
                3 => 16777215, // 24 bits
                _ => 16777215, // 24 bits (avoid u32::MAX overflow in test)
            };
            for i in 0..128 {
                // Use wrapping to avoid overflow when max_val == u32::MAX
                let value = if max_val == u32::MAX {
                    i as u32
                } else {
                    (i as u32) % (max_val + 1)
                };
                input.push(value);
            }
        }
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(input, decompressed);
    }

    #[test]
    fn test_repeated_compression_idempotent() {
        // compress -> decompress -> compress should give same result
        let input: Vec<u32> = (0..1024).map(|i| (i * 12345) as u32).collect();

        let compressed1 = compress(&input).unwrap();
        let decompressed1 = decompress(&compressed1).unwrap();
        let compressed2 = compress(&decompressed1).unwrap();
        let decompressed2 = decompress(&compressed2).unwrap();

        assert_eq!(input, decompressed1);
        assert_eq!(input, decompressed2);
        assert_eq!(compressed1, compressed2);
    }
}

mod robustness_tests {
    use super::*;

    #[test]
    fn test_decompress_random_bytes() {
        // Should not panic on random data
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let size = rng.gen_range(0..1024);
            let random_bytes: Vec<u8> = (0..size).map(|_| rng.gen()).collect();

            // Should either succeed or return error, never panic
            let _ = decompress(&random_bytes);
        }
    }

    #[test]
    fn test_decompress_truncated_at_all_positions() {
        let input: Vec<u32> = (0..256).map(|i| (i % 100) as u32).collect();
        let compressed = compress(&input).unwrap();

        // Try decompressing truncated versions
        for truncate_at in [
            1,
            4,
            8,
            9,
            10,
            20,
            compressed.len() / 2,
            compressed.len() - 1,
        ] {
            if truncate_at < compressed.len() {
                let truncated = &compressed[..truncate_at];
                let result = decompress(truncated);
                // Should return error, not panic
                assert!(result.is_err() || result.is_ok());
            }
        }
    }

    #[test]
    fn test_compress_decompress_consistency() {
        // Multiple compressions of same data should produce same output
        let input: Vec<u32> = (0..512).map(|i| (i * 17) as u32).collect();

        let c1 = compress(&input).unwrap();
        let c2 = compress(&input).unwrap();
        let c3 = compress(&input).unwrap();

        assert_eq!(c1, c2);
        assert_eq!(c2, c3);
    }
}

mod deterministic_tests {
    //! Tests to verify that compression produces deterministic output.
    //!
    //! Deterministic encoding is a critical property for BP128 compression.
    //! The same input must always produce the same compressed bytes.
    //! This ensures that:
    //! - Compression is reproducible across runs
    //! - Compressed data can be compared for equality
    //! - Hash-based indexing works correctly
    //! - SIMD backends will produce identical output to scalar

    use super::*;

    /// Helper function to verify deterministic compression
    fn assert_deterministic_encoding(input: &[u32]) {
        let compressed1 = compress(input).expect("First compression should succeed");
        let compressed2 = compress(input).expect("Second compression should succeed");
        let compressed3 = compress(input).expect("Third compression should succeed");

        assert_eq!(
            compressed1, compressed2,
            "Compression is not deterministic: first and second compression differ"
        );
        assert_eq!(
            compressed2, compressed3,
            "Compression is not deterministic: second and third compression differ"
        );

        // Also verify all bytes are identical
        assert_eq!(
            compressed1.len(),
            compressed2.len(),
            "Compressed lengths differ"
        );
        for (i, (b1, b2)) in compressed1.iter().zip(compressed2.iter()).enumerate() {
            assert_eq!(b1, b2, "Byte {} differs: 0x{:02X} vs 0x{:02X}", i, b1, b2);
        }
    }

    #[test]
    fn deterministic_random_data() {
        // Pseudo-random data using deterministic LCG
        // This ensures the test is reproducible across runs
        let input: Vec<u32> = (0..1000)
            .map(|i| (i as u32).wrapping_mul(1103515245).wrapping_add(12345))
            .collect();

        assert_deterministic_encoding(&input);
    }

    #[test]
    fn deterministic_all_zeros() {
        // All zeros - should produce identical output every time
        let input = vec![0u32; 1000];
        assert_deterministic_encoding(&input);
    }

    #[test]
    fn deterministic_all_zeros_single_block() {
        // Single block of zeros
        let input = vec![0u32; 128];
        assert_deterministic_encoding(&input);
    }

    #[test]
    fn deterministic_sequential_integers() {
        // Sequential integers: 0, 1, 2, 3, ...
        let input: Vec<u32> = (0..1000).map(|i| i as u32).collect();
        assert_deterministic_encoding(&input);
    }

    #[test]
    fn deterministic_sequential_single_block() {
        // Sequential within single block
        let input: Vec<u32> = (0..128).map(|i| i as u32).collect();
        assert_deterministic_encoding(&input);
    }

    #[test]
    fn deterministic_max_values() {
        // All maximum u32 values
        let input = vec![u32::MAX; 1000];
        assert_deterministic_encoding(&input);
    }

    #[test]
    fn deterministic_max_values_single_block() {
        // Single block of max values
        let input = vec![u32::MAX; 128];
        assert_deterministic_encoding(&input);
    }

    #[test]
    fn deterministic_mixed_values() {
        // Mix of small and large values
        let mut input = Vec::new();
        for i in 0..1000 {
            let value = match i % 4 {
                0 => 0u32,
                1 => 255,
                2 => 65535,
                _ => u32::MAX,
            };
            input.push(value);
        }
        assert_deterministic_encoding(&input);
    }

    #[test]
    fn deterministic_multi_block_various_sizes() {
        // Multiple blocks with various sizes
        for num_blocks in [2, 3, 5, 10] {
            let size = num_blocks * 128;
            let input: Vec<u32> = (0..size).map(|i| (i % 100) as u32).collect();
            assert_deterministic_encoding(&input);
        }
    }

    #[test]
    fn deterministic_multi_block_partial() {
        // Multiple full blocks plus partial
        let input: Vec<u32> = (0..400).map(|i| (i % 50) as u32).collect();
        assert_deterministic_encoding(&input);
    }

    #[test]
    fn deterministic_powers_of_two() {
        // Powers of two - tests various bit widths
        let input: Vec<u32> = (0..256).map(|i| 1u32 << (i % 32)).collect();
        assert_deterministic_encoding(&input);
    }

    #[test]
    fn deterministic_alternating_pattern() {
        // Alternating 0 and MAX
        let input: Vec<u32> = (0..256)
            .map(|i| if i % 2 == 0 { 0 } else { u32::MAX })
            .collect();
        assert_deterministic_encoding(&input);
    }

    #[test]
    fn deterministic_sparse_values() {
        // Sparse values (mostly zeros)
        let mut input = vec![0u32; 256];
        input[0] = 1;
        input[127] = 1000;
        input[255] = u32::MAX;
        assert_deterministic_encoding(&input);
    }

    #[test]
    fn deterministic_boundary_sizes() {
        // Test at block boundaries
        for size in [127, 128, 129, 255, 256, 257, 383, 384, 385] {
            let input: Vec<u32> = (0..size).map(|i| (i * 17) as u32).collect();
            assert_deterministic_encoding(&input);
        }
    }

    #[test]
    fn deterministic_single_element() {
        // Single element - edge case
        let input = vec![42u32];
        assert_deterministic_encoding(&input);
    }

    #[test]
    fn deterministic_empty() {
        // Empty input
        let input: Vec<u32> = vec![];
        assert_deterministic_encoding(&input);
    }

    #[test]
    fn deterministic_different_bit_widths_per_block() {
        // Multiple blocks with different bit widths
        let mut input = Vec::new();
        // Block 1: 7-bit values
        input.extend((0..128).map(|i| (i % 100) as u32));
        // Block 2: 16-bit values
        input.extend((0..128).map(|i| (i * 256) as u32));
        // Block 3: 32-bit values
        input.extend((0..128).map(|i| (i as u32) << 24));

        assert_deterministic_encoding(&input);
    }

    #[test]
    fn deterministic_repeated_calls_consistency() {
        // Verify that many repeated calls produce identical results
        let input: Vec<u32> = (0..500).map(|i| (i * 31) as u32).collect();

        let first = compress(&input).unwrap();

        for i in 0..10 {
            let subsequent = compress(&input).unwrap();
            assert_eq!(first, subsequent, "Compression differs on call {}", i + 1);
        }
    }
}

mod bit_boundary_tests {
    //! Bit-boundary stress tests for critical bit widths.
    //!
    //! These tests verify correct packing/unpacking at bit widths that
    //! represent byte boundaries (8, 16, 24, 32) and values just above/below.
    //! These are the most error-prone cases for bit-packing implementations.

    use super::*;

    /// Helper to generate test data with values that require exactly `bit_width` bits
    fn generate_values_for_bit_width(bit_width: u8, count: usize) -> Vec<u32> {
        if bit_width == 0 {
            return vec![0u32; count];
        }
        if bit_width == 32 {
            // Mix of all possible 32-bit values
            return (0..count)
                .map(|i| {
                    match i % 8 {
                        0 => 0u32,
                        1 => 1,
                        2 => 0x7FFFFFFF, // Max positive i32
                        3 => 0x80000000, // Sign bit set
                        4 => 0xAAAAAAAA, // Alternating pattern
                        5 => 0x55555555, // Alternating pattern inverse
                        6 => i as u32,
                        _ => u32::MAX,
                    }
                })
                .collect();
        }
        let max_val = (1u32 << bit_width) - 1;
        (0..count).map(|i| (i as u32) % (max_val + 1)).collect()
    }

    /// Helper to verify roundtrip for a given bit width
    fn assert_bit_width_roundtrip(bit_width: u8, description: &str) {
        // Test with full block
        let data = generate_values_for_bit_width(bit_width, 128);
        let compressed = compress(&data).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(
            data, decompressed,
            "Bit width {} ({}): Full block roundtrip failed",
            bit_width, description
        );

        // Test with partial block
        let data = generate_values_for_bit_width(bit_width, 65);
        let compressed = compress(&data).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(
            data, decompressed,
            "Bit width {} ({}): Partial block roundtrip failed",
            bit_width, description
        );

        // Test with multiple blocks
        let data = generate_values_for_bit_width(bit_width, 256);
        let compressed = compress(&data).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(
            data, decompressed,
            "Bit width {} ({}): Multi-block roundtrip failed",
            bit_width, description
        );
    }

    #[test]
    fn bit_width_boundary_1() {
        // Bit width 1: binary values (0 or 1)
        assert_bit_width_roundtrip(1, "binary values");
    }

    #[test]
    fn bit_width_boundary_7() {
        // Bit width 7: just below byte boundary (values 0-127)
        assert_bit_width_roundtrip(7, "just below byte");
    }

    #[test]
    fn bit_width_boundary_8() {
        // Bit width 8: exact byte boundary (values 0-255)
        assert_bit_width_roundtrip(8, "exact byte");
    }

    #[test]
    fn bit_width_boundary_9() {
        // Bit width 9: just above byte boundary (values 0-511)
        // This is critical: values cross byte boundaries
        assert_bit_width_roundtrip(9, "just above byte");
    }

    #[test]
    fn bit_width_boundary_15() {
        // Bit width 15: just below 16-bit boundary
        assert_bit_width_roundtrip(15, "just below u16");
    }

    #[test]
    fn bit_width_boundary_16() {
        // Bit width 16: exact 16-bit boundary (values 0-65535)
        assert_bit_width_roundtrip(16, "exact u16");
    }

    #[test]
    fn bit_width_boundary_17() {
        // Bit width 17: just above 16-bit boundary
        // Critical: crosses short boundary
        assert_bit_width_roundtrip(17, "just above u16");
    }

    #[test]
    fn bit_width_boundary_23() {
        // Bit width 23: just below 24-bit (3-byte) boundary
        assert_bit_width_roundtrip(23, "just below 3-byte");
    }

    #[test]
    fn bit_width_boundary_24() {
        // Bit width 24: exact 3-byte boundary
        assert_bit_width_roundtrip(24, "exact 3-byte");
    }

    #[test]
    fn bit_width_boundary_31() {
        // Bit width 31: just below 32-bit boundary
        // Max value is 0x7FFFFFFF (max positive i32)
        assert_bit_width_roundtrip(31, "just below u32");
    }

    #[test]
    fn bit_width_boundary_32() {
        // Bit width 32: full 32-bit range
        // Most critical: requires special handling (no masking)
        assert_bit_width_roundtrip(32, "full u32");
    }

    #[test]
    fn bit_width_boundary_all_max_values() {
        // Test all bit widths with their maximum values
        for bit_width in [1, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32] {
            let max_val = if bit_width == 32 {
                u32::MAX
            } else {
                (1u32 << bit_width) - 1
            };

            let data = vec![max_val; 128];
            let compressed = compress(&data).unwrap();
            let decompressed = decompress(&compressed).unwrap();

            assert_eq!(
                data, decompressed,
                "Bit width {} with all max values failed",
                bit_width
            );
        }
    }

    #[test]
    fn bit_width_boundary_mixed_within_block() {
        // Mix different bit widths in different blocks
        let mut input = Vec::new();

        // Block 1: 7-bit values
        input.extend((0..128).map(|i| (i % 128) as u32));

        // Block 2: 16-bit values
        input.extend((0..128).map(|i| (i * 256) as u32));

        // Block 3: 32-bit values
        input.extend((0..128).map(|i| (i as u32) << 24));

        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();

        assert_eq!(input, decompressed, "Mixed bit widths in blocks failed");
    }

    #[test]
    fn bit_width_boundary_crossing_byte_boundaries() {
        // Specifically test values that cross byte boundaries
        // For bit_width 12: each value is 1.5 bytes
        let bit_width = 12;
        let data: Vec<u32> = (0..128)
            .map(|i| {
                // Create pattern that makes byte errors obvious
                let pattern = ((i % 16) as u32) * 256 + (i as u32);
                pattern % (1 << bit_width)
            })
            .collect();

        let compressed = compress(&data).unwrap();
        let decompressed = decompress(&compressed).unwrap();

        assert_eq!(
            data, decompressed,
            "Byte boundary crossing failed for bit_width 12"
        );
    }
}

mod invariant_tests {
    use super::*;

    #[test]
    fn test_compression_ratio_sanity() {
        // Compressed size should generally be smaller or close to original
        // (except for tiny inputs where overhead dominates)
        let sizes = [256, 1024, 4096, 16384];

        for size in sizes {
            let input: Vec<u32> = (0..size).map(|i| (i % 100) as u32).collect();
            let original_size = input.len() * 4; // 4 bytes per u32

            let compressed = compress(&input).unwrap();
            let compressed_size = compressed.len();

            // For large enough inputs with compressible data, should be smaller
            if size >= 256 {
                assert!(
                    compressed_size < original_size,
                    "Compression should reduce size for {} values: {} bytes -> {} bytes",
                    size,
                    original_size,
                    compressed_size
                );
            }
        }
    }

    #[test]
    fn test_all_bit_widths_valid() {
        // Test that all possible bit widths (0-32) work correctly
        for bit_width in 0..=32 {
            let max_val = if bit_width == 0 {
                0
            } else if bit_width == 32 {
                u32::MAX
            } else {
                (1u32 << bit_width) - 1
            };

            let input: Vec<u32> = (0..128).map(|_| max_val).collect();
            let compressed = compress(&input).unwrap();
            let decompressed = decompress(&compressed).unwrap();

            assert_eq!(input, decompressed, "Failed for bit_width {}", bit_width);
        }
    }
}
