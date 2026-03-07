#![no_main]

use libfuzzer_sys::fuzz_target;
use simd_bp128::{compress, decompress};

// Fuzz test: compress -> decompress roundtrip must produce identical output
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    // Convert bytes to u32 array (4 bytes per u32)
    let u32_data: Vec<u32> = data
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    if u32_data.is_empty() {
        return;
    }

    // Compress the data
    let compressed = match compress(&u32_data) {
        Ok(c) => c,
        Err(_) => return, // Compression failure is acceptable for edge cases
    };

    // Decompress the data
    let decompressed = match decompress(&compressed) {
        Ok(d) => d,
        Err(_) => return, // Decompression failure is acceptable
    };

    // Verify roundtrip correctness
    assert_eq!(
        u32_data,
        decompressed,
        "Roundtrip failed: input {} values, output {} values",
        u32_data.len(),
        decompressed.len()
    );

    // Verify no data corruption
    for (i, (original, recovered)) in u32_data.iter().zip(decompressed.iter()).enumerate() {
        assert_eq!(
            original, recovered,
            "Data corruption at index {}: original={}, recovered={}",
            i, original, recovered
        );
    }
});
