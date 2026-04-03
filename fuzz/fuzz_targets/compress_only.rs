#![no_main]

use libfuzzer_sys::fuzz_target;
use packsimd::compress;

// Fuzz test: compression should handle all inputs safely
fuzz_target!(|data: &[u8]| {
    // Convert bytes to u32 array (4 bytes per u32)
    let u32_data: Vec<u32> = data
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    // Attempt to compress arbitrary data
    // Should either succeed or return an error, but NEVER panic
    let _ = compress(&u32_data);

    // If we get here without panicking, the test passes
});
