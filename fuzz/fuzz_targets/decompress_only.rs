#![no_main]

use libfuzzer_sys::fuzz_target;
use packsimd::decompress;

// Fuzz test: decompression should never panic on arbitrary input
fuzz_target!(|data: &[u8]| {
    // Attempt to decompress arbitrary bytes
    // Should either succeed or return an error, but NEVER panic
    let _ = decompress(data);

    // If we get here without panicking, the test passes
    // We don't verify the output - just that it doesn't crash
});
