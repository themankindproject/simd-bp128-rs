# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-13

### Added
- AVX2 backend (`Avx2Backend` in `src/simd/avx2.rs`) with native 256-bit kernels for widths 1 (pack), 4 (unpack), 8, 16, and 32. All other widths delegate to `SseBackend`, preserving byte-identical output across backends.
- Dispatch routing: `BackendType::Avx2` now resolves to `Avx2Backend` (was previously falling through to SSE). `BackendType::Avx512` falls through to `Avx2Backend` until a dedicated AVX-512 kernel ships.
- AVX2-internal tests: byte-for-byte SSE compatibility, scalar parity, roundtrip, boundary patterns, and invalid-input handling, all gated on `is_x86_feature_detected!("avx2")`.
- AVX2 column added to `benches/throughput_comparison.rs` for direct scalar/SSE/AVX2 comparison.
- `cfg(miri)` override in `dispatch::detect_best_backend` so `cargo miri test` exercises the scalar backend without choking on x86 SIMD intrinsics.
- CI matrix expanded to test on Linux, macOS, and Windows; release-mode test job added.
- CI now fails on fuzz crashes (previously masked with `|| true`); fuzz duration bumped from 60s to 300s per target.

### Changed
- Lane-crossing handling: every `_mm256_packus_epi*` is followed by `_mm256_permute4x64_epi64::<0xD8>` to restore linear in-memory order, ensuring AVX2 output is byte-identical to SSE/scalar.
- Doc comments in `compress`, `decompress`, and `lib.rs` now mention AVX2 alongside SSE4.1.
- `internal::SseBackend` re-export is now gated on `target_arch = "x86_64"` so the crate compiles on non-x86_64 targets (aarch64, wasm32, etc.).

[0.2.0]: https://github.com/themankindproject/simd-bp128-rs/releases/tag/v0.2.0

## [0.1.1] - 2026-04-03

### Added
- Detailed docstrings to all public APIs (`compress`, `compress_into`, `max_compressed_size`, `decompress`, `decompress_into`, `decompressed_len`)
- Comprehensive variant-level documentation for `CompressionError` and `DecompressionError` with field descriptions and usage context
- `[package.metadata.docs.rs]` configuration for docs.rs builds

### Fixed
- Broken intra-doc links in error type documentation
- Repository URL pointing to wrong GitHub path

[0.1.1]: https://github.com/themankindproject/simd-bp128-rs/releases/tag/v0.1.1

## [0.1.0] - 2026-04-03

### Added
- BP128 compression and decompression for `u32` arrays
- Scalar backend with full bit width support (1-32)
- SSE4.1 SIMD backend for x86_64 with automatic runtime detection
- Zero-allocation API: `compress_into` / `decompress_into`
- Fast header inspection: `decompressed_len` reads size without decompressing
- Deterministic encoding — same input always produces identical compressed bytes
- Comprehensive error types: `CompressionError`, `DecompressionError`, unified `Error`
- Property-based testing with proptest (128+ tests)
- Fuzz testing targets: roundtrip, compress_only, decompress_only
- Benchmarks: compression, block-level, mixed data, throughput comparison
- Examples package demonstrating API usage and error handling
- Little-endian compile-time guard
- OOM protection: maximum 1 billion decompressed values
- No panics — all error conditions return `Result`

### Performance
- 3-10 GB/s throughput depending on bit width and CPU
- SSE4.1 provides 1.7x-12.6x faster unpack across all bit widths
- Backend dispatch resolved once per call (not per-block)
- Specialized pack/unpack functions for each bit width

### Fixed
- Prevent `decompress_into` panic on empty input
- Zero-fill zero-bit-width blocks during decompression
- Prevent `max_compressed_size` overflow on 32-bit targets
- Eliminate redundant header parsing in decompression
- Remove dead SSE code paths
- Enforce endianness at compile time
- Fuzz CI: use nightly toolchain (required for `-Zsanitizer=address`)
- Fuzz CI: pass binary names instead of file paths to `cargo fuzz run`

### Changed
- Crate renamed from `simd-bp128` to `packsimd`
- Removed `panic=abort` from library profile
- Exposed internal SIMD backends via doc-hidden `internal` module for benchmarks

[0.1.1]: https://github.com/themankindproject/simd-bp128-rs/releases/tag/v0.1.0..V0.1.1
[0.1.0]: https://github.com/themankindproject/simd-bp128-rs/releases/tag/v0.1.0
