# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - Unreleased

### Added
- Detailed docstrings to all public APIs (`compress`, `compress_into`, `max_compressed_size`, `decompress`, `decompress_into`, `decompressed_len`)
- Comprehensive variant-level documentation for `CompressionError` and `DecompressionError` with field descriptions and usage context

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

[0.1.0]: https://github.com/themankindproject/simd-bp128-rs/releases/tag/v0.1.0
