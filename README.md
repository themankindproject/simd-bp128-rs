# packsimd

[![Crates.io](https://img.shields.io/crates/v/packsimd)](https://crates.io/crates/packsimd)
[![Documentation](https://docs.rs/packsimd/badge.svg)](https://docs.rs/packsimd)
[![License](https://img.shields.io/crates/l/packsimd)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/themankindproject/simd-bp128-rs/ci.yml)](https://github.com/themankindproject/packsimd/actions)
![Rust Version](https://img.shields.io/badge/rust-1.70%2B-blue)

High-performance BP128 compression for `u32` integer arrays with **SIMD acceleration**, **zero-allocation APIs**, and **deterministic encoding**. The crate ships **scalar**, **SSE4.1**, and **AVX2** backends and selects the best one at runtime. AVX-512 is planned for a future release.

## Overview

`packsimd` compresses integer arrays by packing each block of 128 values using the minimum bit width required. It automatically detects and uses the best available SIMD backend at runtime.

| Use Case | Example | Typical Ratio |
|----------|---------|---------------|
| **Database Indexing** | Posting lists, doc IDs | 20-40% |
| **Search Systems** | Inverted indices | 20-40% |
| **Time Series** | Timestamp deltas | 30-50% |
| **Network Protocols** | Integer data transfer | Varies |
| **Columnar Storage** | Integer columns | 20-40% |

## Features

- **BP128 Algorithm** — Variable bit-width packing, 128 values per block
- **SIMD Acceleration** — AVX2 and SSE4.1 on x86_64 with automatic runtime detection
- **Scalar Fallback** — Reference implementation for non-SIMD targets
- **Zero-Allocation API** — `compress_into` / `decompress_into` with pre-allocated buffers
- **Fast Header Inspection** — `decompressed_len` reads size without decompressing
- **Deterministic Output** — Same input always produces identical compressed bytes
- **No Dependencies** — Zero runtime dependencies
- **No Panics** — All error conditions return `Result`
- **Extensively Tested** — Property-based testing (proptest), fuzz targets, 128+ tests

## Installation

```toml
[dependencies]
packsimd = "0.2"
```

## Quick Start

```rust
use packsimd::{compress, decompress};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data: Vec<u32> = (0..256).map(|i| i % 1000).collect();
    let compressed = compress(&data)?;
    let decompressed = decompress(&compressed)?;
    assert_eq!(data, decompressed);
    Ok(())
}
```

### Zero-Allocation Path

```rust
use packsimd::{compress_into, decompress_into, max_compressed_size, decompressed_len};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data: Vec<u32> = (0..256).map(|i| i % 1000).collect();

    // Compress
    let mut cbuf = vec![0u8; max_compressed_size(data.len())];
    let cbytes = compress_into(&data, &mut cbuf)?;

    // Decompress
    let dlen = decompressed_len(&cbuf[..cbytes])?;
    let mutdbuf = vec![0u32; dlen];
    decompress_into(&cbuf[..cbytes], &mutdbuf)?;

    Ok(())
}
```

## Documentation

For complete API reference and usage examples, see [USAGE.md](USAGE.md).

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Public API                        │
│  compress / compress_into / max_compressed_size     │
│  decompress / decompress_into / decompressed_len    │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │       Dispatch          │
          │  Runtime CPU detection  │
          │  OnceLock caching       │
          └────────────┬────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
  ┌────┴────┐    ┌─────┴─────┐   ┌────┴────┐
  │ Scalar  │    │   SSE4.1  │   │  AVX2   │
  │Backend  │    │  Backend  │   │ Backend │
  │         │    │           │   │         │
  │Reference│    │  128-bit  │   │ 256-bit │
  │  impl   │    │   SIMD    │   │  SIMD   │
  └─────────┘    └───────────┘   └─────────┘
```

| Component | Responsibility |
|:----------|:---------------|
| **compress** | Bit width calculation, header writing, block packing |
| **decompress** | Header parsing, validation, block unpacking |
| **bitwidth** | `required_bit_width`, block size calculations |
| **dispatch** | Runtime SIMD backend selection and caching |
| **simd/scalar** | Reference scalar implementation (all bit widths) |
| **simd/sse** | SSE4.1-accelerated kernels (x86_64 only) |
| **simd/avx2** | AVX2-accelerated kernels for byte-aligned widths and 1-bit pack (x86_64 only) |

## Performance

Benchmarked on x86_64 (SSE4.1) with LTO and `opt-level=3`.

### Compression Ratios

| Data Pattern | Ratio | Compress | Decompress |
|:-------------|------:|---------:|-----------:|
| **Sequential (0-999)** | 23.65% | 5.4 GiB/s | 1.4 GiB/s |
| **Constant (all same)** | 18.97% | 2.8 GiB/s | 649 MiB/s |
| **Random (full entropy)** | 100.22% | 23.5 GiB/s | 24.0 GiB/s |

### Throughput at Scale (1M values)

| Bit Width | Compress | Decompress |
|:----------|---------:|-----------:|
| 1-bit | 13.7 GiB/s | 11.1 GiB/s |
| 8-bit | 8.9 GiB/s | 16.0 GiB/s |
| 16-bit | 7.1–7.5 GiB/s | 13.8–14.2 GiB/s |
| 32-bit | 7.9–8.3 GiB/s | 9.3–9.8 GiB/s |

SSE4.1 provides **1.7×–12.6× faster unpack** across all bit widths. Scalar pack is competitive for most widths.

Run benchmarks:
```bash
cargo bench
```

## Security

- **No Panics** — All error conditions return `Result`
- **Input Validation** — Header, bit widths, and buffer sizes verified before use
- **OOM Protection** — Maximum 1 billion decompressed values
- **No Undefined Behavior** — Unsafe blocks documented with invariants, covered by fuzz testing

## Examples

See the `examples/` directory:

```bash
cargo run --package packsimd-examples
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests: `cargo test --all-targets`
4. Run clippy: `cargo clippy --all-targets -- -D warnings`
5. Run benchmarks: `cargo bench`
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/themankindproject/packsimd
cd packsimd

# Run tests
cargo test --all-targets

# Run doc tests
cargo test --doc

# Generate documentation
cargo doc --no-deps --open
```

## Roadmap

| Feature | Status |
|:--------|:-------|
| Scalar implementation | Done |
| SSE4.1 backend | Done |
| AVX2 backend | Done |
| BMI2 PDEP/PEXT for irregular widths | Planned |
| AVX-512 backend | Planned |

## License

MIT License - See LICENSE file for details.
