# simd-bp128

[![Crates.io](https://img.shields.io/crates/v/simd-bp128)](https://crates.io/crates/simd-bp128)
[![Documentation](https://docs.rs/simd-bp128/badge.svg)](https://docs.rs/simd-bp128)
[![License](https://img.shields.io/crates/l/simd-bp128)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/themankindproject/simd-bp128/ci.yml)](https://github.com/themankindproject/simd-bp128/actions)
![Rust Version](https://img.shields.io/badge/rust-1.70%2B-blue)

High-performance BP128 compression for `u32` integer arrays with **SIMD acceleration**, **zero-allocation APIs**, and **deterministic encoding**.

## Overview

`simd-bp128` compresses integer arrays by packing each block of 128 values using the minimum bit width required. It automatically detects and uses the best available SIMD backend at runtime.

| Use Case | Example | Typical Ratio |
|----------|---------|---------------|
| **Database Indexing** | Posting lists, doc IDs | 20-40% |
| **Search Systems** | Inverted indices | 20-40% |
| **Time Series** | Timestamp deltas | 30-50% |
| **Network Protocols** | Integer data transfer | Varies |
| **Columnar Storage** | Integer columns | 20-40% |

## Features

- **BP128 Algorithm** — Variable bit-width packing, 128 values per block
- **SIMD Acceleration** — SSE4.1 on x86_64 with automatic runtime detection
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
simd-bp128 = "0.1"
```

## Quick Start

```rust
use simd_bp128::{compress, decompress};

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
use simd_bp128::{compress_into, decompress_into, max_compressed_size, decompressed_len};

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
  │Backend  │    │  Backend  │   │(planned)│
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

## Performance

Benchmarked on x86_64 with SSE4.1:

### Block-Level (128 values)

| Operation | 1-bit | 8-bit | 16-bit | 24-bit | 32-bit |
|:----------|------:|------:|-------:|-------:|-------:|
| Pack | 107 ns | 16 ns | 27 ns | 103 ns | 133 ns |
| Unpack | 102 ns | 15 ns | 32 ns | 90 ns | 26 ns |

### End-to-End Throughput (1M values)

| Bit Width | Compress | Decompress |
|:----------|---------:|-----------:|
| 1-bit | 3-4 GiB/s | 100-170 MiB/s |
| 8-bit | 7-13 GiB/s | 2-5 GiB/s |
| 16-bit | 6-10 GiB/s | 3-6 GiB/s |
| 24-bit | 3-4 GiB/s | 3 GiB/s |
| 32-bit | 2-3 GiB/s | 8-12 GiB/s |

### Compression Ratios

| Data Pattern | Ratio |
|:-------------|------:|
| All zeros | ~2% |
| Sequential (0-999) | 23.65% |
| Constant (same value) | 18.97% |
| Random (full entropy) | ~100% |

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
cargo run --package simd-bp128-examples
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
git clone https://github.com/themankindproject/simd-bp128
cd simd-bp128

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
| AVX2 backend | Planned |
| AVX-512 backend | Planned |
| Streaming compression | Planned |
| `no_std` support | Planned |

## License

MIT License - See LICENSE file for details.
