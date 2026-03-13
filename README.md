<div align="center">

# simd-bp128-rs

**High-performance BP128 compression for u32 integer arrays with SIMD acceleration**

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![CI](https://img.shields.io/github/actions/workflow/status/themankindproject/simd-bp128-rs/ci.yml?style=for-the-badge&label=CI)](https://github.com/themankindproject/simd-bp128-rs/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/themankindproject/simd-bp128-rs?style=for-the-badge&logo=github&color=yellow)](https://github.com/themankindproject/simd-bp128-rs/stargazers)

</div>

simd-bp128-rs is a Rust library that implements the BP128 (Bit Packing 128) compression algorithm with automatic SIMD acceleration. It compresses arrays of `u32` integers by processing data in 128-element blocks and computing the minimum bit width needed per block for optimal packing.

- **Database Indexing** — Compress posting lists and integer arrays
- **Search Systems** — Reduce memory footprint for inverted indices
- **Time Series** — Efficient storage for timestamp deltas
- **Network Protocols** — Minimize bandwidth for integer data transfer

## Quickstart

**Prerequisites**: Rust 1.76+ (`rustup toolchain install stable`)

```bash
# Build & test
cargo test --all

# Run benchmarks
cargo bench

# Run examples
cargo run --package simd-bp128-examples
```

## Usage

```rust
use simd_bp128::{compress, decompress};

let data: Vec<u32> = (0..256).map(|i| i % 1000).collect();
let compressed = compress(&data).unwrap();
let decompressed = decompress(&compressed).unwrap();

assert_eq!(data, decompressed);
```

See [`examples/`](examples/) for more demonstrations.

## Architecture

| Component | Responsibility |
|:----------|:---------------|
| **compress** | Bit width calculation, block packing |
| **decompress** | Block unpacking, validation |
| **dispatch** | Runtime SIMD backend selection |

The library uses a trait-based backend system internally:

- `SimdBackend`: Core trait for packing/unpacking blocks
- `ScalarBackend`: Reference implementation  
- `SseBackend`, `Avx2Backend`, `Avx512Backend`: SIMD-accelerated implementations

## Format Specification

### Binary Layout

```
[version: u8]           // 1 byte  - format version
[input_len: u32 LE]    // 4 bytes - number of original u32 values
[num_blocks: u32 LE]   // 4 bytes - number of blocks
[bit_widths: [u8]]     // N bytes - one bit width per block (0-32)
[packed_data: [u8]]    // variable - bit-packed block data
```

### Block Structure

- **Full block**: 128 values
- **Partial block**: 1-127 values (last block only)

### Bit Width Semantics

- **bit_width = 0**: All values are zero, no packed data follows
- **bit_width = N (1-32)**: Each value uses N bits

## Error Handling

```rust
use simd_bp128::{compress, decompress, Error};

match compress(&data) {
    Ok(compressed) => { /* use compressed data */ }
    Err(e) => { /* handle error */ }
}

match decompress(&compressed) {
    Ok(decompressed) => { /* use decompressed data */ }
    Err(e) => { /* handle error */ }
}
```

## Development

```bash
cargo fmt --check
cargo clippy --all-targets
cargo test --all
cargo bench
```

## Roadmap

| Feature | Status |
|:--------|:-------|
| Scalar implementation | Done |
| SSE backend | Done |
| AVX2 backend | Planned |
| AVX512 backend | Planned |
| SIMD parallel encoding | Planned |
| Streaming compression | Planned |

## Performance

### Block-Level Benchmarks (128 values)

| Operation | 1-bit | 8-bit | 16-bit | 24-bit | 32-bit |
|:----------|------:|------:|-------:|-------:|-------:|
| Pack | 107 ns | 16 ns | 27 ns | 103 ns | 133 ns |
| Unpack | 102 ns | 15 ns | 32 ns | 90 ns | 26 ns |

### Throughput (1M values, 16-bit)

- **Compression**: ~7.8 GiB/s
- **Decompression**: ~4.3 GiB/s

### Compression Ratios

| Data Type | Ratio |
|:----------|------:|
| Sequential (0-999) | 23.65% |
| Constant (all same) | 18.97% |
| Random (full entropy) | 100.22% |

### Typical Throughput by Bit Width

| Bit Width | Compress | Decompress |
|:----------|---------:|----------:|
| 1-bit     | 3-4 GiB/s | 100-170 MiB/s |
| 8-bit | 7-13 GiB/s | 2-5 GiB/s |
| 16-bit | 6-10 GiB/s | 3-6 GiB/s |
| 24-bit | 3-4 GiB/s | 3 GiB/s |
| 32-bit | 2-3 GiB/s | 8-12 GiB/s |

## License

MIT
