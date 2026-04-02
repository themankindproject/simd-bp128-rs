# simd-bp128 Usage Guide

> Complete API reference and examples for high-performance BP128 integer compression

---

## Table of Contents

- [Quick Start](#quick-start)
- [Core API](#core-api)
  - [compress](#compress)
  - [compress_into](#compress_into)
  - [max_compressed_size](#max_compressed_size)
  - [decompress](#decompress)
  - [decompress_into](#decompress_into)
  - [decompressed_len](#decompressed_len)
- [Error Handling](#error-handling)
  - [CompressionError](#compressionerror)
  - [DecompressionError](#decompressionerror)
- [Advanced Usage](#advanced-usage)
  - [Zero-Allocation Hot Path](#zero-allocation-hot-path)
  - [Inspecting Compressed Data](#inspecting-compressed-data)
  - [Working with Partial Blocks](#working-with-partial-blocks)
- [Performance Tips](#performance-tips)
- [Binary Format](#binary-format)

---

## Quick Start

Add the dependency to your `Cargo.toml`:

```toml
[dependencies]
simd-bp128 = "0.1"
```

### Basic Example

```rust
use simd_bp128::{compress, decompress};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data: Vec<u32> = (0..1024).map(|i| i % 1000).collect();

    let compressed = compress(&data)?;
    let decompressed = decompress(&compressed)?;

    assert_eq!(data, decompressed);
    println!("{} bytes -> {} bytes ({:.1}%)",
        data.len() * 4,
        compressed.len(),
        compressed.len() as f64 / (data.len() * 4) as f64 * 100.0
    );

    Ok(())
}
```

### Zero-Allocation Example

```rust
use simd_bp128::{compress_into, decompress_into, max_compressed_size, decompressed_len};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data: Vec<u32> = (0..1024).map(|i| i % 1000).collect();

    // Compress into pre-allocated buffer
    let mut cbuf = vec![0u8; max_compressed_size(data.len())];
    let cbytes = compress_into(&data, &mut cbuf)?;
    cbuf.truncate(cbytes);

    // Decompress into pre-allocated buffer
    let dlen = decompressed_len(&cbuf)?;
    let mutdbuf = vec![0u32; dlen];
    let dcount = decompress_into(&cbuf, &mutdbuf)?;

    assert_eq!(&data[..], &dbuf[..dcount]);
    Ok(())
}
```

---

## Core API

### `compress()`

```rust
pub fn compress(input: &[u32]) -> Result<Vec<u8>, Error>
```

Compresses a `u32` slice into a `Vec<u8>` using BP128 encoding. Allocates the output buffer internally (worst-case size, then truncated).

**Parameters:**
- `input` — Slice of `u32` values to compress

**Returns:**
- `Ok(Vec<u8>)` — Compressed bytes
- `Err(Error)` — If `input.len() > u32::MAX`

**Example:**
```rust
use simd_bp128::compress;

let data: Vec<u32> = (0..256).map(|i| i as u32).collect();
let compressed = compress(&data)?;

assert!(compressed.len() < data.len() * 4);
```

**Performance Note:** For hot paths, prefer `compress_into` with a pre-allocated buffer to avoid repeated allocation.

---

### `compress_into()`

```rust
pub fn compress_into(input: &[u32], output: &mut [u8]) -> Result<usize, Error>
```

Compresses into a caller-owned buffer. Zero allocation on the compression side.

**Parameters:**
- `input` — Slice of `u32` values to compress
- `output` — Mutable buffer; must be at least `max_compressed_size(input.len())` bytes

**Returns:**
- `Ok(usize)` — Number of bytes written
- `Err(Error::CompressionError(OutputTooSmall))` — If `output` is too small
- `Err(Error::CompressionError(InputTooLarge))` — If `input.len() > u32::MAX`

**Example:**
```rust
use simd_bp128::{compress_into, max_compressed_size};

let data: Vec<u32> = (0..256).map(|i| i % 1000).collect();
let mut buffer = vec![0u8; max_compressed_size(data.len())];
let bytes_written = compress_into(&data, &mut buffer)?;
buffer.truncate(bytes_written);
```

---

### `max_compressed_size()`

```rust
pub fn max_compressed_size(input_len: usize) -> usize
```

Returns the worst-case compressed size for `input_len` values. Useful for pre-allocating output buffers.

The return value assumes every block uses 32-bit width. Actual compressed size will be equal to or smaller.

**Example:**
```rust
use simd_bp128::max_compressed_size;

let max = max_compressed_size(1024);
println!("Worst case for 1024 values: {} bytes", max); // 522 bytes
```

---

### `decompress()`

```rust
pub fn decompress(input: &[u8]) -> Result<Vec<u32>, DecompressionError>
```

Decompresses BP128-encoded bytes back into a `Vec<u32>`.

**Parameters:**
- `input` — Compressed bytes produced by `compress` or `compress_into`

**Returns:**
- `Ok(Vec<u32>)` — Decompressed values
- `Err(DecompressionError)` — If input is malformed (see error variants below)

**Example:**
```rust
use simd_bp128::{compress, decompress};

let data: Vec<u32> = (0..256).map(|i| i % 1000).collect();
let compressed = compress(&data)?;
let decompressed = decompress(&compressed)?;

assert_eq!(data, decompressed);
```

---

### `decompress_into()`

```rust
pub fn decompress_into(input: &[u8], output: &mut [u32]) -> Result<usize, DecompressionError>
```

Decompresses into a caller-owned buffer. Use `decompressed_len` to size the buffer.

**Parameters:**
- `input` — Compressed bytes
- `output` — Mutable buffer; must be at least `decompressed_len(input)?` elements

**Returns:**
- `Ok(usize)` — Number of `u32` values written
- `Err(DecompressionError)` — If input is malformed or output is too small

**Example:**
```rust
use simd_bp128::{compress, decompressed_len, decompress_into};

let data: Vec<u32> = (0..256).map(|i| i % 1000).collect();
let compressed = compress(&data)?;

let len = decompressed_len(&compressed)?;
let mut output = vec![0u32; len];
let count = decompress_into(&compressed, &mut output)?;

assert_eq!(&data[..], &output[..count]);
```

---

### `decompressed_len()`

```rust
pub fn decompressed_len(input: &[u8]) -> Result<usize, DecompressionError>
```

Returns the number of `u32` values stored in compressed data, without decompressing. Parses only the header (9 bytes + block count).

Use this to pre-allocate output buffers for `decompress_into`.

**Parameters:**
- `input` — Compressed bytes

**Returns:**
- `Ok(usize)` — Original number of `u32` values
- `Err(DecompressionError)` — If the header is malformed

**Example:**
```rust
use simd_bp128::{compress, decompressed_len};

let data: Vec<u32> = (0..256).collect();
let compressed = compress(&data)?;

let len = decompressed_len(&compressed)?;
assert_eq!(len, 256);
```

---

## Error Handling

All operations return `Result`. The library never panics on malformed input.

### `CompressionError`

Returned by `compress` and `compress_into`.

| Variant | Cause |
|---------|-------|
| `InputTooLarge { max, got }` | `input.len() > u32::MAX` |
| `OutputTooSmall { need, got }` | Output buffer smaller than `max_compressed_size()` |

```rust
use simd_bp128::{compress_into, max_compressed_size, CompressionError};

let data = vec![0u32; 100];
let mut small_buf = vec![0u8; 10];

match compress_into(&data, &mut small_buf) {
    Ok(n) => println!("Wrote {} bytes", n),
    Err(e) => eprintln!("Compression failed: {}", e),
}
```

### `DecompressionError`

Returned by `decompress`, `decompress_into`, and `decompressed_len`.

| Variant | Cause |
|---------|-------|
| `HeaderTooSmall { needed, have }` | Input shorter than 9 bytes |
| `UnsupportedVersion { version }` | Version byte is not 1 |
| `InvalidBitWidth { bit_width }` | A block's bit width exceeds 32 |
| `BlockCountMismatch { expected, found }` | Block count doesn't match value count |
| `TruncatedData { position, needed, have }` | Packed data is shorter than expected |
| `InputTooLarge { max, got }` | Decompressed size exceeds 1 billion values |
| `ExcessiveBlockCount { max, got }` | Block count exceeds safe limits |

```rust
use simd_bp128::{decompress, DecompressionError};

let bad_data = vec![0xFF; 5];

match decompress(&bad_data) {
    Ok(data) => println!("Decompressed {} values", data.len()),
    Err(DecompressionError::HeaderTooSmall { needed, have }) => {
        eprintln!("Need {} bytes for header, got {}", needed, have);
    }
    Err(DecompressionError::TruncatedData { position, .. }) => {
        eprintln!("Data truncated at byte {}", position);
    }
    Err(e) => eprintln!("Decompression error: {}", e),
}
```

---

## Advanced Usage

### Zero-Allocation Hot Path

For maximum throughput, avoid all allocation in the hot loop:

```rust
use simd_bp128::{compress_into, decompress_into, max_compressed_size, decompressed_len};

fn process_batch(chunks: &[Vec<u32>]) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    // Pre-allocate worst-case buffers once
    let max_chunk = chunks.iter().map(|c| c.len()).max().unwrap_or(0);
    let mut cbuf = vec![0u8; max_compressed_size(max_chunk)];

    let mut results = Vec::with_capacity(chunks.len());

    for chunk in chunks {
        let bytes = compress_into(chunk, &mut cbuf)?;
        results.push(cbuf[..bytes].to_vec());
    }

    Ok(results)
}
```

### Inspecting Compressed Data

Read the header to understand compressed data without decompressing:

```rust
use simd_bp128::{compress, decompressed_len};

let data: Vec<u32> = (0..400).map(|i| i % 100).collect();
let compressed = compress(&data)?;

// Parse header
let version = compressed[0];
let input_len = u32::from_le_bytes([
    compressed[1], compressed[2], compressed[3], compressed[4]
]);
let num_blocks = u32::from_le_bytes([
    compressed[5], compressed[6], compressed[7], compressed[8]
]);

// Read per-block bit widths
let bit_widths = &compressed[9..9 + num_blocks as usize];

println!("Version: {}", version);
println!("Values: {}", input_len);
println!("Blocks: {}", num_blocks);
println!("Bit widths: {:?}", bit_widths);
```

### Working with Partial Blocks

The last block in a compressed array may contain fewer than 128 values. This is handled transparently by `compress`/`decompress`, but the format encodes it explicitly:

```rust
use simd_bp128::{compress, decompress};

// 200 values = 1 full block (128) + 1 partial block (72)
let data: Vec<u32> = (0..200).map(|i| i as u32).collect();
let compressed = compress(&data)?;
let decompressed = decompress(&compressed)?;

assert_eq!(data.len(), decompressed.len());
assert_eq!(data, decompressed);
```

---

## Performance Tips

### 1. Pre-allocate Buffers

**Avoid:** Repeated allocation in hot loops
```rust
// Slow: allocates Vec<u8> every call
for chunk in chunks {
    let compressed = compress(chunk)?; // new Vec each time
}
```

**Prefer:** Reuse buffers with `compress_into`
```rust
// Fast: reuses buffer
let mut buf = vec![0u8; max_compressed_size(max_chunk)];
for chunk in chunks {
    let bytes = compress_into(chunk, &mut buf)?;
    // use buf[..bytes]
}
```

### 2. Choose Data Patterns Wisely

BP128 compression is most effective when values in each 128-element block use fewer than 32 bits:

| Pattern | Bit Width | Compression Ratio |
|---------|-----------|-------------------|
| All zeros | 0 | ~2% |
| Small values (0-999) | 7-10 | ~25% |
| Timestamps (deltas) | 10-20 | ~40% |
| Random full-range | 32 | ~100% (no savings) |

Sorting or delta-encoding before compression can dramatically improve ratios.

### 3. Understand Bit Width Impact

Performance varies significantly by bit width:

| Bit Width | Compress | Decompress | Notes |
|-----------|----------|------------|-------|
| 0 | O(1) skip | O(1) fill | All-zero blocks are free |
| 1, 2, 4 | Fast | Fast | Power-of-2, SIMD-friendly |
| 8, 16 | Fastest | Fastest | Byte-aligned, best SIMD |
| 24 | Moderate | Moderate | 3 bytes per value |
| 32 | Slowest | Fastest | No compression, memcpy |

### 4. Use `decompressed_len` to Avoid Guessing

```rust
use simd_bp128::{decompressed_len, decompress_into};

// Good: exact size from header
let len = decompressed_len(&compressed)?;
let mut output = vec![0u32; len];
decompress_into(&compressed, &mut output)?;

// Avoid: guessing or using max size
```

---

## Binary Format

### Header (9 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 1 | `version` | Format version (currently 1) |
| 1 | 4 | `input_len` | Original number of `u32` values (LE) |
| 5 | 4 | `num_blocks` | Number of blocks (LE) |

### Block Directory (N bytes)

One byte per block indicating bit width (0-32):
- `0` — All values are zero, no packed data
- `1-32` — Each value uses that many bits

### Packed Data (variable)

Sequential bit-packed blocks. Each block packs 128 values (or fewer for the last block) using the bit width from the block directory.

### Example Layout (256 values, 7-bit)

```
Offset  Content
------  -------
0       0x01                          (version)
1-4     0x00 0x01 0x00 0x00           (input_len = 256)
5-8     0x02 0x00 0x00 0x00           (num_blocks = 2)
9       0x07                          (block 0 bit_width = 7)
10      0x07                          (block 1 bit_width = 7)
11-234  Packed data (2 × 112 bytes)
```

---

## License

MIT License - See LICENSE file for details.

## Links

- [Crates.io](https://crates.io/crates/simd-bp128)
- [Documentation](https://docs.rs/simd-bp128)
- [Repository](https://github.com/themankindproject/simd-bp128)
