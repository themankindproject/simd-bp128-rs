use simd_bp128::{compress, decompress, DecompressionError, Error};

fn main() {
    basic_example();
    patterns_example();
    error_handling_example();
    large_values_example();
    multiple_blocks_example();
}

fn basic_example() {
    let data: Vec<u32> = (0..128).map(|i| (i % 100) as u32).collect();

    let compressed: Vec<u8> = compress(&data).expect("Compression failed");

    println!("compressed bytes:");
    for b in &compressed {
        print!("{:02x} ", b);
    }
    println!();

    let decompressed: Vec<u32> = decompress(&compressed).expect("Decompression failed");
    assert_eq!(data, decompressed);
}

fn patterns_example() {
    let zeros: Vec<u32> = vec![0u32; 128];
    let small: Vec<u32> = (0..128).map(|i| (i % 256) as u32).collect();
    let large: Vec<u32> = (0..128).map(|i| i as u32 * 10_000_000).collect();

    let cz = compress(&zeros).expect("Compression failed");
    let cs = compress(&small).expect("Compression failed");
    let cl = compress(&large).expect("Compression failed");

    assert_eq!(decompress(&cz).unwrap(), zeros);
    assert_eq!(decompress(&cs).unwrap(), small);
    assert_eq!(decompress(&cl).unwrap(), large);

    println!(
        "Patterns: zeros={}, small={}, large={}",
        cz.len(),
        cs.len(),
        cl.len()
    );
}

fn error_handling_example() {
    let mut truncated = vec![];
    truncated.push(1);
    truncated.extend_from_slice(&1u32.to_le_bytes());
    truncated.extend_from_slice(&1u32.to_le_bytes());
    truncated.push(32);

    let result = decompress(&truncated);
    assert!(matches!(
        result,
        Err(Error::DecompressionError(
            DecompressionError::TruncatedData { .. }
        ))
    ));

    let mut invalid: Vec<u8> = vec![];
    invalid.push(1);
    invalid.extend_from_slice(&1u32.to_le_bytes());
    invalid.extend_from_slice(&1u32.to_le_bytes());
    invalid.push(33);

    let result = decompress(&invalid);
    assert!(matches!(
        result,
        Err(Error::DecompressionError(
            DecompressionError::InvalidBitWidth { bit_width: 33 }
        ))
    ));

    println!("Errors: OK");
}

fn large_values_example() {
    let data = vec![u32::MAX; 128];
    let compressed = compress(&data).expect("Compression failed");
    assert_eq!(decompress(&compressed).unwrap(), data);
    println!("32-bit: {} bytes", compressed.len());
}

fn multiple_blocks_example() {
    let data: Vec<u32> = (0..400).map(|i| (i % 100) as u32).collect();
    let compressed = compress(&data).expect("Compression failed");
    let ratio = compressed.len() as f64 / (data.len() * 4) as f64;

    assert_eq!(decompress(&compressed).unwrap(), data);
    println!(
        "Multi-block: {} bytes ({:.1}%)",
        compressed.len(),
        ratio * 100.0
    );
}
