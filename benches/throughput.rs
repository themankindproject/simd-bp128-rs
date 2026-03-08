use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use simd_bp128::{compress, decompress};
use std::hint::black_box;

fn generate_data_bits(len: usize, bits: u32) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(42);
    let mask = if bits == 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    (0..len).map(|_| rng.gen::<u32>() & mask).collect()
}

fn benchmark_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    let size = 1_000_000usize;
    let data = generate_data_bits(size, 16);
    let compressed = compress(&data).expect("Compression failed");

    let original_bytes = (size * 4) as u64;
    let compressed_bytes = compressed.len() as u64;

    println!(
        "\nCompression ratio: {:.2}% ({} bytes -> {} bytes)",
        (compressed_bytes as f64 / original_bytes as f64) * 100.0,
        original_bytes,
        compressed_bytes,
    );

    group.throughput(Throughput::Bytes(original_bytes));
    group.bench_function("compress_1M_16bit", |b| {
        b.iter(|| {
            let result = compress(black_box(&data)).expect("Compression failed");
            let _ = black_box(result);
        });
    });

    group.throughput(Throughput::Bytes(compressed_bytes));
    group.bench_function("decompress_1M_16bit", |b| {
        b.iter(|| {
            let result = decompress(black_box(&compressed)).expect("Decompression failed");
            let _ = black_box(result);
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_throughput);
criterion_main!(benches);
