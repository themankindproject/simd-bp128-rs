use criterion::{criterion_group, criterion_main, Criterion};
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

fn benchmark_block_level(c: &mut Criterion) {
    // Single block benchmarks (128 values)
    for bits in [1, 8, 16, 24, 32] {
        let data = generate_data_bits(128, bits);
        let compressed = compress(&data).expect("Compression failed");

        c.bench_function(&format!("compress_block_128_{}bit", bits), |b| {
            b.iter(|| {
                let result = compress(black_box(&data)).expect("Compression failed");
                let _ = black_box(result);
            });
        });

        c.bench_function(&format!("decompress_block_128_{}bit", bits), |b| {
            b.iter(|| {
                let result = decompress(black_box(&compressed)).expect("Decompression failed");
                let _ = black_box(result);
            });
        });
    }

    // Partial block benchmarks (64 values)
    for bits in [8, 16] {
        let data = generate_data_bits(64, bits);
        let compressed = compress(&data).expect("Compression failed");

        c.bench_function(&format!("compress_partial_64_{}bit", bits), |b| {
            b.iter(|| {
                let result = compress(black_box(&data)).expect("Compression failed");
                let _ = black_box(result);
            });
        });

        c.bench_function(&format!("decompress_partial_64_{}bit", bits), |b| {
            b.iter(|| {
                let result = decompress(black_box(&compressed)).expect("Decompression failed");
                let _ = black_box(result);
            });
        });
    }
}

criterion_group!(benches, benchmark_block_level);
criterion_main!(benches);
