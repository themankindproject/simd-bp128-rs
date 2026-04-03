use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use packsimd::{compress, decompress};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;

fn generate_data_bits(len: usize, bits: u32, seed: u64) -> Vec<u32> {
    let mut rng: StdRng = StdRng::seed_from_u64(seed);
    let mask = if bits == 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    (0..len).map(|_| rng.gen::<u32>() & mask).collect()
}

fn benchmark_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");

    let sizes = [128usize, 1024, 10_240, 102_400, 1_000_000, 10_000_000];
    let bit_widths = [1u32, 8, 16, 24, 32];

    for &size in &sizes {
        for &bits in &bit_widths {
            let seed = (size as u64) * 1000 + (bits as u64);
            let data: Vec<u32> = generate_data_bits(size, bits, seed);
            let compressed: Vec<u8> = compress(&data).expect("Compression failed");

            let original_bytes = (size * 4) as u64;

            group.throughput(Throughput::Bytes(original_bytes));
            group.bench_function(format!("compress_{}bit_{}", bits, size), |b| {
                b.iter(|| {
                    let result = compress(black_box(&data)).expect("Compression failed");
                    let _ = black_box(result);
                });
            });

            group.throughput(Throughput::Bytes(original_bytes));
            group.bench_function(format!("decompress_{}bit_{}", bits, size), |b| {
                b.iter(|| {
                    let result = decompress(black_box(&compressed)).expect("Decompression failed");
                    let _ = black_box(result);
                });
            });
        }
    }

    group.finish();
}

fn benchmark_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline");

    let sizes = [(1_000_000, "1M"), (10_000_000, "10M")];

    for (size, name) in sizes {
        let data = vec![0u8; size];

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_function(format!("memcpy_{}", name), |b| {
            let mut dest = vec![0u8; size];
            b.iter(|| {
                dest.copy_from_slice(&data);
            });
        });

        group.bench_function(format!("memset_{}", name), |b| {
            let mut dest = vec![0u8; size];
            b.iter(|| {
                dest.fill(0);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_compression, benchmark_baseline);
criterion_main!(benches);
