use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use simd_bp128::{compress, decompress};
use std::hint::black_box;

fn generate_data_bits(len: usize, bits: u32) -> Vec<u32> {
    let mut rng: StdRng = StdRng::seed_from_u64(42);

    let mask = if bits == 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };

    (0..len).map(|_| rng.gen::<u32>() & mask).collect()
}

fn benchmark_compression(c: &mut Criterion) {
    let mut group: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("compression");

    let sizes = [128, 1024, 10_240, 102_400];
    let bit_widths = [1, 8, 16, 24, 32];

    for &size in &sizes {
        for &bits in &bit_widths {
            let data: Vec<u32> = generate_data_bits(size, bits);
            let compressed: Vec<u8> = compress(&data).expect("Compression failed");

            group.throughput(Throughput::Bytes((size * 4) as u64));

            group.bench_function(format!("compress_{}bit_{}", bits, size), |b| {
                b.iter(|| {
                    let result = compress(black_box(&data)).expect("Compression failed");
                    let _ = black_box(result);
                });
            });

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

criterion_group!(benches, benchmark_compression);
criterion_main!(benches);
