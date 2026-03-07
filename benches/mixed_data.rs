use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use simd_bp128::{compress, decompress};
use std::hint::black_box;

fn benchmark_mixed_data(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_data");

    let size = 10_240;

    // sequential pattern (common in posting lists)
    let sequential: Vec<u32> = (0..size).map(|i| (i % 256) as u32).collect();

    // random data
    let mut rng = StdRng::seed_from_u64(42);
    let random: Vec<u32> = (0..size).map(|_| rng.gen()).collect();

    // constant values
    let constant = vec![42u32; size];

    group.throughput(Throughput::Bytes((size * 4) as u64));

    for (name, data) in [
        ("sequential", sequential),
        ("random", random),
        ("constant", constant),
    ] {
        let compressed = compress(&data).expect("Compression failed");

        let ratio = compressed.len() as f64 / (data.len() * 4) as f64;

        println!("\n{} compression ratio: {:.2}%", name, ratio * 100.0);

        group.bench_function(format!("compress_{}", name), |b| {
            b.iter(|| {
                let result = compress(black_box(&data)).expect("Compression failed");
                let _ = black_box(result);
            });
        });

        group.bench_function(format!("decompress_{}", name), |b| {
            b.iter(|| {
                let result = decompress(black_box(&compressed)).expect("Decompression failed");
                let _ = black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_mixed_data);
criterion_main!(benches);
