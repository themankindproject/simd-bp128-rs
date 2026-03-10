//! Comprehensive throughput benchmark for BP128 compression
//!
//! Run with: cargo bench --bench throughput_comparison
//!
//! Expected results on modern x86_64 (SSE4.1 capable):
//! - 32-bit: 10-15 GB/s
//! - 16-bit: 8-12 GB/s  
//! - 8-bit: 8-12 GB/s
//! - 4-bit: 4-6 GB/s
//! - 1-bit: 4-6 GB/s
//! - Odd widths: 3-5 GB/s

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use simd_bp128::internal::{ScalarBackend, SimdBackend};
use std::hint::black_box;
use std::time::Duration;

#[cfg(target_arch = "x86_64")]
use simd_bp128::internal::SseBackend;

fn generate_block(bits: u32) -> [u32; 128] {
    let mut rng = StdRng::seed_from_u64(42);
    let mask = if bits == 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    let mut block = [0u32; 128];
    for v in &mut block {
        *v = rng.gen::<u32>() & mask;
    }
    block
}

fn packed_bytes(bits: u32) -> usize {
    (128 * bits as usize + 7) / 8
}

fn benchmark_scalar_vs_sse(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_vs_sse");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(100);

    // Test different bit widths
    for bits in [1u8, 2, 4, 8, 16, 24, 32] {
        let block = generate_block(bits as u32);
        let bytes_needed = packed_bytes(bits as u32);
        let input_bytes = 128 * 4; // 128 u32 values

        let mut scalar_packed = vec![0u8; bytes_needed];
        let mut scalar_unpacked = [0u32; 128];
        let mut sse_packed = vec![0u8; bytes_needed];
        let mut sse_unpacked = [0u32; 128];

        // Warm up
        ScalarBackend::pack_block(&block, bits, &mut scalar_packed).unwrap();
        #[cfg(target_arch = "x86_64")]
        {
            SseBackend::pack_block(&block, bits, &mut sse_packed).unwrap();
        }

        // Scalar pack
        group.throughput(Throughput::Bytes(input_bytes as u64));
        group.bench_function(format!("scalar_pack_{}bit", bits), |b| {
            b.iter(|| {
                ScalarBackend::pack_block(
                    black_box(&block),
                    black_box(bits),
                    black_box(&mut scalar_packed),
                )
                .unwrap();
            });
        });

        // SSE pack (only on x86_64)
        #[cfg(target_arch = "x86_64")]
        {
            group.throughput(Throughput::Bytes(input_bytes as u64));
            group.bench_function(format!("sse_pack_{}bit", bits), |b| {
                b.iter(|| {
                    SseBackend::pack_block(
                        black_box(&block),
                        black_box(bits),
                        black_box(&mut sse_packed),
                    )
                    .unwrap();
                });
            });
        }

        // Scalar unpack
        group.throughput(Throughput::Bytes(bytes_needed as u64));
        group.bench_function(format!("scalar_unpack_{}bit", bits), |b| {
            b.iter(|| {
                ScalarBackend::unpack_block(
                    black_box(&scalar_packed),
                    black_box(bits),
                    black_box(&mut scalar_unpacked),
                )
                .unwrap();
            });
        });

        // SSE unpack (only on x86_64)
        #[cfg(target_arch = "x86_64")]
        {
            group.throughput(Throughput::Bytes(bytes_needed as u64));
            group.bench_function(format!("sse_unpack_{}bit", bits), |b| {
                b.iter(|| {
                    SseBackend::unpack_block(
                        black_box(&sse_packed),
                        black_box(bits),
                        black_box(&mut sse_unpacked),
                    )
                    .unwrap();
                });
            });
        }
    }

    group.finish();
}

fn benchmark_full_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_compression");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [
        ("1K", 1_000usize),
        ("10K", 10_000usize),
        ("100K", 100_000usize),
        ("1M", 1_000_000usize),
    ];

    for bits in [8u32, 16, 32] {
        for (name, size) in &sizes {
            let mut rng = StdRng::seed_from_u64(42);
            let mask = if bits == 32 {
                u32::MAX
            } else {
                (1u32 << bits) - 1
            };
            let data: Vec<u32> = (0..*size).map(|_| rng.gen::<u32>() & mask).collect();
            let input_bytes = (data.len() * 4) as u64;

            use simd_bp128::{compress, decompress};
            let compressed = compress(&data).expect("Compression failed");
            let compressed_bytes = compressed.len() as u64;

            group.throughput(Throughput::Bytes(input_bytes));
            group.bench_function(format!("compress_{}_{}bit", name, bits), |b| {
                b.iter(|| {
                    let result = compress(black_box(&data)).expect("Compression failed");
                    let _ = black_box(result);
                });
            });

            group.throughput(Throughput::Bytes(compressed_bytes));
            group.bench_function(format!("decompress_{}_{}bit", name, bits), |b| {
                b.iter(|| {
                    let result = decompress(black_box(&compressed)).expect("Decompression failed");
                    let _ = black_box(result);
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, benchmark_scalar_vs_sse, benchmark_full_compression);
criterion_main!(benches);
