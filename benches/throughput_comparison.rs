//! Scalar vs SSE block-level kernel comparison.
//!
//! Run with: cargo bench --bench throughput_comparison
//!
//! This benchmark isolates pack/unpack kernels to measure SIMD speedup
//! independent of allocation and format overhead.

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

    for bits in [1u8, 2, 4, 8, 16, 24, 32] {
        let block = generate_block(bits as u32);
        let bytes_needed = packed_bytes(bits as u32);
        let input_bytes = 128 * 4;

        let mut scalar_packed = vec![0u8; bytes_needed];
        let mut scalar_unpacked = [0u32; 128];
        let mut sse_packed = vec![0u8; bytes_needed];
        let mut sse_unpacked = [0u32; 128];

        ScalarBackend::pack_block(&block, bits, &mut scalar_packed).unwrap();
        #[cfg(target_arch = "x86_64")]
        SseBackend::pack_block(&block, bits, &mut sse_packed).unwrap();

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

criterion_group!(benches, benchmark_scalar_vs_sse);
criterion_main!(benches);
