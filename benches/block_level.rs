use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use simd_bp128::internal::{ScalarBackend, SimdBackend};
use std::hint::black_box;

fn make_block(bits: u32) -> [u32; 128] {
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

fn benchmark_block_level(c: &mut Criterion) {
    for bits in [1u32, 8, 16, 24, 32] {
        let block = make_block(bits);
        let n = packed_bytes(bits);

        let mut packed = vec![0u8; n];
        let mut unpacked = [0u32; 128];

        ScalarBackend::pack_block(&block, bits as u8, &mut packed)
            .expect("pack_block warm-up failed");

        let input_bytes = 128 * 4;
        let mut group = c.benchmark_group(format!("pack_block_128_{}bit", bits));
        group.throughput(Throughput::Bytes(input_bytes as u64));
        group.bench_function("pack", |b| {
            b.iter(|| {
                ScalarBackend::pack_block(
                    black_box(&block),
                    black_box(bits as u8),
                    black_box(&mut packed),
                )
                .expect("pack_block failed");
            });
        });
        group.finish();

        let mut group = c.benchmark_group(format!("unpack_block_128_{}bit", bits));
        group.throughput(Throughput::Bytes(n as u64));
        group.bench_function("unpack", |b| {
            b.iter(|| {
                ScalarBackend::unpack_block(
                    black_box(&packed),
                    black_box(bits as u8),
                    black_box(&mut unpacked),
                )
                .expect("unpack_block failed");
            });
        });
        group.finish();
    }

    for bits in [8u32, 16] {
        let count = 64usize;
        let mut rng = StdRng::seed_from_u64(42);
        let mask = (1u32 << bits) - 1;
        let partial: Vec<u32> = (0..count).map(|_| rng.gen::<u32>() & mask).collect();

        let n = (count * bits as usize + 7) / 8;
        let mut packed = vec![0u8; n];
        let mut unpacked = vec![0u32; count];

        ScalarBackend::pack_partial_block(&partial, bits as u8, &mut packed)
            .expect("pack_partial_block warm-up failed");

        let partial_input_bytes = count * 4;

        let mut group = c.benchmark_group(format!("pack_partial_block_64_{}bit", bits));
        group.throughput(Throughput::Bytes(partial_input_bytes as u64));
        group.bench_function("pack", |b| {
            b.iter(|| {
                ScalarBackend::pack_partial_block(
                    black_box(&partial),
                    black_box(bits as u8),
                    black_box(&mut packed),
                )
                .expect("pack_partial_block failed");
            });
        });
        group.finish();

        let mut group = c.benchmark_group(format!("unpack_partial_block_64_{}bit", bits));
        group.throughput(Throughput::Bytes(n as u64));
        group.bench_function("unpack", |b| {
            b.iter(|| {
                ScalarBackend::unpack_partial_block(
                    black_box(&packed),
                    black_box(bits as u8),
                    black_box(count),
                    black_box(&mut unpacked),
                )
                .expect("unpack_partial_block failed");
            });
        });
        group.finish();
    }
}

criterion_group!(benches, benchmark_block_level);
criterion_main!(benches);
