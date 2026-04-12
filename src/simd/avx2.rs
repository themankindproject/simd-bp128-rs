//! High-performance AVX2 implementation of BP128 bit-packing kernels.
//!
//! # Performance Characteristics
//!
//! Specializes the bit widths where 256-bit operations measurably beat
//! SSE4.1, validated by `benches/throughput_comparison.rs`. All other
//! widths delegate to [`SseBackend`], preserving byte-identical output
//! across backends.
//!
//! ## Specialization map
//!
//! | Width | Pack         | Unpack       |
//! |------:|:-------------|:-------------|
//! |     1 | AVX2 native  | delegate SSE |
//! |     8 | AVX2 native  | AVX2 native  |
//! |    16 | AVX2 native  | AVX2 native  |
//! |    32 | AVX2 native  | AVX2 native  |
//! | other | delegate SSE | delegate SSE |
//!
//! ## Optimization Strategies
//!
//! 1. **Width 32**: raw `_mm256_loadu_si256` / `_mm256_storeu_si256`,
//!    halving the loop count vs SSE.
//! 2. **Widths 8 / 16**: `_mm256_packus_epi32` / `_mm256_packus_epi16`
//!    chains followed by `_mm256_permute4x64_epi64::<0xD8>` to undo the
//!    per-lane interleaving.
//! 3. **Width 1 (pack)**: `_mm256_movemask_epi8` extracts 32 bits per
//!    iteration vs 16 in SSE.
//! 4. **All other widths** (2, 3, 4 unpack, 5–7, 9–15, 17–31): delegate to
//!    `SseBackend`. The hybrid SIMD-load + scalar-accumulator paths are
//!    bottlenecked by the scalar accumulator and gain nothing from wider loads.
//!    The 4-bit unpack chain was implemented and benchmarked but reverted
//!    because the four-level AVX2 expansion ran ~31% slower than SSE.
//!
//! ## Lane-crossing
//!
//! AVX2 pack/unpack ops operate per 128-bit lane independently. Every
//! `_mm256_packus_epi*` is followed by `_mm256_permute4x64_epi64::<0xD8>`
//! to restore linear in-memory order, which is required for byte-for-byte
//! compatibility with SSE/scalar output.
//!
//! # Safety
//!
//! All unsafe functions validate input/output buffer bounds before accessing memory.
//! The AVX2 feature must be checked at runtime before calling these functions
//! (handled automatically by the dispatch module).

use crate::error::Error;
use crate::simd::SimdBackend;

#[cfg(target_arch = "x86_64")]
use crate::simd::sse::SseBackend;

/// Number of values in a full block.
const BLOCK_SIZE: usize = 128;

/// Maximum valid bit width.
const MAX_BIT_WIDTH: u8 = 32;

/// AVX2 implementation of the BP128 bit-packing backend.
///
/// Specializes byte-aligned widths and 1-bit packing using 256-bit operations,
/// delegating non-specialized widths to [`SseBackend`].
#[must_use]
pub struct Avx2Backend;

impl SimdBackend for Avx2Backend {
    fn pack_block(input: &[u32; 128], bit_width: u8, output: &mut [u8]) -> Result<(), Error> {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            avx2_impl::pack_block_avx2(input, bit_width, output)
        }
        #[cfg(not(target_arch = "x86_64"))]
        crate::simd::scalar::ScalarBackend::pack_block(input, bit_width, output)
    }

    fn unpack_block(input: &[u8], bit_width: u8, output: &mut [u32; 128]) -> Result<(), Error> {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            avx2_impl::unpack_block_avx2(input, bit_width, output)
        }
        #[cfg(not(target_arch = "x86_64"))]
        crate::simd::scalar::ScalarBackend::unpack_block(input, bit_width, output)
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2_impl {
    use super::*;
    use std::arch::x86_64::*;

    /// Pack a 128-value block using AVX2 instructions.
    ///
    /// Dispatches to specialized implementations for the bit widths that
    /// benefit from 256-bit operations and delegates the rest to
    /// [`SseBackend::pack_block`].
    ///
    /// # Safety
    ///
    /// This function uses AVX2 intrinsics that require the `avx2` CPU feature.
    /// The caller must ensure this feature is available at runtime (handled by dispatch).
    /// Input/output buffers must be properly sized:
    /// - `input` must contain exactly 128 u32 values
    /// - `output` must have at least `(128 * bit_width + 7) / 8` bytes
    ///
    /// Violating these invariants would cause undefined behavior via out-of-bounds memory access.
    #[target_feature(enable = "avx2")]
    pub unsafe fn pack_block_avx2(
        input: &[u32; 128],
        bit_width: u8,
        output: &mut [u8],
    ) -> Result<(), Error> {
        if bit_width > MAX_BIT_WIDTH {
            return Err(Error::InvalidBitWidth(bit_width));
        }

        if bit_width == 0 {
            return Ok(());
        }

        let required_bytes = (BLOCK_SIZE * bit_width as usize + 7) / 8;
        if output.len() < required_bytes {
            return Err(Error::OutputTooSmall {
                need: required_bytes,
                got: output.len(),
            });
        }

        match bit_width {
            1 => pack_1bit_avx2(input, output),
            8 => pack_8bit_avx2(input, output),
            16 => pack_16bit_avx2(input, output),
            32 => pack_32bit_avx2(input, output),
            // All other widths defer to the SSE kernel. Calls SseBackend
            // directly (NOT crate::dispatch) to avoid infinite recursion
            // on hosts where dispatch resolves to Avx2Backend.
            _ => SseBackend::pack_block(input, bit_width, output),
        }
    }

    /// Unpack a 128-value block using AVX2 instructions.
    ///
    /// # Safety
    ///
    /// Same invariants as [`pack_block_avx2`], with input/output roles reversed:
    /// `input` must have at least `(128 * bit_width + 7) / 8` bytes.
    #[target_feature(enable = "avx2")]
    pub unsafe fn unpack_block_avx2(
        input: &[u8],
        bit_width: u8,
        output: &mut [u32; 128],
    ) -> Result<(), Error> {
        if bit_width > MAX_BIT_WIDTH {
            return Err(Error::InvalidBitWidth(bit_width));
        }

        if bit_width == 0 {
            output.fill(0);
            return Ok(());
        }

        let required_bytes = (BLOCK_SIZE * bit_width as usize + 7) / 8;
        if input.len() < required_bytes {
            return Err(Error::InputTooShort {
                need: required_bytes,
                got: input.len(),
            });
        }

        match bit_width {
            // unpack_4 is delegated to SSE: benchmarks showed an AVX2
            // four-level unpack chain ran ~31% slower than the SSE
            // two-level chain (more permutes per byte produced).
            8 => unpack_8bit_avx2(input, output),
            16 => unpack_16bit_avx2(input, output),
            32 => unpack_32bit_avx2(input, output),
            _ => SseBackend::unpack_block(input, bit_width, output),
        }
    }

    // ============================ Pack helpers ============================

    /// Pack 128 32-bit values via raw 256-bit load/store.
    #[inline]
    unsafe fn pack_32bit_avx2(input: &[u32; 128], output: &mut [u8]) -> Result<(), Error> {
        for i in 0..16 {
            let v = _mm256_loadu_si256(input.as_ptr().add(i * 8) as *const __m256i);
            _mm256_storeu_si256(output.as_mut_ptr().add(i * 32) as *mut __m256i, v);
        }
        Ok(())
    }

    /// Pack 128 16-bit values via `_mm256_packus_epi32` + lane unscramble.
    #[inline]
    unsafe fn pack_16bit_avx2(input: &[u32; 128], output: &mut [u8]) -> Result<(), Error> {
        for i in 0..8 {
            let in_ptr = input.as_ptr().add(i * 16);
            let out_ptr = output.as_mut_ptr().add(i * 32);

            let v0 = _mm256_loadu_si256(in_ptr as *const __m256i);
            let v1 = _mm256_loadu_si256(in_ptr.add(8) as *const __m256i);

            // packus_epi32 is per-lane, so the 64-bit qwords end up as
            // [v0.lo, v1.lo, v0.hi, v1.hi]. permute4x64::<0xD8> rearranges
            // them to [v0.lo, v0.hi, v1.lo, v1.hi] = linear u16[0..16].
            let packed = _mm256_packus_epi32(v0, v1);
            let ordered = _mm256_permute4x64_epi64::<0xD8>(packed);

            _mm256_storeu_si256(out_ptr as *mut __m256i, ordered);
        }
        Ok(())
    }

    /// Pack 128 8-bit values via two-level packus chain + lane unscrambles.
    #[inline]
    unsafe fn pack_8bit_avx2(input: &[u32; 128], output: &mut [u8]) -> Result<(), Error> {
        for i in 0..4 {
            let in_ptr = input.as_ptr().add(i * 32);
            let out_ptr = output.as_mut_ptr().add(i * 32);

            let v0 = _mm256_loadu_si256(in_ptr as *const __m256i);
            let v1 = _mm256_loadu_si256(in_ptr.add(8) as *const __m256i);
            let v2 = _mm256_loadu_si256(in_ptr.add(16) as *const __m256i);
            let v3 = _mm256_loadu_si256(in_ptr.add(24) as *const __m256i);

            // Level 1: u32 -> u16 with per-lane unscramble.
            let p01 = _mm256_packus_epi32(v0, v1);
            let p01 = _mm256_permute4x64_epi64::<0xD8>(p01);
            let p23 = _mm256_packus_epi32(v2, v3);
            let p23 = _mm256_permute4x64_epi64::<0xD8>(p23);

            // Level 2: u16 -> u8 with final per-lane unscramble.
            let packed = _mm256_packus_epi16(p01, p23);
            let ordered = _mm256_permute4x64_epi64::<0xD8>(packed);

            _mm256_storeu_si256(out_ptr as *mut __m256i, ordered);
        }
        Ok(())
    }

    /// Pack 128 1-bit values via `_mm256_movemask_epi8`, 32 bits per iteration.
    #[inline]
    unsafe fn pack_1bit_avx2(input: &[u32; 128], output: &mut [u8]) -> Result<(), Error> {
        let mask1 = _mm256_set1_epi32(1);

        for i in 0..4 {
            let in_ptr = input.as_ptr().add(i * 32);

            let v0 = _mm256_loadu_si256(in_ptr as *const __m256i);
            let v1 = _mm256_loadu_si256(in_ptr.add(8) as *const __m256i);
            let v2 = _mm256_loadu_si256(in_ptr.add(16) as *const __m256i);
            let v3 = _mm256_loadu_si256(in_ptr.add(24) as *const __m256i);

            let m0 = _mm256_and_si256(v0, mask1);
            let m1 = _mm256_and_si256(v1, mask1);
            let m2 = _mm256_and_si256(v2, mask1);
            let m3 = _mm256_and_si256(v3, mask1);

            let c0 = _mm256_cmpeq_epi32(m0, mask1);
            let c1 = _mm256_cmpeq_epi32(m1, mask1);
            let c2 = _mm256_cmpeq_epi32(m2, mask1);
            let c3 = _mm256_cmpeq_epi32(m3, mask1);

            // Same per-lane packs+permute pattern as pack_8bit.
            let p01 = _mm256_packs_epi32(c0, c1);
            let p01 = _mm256_permute4x64_epi64::<0xD8>(p01);
            let p23 = _mm256_packs_epi32(c2, c3);
            let p23 = _mm256_permute4x64_epi64::<0xD8>(p23);

            let bytes = _mm256_packs_epi16(p01, p23);
            let bytes = _mm256_permute4x64_epi64::<0xD8>(bytes);

            // Each byte is now 0xFF (LSB was set) or 0x00. movemask_epi8
            // picks the MSB of each byte, giving a 32-bit mask. The
            // little-endian u32 store of mask is byte-identical to two
            // consecutive SSE iterations of (mask & 0xFF, mask >> 8).
            let mask = _mm256_movemask_epi8(bytes) as u32;
            (output.as_mut_ptr().add(i * 4) as *mut u32).write_unaligned(mask);
        }
        Ok(())
    }

    // =========================== Unpack helpers ===========================

    /// Unpack 128 32-bit values via raw 256-bit load/store.
    #[inline]
    unsafe fn unpack_32bit_avx2(input: &[u8], output: &mut [u32; 128]) -> Result<(), Error> {
        for i in 0..16 {
            let v = _mm256_loadu_si256(input.as_ptr().add(i * 32) as *const __m256i);
            _mm256_storeu_si256(output.as_mut_ptr().add(i * 8) as *mut __m256i, v);
        }
        Ok(())
    }

    /// Unpack 128 16-bit values via `_mm256_unpacklo/hi_epi16` + `permute2x128`.
    #[inline]
    unsafe fn unpack_16bit_avx2(input: &[u8], output: &mut [u32; 128]) -> Result<(), Error> {
        let zeros = _mm256_setzero_si256();
        for i in 0..8 {
            let in_ptr = input.as_ptr().add(i * 32);
            let out_ptr = output.as_mut_ptr().add(i * 16);

            let bytes = _mm256_loadu_si256(in_ptr as *const __m256i);

            // u16 -> u32, per-lane:
            //   lo.lane0 = u32 of u16[0..4],  lo.lane1 = u32 of u16[8..12]
            //   hi.lane0 = u32 of u16[4..8],  hi.lane1 = u32 of u16[12..16]
            let lo = _mm256_unpacklo_epi16(bytes, zeros);
            let hi = _mm256_unpackhi_epi16(bytes, zeros);

            // Recombine lanes for linear order.
            let out0_7 = _mm256_permute2x128_si256::<0x20>(lo, hi);
            let out8_15 = _mm256_permute2x128_si256::<0x31>(lo, hi);

            _mm256_storeu_si256(out_ptr as *mut __m256i, out0_7);
            _mm256_storeu_si256(out_ptr.add(8) as *mut __m256i, out8_15);
        }
        Ok(())
    }

    /// Unpack 128 8-bit values via `_mm256_unpacklo/hi_epi8` chains + `permute2x128`.
    #[inline]
    unsafe fn unpack_8bit_avx2(input: &[u8], output: &mut [u32; 128]) -> Result<(), Error> {
        let zeros = _mm256_setzero_si256();
        for i in 0..4 {
            let in_ptr = input.as_ptr().add(i * 32);
            let out_ptr = output.as_mut_ptr().add(i * 32);

            let bytes = _mm256_loadu_si256(in_ptr as *const __m256i);

            // u8 -> u16, per-lane:
            //   lo16.lane0 = u16 of bytes[0..8],  lo16.lane1 = u16 of bytes[16..24]
            //   hi16.lane0 = u16 of bytes[8..16], hi16.lane1 = u16 of bytes[24..32]
            let lo16 = _mm256_unpacklo_epi8(bytes, zeros);
            let hi16 = _mm256_unpackhi_epi8(bytes, zeros);

            // u16 -> u32, per-lane.
            let exp_lo_lo = _mm256_unpacklo_epi16(lo16, zeros); // bytes[0..4]   | bytes[16..20]
            let exp_lo_hi = _mm256_unpackhi_epi16(lo16, zeros); // bytes[4..8]   | bytes[20..24]
            let exp_hi_lo = _mm256_unpacklo_epi16(hi16, zeros); // bytes[8..12]  | bytes[24..28]
            let exp_hi_hi = _mm256_unpackhi_epi16(hi16, zeros); // bytes[12..16] | bytes[28..32]

            // Recombine lanes for linear order.
            let s0 = _mm256_permute2x128_si256::<0x20>(exp_lo_lo, exp_lo_hi); // u32[0..8]
            let s1 = _mm256_permute2x128_si256::<0x20>(exp_hi_lo, exp_hi_hi); // u32[8..16]
            let s2 = _mm256_permute2x128_si256::<0x31>(exp_lo_lo, exp_lo_hi); // u32[16..24]
            let s3 = _mm256_permute2x128_si256::<0x31>(exp_hi_lo, exp_hi_hi); // u32[24..32]

            _mm256_storeu_si256(out_ptr as *mut __m256i, s0);
            _mm256_storeu_si256(out_ptr.add(8) as *mut __m256i, s1);
            _mm256_storeu_si256(out_ptr.add(16) as *mut __m256i, s2);
            _mm256_storeu_si256(out_ptr.add(24) as *mut __m256i, s3);
        }
        Ok(())
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::simd::scalar::ScalarBackend;

    fn random_input(bit_width: u8) -> [u32; 128] {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mask = if bit_width == 0 {
            0
        } else if bit_width >= 32 {
            u32::MAX
        } else {
            (1u32 << bit_width) - 1
        };
        std::array::from_fn(|_| rng.gen::<u32>() & mask)
    }

    #[test]
    fn test_avx2_pack_matches_sse_byte_for_byte() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 not available, skipping");
            return;
        }
        for _ in 0..10 {
            for bit_width in 0..=MAX_BIT_WIDTH {
                let input = random_input(bit_width);
                let len = ((BLOCK_SIZE * bit_width as usize + 7) / 8).max(1);
                let mut sse_packed = vec![0u8; len];
                let mut avx2_packed = vec![0u8; len];

                SseBackend::pack_block(&input, bit_width, &mut sse_packed).unwrap();
                Avx2Backend::pack_block(&input, bit_width, &mut avx2_packed).unwrap();

                assert_eq!(
                    sse_packed, avx2_packed,
                    "pack mismatch at bit_width={bit_width}"
                );
            }
        }
    }

    #[test]
    fn test_avx2_unpack_matches_sse() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 not available, skipping");
            return;
        }
        for _ in 0..10 {
            for bit_width in 0..=MAX_BIT_WIDTH {
                let input = random_input(bit_width);
                let len = ((BLOCK_SIZE * bit_width as usize + 7) / 8).max(1);
                let mut packed = vec![0u8; len];
                SseBackend::pack_block(&input, bit_width, &mut packed).unwrap();

                let mut sse_out = [0u32; 128];
                let mut avx2_out = [0u32; 128];
                SseBackend::unpack_block(&packed, bit_width, &mut sse_out).unwrap();
                Avx2Backend::unpack_block(&packed, bit_width, &mut avx2_out).unwrap();

                assert_eq!(
                    sse_out, avx2_out,
                    "unpack mismatch at bit_width={bit_width}"
                );
            }
        }
    }

    #[test]
    fn test_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 not available, skipping");
            return;
        }
        for bit_width in 0..=MAX_BIT_WIDTH {
            let input = random_input(bit_width);
            let len = ((BLOCK_SIZE * bit_width as usize + 7) / 8).max(1);
            let mut scalar_packed = vec![0u8; len];
            let mut avx2_packed = vec![0u8; len];

            ScalarBackend::pack_block(&input, bit_width, &mut scalar_packed).unwrap();
            Avx2Backend::pack_block(&input, bit_width, &mut avx2_packed).unwrap();
            assert_eq!(
                scalar_packed, avx2_packed,
                "pack mismatch vs scalar at bit_width={bit_width}"
            );

            let mut scalar_out = [0u32; 128];
            let mut avx2_out = [0u32; 128];
            ScalarBackend::unpack_block(&scalar_packed, bit_width, &mut scalar_out).unwrap();
            Avx2Backend::unpack_block(&avx2_packed, bit_width, &mut avx2_out).unwrap();
            assert_eq!(
                scalar_out, avx2_out,
                "unpack mismatch vs scalar at bit_width={bit_width}"
            );
        }
    }

    #[test]
    fn test_avx2_roundtrip() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 not available, skipping");
            return;
        }
        for bit_width in 0..=MAX_BIT_WIDTH {
            let input = random_input(bit_width);
            let len = ((BLOCK_SIZE * bit_width as usize + 7) / 8).max(1);
            let mut packed = vec![0u8; len];
            let mut output = [0u32; 128];

            Avx2Backend::pack_block(&input, bit_width, &mut packed).unwrap();
            Avx2Backend::unpack_block(&packed, bit_width, &mut output).unwrap();
            assert_eq!(input, output, "roundtrip failed at bit_width={bit_width}");
        }
    }

    #[test]
    fn test_avx2_invalid_bit_width() {
        let input = [0u32; 128];
        let mut output = [0u8; 1024];
        let result = Avx2Backend::pack_block(&input, 33, &mut output);
        assert!(matches!(result, Err(Error::InvalidBitWidth(33))));

        let mut unpacked = [0u32; 128];
        let result = Avx2Backend::unpack_block(&[], 33, &mut unpacked);
        assert!(matches!(result, Err(Error::InvalidBitWidth(33))));
    }

    #[test]
    fn test_avx2_boundary_values() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // Min, max, and alternating patterns at every specialized width.
        for bit_width in [1u8, 4, 8, 16, 32] {
            let max_val = if bit_width >= 32 {
                u32::MAX
            } else {
                (1u32 << bit_width) - 1
            };
            let patterns: [[u32; 128]; 4] = [
                [0u32; 128],
                [max_val; 128],
                std::array::from_fn(|i| if i % 2 == 0 { 0 } else { max_val }),
                std::array::from_fn(|i| (i as u32) & max_val),
            ];
            for input in &patterns {
                let len = ((BLOCK_SIZE * bit_width as usize + 7) / 8).max(1);
                let mut sse_packed = vec![0u8; len];
                let mut avx2_packed = vec![0u8; len];
                SseBackend::pack_block(input, bit_width, &mut sse_packed).unwrap();
                Avx2Backend::pack_block(input, bit_width, &mut avx2_packed).unwrap();
                assert_eq!(sse_packed, avx2_packed, "pack diverged at bw={bit_width}");

                let mut sse_out = [0u32; 128];
                let mut avx2_out = [0u32; 128];
                SseBackend::unpack_block(&sse_packed, bit_width, &mut sse_out).unwrap();
                Avx2Backend::unpack_block(&avx2_packed, bit_width, &mut avx2_out).unwrap();
                assert_eq!(sse_out, avx2_out, "unpack diverged at bw={bit_width}");
                assert_eq!(*input, avx2_out, "roundtrip failed at bw={bit_width}");
            }
        }
    }
}
