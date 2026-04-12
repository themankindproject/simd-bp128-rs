//! High-performance SSE4.1 implementation of BP128 bit-packing kernels.
//!
//! # Performance Characteristics
//!
//! This implementation targets 20-40 GB/s throughput for byte-aligned widths,
//! 5-15 GB/s for non-byte-aligned widths.
//!
//! ## Optimization Strategies
//!
//! 1. **Byte-aligned widths (8, 16, 24, 32)**: Direct SIMD loads/stores with minimal shuffling
//! 2. **Power-of-2 sub-byte (1, 2, 4)**: Parallel bit extraction using movemask and horizontal operations
//! 3. **Non-byte-aligned (3, 5-7, 9-15, 17-23, 25-31)**: SIMD where profitable, scalar for complex cases
//!
//! # Safety
//!
//! All unsafe functions validate input/output buffer bounds before accessing memory.
//! The SSE4.1 feature must be checked at runtime before calling these functions
//! (handled automatically by the dispatch module).

use crate::error::Error;
use crate::simd::scalar::ScalarBackend;
use crate::simd::SimdBackend;

/// Number of values in a full block.
const BLOCK_SIZE: usize = 128;

/// Maximum valid bit width.
const MAX_BIT_WIDTH: u8 = 32;

/// SSE4.1 implementation of the BP128 bit-packing backend.
///
/// Provides SIMD-accelerated packing/unpacking for 128-value blocks.
/// Automatically falls back to scalar implementation on non-x86_64 platforms.
#[must_use]
pub struct SseBackend;

impl SimdBackend for SseBackend {
    fn pack_block(input: &[u32; 128], bit_width: u8, output: &mut [u8]) -> Result<(), Error> {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            sse_impl::pack_block_sse41(input, bit_width, output)
        }
        #[cfg(not(target_arch = "x86_64"))]
        ScalarBackend::pack_block(input, bit_width, output)
    }

    fn unpack_block(input: &[u8], bit_width: u8, output: &mut [u32; 128]) -> Result<(), Error> {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            sse_impl::unpack_block_sse41(input, bit_width, output)
        }
        #[cfg(not(target_arch = "x86_64"))]
        ScalarBackend::unpack_block(input, bit_width, output)
    }
}

#[cfg(target_arch = "x86_64")]
mod sse_impl {
    use super::*;
    use std::arch::x86_64::*;

    /// Pack a 128-value block using SSE4.1 instructions.
    ///
    /// Dispatches to specialized implementations based on bit width.
    ///
    /// # Safety
    ///
    /// This function uses SSE4.1 intrinsics that require the `sse4.1` CPU feature.
    /// The caller must ensure this feature is available at runtime (handled by dispatch).
    /// Input/output buffers must be properly sized:
    /// - `input` must contain exactly 128 u32 values
    /// - `output` must have at least `(128 * bit_width + 7) / 8` bytes
    ///
    /// Violating these invariants would cause undefined behavior via out-of-bounds memory access.
    #[target_feature(enable = "sse4.1")]
    pub unsafe fn pack_block_sse41(
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
            1 => pack_1bit(input, output),
            2 => pack_2bit(input, output),
            // 3,5,6,7: scalar accumulator is identical to ScalarBackend
            3 | 5 | 6 | 7 => ScalarBackend::pack_block(input, bit_width, output),
            4 => pack_4bit(input, output),
            8 => pack_8bit(input, output),
            9..=15 => pack_9to23bit(input, output, bit_width),
            16 => pack_16bit(input, output),
            17..=23 => pack_9to23bit(input, output, bit_width),
            24 => pack_24bit(input, output),
            // 25-31: scalar accumulator is identical to ScalarBackend
            25..=31 => ScalarBackend::pack_block(input, bit_width, output),
            32 => pack_32bit(input, output),
            _ => unreachable!("bit_width validated above, must be 0-32"),
        }
    }

    /// Pack 128 32-bit values using direct SIMD load/store.
    #[inline]
    unsafe fn pack_32bit(input: &[u32; 128], output: &mut [u8]) -> Result<(), Error> {
        for i in 0..32 {
            let values = _mm_loadu_si128(input.as_ptr().add(i * 4) as *const __m128i);
            _mm_storeu_si128(output.as_mut_ptr().add(i * 16) as *mut __m128i, values);
        }
        Ok(())
    }

    /// Pack 128 8-bit values using `_mm_packus_epi32` and `_mm_packus_epi16`.
    #[inline]
    unsafe fn pack_8bit(input: &[u32; 128], output: &mut [u8]) -> Result<(), Error> {
        for i in 0..8 {
            let in_ptr = input.as_ptr().add(i * 16);
            let out_ptr = output.as_mut_ptr().add(i * 16);

            let v0 = _mm_loadu_si128(in_ptr as *const __m128i);
            let v1 = _mm_loadu_si128(in_ptr.add(4) as *const __m128i);
            let v2 = _mm_loadu_si128(in_ptr.add(8) as *const __m128i);
            let v3 = _mm_loadu_si128(in_ptr.add(12) as *const __m128i);

            let p0 = _mm_packus_epi32(v0, v1);
            let p1 = _mm_packus_epi32(v2, v3);
            let result = _mm_packus_epi16(p0, p1);

            _mm_storeu_si128(out_ptr as *mut __m128i, result);
        }
        Ok(())
    }

    /// Pack 128 16-bit values using `_mm_packus_epi32`.
    #[inline]
    unsafe fn pack_16bit(input: &[u32; 128], output: &mut [u8]) -> Result<(), Error> {
        for i in 0..16 {
            let in_ptr = input.as_ptr().add(i * 8);
            let out_ptr = output.as_mut_ptr().add(i * 16);

            let v0 = _mm_loadu_si128(in_ptr as *const __m128i);
            let v1 = _mm_loadu_si128(in_ptr.add(4) as *const __m128i);
            let result = _mm_packus_epi32(v0, v1);

            _mm_storeu_si128(out_ptr as *mut __m128i, result);
        }
        Ok(())
    }

    /// Pack 128 24-bit values using scalar extraction.
    #[inline]
    unsafe fn pack_24bit(input: &[u32; 128], output: &mut [u8]) -> Result<(), Error> {
        for i in 0..32 {
            let in_ptr = input.as_ptr().add(i * 4);
            let out_ptr = output.as_mut_ptr().add(i * 12);

            let v = _mm_loadu_si128(in_ptr as *const __m128i);

            let val0 = _mm_extract_epi32(v, 0) as u32;
            let val1 = _mm_extract_epi32(v, 1) as u32;
            let val2 = _mm_extract_epi32(v, 2) as u32;
            let val3 = _mm_extract_epi32(v, 3) as u32;

            *out_ptr.add(0) = val0 as u8;
            *out_ptr.add(1) = (val0 >> 8) as u8;
            *out_ptr.add(2) = (val0 >> 16) as u8;
            *out_ptr.add(3) = val1 as u8;
            *out_ptr.add(4) = (val1 >> 8) as u8;
            *out_ptr.add(5) = (val1 >> 16) as u8;
            *out_ptr.add(6) = val2 as u8;
            *out_ptr.add(7) = (val2 >> 8) as u8;
            *out_ptr.add(8) = (val2 >> 16) as u8;
            *out_ptr.add(9) = val3 as u8;
            *out_ptr.add(10) = (val3 >> 8) as u8;
            *out_ptr.add(11) = (val3 >> 16) as u8;
        }
        Ok(())
    }

    /// Pack 128 1-bit values using `_mm_movemask_epi8` for parallel bit extraction.
    #[inline]
    unsafe fn pack_1bit(input: &[u32; 128], output: &mut [u8]) -> Result<(), Error> {
        let mask1 = _mm_set1_epi32(1);

        for i in 0..8 {
            let in_ptr = input.as_ptr().add(i * 16);

            let v0 = _mm_loadu_si128(in_ptr as *const __m128i);
            let v1 = _mm_loadu_si128(in_ptr.add(4) as *const __m128i);
            let v2 = _mm_loadu_si128(in_ptr.add(8) as *const __m128i);
            let v3 = _mm_loadu_si128(in_ptr.add(12) as *const __m128i);

            let m0 = _mm_and_si128(v0, mask1);
            let m1 = _mm_and_si128(v1, mask1);
            let m2 = _mm_and_si128(v2, mask1);
            let m3 = _mm_and_si128(v3, mask1);

            let c0 = _mm_cmpeq_epi32(m0, mask1);
            let c1 = _mm_cmpeq_epi32(m1, mask1);
            let c2 = _mm_cmpeq_epi32(m2, mask1);
            let c3 = _mm_cmpeq_epi32(m3, mask1);

            let p0 = _mm_packs_epi32(c0, c1);
            let p1 = _mm_packs_epi32(c2, c3);
            let bytes = _mm_packs_epi16(p0, p1);

            let mask = _mm_movemask_epi8(bytes) as u16;
            output[i * 2] = (mask & 0xFF) as u8;
            output[i * 2 + 1] = (mask >> 8) as u8;
        }
        Ok(())
    }

    /// Pack 128 values into 2-bit packed format using SIMD load + scalar pack.
    #[inline]
    unsafe fn pack_2bit(input: &[u32; 128], output: &mut [u8]) -> Result<(), Error> {
        let mask3 = _mm_set1_epi32(3);

        for i in 0..16 {
            let in_ptr = input.as_ptr().add(i * 8);
            let out_ptr = output.as_mut_ptr().add(i * 2);

            let v0 = _mm_loadu_si128(in_ptr as *const __m128i);
            let v1 = _mm_loadu_si128(in_ptr.add(4) as *const __m128i);

            let m0 = _mm_and_si128(v0, mask3);
            let m1 = _mm_and_si128(v1, mask3);

            let val0 = _mm_extract_epi32(m0, 0) as u8;
            let val1 = _mm_extract_epi32(m0, 1) as u8;
            let val2 = _mm_extract_epi32(m0, 2) as u8;
            let val3 = _mm_extract_epi32(m0, 3) as u8;
            *out_ptr.add(0) = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);

            let val4 = _mm_extract_epi32(m1, 0) as u8;
            let val5 = _mm_extract_epi32(m1, 1) as u8;
            let val6 = _mm_extract_epi32(m1, 2) as u8;
            let val7 = _mm_extract_epi32(m1, 3) as u8;
            *out_ptr.add(1) = val4 | (val5 << 2) | (val6 << 4) | (val7 << 6);
        }
        Ok(())
    }

    /// Pack 128 4-bit values using SIMD load + scalar pack.
    #[inline]
    unsafe fn pack_4bit(input: &[u32; 128], output: &mut [u8]) -> Result<(), Error> {
        let mask_f = _mm_set1_epi32(0x0F);

        for i in 0..16 {
            let in_ptr = input.as_ptr().add(i * 8);
            let out_ptr = output.as_mut_ptr().add(i * 4);

            let v0 = _mm_loadu_si128(in_ptr as *const __m128i);
            let v1 = _mm_loadu_si128(in_ptr.add(4) as *const __m128i);

            let m0 = _mm_and_si128(v0, mask_f);
            let m1 = _mm_and_si128(v1, mask_f);

            let val0 = _mm_extract_epi32(m0, 0) as u8;
            let val1 = _mm_extract_epi32(m0, 1) as u8;
            let val2 = _mm_extract_epi32(m0, 2) as u8;
            let val3 = _mm_extract_epi32(m0, 3) as u8;
            let val4 = _mm_extract_epi32(m1, 0) as u8;
            let val5 = _mm_extract_epi32(m1, 1) as u8;
            let val6 = _mm_extract_epi32(m1, 2) as u8;
            let val7 = _mm_extract_epi32(m1, 3) as u8;

            *out_ptr.add(0) = val0 | (val1 << 4);
            *out_ptr.add(1) = val2 | (val3 << 4);
            *out_ptr.add(2) = val4 | (val5 << 4);
            *out_ptr.add(3) = val6 | (val7 << 4);
        }
        Ok(())
    }

    /// Pack values with bit widths 9-23 using SIMD masking + scalar accumulator.
    ///
    /// This consolidated function handles both 9-15 bit and 17-23 bit widths,
    /// eliminating code duplication between the previous separate implementations.
    #[inline]
    unsafe fn pack_9to23bit(input: &[u32; 128], output: &mut [u8], bits: u8) -> Result<(), Error> {
        let mask = (1u32 << bits) - 1;
        let mask_vec = _mm_set1_epi32(mask as i32);

        let required = (BLOCK_SIZE * bits as usize + 7) / 8;
        let output = &mut output[..required];

        let mut acc: u64 = 0;
        let mut acc_bits: usize = 0;
        let mut out_idx: usize = 0;
        let mut tmp = [0u32; 4];

        for i in 0..32 {
            let in_ptr = input.as_ptr().add(i * 4);

            let v = _mm_loadu_si128(in_ptr as *const __m128i);
            let masked = _mm_and_si128(v, mask_vec);
            _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, masked);

            for &val in tmp.iter() {
                acc |= (val as u64) << acc_bits;
                acc_bits += bits as usize;
                while acc_bits >= 8 {
                    output[out_idx] = acc as u8;
                    out_idx += 1;
                    acc >>= 8;
                    acc_bits -= 8;
                }
            }
        }

        if acc_bits > 0 {
            output[out_idx] = acc as u8;
        }

        Ok(())
    }

    /// Unpack a 128-value block using SSE4.1 instructions.
    ///
    /// Dispatches to specialized implementations based on bit width.
    ///
    /// # Safety
    ///
    /// This function uses SSE4.1 intrinsics that require the `sse4.1` CPU feature.
    /// The caller must ensure this feature is available at runtime (handled by dispatch).
    /// Input/output buffers must be properly sized:
    /// - `input` must have at least `(128 * bit_width + 7) / 8` bytes
    /// - `output` must be able to hold exactly 128 u32 values
    ///
    /// Violating these invariants would cause undefined behavior via out-of-bounds memory access.
    #[target_feature(enable = "sse4.1")]
    pub unsafe fn unpack_block_sse41(
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
            1 => unpack_1bit(input, output),
            2 => unpack_2bit(input, output),
            // 3,5,6,7: scalar accumulator matches ScalarBackend
            3 | 5 | 6 | 7 => ScalarBackend::unpack_block(input, bit_width, output),
            4 => unpack_4bit(input, output),
            8 => unpack_8bit(input, output),
            9..=15 => unpack_9to23bit(input, output, bit_width),
            16 => unpack_16bit(input, output),
            17..=23 => unpack_9to23bit(input, output, bit_width),
            24 => unpack_24bit(input, output),
            // 25-31: scalar accumulator matches ScalarBackend
            25..=31 => ScalarBackend::unpack_block(input, bit_width, output),
            32 => unpack_32bit(input, output),
            _ => unreachable!("bit_width validated above, must be 0-32"),
        }
    }

    /// Unpack 128 32-bit values using direct SIMD load/store.
    #[inline]
    unsafe fn unpack_32bit(input: &[u8], output: &mut [u32; 128]) -> Result<(), Error> {
        for i in 0..32 {
            let values = _mm_loadu_si128(input.as_ptr().add(i * 16) as *const __m128i);
            _mm_storeu_si128(output.as_mut_ptr().add(i * 4) as *mut __m128i, values);
        }
        Ok(())
    }

    /// Unpack 8-bit packed values using `_mm_unpacklo_epi8` and `_mm_unpackhi_epi8`.
    #[inline]
    unsafe fn unpack_8bit(input: &[u8], output: &mut [u32; 128]) -> Result<(), Error> {
        for i in 0..8 {
            let in_ptr = input.as_ptr().add(i * 16);
            let out_ptr = output.as_mut_ptr().add(i * 16);

            let bytes = _mm_loadu_si128(in_ptr as *const __m128i);
            let zeros = _mm_setzero_si128();
            let lo16 = _mm_unpacklo_epi8(bytes, zeros);
            let hi16 = _mm_unpackhi_epi8(bytes, zeros);

            let v0 = _mm_unpacklo_epi16(lo16, zeros);
            let v1 = _mm_unpackhi_epi16(lo16, zeros);
            let v2 = _mm_unpacklo_epi16(hi16, zeros);
            let v3 = _mm_unpackhi_epi16(hi16, zeros);

            _mm_storeu_si128(out_ptr as *mut __m128i, v0);
            _mm_storeu_si128(out_ptr.add(4) as *mut __m128i, v1);
            _mm_storeu_si128(out_ptr.add(8) as *mut __m128i, v2);
            _mm_storeu_si128(out_ptr.add(12) as *mut __m128i, v3);
        }
        Ok(())
    }

    /// Unpack 16-bit packed values using `_mm_unpacklo_epi16` and `_mm_unpackhi_epi16`.
    #[inline]
    unsafe fn unpack_16bit(input: &[u8], output: &mut [u32; 128]) -> Result<(), Error> {
        for i in 0..16 {
            let in_ptr = input.as_ptr().add(i * 16);
            let out_ptr = output.as_mut_ptr().add(i * 8);

            let bytes = _mm_loadu_si128(in_ptr as *const __m128i);
            let zeros = _mm_setzero_si128();
            let lo = _mm_unpacklo_epi16(bytes, zeros);
            let hi = _mm_unpackhi_epi16(bytes, zeros);

            _mm_storeu_si128(out_ptr as *mut __m128i, lo);
            _mm_storeu_si128(out_ptr.add(4) as *mut __m128i, hi);
        }
        Ok(())
    }

    /// Unpack 24-bit packed values using scalar extraction.
    ///
    /// Uses scalar byte extraction rather than SIMD shuffles because:
    /// - Shuffle patterns were slower (110ns vs 54ns)
    /// - 32/64-bit loads have alignment/ordering issues
    #[inline]
    unsafe fn unpack_24bit(input: &[u8], output: &mut [u32; 128]) -> Result<(), Error> {
        for i in 0..32 {
            let in_ptr = input.as_ptr().add(i * 12);
            let out_ptr = output.as_mut_ptr().add(i * 4);

            let val0 = (*in_ptr.add(0) as u32)
                | ((*in_ptr.add(1) as u32) << 8)
                | ((*in_ptr.add(2) as u32) << 16);
            let val1 = (*in_ptr.add(3) as u32)
                | ((*in_ptr.add(4) as u32) << 8)
                | ((*in_ptr.add(5) as u32) << 16);
            let val2 = (*in_ptr.add(6) as u32)
                | ((*in_ptr.add(7) as u32) << 8)
                | ((*in_ptr.add(8) as u32) << 16);
            let val3 = (*in_ptr.add(9) as u32)
                | ((*in_ptr.add(10) as u32) << 8)
                | ((*in_ptr.add(11) as u32) << 16);

            let v = _mm_set_epi32(val3 as i32, val2 as i32, val1 as i32, val0 as i32);
            _mm_storeu_si128(out_ptr as *mut __m128i, v);
        }
        Ok(())
    }

    /// Unpack 1-bit packed values using scalar bit extraction.
    ///
    /// NOTE: Scalar — SIMD patterns for 1-bit are slower due to the cost of
    /// expanding 16 bytes into 128 individual u32 values. This is already
    /// near-optimal at ~100 ns/block.
    #[inline]
    unsafe fn unpack_1bit(input: &[u8], output: &mut [u32; 128]) -> Result<(), Error> {
        for i in 0..4 {
            let in_ptr = input.as_ptr().add(i * 4);
            let out_ptr = output.as_mut_ptr().add(i * 32);

            for j in 0..4 {
                let byte = *in_ptr.add(j) as u32;
                let byte_out_ptr = out_ptr.add(j * 8);

                *byte_out_ptr.add(0) = byte & 1;
                *byte_out_ptr.add(1) = (byte >> 1) & 1;
                *byte_out_ptr.add(2) = (byte >> 2) & 1;
                *byte_out_ptr.add(3) = (byte >> 3) & 1;
                *byte_out_ptr.add(4) = (byte >> 4) & 1;
                *byte_out_ptr.add(5) = (byte >> 5) & 1;
                *byte_out_ptr.add(6) = (byte >> 6) & 1;
                *byte_out_ptr.add(7) = (byte >> 7) & 1;
            }
        }

        Ok(())
    }

    /// Unpack 2-bit packed values using scalar extraction.
    ///
    /// NOTE: Scalar — the byte-by-byte nibble extraction is already efficient.
    #[inline]
    unsafe fn unpack_2bit(input: &[u8], output: &mut [u32; 128]) -> Result<(), Error> {
        for i in 0..8 {
            let in_ptr = input.as_ptr().add(i * 4);
            let out_ptr = output.as_mut_ptr().add(i * 16);

            for j in 0..4 {
                let byte = *in_ptr.add(j);
                let byte_out_ptr = out_ptr.add(j * 4);

                *byte_out_ptr.add(0) = (byte & 0x03) as u32;
                *byte_out_ptr.add(1) = ((byte >> 2) & 0x03) as u32;
                *byte_out_ptr.add(2) = ((byte >> 4) & 0x03) as u32;
                *byte_out_ptr.add(3) = ((byte >> 6) & 0x03) as u32;
            }
        }
        Ok(())
    }

    /// Unpack 4-bit packed values using `_mm_unpacklo_epi8` and nibble extraction.
    #[inline]
    unsafe fn unpack_4bit(input: &[u8], output: &mut [u32; 128]) -> Result<(), Error> {
        for i in 0..8 {
            let in_ptr = input.as_ptr().add(i * 8);
            let out_ptr = output.as_mut_ptr().add(i * 16);

            let v = _mm_loadl_epi64(in_ptr as *const __m128i);
            let bytes = _mm_unpacklo_epi8(v, _mm_setzero_si128());

            let low_nibble_mask = _mm_set1_epi16(0x0F);
            let low = _mm_and_si128(bytes, low_nibble_mask);
            let high = _mm_srli_epi16(bytes, 4);
            let high_masked = _mm_and_si128(high, low_nibble_mask);

            let unpacked_lo = _mm_unpacklo_epi16(low, high_masked);
            let unpacked_hi = _mm_unpackhi_epi16(low, high_masked);

            let zeros = _mm_setzero_si128();
            let out0 = _mm_unpacklo_epi16(unpacked_lo, zeros);
            let out1 = _mm_unpackhi_epi16(unpacked_lo, zeros);
            let out2 = _mm_unpacklo_epi16(unpacked_hi, zeros);
            let out3 = _mm_unpackhi_epi16(unpacked_hi, zeros);

            _mm_storeu_si128(out_ptr as *mut __m128i, out0);
            _mm_storeu_si128(out_ptr.add(4) as *mut __m128i, out1);
            _mm_storeu_si128(out_ptr.add(8) as *mut __m128i, out2);
            _mm_storeu_si128(out_ptr.add(12) as *mut __m128i, out3);
        }
        Ok(())
    }

    /// Unpack values with bit widths 9-23 using scalar accumulator + SIMD stores.
    ///
    /// This consolidated function handles both 9-15 bit and 17-23 bit widths,
    /// eliminating code duplication between the previous separate implementations.
    #[inline]
    #[allow(clippy::needless_range_loop)] // Indexed loop with raw pointers is faster
    unsafe fn unpack_9to23bit(
        input: &[u8],
        output: &mut [u32; 128],
        bits: u8,
    ) -> Result<(), Error> {
        let mask = (1u64 << bits) - 1;

        let required = (BLOCK_SIZE * bits as usize + 7) / 8;
        let input = &input[..required];

        let mut acc: u64 = 0;
        let mut acc_bits: usize = 0;
        let mut in_idx: usize = 0;

        for i in 0..32 {
            let out_ptr = output.as_mut_ptr().add(i * 4);
            let mut vals = [0u32; 4];

            for j in 0..4 {
                while acc_bits < bits as usize {
                    acc |= (input[in_idx] as u64) << acc_bits;
                    acc_bits += 8;
                    in_idx += 1;
                }

                vals[j] = (acc & mask) as u32;
                acc >>= bits;
                acc_bits -= bits as usize;
            }

            let v = _mm_set_epi32(
                vals[3] as i32,
                vals[2] as i32,
                vals[1] as i32,
                vals[0] as i32,
            );
            _mm_storeu_si128(out_ptr as *mut __m128i, v);
        }

        Ok(())
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn test_sse_matches_scalar() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for bit_width in 1..=32 {
            let max_val = if bit_width >= 32 {
                u32::MAX
            } else {
                (1u32 << bit_width) - 1
            };

            let input: [u32; 128] = std::array::from_fn(|_| rng.gen::<u32>() & max_val);

            let mut scalar_packed = vec![0u8; (128 * bit_width as usize + 7) / 8];
            let mut sse_packed = vec![0u8; (128 * bit_width as usize + 7) / 8];

            ScalarBackend::pack_block(&input, bit_width, &mut scalar_packed).unwrap();
            SseBackend::pack_block(&input, bit_width, &mut sse_packed).unwrap();

            assert_eq!(
                scalar_packed, sse_packed,
                "Bit width {} produced different packed output",
                bit_width
            );

            let mut scalar_output = [0u32; 128];
            let mut sse_output = [0u32; 128];

            ScalarBackend::unpack_block(&scalar_packed, bit_width, &mut scalar_output).unwrap();
            SseBackend::unpack_block(&sse_packed, bit_width, &mut sse_output).unwrap();

            assert_eq!(
                scalar_output, sse_output,
                "Bit width {} produced different unpacked output",
                bit_width
            );
        }
    }

    #[test]
    fn test_sse_roundtrip() {
        if !is_x86_feature_detected!("sse4.1") {
            println!("SSE4.1 not available, skipping test");
            return;
        }

        use rand::Rng;
        let mut rng = rand::thread_rng();

        for bit_width in [1, 2, 4, 8, 16, 24, 32] {
            let max_val = if bit_width >= 32 {
                u32::MAX
            } else {
                (1u32 << bit_width) - 1
            };

            let input: [u32; 128] = std::array::from_fn(|_| rng.gen::<u32>() & max_val);
            let mut packed = vec![0u8; (128 * bit_width as usize + 7) / 8];
            let mut output = [0u32; 128];

            SseBackend::pack_block(&input, bit_width as u8, &mut packed).unwrap();
            SseBackend::unpack_block(&packed, bit_width as u8, &mut output).unwrap();

            assert_eq!(
                input, output,
                "SSE roundtrip failed for bit_width={}",
                bit_width
            );
        }
    }

    #[test]
    fn test_sse_byte_aligned_performance() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }

        let input: [u32; 128] = [0xDEADBEEF; 128];
        let mut packed = [0u8; 512];
        let mut output = [0u32; 128];

        SseBackend::pack_block(&input, 32, &mut packed).unwrap();
        SseBackend::unpack_block(&packed, 32, &mut output).unwrap();

        assert_eq!(input, output);
    }
}
