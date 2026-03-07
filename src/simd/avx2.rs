use crate::error::Error;
use crate::simd::SimdBackend;

#[allow(dead_code)]
pub struct Avx2Backend;

impl SimdBackend for Avx2Backend {
    fn pack_block(_input: &[u32; 128], _bit_width: u8, _output: &mut [u8]) -> Result<(), Error> {
        todo!("Implement AVX2 bit packing using _mm256 intrinsics")
    }

    fn unpack_block(_input: &[u8], _bit_width: u8, _output: &mut [u32; 128]) -> Result<(), Error> {
        todo!("Implement AVX2 bit unpacking using _mm256 intrinsics")
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    #[test]
    #[ignore = "AVX2 implementation not yet complete"]
    fn test_avx2_available() {
        if is_x86_feature_detected!("avx2") {
            println!("AVX2 is available on this CPU");
        } else {
            println!("AVX2 is not available on this CPU");
        }
    }
}
