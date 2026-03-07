use crate::error::Error;
use crate::simd::SimdBackend;

#[allow(dead_code)]
pub struct SseBackend;

impl SimdBackend for SseBackend {
    fn pack_block(_input: &[u32; 128], _bit_width: u8, _output: &mut [u8]) -> Result<(), Error> {
        todo!("Implement SSE4.1 bit packing using _mm intrinsics")
    }

    fn unpack_block(_input: &[u8], _bit_width: u8, _output: &mut [u32; 128]) -> Result<(), Error> {
        todo!("Implement SSE4.1 bit unpacking using _mm intrinsics")
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    #[test]
    #[ignore = "SSE implementation not yet complete"]
    fn test_sse_available() {
        if is_x86_feature_detected!("sse4.1") {
            println!("SSE4.1 is available on this CPU");
        } else {
            println!("SSE4.1 is not available on this CPU");
        }
    }
}
