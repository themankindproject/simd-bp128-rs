use crate::error::Error;

#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512;
pub mod scalar;
#[cfg(target_arch = "x86_64")]
pub mod sse;

pub trait SimdBackend: Send + Sync {
    fn pack_block(input: &[u32; 128], bit_width: u8, output: &mut [u8]) -> Result<(), Error>;

    fn unpack_block(input: &[u8], bit_width: u8, output: &mut [u32; 128]) -> Result<(), Error>;
}
