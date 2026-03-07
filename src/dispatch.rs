use crate::error::Error;
use crate::simd::scalar::ScalarBackend;
use crate::simd::SimdBackend;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Sse,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

pub fn detect_best_backend() -> BackendType {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return BackendType::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return BackendType::Avx2;
        }
        if is_x86_feature_detected!("sse4.1") {
            return BackendType::Sse;
        }
    }

    BackendType::Scalar
}

pub fn get_backend() -> BackendType {
    use std::sync::OnceLock;

    static BACKEND: OnceLock<BackendType> = OnceLock::new();
    *BACKEND.get_or_init(detect_best_backend)
}

pub fn pack_block_dispatch(
    input: &[u32; 128],
    bit_width: u8,
    output: &mut [u8],
) -> Result<(), Error> {
    let backend: BackendType = get_backend();

    match backend {
        BackendType::Scalar => ScalarBackend::pack_block(input, bit_width, output),
        #[cfg(target_arch = "x86_64")]
        BackendType::Sse => ScalarBackend::pack_block(input, bit_width, output),
        #[cfg(target_arch = "x86_64")]
        BackendType::Avx2 => ScalarBackend::pack_block(input, bit_width, output),
        #[cfg(target_arch = "x86_64")]
        BackendType::Avx512 => ScalarBackend::pack_block(input, bit_width, output),
    }
}

pub fn unpack_block_dispatch(
    input: &[u8],
    bit_width: u8,
    output: &mut [u32; 128],
) -> Result<(), Error> {
    let backend = get_backend();

    match backend {
        BackendType::Scalar => ScalarBackend::unpack_block(input, bit_width, output),
        #[cfg(target_arch = "x86_64")]
        BackendType::Sse => ScalarBackend::unpack_block(input, bit_width, output),
        #[cfg(target_arch = "x86_64")]
        BackendType::Avx2 => ScalarBackend::unpack_block(input, bit_width, output),
        #[cfg(target_arch = "x86_64")]
        BackendType::Avx512 => ScalarBackend::unpack_block(input, bit_width, output),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_backend() {
        let _backend: BackendType = detect_best_backend();
    }

    #[test]
    fn test_get_backend_cached() {
        let backend1: BackendType = get_backend();
        let backend2: BackendType = get_backend();
        assert_eq!(backend1, backend2);
    }
}
