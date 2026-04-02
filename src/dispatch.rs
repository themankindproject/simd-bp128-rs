use crate::error::Error;
use crate::simd::scalar::ScalarBackend;
use crate::simd::SimdBackend;

#[cfg(target_arch = "x86_64")]
use crate::simd::sse::SseBackend;

/// Function pointer type for packing a 128-value block.
pub(crate) type PackFn = fn(&[u32; 128], u8, &mut [u8]) -> Result<(), Error>;

/// Function pointer type for unpacking a 128-value block.
pub(crate) type UnpackFn = fn(&[u8], u8, &mut [u32; 128]) -> Result<(), Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendType {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Sse,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

pub(crate) fn detect_best_backend() -> BackendType {
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

pub(crate) fn get_backend() -> BackendType {
    use std::sync::OnceLock;

    static BACKEND: OnceLock<BackendType> = OnceLock::new();
    *BACKEND.get_or_init(detect_best_backend)
}

/// Returns the pack function pointer for the best available backend.
///
/// Resolves the backend once so callers can use the pointer in a tight loop
/// without repeated atomic loads from `OnceLock`.
#[inline(always)]
pub(crate) fn get_pack_fn() -> PackFn {
    match get_backend() {
        BackendType::Scalar => ScalarBackend::pack_block,
        #[cfg(target_arch = "x86_64")]
        BackendType::Sse | BackendType::Avx2 | BackendType::Avx512 => SseBackend::pack_block,
    }
}

/// Returns the unpack function pointer for the best available backend.
#[inline(always)]
pub(crate) fn get_unpack_fn() -> UnpackFn {
    match get_backend() {
        BackendType::Scalar => ScalarBackend::unpack_block,
        #[cfg(target_arch = "x86_64")]
        BackendType::Sse | BackendType::Avx2 | BackendType::Avx512 => SseBackend::unpack_block,
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
