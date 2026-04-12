use crate::error::Error;
use crate::simd::scalar::ScalarBackend;
use crate::simd::SimdBackend;

#[cfg(target_arch = "x86_64")]
use crate::simd::avx2::Avx2Backend;
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
    // AVX2 has a dedicated kernel (specializes byte-aligned widths and 1-bit
    // pack, delegates the rest to SseBackend). AVX-512 detection currently
    // falls through to the AVX2 backend until a dedicated kernel lands.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

// Miri cannot interpret x86 SIMD intrinsics, so force the scalar
// reference under miri. This lets `cargo miri test` validate the
// public API and the scalar backend without choking on intrinsics.
#[cfg(miri)]
pub(crate) fn detect_best_backend() -> BackendType {
    BackendType::Scalar
}

#[cfg(not(miri))]
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
#[inline]
pub(crate) fn get_pack_fn() -> PackFn {
    match get_backend() {
        BackendType::Scalar => ScalarBackend::pack_block,
        #[cfg(target_arch = "x86_64")]
        BackendType::Sse => SseBackend::pack_block,
        #[cfg(target_arch = "x86_64")]
        BackendType::Avx2 | BackendType::Avx512 => Avx2Backend::pack_block,
    }
}

/// Returns the unpack function pointer for the best available backend.
#[inline]
pub(crate) fn get_unpack_fn() -> UnpackFn {
    match get_backend() {
        BackendType::Scalar => ScalarBackend::unpack_block,
        #[cfg(target_arch = "x86_64")]
        BackendType::Sse => SseBackend::unpack_block,
        #[cfg(target_arch = "x86_64")]
        BackendType::Avx2 | BackendType::Avx512 => Avx2Backend::unpack_block,
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
