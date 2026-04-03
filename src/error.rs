/// Unified error type for compression and decompression operations.
///
/// This enum serves as the top-level error container that can represent either
/// a compression or decompression failure. Use pattern matching to handle
/// specific error cases, or convert to a string using `Display` formatting.
///
/// # Example
///
/// ```
/// use simd_bp128::Error;
///
/// fn handle_error(err: &Error) -> &'static str {
///     match err {
///         Error::InvalidBitWidth(_) => "Invalid bit width",
///         Error::InputTooShort { .. } => "Input buffer too short",
///         Error::OutputTooSmall { .. } => "Output buffer too small",
///         Error::CompressionError(_) => "Compression failed",
///         Error::DecompressionError(_) => "Decompression failed",
///         _ => "Unknown error",
///     }
/// }
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The specified bit width is invalid (must be 0-32).
    InvalidBitWidth(u8),
    /// The input buffer does not contain enough data for the operation.
    InputTooShort { need: usize, got: usize },
    /// The output buffer is too small to hold the result.
    OutputTooSmall { need: usize, got: usize },
    /// A compression-specific error occurred.
    CompressionError(CompressionError),
    /// A decompression-specific error occurred.
    DecompressionError(DecompressionError),
}

/// Errors that can occur during compression operations.
///
/// These errors indicate problems with the input data or output buffer
/// during the compression process.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressionError {
    /// The input array exceeds the maximum supported size.
    InputTooLarge { max: usize, got: usize },
    /// The output buffer is too small to hold the compressed data.
    OutputTooSmall { need: usize, got: usize },
}

/// Errors that can occur during decompression operations.
///
/// These errors indicate problems with the compressed data format or
/// insufficient output buffer space during decompression.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecompressionError {
    /// The compressed data header is incomplete.
    HeaderTooSmall { needed: usize, have: usize },
    /// The compressed data is truncated and missing expected bytes.
    TruncatedData {
        position: usize,
        needed: usize,
        have: usize,
    },
    /// A bit width value exceeds the valid range (0-32).
    InvalidBitWidth { bit_width: u8 },
    /// The number of blocks in the header doesn't match the value count.
    BlockCountMismatch { expected: usize, found: usize },
    /// The decompressed value count would exceed safe limits.
    InputTooLarge { max: usize, got: usize },
    /// The number of blocks exceeds safe limits (possible malformed data).
    ExcessiveBlockCount { max: usize, got: usize },
    /// The format version byte is not recognized.
    UnsupportedVersion { version: u8 },
    /// The output buffer is too small to hold the decompressed values.
    OutputTooSmall { need: usize, got: usize },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidBitWidth(bw) => {
                write!(f, "Invalid bit width: {} (must be 0-32)", bw)
            }
            Error::InputTooShort { need, got } => {
                write!(
                    f,
                    "Input buffer too small: need {} bytes, got {}",
                    need, got
                )
            }
            Error::OutputTooSmall { need, got } => {
                write!(
                    f,
                    "Output buffer too small: need {} bytes, got {}",
                    need, got
                )
            }
            Error::CompressionError(e) => write!(f, "{}", e),
            Error::DecompressionError(e) => write!(f, "{}", e),
        }
    }
}

impl std::fmt::Display for CompressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompressionError::InputTooLarge { max, got } => {
                write!(f, "Input too large: maximum {} values, got {}", max, got)
            }
            CompressionError::OutputTooSmall { need, got } => {
                write!(
                    f,
                    "Output buffer too small: need {} bytes, got {}",
                    need, got
                )
            }
        }
    }
}

impl std::fmt::Display for DecompressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecompressionError::HeaderTooSmall { needed, have } => {
                write!(
                    f,
                    "Input too small for header: need {} bytes, have {}",
                    needed, have
                )
            }
            DecompressionError::TruncatedData {
                position,
                needed,
                have,
            } => {
                write!(
                    f,
                    "Truncated data at position {}: need {} bytes, have {}",
                    position, needed, have
                )
            }
            DecompressionError::InvalidBitWidth { bit_width } => {
                write!(f, "Invalid bit width: {} (must be 0-32)", bit_width)
            }
            DecompressionError::BlockCountMismatch { expected, found } => {
                write!(
                    f,
                    "Block count mismatch: expected {} values, found {}",
                    expected, found
                )
            }
            DecompressionError::InputTooLarge { max, got } => {
                write!(f, "Input too large: maximum {} values, got {}", max, got)
            }
            DecompressionError::ExcessiveBlockCount { max, got } => {
                write!(
                    f,
                    "Excessive block count: maximum {} blocks, got {}",
                    max, got
                )
            }
            DecompressionError::UnsupportedVersion { version } => {
                write!(f, "Unsupported format version: {} (expected 1)", version)
            }
            DecompressionError::OutputTooSmall { need, got } => {
                write!(
                    f,
                    "Output buffer too small: need {} values, got {}",
                    need, got
                )
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::CompressionError(e) => Some(e),
            Error::DecompressionError(e) => Some(e),
            _ => None,
        }
    }
}
impl std::error::Error for CompressionError {}
impl std::error::Error for DecompressionError {}

impl From<CompressionError> for Error {
    fn from(e: CompressionError) -> Self {
        Error::CompressionError(e)
    }
}

impl From<DecompressionError> for Error {
    fn from(e: DecompressionError) -> Self {
        Error::DecompressionError(e)
    }
}
