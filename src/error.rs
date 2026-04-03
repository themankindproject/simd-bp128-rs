/// Unified error type for compression and decompression operations.
///
/// This enum serves as the top-level error container that can represent either
/// a compression or decompression failure. Use pattern matching to handle
/// specific error cases, or convert to a string using `Display` formatting.
///
/// # Example
///
/// ```
/// use packsimd::Error;
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
/// during the compression process. This enum is `#[non_exhaustive]` to
/// allow future variants to be added without a breaking change.
///
/// # Example
///
/// ```
/// use packsimd::CompressionError;
///
/// fn handle_compression_error(err: &CompressionError) -> &'static str {
///     match err {
///         CompressionError::InputTooLarge { .. } => "Input exceeds u32::MAX values",
///         CompressionError::OutputTooSmall { .. } => "Output buffer too small",
///         _ => "Unknown compression error",
///     }
/// }
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressionError {
    /// The input array exceeds the maximum supported size of `u32::MAX` values.
    ///
    /// This limit exists because the compressed header stores the input length
    /// as a 32-bit unsigned integer. Inputs larger than this cannot be represented
    /// in the binary format.
    ///
    /// # Fields
    ///
    /// - `max`: The maximum allowed number of values (`u32::MAX`).
    /// - `got`: The actual number of values in the input.
    InputTooLarge { max: usize, got: usize },
    /// The output buffer is too small to hold the compressed data.
    ///
    /// The output buffer must be at least `max_compressed_size(input.len())`
    /// bytes. Use that function to compute the minimum required size before
    /// calling `compress_into`.
    ///
    /// # Fields
    ///
    /// - `need`: The minimum number of bytes required.
    /// - `got`: The actual size of the output buffer.
    OutputTooSmall { need: usize, got: usize },
}

/// Errors that can occur during decompression operations.
///
/// These errors indicate problems with the compressed data format, data integrity,
/// or insufficient output buffer space during decompression. This enum is
/// `#[non_exhaustive]` to allow future variants to be added without a breaking change.
///
/// # Example
///
/// ```
/// use packsimd::DecompressionError;
///
/// fn handle_decompression_error(err: &DecompressionError) -> &'static str {
///     match err {
///         DecompressionError::HeaderTooSmall { .. } => "Compressed data header is incomplete",
///         DecompressionError::TruncatedData { .. } => "Compressed data is truncated",
///         DecompressionError::InvalidBitWidth { .. } => "Invalid bit width in block directory",
///         DecompressionError::UnsupportedVersion { .. } => "Unknown format version",
///         DecompressionError::OutputTooSmall { .. } => "Output buffer too small",
///         _ => "Unknown decompression error",
///     }
/// }
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecompressionError {
    /// The compressed data header is incomplete — fewer than 9 bytes provided.
    ///
    /// The header consists of: 1 byte version + 4 bytes input length + 4 bytes block count.
    ///
    /// # Fields
    ///
    /// - `needed`: The minimum number of bytes required (9).
    /// - `have`: The actual number of bytes in the input.
    HeaderTooSmall { needed: usize, have: usize },
    /// The compressed data is truncated and missing expected bytes.
    ///
    /// This error occurs when the input ends before all packed block data could
    /// be read. The compressed data may have been cut off during transmission
    /// or storage.
    ///
    /// # Fields
    ///
    /// - `position`: The byte offset in the input where the truncation was detected.
    /// - `needed`: The number of additional bytes required.
    /// - `have`: The number of bytes actually available from `position`.
    TruncatedData {
        position: usize,
        needed: usize,
        have: usize,
    },
    /// A bit width value in the block directory exceeds the valid range (0–32).
    ///
    /// Each block stores its values using a bit width between 0 (all zeros) and
    /// 32 (full `u32` values). A value outside this range indicates corrupted
    /// or malformed data.
    ///
    /// # Fields
    ///
    /// - `bit_width`: The invalid bit width value found in the data.
    InvalidBitWidth { bit_width: u8 },
    /// The number of blocks in the header doesn't match the expected count for the value count.
    ///
    /// The expected block count is `ceil(input_len / 128)`. A mismatch indicates
    /// the header fields are inconsistent, which suggests corrupted data.
    ///
    /// # Fields
    ///
    /// - `expected`: The expected number of blocks based on `input_len`.
    /// - `found`: The actual number of blocks stored in the header.
    BlockCountMismatch { expected: usize, found: usize },
    /// The decompressed value count would exceed the safe limit of 1 billion values.
    ///
    /// This safeguard prevents denial-of-service attacks where a malicious header
    /// claims an enormous number of values, causing an out-of-memory condition.
    ///
    /// # Fields
    ///
    /// - `max`: The maximum allowed number of values (1,000,000,000).
    /// - `got`: The value count claimed by the header.
    InputTooLarge { max: usize, got: usize },
    /// The number of blocks exceeds the safe limit.
    ///
    /// This indicates either corrupted data or an attempt to trigger excessive
    /// processing. The limit is derived from `MAX_DECOMPRESSED_VALUES / 128 + 1`.
    ///
    /// # Fields
    ///
    /// - `max`: The maximum allowed number of blocks.
    /// - `got`: The actual number of blocks stored in the header.
    ExcessiveBlockCount { max: usize, got: usize },
    /// The format version byte is not recognized.
    ///
    /// Currently only version 1 is supported. Data compressed with a different
    /// version of the library may not be compatible.
    ///
    /// # Fields
    ///
    /// - `version`: The unsupported version byte found in the header.
    UnsupportedVersion { version: u8 },
    /// The output buffer is too small to hold the decompressed values.
    ///
    /// Use [`decompressed_len`](crate::decompressed_len) to determine the required
    /// output buffer size before calling [`decompress_into`](crate::decompress_into).
    ///
    /// # Fields
    ///
    /// - `need`: The minimum number of `u32` values the buffer must hold.
    /// - `got`: The actual capacity of the output buffer.
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
