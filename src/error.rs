/// Unified error type for compression and decompression operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidBitWidth(u8),
    InputTooShort { need: usize, got: usize },
    OutputTooSmall { need: usize, got: usize },
    CompressionError(CompressionError),
    DecompressionError(DecompressionError),
}

/// Compression errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressionError {
    InputTooLarge { max: usize, got: usize },
    InvalidBitWidth { bit_width: u8 },
    OutputTooSmall { need: usize, got: usize },
    BlockSizeMismatch { expected: usize, got: usize },
}

/// Decompression errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecompressionError {
    HeaderTooSmall {
        needed: usize,
        have: usize,
    },
    TruncatedData {
        position: usize,
        needed: usize,
        have: usize,
    },
    InvalidBitWidth {
        bit_width: u8,
    },
    BlockCountMismatch {
        expected: usize,
        found: usize,
    },
    InputTooLarge {
        max: usize,
        got: usize,
    },
    ExcessiveBlockCount {
        max: usize,
        got: usize,
    },
    UnsupportedVersion {
        version: u8,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidBitWidth(bw) => {
                write!(f, "Invalid bit width: {} (must be 1-32)", bw)
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
            CompressionError::InvalidBitWidth { bit_width } => {
                write!(f, "Invalid bit width: {} (must be 1-32)", bit_width)
            }
            CompressionError::OutputTooSmall { need, got } => {
                write!(
                    f,
                    "Output buffer too small: need {} bytes, got {}",
                    need, got
                )
            }
            CompressionError::BlockSizeMismatch { expected, got } => {
                write!(
                    f,
                    "Block size mismatch: expected {} values, got {}",
                    expected, got
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
                write!(f, "Invalid bit width: {} (must be 1-32)", bit_width)
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
        }
    }
}

impl std::error::Error for Error {}
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
