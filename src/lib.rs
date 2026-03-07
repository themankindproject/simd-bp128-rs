pub use compress::compress;
pub use decompress::decompress;
pub use error::{CompressionError, DecompressionError, Error};

pub(crate) mod bitwidth;
pub(crate) mod compress;
pub(crate) mod decompress;
pub(crate) mod dispatch;
pub(crate) mod error;
pub(crate) mod simd;

pub(crate) const BLOCK_SIZE: usize = 128;
