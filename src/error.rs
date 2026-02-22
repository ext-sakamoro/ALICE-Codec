//! Error types for ALICE-Codec
//!
//! All public APIs that can fail return `Result<T, CodecError>`.

#[cfg(not(feature = "std"))]
use alloc::string::String;

use core::fmt;

/// Errors that can occur during encoding, decoding, or buffer operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodecError {
    /// Input buffer size does not match the declared dimensions.
    InvalidBufferSize { expected: usize, got: usize },
    /// Width or height is zero.
    InvalidDimensions { width: u32, height: u32 },
    /// Dimensions overflow `usize` when multiplied together.
    DimensionOverflow,
    /// The compressed bitstream is malformed or truncated.
    InvalidBitstream(String),
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBufferSize { expected, got } => {
                write!(f, "buffer size mismatch: expected {expected}, got {got}")
            }
            Self::InvalidDimensions { width, height } => {
                write!(f, "invalid dimensions: {width}x{height}")
            }
            Self::DimensionOverflow => write!(f, "dimensions overflow usize"),
            Self::InvalidBitstream(msg) => write!(f, "invalid bitstream: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CodecError {}
