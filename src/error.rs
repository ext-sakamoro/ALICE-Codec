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
    /// Quantization step size is not positive.
    InvalidQuantStep(i32),
    /// A quantized coefficient does not fit the 8-bit zigzag symbol alphabet
    /// (`|q| ≤ 127`); the encoder chooses the step so this cannot happen.
    SymbolOverflow(i32),
    /// A measure was given input it cannot measure: nothing to compare, a non-finite value, a zero-length
    /// normal or a flat image for a period (see [`crate::appearance`])
    InvalidInput(String),
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
            Self::InvalidQuantStep(step) => {
                write!(f, "quantization step must be positive, got {step}")
            }
            Self::SymbolOverflow(q) => {
                write!(
                    f,
                    "quantized coefficient {q} exceeds the 8-bit symbol range (|q| <= 127)"
                )
            }
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CodecError {}
