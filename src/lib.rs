//! ALICE-Codec: Hyper-Fast 3D Wavelet Video/Audio Codec
//!
//! > "Time is just another spatial dimension."
//!
//! A radical codec that treats video as a 3D volume (x, y, t) and compresses
//! it using Integer Wavelet Transform + rANS entropy coding.
//!
//! # Architecture
//!
//! ```text
//! RGB Frames → YCoCg-R → 3D Wavelet → Quantize → rANS → Bitstream
//! ```
//!
//! # Key Innovations
//!
//! - **No I/P/B Frames**: Entire chunk processed as 3D volume
//! - **Integer Lifting**: No floating-point, perfect reconstruction
//! - **Analytical RDO**: One-shot optimal quantization, no iteration
//! - **rANS**: Near Shannon-limit compression at memory bandwidth speed
//!
//! # Example
//!
//! ```rust
//! use alice_codec::wavelet::{Wavelet1D, LiftingStep};
//!
//! // 1D Integer CDF 9/7 Wavelet
//! let mut signal = [1i32, 2, 3, 4, 5, 6, 7, 8];
//! let wavelet = Wavelet1D::cdf97();
//! wavelet.forward(&mut signal);
//!
//! // signal now contains [low-pass..., high-pass...]
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::unused_self
)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod color;
pub mod error;
pub mod pipeline;
pub mod quant;
pub mod rans;
pub mod segment;
pub mod wavelet;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "ml")]
pub mod ml_bridge;

#[cfg(feature = "db")]
pub mod db_bridge;

#[cfg(feature = "crypto")]
pub mod crypto_bridge;

#[cfg(feature = "cache")]
pub mod cache_bridge;

// Re-exports
pub use color::{rgb_to_ycocg_r, ycocg_r_to_rgb};
pub use error::CodecError;
pub use pipeline::{EncodedChunk, FrameDecoder, FrameEncoder, WaveletType};
pub use quant::{AnalyticalRDO, Quantizer};
pub use rans::{RansDecoder, RansEncoder, RansState};
pub use segment::{segment_by_chroma, segment_by_motion, SegmentConfig, SegmentResult};
pub use wavelet::{Wavelet1D, Wavelet2D, Wavelet3D};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default chunk size for 3D wavelet (frames)
pub const DEFAULT_CHUNK_SIZE: usize = 64;

/// Sub-band index for 3D wavelet decomposition
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SubBand3D {
    /// Low-Low-Low (static background, highest compression)
    LLL = 0,
    /// Low-Low-High (slow temporal change)
    LLH = 1,
    /// Low-High-Low (horizontal edges)
    LHL = 2,
    /// Low-High-High (horizontal + temporal)
    LHH = 3,
    /// High-Low-Low (vertical edges)
    HLL = 4,
    /// High-Low-High (vertical + temporal)
    HLH = 5,
    /// High-High-Low (diagonal edges)
    HHL = 6,
    /// High-High-High (highest frequency noise)
    HHH = 7,
}

impl SubBand3D {
    /// Returns true if this sub-band contains temporal high-frequency (motion)
    #[must_use]
    #[inline]
    pub const fn is_temporal_high(&self) -> bool {
        matches!(
            self,
            SubBand3D::LLH | SubBand3D::LHH | SubBand3D::HLH | SubBand3D::HHH
        )
    }

    /// Returns true if this is the lowest frequency sub-band
    #[must_use]
    #[inline]
    pub const fn is_dc(&self) -> bool {
        matches!(self, SubBand3D::LLL)
    }

    /// Recommended quantization strength (higher = more aggressive)
    #[must_use]
    #[inline]
    pub const fn quant_strength(&self) -> u8 {
        match self {
            SubBand3D::LLL => 1, // Preserve DC
            SubBand3D::LLH | SubBand3D::LHL | SubBand3D::HLL => 2,
            SubBand3D::LHH | SubBand3D::HLH | SubBand3D::HHL => 4,
            SubBand3D::HHH => 8, // Most aggressive
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subband_classification() {
        assert!(SubBand3D::LLL.is_dc());
        assert!(!SubBand3D::LLH.is_dc());

        assert!(SubBand3D::LLH.is_temporal_high());
        assert!(SubBand3D::HHH.is_temporal_high());
        assert!(!SubBand3D::LLL.is_temporal_high());
        assert!(!SubBand3D::HHL.is_temporal_high());
    }

    #[test]
    fn test_quant_strength() {
        assert_eq!(SubBand3D::LLL.quant_strength(), 1);
        assert_eq!(SubBand3D::HHH.quant_strength(), 8);
    }

    #[test]
    fn test_subband_all_variants_temporal_high() {
        // Exhaustive: exactly 4 sub-bands are temporal-high
        let temporal_high = [
            SubBand3D::LLH,
            SubBand3D::LHH,
            SubBand3D::HLH,
            SubBand3D::HHH,
        ];
        let temporal_low = [
            SubBand3D::LLL,
            SubBand3D::LHL,
            SubBand3D::HLL,
            SubBand3D::HHL,
        ];

        for sb in temporal_high {
            assert!(sb.is_temporal_high(), "{sb:?} should be temporal-high");
        }
        for sb in temporal_low {
            assert!(!sb.is_temporal_high(), "{sb:?} should NOT be temporal-high");
        }
    }

    #[test]
    fn test_subband_only_lll_is_dc() {
        let all = [
            SubBand3D::LLL,
            SubBand3D::LLH,
            SubBand3D::LHL,
            SubBand3D::LHH,
            SubBand3D::HLL,
            SubBand3D::HLH,
            SubBand3D::HHL,
            SubBand3D::HHH,
        ];
        for sb in all {
            if matches!(sb, SubBand3D::LLL) {
                assert!(sb.is_dc());
            } else {
                assert!(!sb.is_dc(), "{sb:?} should not be DC");
            }
        }
    }

    #[test]
    fn test_quant_strength_all_subbands() {
        // quant_strength must be monotonically non-decreasing with frequency content
        assert_eq!(SubBand3D::LLL.quant_strength(), 1);
        assert_eq!(SubBand3D::LLH.quant_strength(), 2);
        assert_eq!(SubBand3D::LHL.quant_strength(), 2);
        assert_eq!(SubBand3D::LHH.quant_strength(), 4);
        assert_eq!(SubBand3D::HLL.quant_strength(), 2);
        assert_eq!(SubBand3D::HLH.quant_strength(), 4);
        assert_eq!(SubBand3D::HHL.quant_strength(), 4);
        assert_eq!(SubBand3D::HHH.quant_strength(), 8);
    }

    #[test]
    fn test_subband_repr_u8_ordering() {
        // Verify repr(u8) values match expected ordering 0..7
        assert_eq!(SubBand3D::LLL as u8, 0);
        assert_eq!(SubBand3D::LLH as u8, 1);
        assert_eq!(SubBand3D::LHL as u8, 2);
        assert_eq!(SubBand3D::LHH as u8, 3);
        assert_eq!(SubBand3D::HLL as u8, 4);
        assert_eq!(SubBand3D::HLH as u8, 5);
        assert_eq!(SubBand3D::HHL as u8, 6);
        assert_eq!(SubBand3D::HHH as u8, 7);
    }

    #[test]
    fn test_version_and_constants() {
        assert!(!VERSION.is_empty());
        assert_eq!(DEFAULT_CHUNK_SIZE, 64);
    }
}
