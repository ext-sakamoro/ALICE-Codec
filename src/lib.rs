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

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod wavelet;
pub mod color;
pub mod rans;
pub mod quant;
pub mod segment;

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
pub use wavelet::{Wavelet1D, Wavelet2D, Wavelet3D};
pub use color::{rgb_to_ycocg_r, ycocg_r_to_rgb};
pub use rans::{RansEncoder, RansDecoder, RansState};
pub use quant::{Quantizer, AnalyticalRDO};
pub use segment::{SegmentConfig, SegmentResult, segment_by_motion, segment_by_chroma};

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
    #[inline]
    pub const fn is_temporal_high(&self) -> bool {
        matches!(self, SubBand3D::LLH | SubBand3D::LHH | SubBand3D::HLH | SubBand3D::HHH)
    }

    /// Returns true if this is the lowest frequency sub-band
    #[inline]
    pub const fn is_dc(&self) -> bool {
        matches!(self, SubBand3D::LLL)
    }

    /// Recommended quantization strength (higher = more aggressive)
    #[inline]
    pub const fn quant_strength(&self) -> u8 {
        match self {
            SubBand3D::LLL => 1,   // Preserve DC
            SubBand3D::LLH => 2,
            SubBand3D::LHL => 2,
            SubBand3D::LHH => 4,
            SubBand3D::HLL => 2,
            SubBand3D::HLH => 4,
            SubBand3D::HHL => 4,
            SubBand3D::HHH => 8,   // Most aggressive
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
}
