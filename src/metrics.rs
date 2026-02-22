//! Quality metrics for codec evaluation.
//!
//! Provides PSNR (Peak Signal-to-Noise Ratio) and MSE (Mean Squared Error)
//! measurements between original and reconstructed buffers.

use crate::error::CodecError;

/// Compute MSE (Mean Squared Error) between two byte buffers.
///
/// Returns `0.0` for empty buffers.
///
/// # Errors
///
/// Returns [`CodecError::InvalidBufferSize`] if `a` and `b` have different lengths.
#[inline]
pub fn mse(a: &[u8], b: &[u8]) -> Result<f64, CodecError> {
    if a.len() != b.len() {
        return Err(CodecError::InvalidBufferSize {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Ok(0.0);
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = f64::from(x) - f64::from(y);
            diff * diff
        })
        .sum();
    Ok(sum / a.len() as f64)
}

/// Compute PSNR (Peak Signal-to-Noise Ratio) between two byte buffers.
///
/// Returns `f64::INFINITY` when the buffers are identical (MSE = 0) or empty.
/// Higher values indicate better quality; typical video is 30–50 dB.
///
/// # Errors
///
/// Returns [`CodecError::InvalidBufferSize`] if `a` and `b` have different lengths.
///
/// # Example
///
/// ```
/// use alice_codec::metrics::psnr;
///
/// let original = [100u8, 150, 200];
/// let decoded  = [101u8, 149, 198];
/// let db = psnr(&original, &decoded).unwrap();
/// assert!(db > 30.0);
/// ```
#[inline]
pub fn psnr(a: &[u8], b: &[u8]) -> Result<f64, CodecError> {
    let mse_val = mse(a, b)?;
    if mse_val == 0.0 {
        return Ok(f64::INFINITY);
    }
    Ok(10.0 * libm::log10(255.0_f64 * 255.0 / mse_val))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psnr_identical() {
        let buf = [10u8, 20, 30, 40];
        assert!(psnr(&buf, &buf).unwrap().is_infinite());
    }

    #[test]
    fn test_psnr_empty() {
        let a: [u8; 0] = [];
        assert!(psnr(&a, &a).unwrap().is_infinite());
    }

    #[test]
    fn test_psnr_known_value() {
        // MSE = 1.0 → PSNR = 10 * log10(65025) ≈ 48.13
        let a = [100u8];
        let b = [101u8];
        let db = psnr(&a, &b).unwrap();
        assert!((db - 48.13).abs() < 0.1, "PSNR = {db}");
    }

    #[test]
    fn test_psnr_mismatched_lengths() {
        let a = [1u8, 2, 3];
        let b = [1u8, 2];
        assert!(psnr(&a, &b).is_err());
    }

    #[test]
    fn test_mse_zero() {
        let buf = [50u8; 10];
        assert!(mse(&buf, &buf).unwrap().abs() < f64::EPSILON);
    }

    #[test]
    fn test_mse_known_value() {
        let a = [0u8, 0];
        let b = [3u8, 4];
        // MSE = (9 + 16) / 2 = 12.5
        assert!((mse(&a, &b).unwrap() - 12.5).abs() < 1e-10);
    }

    #[test]
    fn test_psnr_symmetry() {
        let a = [10u8, 20, 30];
        let b = [15u8, 25, 35];
        let ab = psnr(&a, &b).unwrap();
        let ba = psnr(&b, &a).unwrap();
        assert!((ab - ba).abs() < f64::EPSILON);
    }

    #[test]
    fn test_psnr_quality_range() {
        // Small difference → high PSNR (good quality)
        let a = [128u8; 100];
        let mut b = [128u8; 100];
        b[0] = 129;
        let db = psnr(&a, &b).unwrap();
        assert!(db > 40.0, "Small diff should give high PSNR, got {db}");
    }
}
