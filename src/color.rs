//! YCoCg-R Color Space Conversion
//!
//! Reversible integer color space with better decorrelation than YCbCr.
//!
//! # Transform
//!
//! ```text
//! RGB → YCoCg-R (Forward):
//!   Co = R - B
//!   t  = B + (Co >> 1)
//!   Cg = G - t
//!   Y  = t + (Cg >> 1)
//!
//! YCoCg-R → RGB (Inverse):
//!   t  = Y - (Cg >> 1)
//!   G  = Cg + t
//!   B  = t - (Co >> 1)
//!   R  = Co + B
//! ```
//!
//! # Properties
//!
//! - **Lossless**: Perfect round-trip for all 8-bit RGB values
//! - **Better decorrelation**: Y captures ~95% of energy
//! - **Simple**: Only shifts and adds, no multiplication

use crate::error::CodecError;

/// YCoCg-R pixel (Y, Co, Cg components)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct YCoCgR {
    pub y: i16,
    pub co: i16,
    pub cg: i16,
}

/// RGB pixel (R, G, B components)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RGB {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl RGB {
    /// Create a new RGB pixel.
    #[must_use]
    #[inline]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }
}

impl YCoCgR {
    /// Create a new YCoCg-R pixel.
    #[must_use]
    #[inline]
    pub const fn new(y: i16, co: i16, cg: i16) -> Self {
        Self { y, co, cg }
    }
}

/// Convert single RGB pixel to YCoCg-R
///
/// # Example
///
/// ```
/// use alice_codec::color::{rgb_to_ycocg_r_pixel, RGB};
///
/// let rgb = RGB::new(100, 150, 200);
/// let ycocg = rgb_to_ycocg_r_pixel(rgb);
/// ```
#[must_use]
#[inline]
pub fn rgb_to_ycocg_r_pixel(rgb: RGB) -> YCoCgR {
    let r = rgb.r as i16;
    let g = rgb.g as i16;
    let b = rgb.b as i16;

    let co = r - b;
    let t = b + (co >> 1);
    let cg = g - t;
    let y = t + (cg >> 1);

    YCoCgR { y, co, cg }
}

/// Convert single YCoCg-R pixel back to RGB
///
/// # Example
///
/// ```
/// use alice_codec::color::{ycocg_r_to_rgb_pixel, YCoCgR};
///
/// let ycocg = YCoCgR::new(150, -100, 0);
/// let rgb = ycocg_r_to_rgb_pixel(ycocg);
/// ```
#[must_use]
#[inline]
pub fn ycocg_r_to_rgb_pixel(ycocg: YCoCgR) -> RGB {
    let t = ycocg.y - (ycocg.cg >> 1);
    let g = ycocg.cg + t;
    let b = t - (ycocg.co >> 1);
    let r = ycocg.co + b;

    // Clamp to [0, 255] for safety (should be exact for valid inputs)
    RGB {
        r: r.clamp(0, 255) as u8,
        g: g.clamp(0, 255) as u8,
        b: b.clamp(0, 255) as u8,
    }
}

/// Convert RGB buffer to YCoCg-R (in-place capable via separate buffers)
///
/// # Arguments
///
/// * `rgb` - Input RGB pixels
/// * `y_out` - Output Y channel
/// * `co_out` - Output Co channel
/// * `cg_out` - Output Cg channel
///
/// # Errors
///
/// Returns `CodecError::InvalidBufferSize` if any output buffer is smaller
/// than the input.
pub fn rgb_to_ycocg_r(
    rgb: &[RGB],
    y_out: &mut [i16],
    co_out: &mut [i16],
    cg_out: &mut [i16],
) -> Result<(), CodecError> {
    let n = rgb.len();
    if y_out.len() < n || co_out.len() < n || cg_out.len() < n {
        let min_out = y_out.len().min(co_out.len()).min(cg_out.len());
        return Err(CodecError::InvalidBufferSize {
            expected: n,
            got: min_out,
        });
    }

    for (i, &pixel) in rgb.iter().enumerate() {
        let ycocg = rgb_to_ycocg_r_pixel(pixel);
        y_out[i] = ycocg.y;
        co_out[i] = ycocg.co;
        cg_out[i] = ycocg.cg;
    }
    Ok(())
}

/// Convert YCoCg-R channels back to RGB
///
/// # Arguments
///
/// * `y` - Y channel
/// * `co` - Co channel
/// * `cg` - Cg channel
/// * `rgb_out` - Output RGB pixels
///
/// # Errors
///
/// Returns `CodecError::InvalidBufferSize` if channels have different lengths
/// or if `rgb_out` is smaller than the channel length.
pub fn ycocg_r_to_rgb(
    y: &[i16],
    co: &[i16],
    cg: &[i16],
    rgb_out: &mut [RGB],
) -> Result<(), CodecError> {
    let n = y.len();
    if co.len() != n || cg.len() != n {
        return Err(CodecError::InvalidBufferSize {
            expected: n,
            got: co.len().min(cg.len()),
        });
    }
    if rgb_out.len() < n {
        return Err(CodecError::InvalidBufferSize {
            expected: n,
            got: rgb_out.len(),
        });
    }

    for i in 0..n {
        let ycocg = YCoCgR::new(y[i], co[i], cg[i]);
        rgb_out[i] = ycocg_r_to_rgb_pixel(ycocg);
    }
    Ok(())
}

/// Convert interleaved RGB bytes to planar YCoCg-R
///
/// Input format: `[R0, G0, B0, R1, G1, B1, ...]`
///
/// # Errors
///
/// Returns `CodecError::InvalidBufferSize` if `rgb_bytes.len()` is not a
/// multiple of 3, or if any output buffer is smaller than the pixel count.
pub fn rgb_bytes_to_ycocg_r(
    rgb_bytes: &[u8],
    y_out: &mut [i16],
    co_out: &mut [i16],
    cg_out: &mut [i16],
) -> Result<(), CodecError> {
    if !rgb_bytes.len().is_multiple_of(3) {
        return Err(CodecError::InvalidBufferSize {
            expected: (rgb_bytes.len() / 3 + 1) * 3,
            got: rgb_bytes.len(),
        });
    }
    let n_pixels = rgb_bytes.len() / 3;
    if y_out.len() < n_pixels || co_out.len() < n_pixels || cg_out.len() < n_pixels {
        let min_out = y_out.len().min(co_out.len()).min(cg_out.len());
        return Err(CodecError::InvalidBufferSize {
            expected: n_pixels,
            got: min_out,
        });
    }

    for i in 0..n_pixels {
        let r = rgb_bytes[i * 3] as i16;
        let g = rgb_bytes[i * 3 + 1] as i16;
        let b = rgb_bytes[i * 3 + 2] as i16;

        let co = r - b;
        let t = b + (co >> 1);
        let cg = g - t;
        let y = t + (cg >> 1);

        y_out[i] = y;
        co_out[i] = co;
        cg_out[i] = cg;
    }
    Ok(())
}

/// Convert planar YCoCg-R to interleaved RGB bytes
///
/// Output format: `[R0, G0, B0, R1, G1, B1, ...]`
///
/// # Errors
///
/// Returns `CodecError::InvalidBufferSize` if channels have different lengths
/// or if `rgb_out` is smaller than `y.len() * 3`.
pub fn ycocg_r_to_rgb_bytes(
    y: &[i16],
    co: &[i16],
    cg: &[i16],
    rgb_out: &mut [u8],
) -> Result<(), CodecError> {
    let n = y.len();
    if co.len() != n || cg.len() != n {
        return Err(CodecError::InvalidBufferSize {
            expected: n,
            got: co.len().min(cg.len()),
        });
    }
    if rgb_out.len() < n * 3 {
        return Err(CodecError::InvalidBufferSize {
            expected: n * 3,
            got: rgb_out.len(),
        });
    }

    for i in 0..n {
        let t = y[i] - (cg[i] >> 1);
        let g = cg[i] + t;
        let b = t - (co[i] >> 1);
        let r = co[i] + b;

        rgb_out[i * 3] = r.clamp(0, 255) as u8;
        rgb_out[i * 3 + 1] = g.clamp(0, 255) as u8;
        rgb_out[i * 3 + 2] = b.clamp(0, 255) as u8;
    }
    Ok(())
}

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
mod simd {
    use core::arch::x86_64::*;

    /// SIMD RGB to YCoCg-R conversion (8 pixels at a time)
    ///
    /// # Safety
    ///
    /// Requires AVX2 support.
    ///
    /// # Panics
    ///
    /// Panics if the input and output slices have different lengths.
    #[target_feature(enable = "avx2")]
    pub unsafe fn rgb_to_ycocg_r_avx2(
        r: &[i16],
        g: &[i16],
        b: &[i16],
        y_out: &mut [i16],
        co_out: &mut [i16],
        cg_out: &mut [i16],
    ) {
        let n = r.len();
        assert_eq!(n, g.len());
        assert_eq!(n, b.len());
        assert!(y_out.len() >= n);
        assert!(co_out.len() >= n);
        assert!(cg_out.len() >= n);

        let chunks = n / 16;

        for i in 0..chunks {
            let offset = i * 16;

            // Load 16 pixels
            let r_vec = _mm256_loadu_si256(r.as_ptr().add(offset) as *const __m256i);
            let g_vec = _mm256_loadu_si256(g.as_ptr().add(offset) as *const __m256i);
            let b_vec = _mm256_loadu_si256(b.as_ptr().add(offset) as *const __m256i);

            // Co = R - B
            let co_vec = _mm256_sub_epi16(r_vec, b_vec);

            // t = B + (Co >> 1)
            let co_shift = _mm256_srai_epi16(co_vec, 1);
            let t_vec = _mm256_add_epi16(b_vec, co_shift);

            // Cg = G - t
            let cg_vec = _mm256_sub_epi16(g_vec, t_vec);

            // Y = t + (Cg >> 1)
            let cg_shift = _mm256_srai_epi16(cg_vec, 1);
            let y_vec = _mm256_add_epi16(t_vec, cg_shift);

            // Store results
            _mm256_storeu_si256(y_out.as_mut_ptr().add(offset) as *mut __m256i, y_vec);
            _mm256_storeu_si256(co_out.as_mut_ptr().add(offset) as *mut __m256i, co_vec);
            _mm256_storeu_si256(cg_out.as_mut_ptr().add(offset) as *mut __m256i, cg_vec);
        }

        // Handle remaining pixels
        let remainder = chunks * 16;
        for i in remainder..n {
            let co = r[i] - b[i];
            let t = b[i] + (co >> 1);
            let cg = g[i] - t;
            let y = t + (cg >> 1);

            y_out[i] = y;
            co_out[i] = co;
            cg_out[i] = cg;
        }
    }

    /// SIMD YCoCg-R to RGB conversion (8 pixels at a time)
    ///
    /// # Safety
    ///
    /// Requires AVX2 support.
    ///
    /// # Panics
    ///
    /// Panics if the input and output slices have different lengths.
    #[target_feature(enable = "avx2")]
    pub unsafe fn ycocg_r_to_rgb_avx2(
        y: &[i16],
        co: &[i16],
        cg: &[i16],
        r_out: &mut [i16],
        g_out: &mut [i16],
        b_out: &mut [i16],
    ) {
        let n = y.len();
        assert_eq!(n, co.len());
        assert_eq!(n, cg.len());
        assert!(r_out.len() >= n);
        assert!(g_out.len() >= n);
        assert!(b_out.len() >= n);

        let chunks = n / 16;

        for i in 0..chunks {
            let offset = i * 16;

            // Load 16 pixels
            let y_vec = _mm256_loadu_si256(y.as_ptr().add(offset) as *const __m256i);
            let co_vec = _mm256_loadu_si256(co.as_ptr().add(offset) as *const __m256i);
            let cg_vec = _mm256_loadu_si256(cg.as_ptr().add(offset) as *const __m256i);

            // t = Y - (Cg >> 1)
            let cg_shift = _mm256_srai_epi16(cg_vec, 1);
            let t_vec = _mm256_sub_epi16(y_vec, cg_shift);

            // G = Cg + t
            let g_vec = _mm256_add_epi16(cg_vec, t_vec);

            // B = t - (Co >> 1)
            let co_shift = _mm256_srai_epi16(co_vec, 1);
            let b_vec = _mm256_sub_epi16(t_vec, co_shift);

            // R = Co + B
            let r_vec = _mm256_add_epi16(co_vec, b_vec);

            // Store results
            _mm256_storeu_si256(r_out.as_mut_ptr().add(offset) as *mut __m256i, r_vec);
            _mm256_storeu_si256(g_out.as_mut_ptr().add(offset) as *mut __m256i, g_vec);
            _mm256_storeu_si256(b_out.as_mut_ptr().add(offset) as *mut __m256i, b_vec);
        }

        // Handle remaining pixels
        let remainder = chunks * 16;
        for i in remainder..n {
            let t = y[i] - (cg[i] >> 1);
            let g = cg[i] + t;
            let b = t - (co[i] >> 1);
            let r = co[i] + b;

            r_out[i] = r;
            g_out[i] = g;
            b_out[i] = b;
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
pub use simd::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_pixel() {
        // Test all corner cases
        let test_cases = [
            RGB::new(0, 0, 0),
            RGB::new(255, 255, 255),
            RGB::new(255, 0, 0),
            RGB::new(0, 255, 0),
            RGB::new(0, 0, 255),
            RGB::new(128, 128, 128),
            RGB::new(100, 150, 200),
        ];

        for rgb in test_cases {
            let ycocg = rgb_to_ycocg_r_pixel(rgb);
            let back = ycocg_r_to_rgb_pixel(ycocg);
            assert_eq!(rgb, back, "Roundtrip failed for {rgb:?}");
        }
    }

    #[test]
    fn test_roundtrip_exhaustive_samples() {
        // Test a sample of the full RGB space
        for r in (0..=255).step_by(17) {
            for g in (0..=255).step_by(17) {
                for b in (0..=255).step_by(17) {
                    let rgb = RGB::new(r, g, b);
                    let ycocg = rgb_to_ycocg_r_pixel(rgb);
                    let back = ycocg_r_to_rgb_pixel(ycocg);
                    assert_eq!(rgb, back, "Roundtrip failed for {rgb:?}");
                }
            }
        }
    }

    #[test]
    fn test_buffer_conversion() {
        let rgb = [
            RGB::new(100, 150, 200),
            RGB::new(50, 100, 150),
            RGB::new(200, 100, 50),
        ];

        let mut y = [0i16; 3];
        let mut co = [0i16; 3];
        let mut cg = [0i16; 3];
        let mut rgb_out = [RGB::new(0, 0, 0); 3];

        rgb_to_ycocg_r(&rgb, &mut y, &mut co, &mut cg).unwrap();
        ycocg_r_to_rgb(&y, &co, &cg, &mut rgb_out).unwrap();

        assert_eq!(rgb, rgb_out);
    }

    #[test]
    fn test_byte_conversion() {
        let rgb_bytes = [100u8, 150, 200, 50, 100, 150, 200, 100, 50];

        let mut y = [0i16; 3];
        let mut co = [0i16; 3];
        let mut cg = [0i16; 3];
        let mut rgb_out = [0u8; 9];

        rgb_bytes_to_ycocg_r(&rgb_bytes, &mut y, &mut co, &mut cg).unwrap();
        ycocg_r_to_rgb_bytes(&y, &co, &cg, &mut rgb_out).unwrap();

        assert_eq!(rgb_bytes, rgb_out);
    }

    #[test]
    fn test_y_dominance() {
        // Y channel should capture most of the energy for typical images
        let rgb = RGB::new(128, 128, 128);
        let ycocg = rgb_to_ycocg_r_pixel(rgb);

        // For gray, Co and Cg should be 0
        assert_eq!(ycocg.co, 0);
        assert_eq!(ycocg.cg, 0);
        assert_eq!(ycocg.y, 128);
    }

    #[test]
    fn test_empty_buffer_conversion() {
        // Zero-length buffers should work without error
        let rgb: [RGB; 0] = [];
        let mut y = [];
        let mut co = [];
        let mut cg = [];
        let mut rgb_out: [RGB; 0] = [];

        rgb_to_ycocg_r(&rgb, &mut y, &mut co, &mut cg).unwrap();
        ycocg_r_to_rgb(&y, &co, &cg, &mut rgb_out).unwrap();
    }

    #[test]
    fn test_empty_byte_conversion() {
        let rgb_bytes: [u8; 0] = [];
        let mut y: [i16; 0] = [];
        let mut co: [i16; 0] = [];
        let mut cg: [i16; 0] = [];
        let mut rgb_out: [u8; 0] = [];

        rgb_bytes_to_ycocg_r(&rgb_bytes, &mut y, &mut co, &mut cg).unwrap();
        ycocg_r_to_rgb_bytes(&y, &co, &cg, &mut rgb_out).unwrap();
    }

    #[test]
    fn test_buffer_size_error() {
        let rgb = [RGB::new(100, 150, 200); 3];
        let mut y = [0i16; 2]; // too small
        let mut co = [0i16; 3];
        let mut cg = [0i16; 3];

        let result = rgb_to_ycocg_r(&rgb, &mut y, &mut co, &mut cg);
        assert!(result.is_err());
    }

    #[test]
    fn test_byte_not_multiple_of_3() {
        let rgb_bytes = [1u8, 2, 3, 4]; // 4 bytes, not multiple of 3
        let mut y = [0i16; 2];
        let mut co = [0i16; 2];
        let mut cg = [0i16; 2];

        let result = rgb_bytes_to_ycocg_r(&rgb_bytes, &mut y, &mut co, &mut cg);
        assert!(result.is_err());
    }

    #[test]
    fn test_pure_red_pixel() {
        let rgb = RGB::new(255, 0, 0);
        let ycocg = rgb_to_ycocg_r_pixel(rgb);
        // Co = R - B = 255, t = 0 + 127 = 127, Cg = 0 - 127 = -127, Y = 127 + (-64) = 63
        assert_eq!(ycocg.co, 255);
        let back = ycocg_r_to_rgb_pixel(ycocg);
        assert_eq!(back, rgb);
    }

    #[test]
    fn test_pure_green_pixel() {
        let rgb = RGB::new(0, 255, 0);
        let ycocg = rgb_to_ycocg_r_pixel(rgb);
        let back = ycocg_r_to_rgb_pixel(ycocg);
        assert_eq!(back, rgb);
        // Green should produce high Cg
        assert!(ycocg.cg > 0, "Cg should be positive for pure green");
    }

    #[test]
    fn test_pure_blue_pixel() {
        let rgb = RGB::new(0, 0, 255);
        let ycocg = rgb_to_ycocg_r_pixel(rgb);
        let back = ycocg_r_to_rgb_pixel(ycocg);
        assert_eq!(back, rgb);
        // Blue: Co = R - B = -255 (negative)
        assert!(ycocg.co < 0, "Co should be negative for pure blue");
    }

    #[test]
    fn test_grayscale_decorrelation() {
        // All grayscale pixels should have Co=0, Cg=0
        for v in (0..=255).step_by(5) {
            let rgb = RGB::new(v, v, v);
            let ycocg = rgb_to_ycocg_r_pixel(rgb);
            assert_eq!(ycocg.co, 0, "Co should be 0 for gray level {v}");
            assert_eq!(ycocg.cg, 0, "Cg should be 0 for gray level {v}");
            assert_eq!(ycocg.y, v as i16, "Y should equal gray level {v}");
        }
    }
}
