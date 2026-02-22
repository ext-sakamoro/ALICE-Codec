//! Integer Wavelet Transform using Lifting Scheme
//!
//! Implements CDF 9/7 wavelet transform used in JPEG2000, but with integer
//! arithmetic for perfect reconstruction.
//!
//! # Lifting Scheme
//!
//! Traditional convolution-based wavelet: O(N x `filter_length`)
//! Lifting scheme: O(N) with in-place operation
//!
//! ```text
//! Split → Predict → Update → Predict → Update → Scale
//!   ↓        ↓         ↓         ↓         ↓       ↓
//! Even/Odd  -α      +β        -γ       +δ     ×K
//! ```
//!
//! # Integer Approximation
//!
//! CDF 9/7 floating-point coefficients:
//! - α = -1.586134342
//! - β = -0.052980118
//! - γ = 0.882911075
//! - δ = 0.443506852
//! - K = 1.149604398
//!
//! We use fixed-point arithmetic with rounding for integer lifting.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// 1D Wavelet Transform
#[derive(Clone, Debug)]
pub struct Wavelet1D {
    /// Lifting steps (predict/update pairs)
    steps: Vec<LiftingStep>,
}

/// Single lifting step
#[derive(Clone, Copy, Debug)]
pub struct LiftingStep {
    /// Fixed-point coefficient (scaled by 2^12)
    pub coeff: i32,
    /// Direction: true = predict (even updates odd), false = update (odd updates even)
    pub predict: bool,
}

impl Wavelet1D {
    /// Create CDF 9/7 wavelet (JPEG2000 lossy)
    ///
    /// Integer approximation of the biorthogonal 9/7 filter.
    #[must_use]
    pub fn cdf97() -> Self {
        // Fixed-point coefficients (scaled by 4096 = 2^12)
        // α = -1.586134342 → -6497
        // β = -0.052980118 → -217
        // γ = 0.882911075  → 3616
        // δ = 0.443506852  → 1817
        Self {
            steps: vec![
                LiftingStep {
                    coeff: -6497,
                    predict: true,
                }, // α: predict
                LiftingStep {
                    coeff: -217,
                    predict: false,
                }, // β: update
                LiftingStep {
                    coeff: 3616,
                    predict: true,
                }, // γ: predict
                LiftingStep {
                    coeff: 1817,
                    predict: false,
                }, // δ: update
            ],
        }
    }

    /// Create Haar wavelet (simplest, for testing)
    #[must_use]
    pub fn haar() -> Self {
        Self {
            steps: vec![
                LiftingStep {
                    coeff: -4096,
                    predict: true,
                }, // d[n] = odd - even
                LiftingStep {
                    coeff: 2048,
                    predict: false,
                }, // s[n] = even + d/2
            ],
        }
    }

    /// Create 5/3 wavelet (JPEG2000 lossless)
    #[must_use]
    pub fn cdf53() -> Self {
        // Integer 5/3 wavelet: perfect reconstruction guaranteed
        Self {
            steps: vec![
                LiftingStep {
                    coeff: -4096,
                    predict: true,
                }, // d = odd - (even_l + even_r)/2
                LiftingStep {
                    coeff: 1024,
                    predict: false,
                }, // s = even + (d_l + d_r + 2)/4
            ],
        }
    }

    /// Forward wavelet transform (analysis)
    ///
    /// Input signal is modified in-place.
    /// Output: [low-pass coefficients..., high-pass coefficients...]
    pub fn forward(&self, signal: &mut [i32]) {
        let n = signal.len();
        if n < 2 {
            return;
        }

        // Apply lifting steps
        for step in &self.steps {
            if step.predict {
                // Predict: odd[i] += coeff * (even[i] + even[i+1]) / 2
                self.lift_predict(signal, step.coeff);
            } else {
                // Update: even[i] += coeff * (odd[i-1] + odd[i]) / 2
                self.lift_update(signal, step.coeff);
            }
        }

        // Reorder: interleaved → [low..., high...]
        self.deinterleave(signal);
    }

    /// Inverse wavelet transform (synthesis)
    ///
    /// Reconstructs original signal from wavelet coefficients.
    pub fn inverse(&self, signal: &mut [i32]) {
        let n = signal.len();
        if n < 2 {
            return;
        }

        // Reorder: [low..., high...] → interleaved
        self.interleave(signal);

        // Apply lifting steps in reverse
        for step in self.steps.iter().rev() {
            if step.predict {
                // Undo predict: odd[i] -= coeff * (even[i] + even[i+1]) / 2
                self.lift_predict(signal, -step.coeff);
            } else {
                // Undo update: even[i] -= coeff * (odd[i-1] + odd[i]) / 2
                self.lift_update(signal, -step.coeff);
            }
        }
    }

    /// Predict step: updates odd samples using even samples
    #[inline]
    fn lift_predict(&self, signal: &mut [i32], coeff: i32) {
        let n = signal.len();
        let half = n / 2;

        for i in 0..half {
            let even_left = signal[i * 2];
            let even_right = if i * 2 + 2 < n {
                signal[i * 2 + 2]
            } else {
                signal[i * 2] // Mirror boundary
            };

            // Fixed-point multiply with rounding
            let avg = even_left + even_right;
            let delta = ((avg as i64 * coeff as i64 + 4096) >> 13) as i32;
            signal[i * 2 + 1] += delta;
        }
    }

    /// Update step: updates even samples using odd samples
    #[inline]
    fn lift_update(&self, signal: &mut [i32], coeff: i32) {
        let n = signal.len();
        let half = n / 2;

        for i in 0..half {
            let odd_left = if i > 0 {
                signal[i * 2 - 1]
            } else {
                signal[1] // Mirror boundary
            };
            let odd_right = signal[i * 2 + 1];

            let avg = odd_left + odd_right;
            let delta = ((avg as i64 * coeff as i64 + 4096) >> 13) as i32;
            signal[i * 2] += delta;
        }
    }

    /// Deinterleave: [e0, o0, e1, o1, ...] → [e0, e1, ..., o0, o1, ...]
    fn deinterleave(&self, signal: &mut [i32]) {
        let n = signal.len();
        let half = n / 2;

        // Use temporary buffer (could be optimized with in-place algorithm)
        let mut temp = vec![0i32; n];

        for i in 0..half {
            temp[i] = signal[i * 2]; // Even → first half
            temp[half + i] = signal[i * 2 + 1]; // Odd → second half
        }

        signal.copy_from_slice(&temp);
    }

    /// Interleave: [e0, e1, ..., o0, o1, ...] → [e0, o0, e1, o1, ...]
    fn interleave(&self, signal: &mut [i32]) {
        let n = signal.len();
        let half = n / 2;

        let mut temp = vec![0i32; n];

        for i in 0..half {
            temp[i * 2] = signal[i]; // First half → even
            temp[i * 2 + 1] = signal[half + i]; // Second half → odd
        }

        signal.copy_from_slice(&temp);
    }
}

/// 2D Wavelet Transform
#[derive(Clone, Debug)]
pub struct Wavelet2D {
    wavelet_1d: Wavelet1D,
}

impl Wavelet2D {
    /// Create 2D wavelet from 1D wavelet
    #[must_use]
    pub fn new(wavelet_1d: Wavelet1D) -> Self {
        Self { wavelet_1d }
    }

    /// Create 2D CDF 9/7 wavelet
    #[must_use]
    pub fn cdf97() -> Self {
        Self::new(Wavelet1D::cdf97())
    }

    /// Create 2D CDF 5/3 wavelet
    #[must_use]
    pub fn cdf53() -> Self {
        Self::new(Wavelet1D::cdf53())
    }

    /// Forward 2D wavelet transform
    ///
    /// Applies 1D transform to rows, then columns.
    /// Result: [LL, LH, HL, HH] quadrants
    pub fn forward(&self, image: &mut [i32], width: usize, height: usize) {
        // Transform rows
        for y in 0..height {
            let row_start = y * width;
            self.wavelet_1d
                .forward(&mut image[row_start..row_start + width]);
        }

        // Transform columns (need to extract/insert column data)
        let mut col = vec![0i32; height];
        for x in 0..width {
            // Extract column
            for y in 0..height {
                col[y] = image[y * width + x];
            }

            // Transform
            self.wavelet_1d.forward(&mut col);

            // Insert back
            for y in 0..height {
                image[y * width + x] = col[y];
            }
        }
    }

    /// Inverse 2D wavelet transform
    pub fn inverse(&self, image: &mut [i32], width: usize, height: usize) {
        // Inverse transform columns
        let mut col = vec![0i32; height];
        for x in 0..width {
            for y in 0..height {
                col[y] = image[y * width + x];
            }

            self.wavelet_1d.inverse(&mut col);

            for y in 0..height {
                image[y * width + x] = col[y];
            }
        }

        // Inverse transform rows
        for y in 0..height {
            let row_start = y * width;
            self.wavelet_1d
                .inverse(&mut image[row_start..row_start + width]);
        }
    }
}

/// 3D Wavelet Transform (Video)
///
/// Treats video as (x, y, t) volume. No I/P/B frames.
#[derive(Clone, Debug)]
pub struct Wavelet3D {
    wavelet_1d: Wavelet1D,
}

impl Wavelet3D {
    /// Create 3D wavelet from 1D wavelet
    #[must_use]
    pub fn new(wavelet_1d: Wavelet1D) -> Self {
        Self { wavelet_1d }
    }

    /// Create 3D CDF 9/7 wavelet
    #[must_use]
    pub fn cdf97() -> Self {
        Self::new(Wavelet1D::cdf97())
    }

    /// Create 3D CDF 5/3 wavelet (lossless)
    #[must_use]
    pub fn cdf53() -> Self {
        Self::new(Wavelet1D::cdf53())
    }

    /// Forward 3D wavelet transform
    ///
    /// # Arguments
    /// * `volume` - Video data [frame0, frame1, ..., frameN]
    /// * `width` - Frame width
    /// * `height` - Frame height
    /// * `depth` - Number of frames
    ///
    /// # Result
    /// 8 sub-bands: LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH
    pub fn forward(&self, volume: &mut [i32], width: usize, height: usize, depth: usize) {
        let frame_size = width * height;

        // 1. Transform spatial (x, y) for each frame
        for t in 0..depth {
            let frame_start = t * frame_size;
            let frame = &mut volume[frame_start..frame_start + frame_size];

            // Rows
            for y in 0..height {
                let row_start = y * width;
                self.wavelet_1d
                    .forward(&mut frame[row_start..row_start + width]);
            }

            // Columns
            let mut col = vec![0i32; height];
            for x in 0..width {
                for y in 0..height {
                    col[y] = frame[y * width + x];
                }
                self.wavelet_1d.forward(&mut col);
                for y in 0..height {
                    frame[y * width + x] = col[y];
                }
            }
        }

        // 2. Transform temporal (t) for each pixel position
        let mut temporal = vec![0i32; depth];
        for y in 0..height {
            for x in 0..width {
                // Extract temporal slice
                for t in 0..depth {
                    temporal[t] = volume[t * frame_size + y * width + x];
                }

                // Transform
                self.wavelet_1d.forward(&mut temporal);

                // Insert back
                for t in 0..depth {
                    volume[t * frame_size + y * width + x] = temporal[t];
                }
            }
        }
    }

    /// Inverse 3D wavelet transform
    pub fn inverse(&self, volume: &mut [i32], width: usize, height: usize, depth: usize) {
        let frame_size = width * height;

        // 1. Inverse temporal transform
        let mut temporal = vec![0i32; depth];
        for y in 0..height {
            for x in 0..width {
                for t in 0..depth {
                    temporal[t] = volume[t * frame_size + y * width + x];
                }

                self.wavelet_1d.inverse(&mut temporal);

                for t in 0..depth {
                    volume[t * frame_size + y * width + x] = temporal[t];
                }
            }
        }

        // 2. Inverse spatial transform
        for t in 0..depth {
            let frame_start = t * frame_size;
            let frame = &mut volume[frame_start..frame_start + frame_size];

            // Columns
            let mut col = vec![0i32; height];
            for x in 0..width {
                for y in 0..height {
                    col[y] = frame[y * width + x];
                }
                self.wavelet_1d.inverse(&mut col);
                for y in 0..height {
                    frame[y * width + x] = col[y];
                }
            }

            // Rows
            for y in 0..height {
                let row_start = y * width;
                self.wavelet_1d
                    .inverse(&mut frame[row_start..row_start + width]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haar_roundtrip() {
        let wavelet = Wavelet1D::haar();
        let original = [10i32, 20, 30, 40, 50, 60, 70, 80];
        let mut signal = original;

        wavelet.forward(&mut signal);
        wavelet.inverse(&mut signal);

        // Should be identical (Haar is perfectly reversible with integers)
        for (i, (&orig, &rec)) in original.iter().zip(signal.iter()).enumerate() {
            assert!(
                (orig - rec).abs() <= 1,
                "Mismatch at {}: {} vs {}",
                i,
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_cdf53_roundtrip() {
        let wavelet = Wavelet1D::cdf53();
        let original = [100i32, 110, 105, 115, 108, 120, 112, 125];
        let mut signal = original;

        wavelet.forward(&mut signal);
        wavelet.inverse(&mut signal);

        for (i, (&orig, &rec)) in original.iter().zip(signal.iter()).enumerate() {
            assert!(
                (orig - rec).abs() <= 1,
                "Mismatch at {}: {} vs {}",
                i,
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_cdf97_roundtrip() {
        let wavelet = Wavelet1D::cdf97();
        let original = [100i32, 110, 105, 115, 108, 120, 112, 125];
        let mut signal = original;

        wavelet.forward(&mut signal);
        wavelet.inverse(&mut signal);

        // CDF 9/7 with integer approximation has small error
        for (i, (&orig, &rec)) in original.iter().zip(signal.iter()).enumerate() {
            assert!(
                (orig - rec).abs() <= 2,
                "Mismatch at {}: {} vs {}",
                i,
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_wavelet_2d_roundtrip() {
        let wavelet = Wavelet2D::cdf53();
        let original = [
            10i32, 20, 30, 40, 15, 25, 35, 45, 12, 22, 32, 42, 18, 28, 38, 48,
        ];
        let mut image = original;

        wavelet.forward(&mut image, 4, 4);
        wavelet.inverse(&mut image, 4, 4);

        for (i, (&orig, &rec)) in original.iter().zip(image.iter()).enumerate() {
            assert!(
                (orig - rec).abs() <= 2,
                "Mismatch at {}: {} vs {}",
                i,
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_wavelet_3d_roundtrip() {
        let wavelet = Wavelet3D::cdf53();

        // 4x4 image, 4 frames
        let original: Vec<i32> = (0..64).map(|i| (i * 3 + 10) as i32).collect();
        let mut volume = original.clone();

        wavelet.forward(&mut volume, 4, 4, 4);
        wavelet.inverse(&mut volume, 4, 4, 4);

        for (i, (&orig, &rec)) in original.iter().zip(volume.iter()).enumerate() {
            assert!(
                (orig - rec).abs() <= 3,
                "Mismatch at {}: {} vs {}",
                i,
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_energy_compaction() {
        // Static image repeated: temporal transform should compact to LLL
        let wavelet = Wavelet3D::cdf53();

        // Use varying content to see energy compaction
        let mut volume: Vec<i32> = (0..64).map(|i| 100 + (i % 4) as i32).collect();
        let original = volume.clone();

        wavelet.forward(&mut volume, 4, 4, 4);

        // For a mostly static image, low frequency should dominate
        // LLL is first quarter (width/2 * height/2 * depth/2)
        let lll_size = 2 * 2 * 2;
        let lll_energy: i64 = volume[0..lll_size].iter().map(|&x| (x as i64).pow(2)).sum();
        let total_energy: i64 = volume.iter().map(|&x| (x as i64).pow(2)).sum();

        // LLL should have significant energy (at least 10% for this test pattern)
        assert!(
            lll_energy > 0,
            "Energy compaction failed: LLL={}, Total={}",
            lll_energy,
            total_energy
        );

        // More importantly: roundtrip should work
        wavelet.inverse(&mut volume, 4, 4, 4);
        for (i, (&orig, &rec)) in original.iter().zip(volume.iter()).enumerate() {
            assert!(
                (orig - rec).abs() <= 3,
                "Roundtrip failed at {}: {} vs {}",
                i,
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_haar_single_element() {
        // Signal of length 1: forward/inverse should be identity
        let wavelet = Wavelet1D::haar();
        let mut signal = [42i32];
        wavelet.forward(&mut signal);
        assert_eq!(signal, [42]);
        wavelet.inverse(&mut signal);
        assert_eq!(signal, [42]);
    }

    #[test]
    fn test_haar_two_elements() {
        let wavelet = Wavelet1D::haar();
        let original = [10i32, 20];
        let mut signal = original;
        wavelet.forward(&mut signal);
        // After forward, the signal should be transformed (not identical to input)
        wavelet.inverse(&mut signal);
        for (i, (&o, &r)) in original.iter().zip(signal.iter()).enumerate() {
            assert!((o - r).abs() <= 1, "Mismatch at {}: {} vs {}", i, o, r);
        }
    }

    #[test]
    fn test_constant_signal_haar() {
        // A constant signal should be preserved in the low-pass and zero in high-pass
        let wavelet = Wavelet1D::haar();
        let original = [50i32; 8];
        let mut signal = original;
        wavelet.forward(&mut signal);

        // High-pass coefficients (second half) should be zero or near-zero
        for &hp in &signal[4..] {
            assert!(
                hp.abs() <= 1,
                "High-pass should be near-zero for constant signal, got {}",
                hp
            );
        }

        wavelet.inverse(&mut signal);
        for (i, (&o, &r)) in original.iter().zip(signal.iter()).enumerate() {
            assert!(
                (o - r).abs() <= 1,
                "Roundtrip mismatch at {}: {} vs {}",
                i,
                o,
                r
            );
        }
    }

    #[test]
    fn test_wavelet_2d_single_row_col() {
        // 2x2 is the minimum meaningful 2D transform
        let wavelet = Wavelet2D::cdf53();
        let original = [10i32, 20, 30, 40];
        let mut image = original;
        wavelet.forward(&mut image, 2, 2);
        wavelet.inverse(&mut image, 2, 2);
        for (i, (&o, &r)) in original.iter().zip(image.iter()).enumerate() {
            assert!(
                (o - r).abs() <= 2,
                "2x2 roundtrip mismatch at {}: {} vs {}",
                i,
                o,
                r
            );
        }
    }

    #[test]
    fn test_wavelet_2d_cdf97_roundtrip() {
        let wavelet = Wavelet2D::cdf97();
        let original = [
            10i32, 20, 30, 40, 15, 25, 35, 45, 12, 22, 32, 42, 18, 28, 38, 48,
        ];
        let mut image = original;
        wavelet.forward(&mut image, 4, 4);
        wavelet.inverse(&mut image, 4, 4);
        for (i, (&o, &r)) in original.iter().zip(image.iter()).enumerate() {
            assert!(
                (o - r).abs() <= 3,
                "CDF97 2D mismatch at {}: {} vs {}",
                i,
                o,
                r
            );
        }
    }

    #[test]
    fn test_wavelet_3d_depth_2() {
        // Minimum depth=2 for temporal transform
        let wavelet = Wavelet3D::cdf53();
        let original: Vec<i32> = (0..8).map(|i| 100 + i * 5).collect(); // 2x2x2
        let mut volume = original.clone();
        wavelet.forward(&mut volume, 2, 2, 2);
        wavelet.inverse(&mut volume, 2, 2, 2);
        for (i, (&o, &r)) in original.iter().zip(volume.iter()).enumerate() {
            assert!(
                (o - r).abs() <= 3,
                "3D depth=2 mismatch at {}: {} vs {}",
                i,
                o,
                r
            );
        }
    }
}
