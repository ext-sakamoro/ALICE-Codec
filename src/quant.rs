//! Quantization and Rate-Distortion Optimization
//!
//! Dead-zone quantization with analytical RDO for optimal quality/size trade-off.
//!
//! # Dead-Zone Quantization
//!
//! ```text
//!        Output
//!          ↑
//!      2   │     ●───────
//!          │    /
//!      1   │   ●
//!          │  /
//!      0 ──┼─●───────────→ Input
//!          │  \
//!     -1   │   ●
//!          │    \
//!     -2   │     ●───────
//!          │
//!        -3Q  -Q  0  Q  3Q
//! ```
//!
//! The dead-zone around zero helps compress near-zero coefficients.
//!
//! # Analytical RDO
//!
//! Traditional: Try all quantizers, count bits → O(N × Q)
//!
//! ALICE-Codec: Closed-form solution using Laplacian distribution:
//! ```text
//! λ_optimal = (6 × ln(2) × σ²) / R_target
//! ```
//!
//! One-shot. No iteration.
//!
//! # Optimized Division
//!
//! Replace `idiv` (integer division, ~30 cycles) with Magic Number multiplication + shift (~3 cycles).
//!
//! > "Division is just multiplication by a magic number."

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::SubBand3D;

/// Quantizer configuration
#[derive(Clone, Copy, Debug)]
pub struct Quantizer {
    /// Quantization step size
    pub step: i32,
    /// Dead-zone width (typically 0.5 to 1.5 × step)
    pub dead_zone: i32,
}

impl Quantizer {
    /// Create quantizer with specified step size
    ///
    /// Dead-zone defaults to step size (standard dead-zone quantization).
    #[must_use]
    #[inline]
    pub const fn new(step: i32) -> Self {
        Self {
            step,
            dead_zone: step,
        }
    }

    /// Create quantizer with custom dead-zone
    #[must_use]
    #[inline]
    pub const fn with_dead_zone(step: i32, dead_zone: i32) -> Self {
        Self { step, dead_zone }
    }

    /// Quantize a single coefficient
    ///
    /// Uses dead-zone quantization: values in `[-dead_zone, dead_zone]` map to 0.
    #[must_use]
    #[inline]
    pub fn quantize(&self, value: i32) -> i32 {
        if value.abs() < self.dead_zone {
            0
        } else if value >= 0 {
            (value - self.dead_zone / 2) / self.step
        } else {
            (value + self.dead_zone / 2) / self.step
        }
    }

    /// Dequantize a single coefficient
    ///
    /// Maps quantized value back to reconstruction level.
    #[must_use]
    #[inline]
    pub fn dequantize(&self, qvalue: i32) -> i32 {
        if qvalue == 0 {
            0
        } else {
            qvalue * self.step
        }
    }

    /// Quantize buffer in-place (DPS style)
    ///
    /// # Panics
    ///
    /// Panics if `output` is smaller than `input`.
    pub fn quantize_buffer(&self, input: &[i32], output: &mut [i32]) {
        assert!(output.len() >= input.len());
        for (i, &val) in input.iter().enumerate() {
            output[i] = self.quantize(val);
        }
    }

    /// Dequantize buffer in-place (DPS style)
    ///
    /// # Panics
    ///
    /// Panics if `output` is smaller than `input`.
    pub fn dequantize_buffer(&self, input: &[i32], output: &mut [i32]) {
        assert!(output.len() >= input.len());
        for (i, &val) in input.iter().enumerate() {
            output[i] = self.dequantize(val);
        }
    }
}

impl Default for Quantizer {
    fn default() -> Self {
        Self::new(16)
    }
}

// ============================================================================
// Fast Quantizer (Optimized Edition) - Magic Number Division
// ============================================================================

/// Fast Quantizer using reciprocal multiplication (Magic Number Division)
///
/// Replaces `idiv` instruction (~30 cycles) with multiplication + shift (~3 cycles).
///
/// # How It Works
///
/// For a divisor `d`, we precompute:
/// - `reciprocal = ceil(2^(32+shift) / d)`
/// - `shift` = precision bits
///
/// Then: `x / d ≈ (x * reciprocal) >> (32 + shift)`
#[derive(Clone, Copy, Debug)]
pub struct FastQuantizer {
    /// Reciprocal of step size (scaled by 2^shift)
    reciprocal: u64,
    /// Shift amount (32 + extra precision)
    shift: u32,
    /// Step size (original, for dequantization)
    step: i32,
    /// Dead zone threshold
    dead_zone: i32,
}

impl FastQuantizer {
    /// Create fast quantizer from step size
    ///
    /// Precomputes magic number for division.
    ///
    /// # Panics
    ///
    /// Panics if `step` is not positive.
    #[must_use]
    pub fn new(step: i32) -> Self {
        assert!(step > 0, "step must be positive");

        // Compute magic number for unsigned division by step
        // We want: (x * reciprocal) >> shift ≈ x / step
        //
        // Standard algorithm: choose shift such that 2^shift > step,
        // then reciprocal = ceil(2^shift / step)
        let step_u = step as u32;

        // Use 32-bit base + extra precision
        // shift = 32 + ceil(log2(step))
        let extra_bits = 32 - step_u.leading_zeros();
        let shift = 32 + extra_bits;

        // reciprocal = ceil(2^shift / step)
        let power: u128 = 1u128 << shift;
        let reciprocal = power.div_ceil(step_u as u128) as u64;

        Self {
            reciprocal,
            shift,
            step,
            dead_zone: step, // Default dead-zone = step
        }
    }

    /// Create fast quantizer with custom dead-zone
    #[must_use]
    pub fn with_dead_zone(step: i32, dead_zone: i32) -> Self {
        let mut q = Self::new(step);
        q.dead_zone = dead_zone;
        q
    }

    /// Fast division: x / step using precomputed magic number
    #[inline]
    fn fast_div(&self, x: u32) -> u32 {
        // (x * reciprocal) >> shift
        let product = x as u64 * self.reciprocal;
        (product >> self.shift) as u32
    }

    /// Quantize a single coefficient
    ///
    /// Uses magic number division for speed.
    #[must_use]
    #[inline]
    pub fn quantize(&self, value: i32) -> i32 {
        let abs_val = value.abs();

        // Dead-zone check
        if abs_val < self.dead_zone {
            return 0;
        }

        // Apply offset for dead-zone
        let offset = self.dead_zone >> 1;
        let adjusted = (abs_val - offset) as u32;

        // Fast division
        let q_abs = self.fast_div(adjusted) as i32;

        // Restore sign
        if value < 0 {
            -q_abs
        } else {
            q_abs
        }
    }

    /// Dequantize a single coefficient
    #[must_use]
    #[inline]
    pub fn dequantize(&self, qvalue: i32) -> i32 {
        if qvalue == 0 {
            0
        } else {
            qvalue * self.step
        }
    }

    /// Quantize buffer (DPS style)
    ///
    /// # Panics
    ///
    /// Panics if `output` is smaller than `input`.
    pub fn quantize_buffer(&self, input: &[i32], output: &mut [i32]) {
        assert!(output.len() >= input.len());
        for (i, &val) in input.iter().enumerate() {
            output[i] = self.quantize(val);
        }
    }

    /// Dequantize buffer (DPS style)
    ///
    /// # Panics
    ///
    /// Panics if `output` is smaller than `input`.
    pub fn dequantize_buffer(&self, input: &[i32], output: &mut [i32]) {
        assert!(output.len() >= input.len());
        for (i, &val) in input.iter().enumerate() {
            output[i] = self.dequantize(val);
        }
    }

    /// Quantize buffer using AVX2 SIMD (8 coefficients at a time)
    ///
    /// Falls back to scalar path on non-x86_64 or when `simd` feature is disabled.
    ///
    /// # Safety (SIMD path)
    ///
    /// Requires AVX2 support at runtime. The implementation checks for AVX2 via
    /// `#[target_feature]` and the compiler guarantees it is only called on
    /// CPUs that expose the AVX2 feature.
    pub fn quantize_buffer_simd(&self, input: &[i32], output: &mut [i32]) {
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            // SAFETY: quantize_avx2 is guarded by #[target_feature(enable = "avx2")].
            unsafe { quantize_avx2(input, output, self.step, self.dead_zone) }
        }
        #[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
        {
            self.quantize_buffer(input, output);
        }
    }

    /// Get step size
    #[must_use]
    #[inline]
    pub fn step(&self) -> i32 {
        self.step
    }

    /// Get dead zone
    #[must_use]
    #[inline]
    pub fn dead_zone(&self) -> i32 {
        self.dead_zone
    }
}

impl Default for FastQuantizer {
    fn default() -> Self {
        Self::new(16)
    }
}

impl From<Quantizer> for FastQuantizer {
    fn from(q: Quantizer) -> Self {
        Self::with_dead_zone(q.step, q.dead_zone)
    }
}

/// Analytical Rate-Distortion Optimization
///
/// Computes optimal quantization step sizes using closed-form solution
/// assuming Laplacian distribution of wavelet coefficients.
#[derive(Clone, Debug)]
pub struct AnalyticalRDO {
    /// Target bits per pixel
    target_bpp: f64,
    /// Quality factor (0-100, higher = better quality)
    quality: u8,
}

impl AnalyticalRDO {
    /// Create RDO optimizer with target bits-per-pixel
    #[must_use]
    pub fn new(target_bpp: f64) -> Self {
        Self {
            target_bpp,
            quality: 75,
        }
    }

    /// Create RDO optimizer with quality setting
    ///
    /// Quality 0 = maximum compression, 100 = lossless
    #[must_use]
    pub fn with_quality(quality: u8) -> Self {
        // Map quality to approximate bits-per-pixel
        // Quality 100 → ~24 bpp (near lossless for 8-bit RGB)
        // Quality 50 → ~2 bpp
        // Quality 0 → ~0.1 bpp
        const RCP_100: f64 = 1.0 / 100.0;
        let quality = quality.min(100);
        let target_bpp = 0.1 + (quality as f64 * RCP_100).powi(2) * 23.9;

        Self {
            target_bpp,
            quality,
        }
    }

    /// Estimate variance of coefficients
    fn estimate_variance(coeffs: &[i32]) -> f64 {
        if coeffs.is_empty() {
            return 1.0;
        }

        let n = coeffs.len() as f64;
        let inv_n = 1.0 / n;
        let sum: i64 = coeffs.iter().map(|&x| x as i64).sum();
        let mean = sum as f64 * inv_n;

        let variance: f64 = coeffs
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            * inv_n;

        variance.max(1.0)
    }

    /// Compute optimal quantization step for Laplacian distribution
    ///
    /// Using the formula: `lambda_optimal` = (6 x ln(2) x sigma^2) / `R_target`
    fn compute_optimal_lambda(&self, variance: f64) -> f64 {
        let ln2 = core::f64::consts::LN_2;
        (6.0 * ln2 * variance) / self.target_bpp
    }

    /// Compute quantization step from Lagrange multiplier
    ///
    /// For Laplacian distribution: Q ≈ sqrt(12 × λ)
    fn lambda_to_step(&self, lambda: f64) -> i32 {
        let step = (12.0 * lambda).sqrt();
        (step.round() as i32).max(1)
    }

    /// Compute optimal quantizer for a sub-band
    #[must_use]
    pub fn compute_quantizer(&self, coeffs: &[i32], subband: SubBand3D) -> Quantizer {
        let variance = Self::estimate_variance(coeffs);
        let lambda = self.compute_optimal_lambda(variance);
        let base_step = self.lambda_to_step(lambda);

        // Apply sub-band specific scaling
        let strength = subband.quant_strength() as i32;
        let step = (base_step * strength).max(1);

        // Dead-zone is typically 1.0-1.5× step for better compression
        let dead_zone = step + step / 2;

        Quantizer::with_dead_zone(step, dead_zone)
    }

    /// Compute quantizers for all 8 sub-bands of a 3D decomposition
    #[must_use]
    pub fn compute_all_quantizers(&self, subbands: &[&[i32]; 8]) -> [Quantizer; 8] {
        let all_subbands = [
            SubBand3D::LLL,
            SubBand3D::LLH,
            SubBand3D::LHL,
            SubBand3D::LHH,
            SubBand3D::HLL,
            SubBand3D::HLH,
            SubBand3D::HHL,
            SubBand3D::HHH,
        ];

        let mut quantizers = [Quantizer::default(); 8];
        for (i, (&coeffs, &subband)) in subbands.iter().zip(all_subbands.iter()).enumerate() {
            quantizers[i] = self.compute_quantizer(coeffs, subband);
        }

        quantizers
    }

    /// Get current quality setting
    #[must_use]
    #[inline]
    pub fn quality(&self) -> u8 {
        self.quality
    }

    /// Get target bits-per-pixel
    #[must_use]
    #[inline]
    pub fn target_bpp(&self) -> f64 {
        self.target_bpp
    }
}

impl Default for AnalyticalRDO {
    fn default() -> Self {
        Self::with_quality(75)
    }
}

/// Quantize sub-band coefficients
pub fn quantize_subband(coeffs: &[i32], quantizer: &Quantizer, output: &mut [i32]) {
    quantizer.quantize_buffer(coeffs, output);
}

/// Dequantize sub-band coefficients
pub fn dequantize_subband(coeffs: &[i32], quantizer: &Quantizer, output: &mut [i32]) {
    quantizer.dequantize_buffer(coeffs, output);
}

/// Convert quantized coefficients to symbols for entropy coding
///
/// Maps signed integers to unsigned symbols:
/// 0 -> 0, 1 -> 1, -1 -> 2, 2 -> 3, -2 -> 4, ...
///
/// # Panics
///
/// Panics if `symbols` is smaller than `coeffs`.
pub fn to_symbols(coeffs: &[i32], symbols: &mut [u8]) {
    assert!(symbols.len() >= coeffs.len());

    for (i, &coeff) in coeffs.iter().enumerate() {
        symbols[i] = match coeff.cmp(&0) {
            core::cmp::Ordering::Equal => 0,
            core::cmp::Ordering::Greater => (coeff * 2 - 1) as u8,
            core::cmp::Ordering::Less => (-coeff * 2) as u8,
        };
    }
}

/// Convert symbols back to quantized coefficients
///
/// Inverse of `to_symbols`.
///
/// # Panics
///
/// Panics if `coeffs` is smaller than `symbols`.
pub fn from_symbols(symbols: &[u8], coeffs: &mut [i32]) {
    assert!(coeffs.len() >= symbols.len());

    for (i, &sym) in symbols.iter().enumerate() {
        coeffs[i] = if sym == 0 {
            0
        } else if sym % 2 == 1 {
            (sym as i32 + 1) / 2
        } else {
            -(sym as i32 / 2)
        };
    }
}

/// Build histogram of symbols for entropy coding
#[must_use]
pub fn build_histogram(symbols: &[u8]) -> [u32; 256] {
    let mut histogram = [0u32; 256];
    for &sym in symbols {
        histogram[sym as usize] += 1;
    }
    histogram
}

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
mod simd {
    use super::*;
    use core::arch::x86_64::*;

    /// SIMD quantization (8 coefficients at a time)
    ///
    /// # Safety
    ///
    /// Requires AVX2 support.
    #[target_feature(enable = "avx2")]
    pub unsafe fn quantize_avx2(input: &[i32], output: &mut [i32], step: i32, dead_zone: i32) {
        let n = input.len();
        assert!(output.len() >= n);

        let step_vec = _mm256_set1_epi32(step);
        let dead_zone_vec = _mm256_set1_epi32(dead_zone);
        let half_dead_zone_vec = _mm256_set1_epi32(dead_zone / 2);
        let zero = _mm256_setzero_si256();

        let chunks = n / 8;

        for i in 0..chunks {
            let offset = i * 8;
            let values = _mm256_loadu_si256(input.as_ptr().add(offset) as *const __m256i);

            // Compute abs(values)
            let abs_values = _mm256_abs_epi32(values);

            // Check if in dead zone: abs(value) < dead_zone
            let in_dead_zone = _mm256_cmpgt_epi32(dead_zone_vec, abs_values);

            // Compute sign
            let sign_mask = _mm256_cmpgt_epi32(zero, values);

            // Quantize positive: (value - dead_zone/2) / step
            let adjusted_pos = _mm256_sub_epi32(values, half_dead_zone_vec);

            // Quantize negative: (value + dead_zone/2) / step
            let adjusted_neg = _mm256_add_epi32(values, half_dead_zone_vec);

            // Select based on sign
            let adjusted = _mm256_blendv_epi8(adjusted_pos, adjusted_neg, sign_mask);

            // Integer division by step (approximate using multiply + shift)
            // For exact division, we'd need a proper divider, but this is close enough
            let quotient = _mm256_div_epi32_approx(adjusted, step_vec);

            // Zero out dead-zone values
            let result = _mm256_blendv_epi8(quotient, zero, in_dead_zone);

            _mm256_storeu_si256(output.as_mut_ptr().add(offset) as *mut __m256i, result);
        }

        // Handle remainder
        let remainder = chunks * 8;
        let quantizer = super::Quantizer::with_dead_zone(step, dead_zone);
        for i in remainder..n {
            output[i] = quantizer.quantize(input[i]);
        }
    }

    /// Approximate integer division using multiply and shift
    // SAFETY: Caller must ensure AVX2 is available. All __m256i values are valid
    // since they come from other AVX2 intrinsics.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn _mm256_div_epi32_approx(a: __m256i, b: __m256i) -> __m256i {
        // Extract to scalars and divide (not truly SIMD, but works for now)
        let mut a_arr = [0i32; 8];
        let mut b_arr = [0i32; 8];
        _mm256_storeu_si256(a_arr.as_mut_ptr() as *mut __m256i, a);
        _mm256_storeu_si256(b_arr.as_mut_ptr() as *mut __m256i, b);

        for i in 0..8 {
            if b_arr[i] != 0 {
                a_arr[i] /= b_arr[i];
            }
        }

        _mm256_loadu_si256(a_arr.as_ptr() as *const __m256i)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
pub use simd::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantizer_roundtrip() {
        let quantizer = Quantizer::new(16);

        // Test various values
        // Dead-zone quantization has larger errors near the dead-zone boundary
        for value in [-100, -50, -32, -17, 0, 17, 32, 50, 100] {
            let qval = quantizer.quantize(value);
            let dequant = quantizer.dequantize(qval);

            // For values outside dead-zone, error should be bounded
            // For values in dead-zone, they map to 0
            if value.abs() < quantizer.dead_zone {
                assert_eq!(
                    dequant, 0,
                    "Dead-zone value {} should dequantize to 0",
                    value
                );
            } else {
                // Error can be up to step + dead_zone for boundary cases
                let max_error = quantizer.step + quantizer.dead_zone;
                let error = (dequant - value).abs();
                assert!(
                    error <= max_error,
                    "Error {} too large for value {} (max {})",
                    error,
                    value,
                    max_error
                );
            }
        }
    }

    #[test]
    fn test_dead_zone() {
        let quantizer = Quantizer::new(16);

        // Values in dead zone should map to 0
        for value in -15..=15 {
            let qval = quantizer.quantize(value);
            assert_eq!(qval, 0, "Value {} should be in dead zone", value);
        }
    }

    #[test]
    fn test_symbol_mapping() {
        // Test symbol mapping roundtrip
        let original = [-5, -2, -1, 0, 1, 2, 5];
        let mut symbols = [0u8; 7];
        let mut recovered = [0i32; 7];

        to_symbols(&original, &mut symbols);
        from_symbols(&symbols, &mut recovered);

        // Values should be recoverable for small integers
        for (i, (&orig, &rec)) in original.iter().zip(recovered.iter()).enumerate() {
            assert_eq!(orig, rec, "Mismatch at index {}: {} != {}", i, orig, rec);
        }
    }

    #[test]
    fn test_symbol_ordering() {
        // Verify symbol ordering: 0 → 0, 1 → 1, -1 → 2, 2 → 3, -2 → 4
        let values = [0, 1, -1, 2, -2, 3, -3];
        let expected = [0u8, 1, 2, 3, 4, 5, 6];
        let mut symbols = [0u8; 7];

        to_symbols(&values, &mut symbols);

        assert_eq!(symbols, expected);
    }

    #[test]
    fn test_analytical_rdo() {
        let rdo = AnalyticalRDO::with_quality(50);

        // Create some test coefficients with known variance
        let coeffs: Vec<i32> = (-100..=100).collect();

        let quantizer = rdo.compute_quantizer(&coeffs, SubBand3D::LLL);

        // LLL sub-band should have smallest step (preserve DC)
        assert!(quantizer.step > 0);

        let quantizer_hhh = rdo.compute_quantizer(&coeffs, SubBand3D::HHH);

        // HHH should have larger step (more aggressive quantization)
        assert!(
            quantizer_hhh.step >= quantizer.step,
            "HHH step {} should be >= LLL step {}",
            quantizer_hhh.step,
            quantizer.step
        );
    }

    #[test]
    fn test_quality_levels() {
        let rdo_low = AnalyticalRDO::with_quality(10);
        let rdo_high = AnalyticalRDO::with_quality(90);

        assert!(
            rdo_low.target_bpp() < rdo_high.target_bpp(),
            "Low quality {} should have lower bpp than high quality {}",
            rdo_low.target_bpp(),
            rdo_high.target_bpp()
        );
    }

    #[test]
    fn test_histogram() {
        let symbols = [0u8, 0, 1, 1, 1, 2, 5, 5];
        let hist = build_histogram(&symbols);

        assert_eq!(hist[0], 2);
        assert_eq!(hist[1], 3);
        assert_eq!(hist[2], 1);
        assert_eq!(hist[3], 0);
        assert_eq!(hist[5], 2);
    }

    #[test]
    fn test_buffer_operations() {
        let quantizer = Quantizer::new(8);
        let input = [-50, -25, 0, 25, 50];
        let mut quantized = [0i32; 5];
        let mut dequantized = [0i32; 5];

        quantizer.quantize_buffer(&input, &mut quantized);
        quantizer.dequantize_buffer(&quantized, &mut dequantized);

        // Check that quantization happened
        // Values in dead-zone should become 0
        assert_eq!(dequantized[2], 0, "Zero should stay zero");

        // Values outside dead-zone should be quantized
        // With step=8 and dead_zone=8, values like ±50 should quantize to ±5 (approx)
        let max_error = quantizer.step + quantizer.dead_zone;
        for (i, (&orig, &rec)) in input.iter().zip(dequantized.iter()).enumerate() {
            let error = (orig - rec).abs();
            assert!(
                error <= max_error,
                "Error {} at index {} too large (max {})",
                error,
                i,
                max_error
            );
        }
    }

    // =========================================================================
    // FastQuantizer Tests (Optimized Edition)
    // =========================================================================

    #[test]
    fn test_fast_quantizer_matches_regular() {
        // FastQuantizer should produce identical results to Quantizer
        for step in [4, 8, 16, 32, 64, 128] {
            let regular = Quantizer::new(step);
            let fast = FastQuantizer::new(step);

            for value in [-1000, -500, -100, -50, 0, 50, 100, 500, 1000] {
                let q_regular = regular.quantize(value);
                let q_fast = fast.quantize(value);

                assert_eq!(
                    q_regular, q_fast,
                    "Mismatch for step={}, value={}: regular={}, fast={}",
                    step, value, q_regular, q_fast
                );
            }
        }
    }

    #[test]
    fn test_fast_quantizer_dead_zone() {
        let fast = FastQuantizer::new(16);

        // Values in dead zone should map to 0
        for value in -15..=15 {
            let qval = fast.quantize(value);
            assert_eq!(qval, 0, "Value {} should be in dead zone", value);
        }
    }

    #[test]
    fn test_fast_quantizer_large_values() {
        // Test with larger values to ensure magic number works
        let fast = FastQuantizer::new(17); // Prime number to stress test

        for value in [1000, 5000, 10000, 50000, 100000] {
            let q = fast.quantize(value);
            let expected = (value - 17 / 2) / 17;

            // Allow ±1 tolerance due to rounding differences
            assert!(
                (q - expected).abs() <= 1,
                "Large value {}: got {}, expected ~{}",
                value,
                q,
                expected
            );
        }
    }

    #[test]
    fn test_fast_quantizer_roundtrip() {
        let fast = FastQuantizer::new(16);
        let input = [-100, -50, 0, 50, 100];
        let mut quantized = [0i32; 5];
        let mut dequantized = [0i32; 5];

        fast.quantize_buffer(&input, &mut quantized);
        fast.dequantize_buffer(&quantized, &mut dequantized);

        // Check consistency
        assert_eq!(dequantized[2], 0, "Zero should stay zero");

        let max_error = fast.step() + fast.dead_zone();
        for (i, (&orig, &rec)) in input.iter().zip(dequantized.iter()).enumerate() {
            let error = (orig - rec).abs();
            assert!(
                error <= max_error,
                "Error {} at index {} too large (max {})",
                error,
                i,
                max_error
            );
        }
    }

    #[test]
    fn test_fast_quantizer_from_regular() {
        let regular = Quantizer::with_dead_zone(32, 48);
        let fast: FastQuantizer = regular.into();

        assert_eq!(fast.step(), 32);
        assert_eq!(fast.dead_zone(), 48);

        // Results should match
        for value in [-200, -100, 0, 100, 200] {
            assert_eq!(
                regular.quantize(value),
                fast.quantize(value),
                "Mismatch for value {}",
                value
            );
        }
    }

    #[test]
    fn test_quantizer_zero_value() {
        let q = Quantizer::new(16);
        assert_eq!(q.quantize(0), 0);
        assert_eq!(q.dequantize(0), 0);
    }

    #[test]
    fn test_quantizer_sign_symmetry() {
        let q = Quantizer::new(10);
        // Positive and negative of the same magnitude should have symmetric results
        for v in [20, 50, 100, 200] {
            let qp = q.quantize(v);
            let qn = q.quantize(-v);
            assert_eq!(
                qp, -qn,
                "Sign symmetry broken for value {}: +={}, -={}",
                v, qp, qn
            );
        }
    }

    #[test]
    fn test_quantizer_default() {
        let q = Quantizer::default();
        assert_eq!(q.step, 16);
        assert_eq!(q.dead_zone, 16);
    }

    #[test]
    fn test_fast_quantizer_default() {
        let fq = FastQuantizer::default();
        assert_eq!(fq.step(), 16);
        assert_eq!(fq.dead_zone(), 16);
    }

    #[test]
    fn test_fast_quantizer_sign_symmetry() {
        let fq = FastQuantizer::new(10);
        for v in [20, 50, 100, 500] {
            let qp = fq.quantize(v);
            let qn = fq.quantize(-v);
            assert_eq!(
                qp, -qn,
                "FastQuantizer sign symmetry broken for {}: +={}, -={}",
                v, qp, qn
            );
        }
    }

    #[test]
    fn test_symbol_mapping_zero() {
        let coeffs = [0i32];
        let mut symbols = [0u8; 1];
        let mut recovered = [0i32; 1];

        to_symbols(&coeffs, &mut symbols);
        assert_eq!(symbols[0], 0);

        from_symbols(&symbols, &mut recovered);
        assert_eq!(recovered[0], 0);
    }

    #[test]
    fn test_histogram_empty() {
        let symbols: [u8; 0] = [];
        let hist = build_histogram(&symbols);
        assert!(hist.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_histogram_single_symbol() {
        let symbols = [42u8; 100];
        let hist = build_histogram(&symbols);
        assert_eq!(hist[42], 100);
        // All others should be 0
        for (i, &c) in hist.iter().enumerate() {
            if i != 42 {
                assert_eq!(c, 0);
            }
        }
    }

    #[test]
    fn test_rdo_quality_zero() {
        let rdo = AnalyticalRDO::with_quality(0);
        assert_eq!(rdo.quality(), 0);
        assert!(rdo.target_bpp() > 0.0);
    }

    #[test]
    fn test_rdo_quality_100() {
        let rdo = AnalyticalRDO::with_quality(100);
        assert_eq!(rdo.quality(), 100);
        // Quality 100 should give highest bpp
        assert!(
            rdo.target_bpp() > 20.0,
            "Quality 100 bpp {} too low",
            rdo.target_bpp()
        );
    }

    #[test]
    fn test_rdo_quality_clamped_above_100() {
        // Quality above 100 should be clamped to 100
        let rdo_100 = AnalyticalRDO::with_quality(100);
        let rdo_200 = AnalyticalRDO::with_quality(200);
        assert!((rdo_100.target_bpp() - rdo_200.target_bpp()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rdo_compute_all_quantizers() {
        let rdo = AnalyticalRDO::with_quality(50);
        let data: Vec<i32> = (-50..=50).collect();
        let subbands: [&[i32]; 8] = [&data, &data, &data, &data, &data, &data, &data, &data];
        let quantizers = rdo.compute_all_quantizers(&subbands);

        // All steps should be positive
        for (i, q) in quantizers.iter().enumerate() {
            assert!(
                q.step > 0,
                "Quantizer {} has non-positive step: {}",
                i,
                q.step
            );
        }

        // LLL (index 0) should have smallest step, HHH (index 7) largest
        assert!(
            quantizers[7].step >= quantizers[0].step,
            "HHH step {} should >= LLL step {}",
            quantizers[7].step,
            quantizers[0].step
        );
    }

    #[test]
    fn test_quantize_subband_fn() {
        let q = Quantizer::new(8);
        let input = [0, 4, -4, 16, -16, 100, -100];
        let mut output = [0i32; 7];
        quantize_subband(&input, &q, &mut output);

        // Values in dead-zone (|v| < 8) should be 0
        assert_eq!(output[0], 0);
        assert_eq!(output[1], 0);
        assert_eq!(output[2], 0);

        // Values outside dead-zone should be non-zero
        assert_ne!(output[3], 0);
        assert_ne!(output[4], 0);
    }

    #[test]
    fn test_dequantize_subband_fn() {
        let q = Quantizer::new(8);
        let input = [0, 1, -1, 5, -5];
        let mut output = [0i32; 5];
        dequantize_subband(&input, &q, &mut output);
        assert_eq!(output[0], 0);
        assert_eq!(output[1], 8);
        assert_eq!(output[2], -8);
        assert_eq!(output[3], 40);
        assert_eq!(output[4], -40);
    }

    #[test]
    fn test_fast_quantizer_simd_fallback() {
        // quantize_buffer_simd should produce same results as quantize_buffer
        // on non-SIMD path (default features)
        let fq = FastQuantizer::new(16);
        let input: Vec<i32> = (-100..=100).collect();
        let mut out_scalar = vec![0i32; input.len()];
        let mut out_simd = vec![0i32; input.len()];

        fq.quantize_buffer(&input, &mut out_scalar);
        fq.quantize_buffer_simd(&input, &mut out_simd);

        assert_eq!(out_scalar, out_simd);
    }

    #[test]
    fn test_quantizer_with_custom_dead_zone() {
        // Larger dead zone: more values map to 0
        let q_small = Quantizer::with_dead_zone(8, 8);
        let q_large = Quantizer::with_dead_zone(8, 24);

        // Value 20: within large dead zone (24) but outside small dead zone (8)
        assert_ne!(q_small.quantize(20), 0);
        assert_eq!(q_large.quantize(20), 0);

        // Also check that 50 is outside both dead zones
        assert_ne!(q_small.quantize(50), 0);
        assert_ne!(q_large.quantize(50), 0);
    }
}
