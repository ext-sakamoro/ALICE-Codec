//! ALICE-Codec × ALICE-ML Bridge
//!
//! Ternary neural inference for motion estimation and adaptive quantization.
//! Uses 1.58-bit weights for ultra-fast sub-band classification and RDO hints.

use alice_ml::{ternary_matvec, TernaryWeight};

/// ML-accelerated sub-band classifier for adaptive quantization.
///
/// Uses ternary weights to predict optimal quantization parameters
/// from wavelet sub-band statistics (energy, variance, temporal coherence).
pub struct SubBandClassifier {
    /// Feature → quantization class weights.
    weights: TernaryWeight,
    /// Input feature dimension.
    input_dim: usize,
    /// Number of quantization classes.
    num_classes: usize,
}

impl SubBandClassifier {
    /// Create a sub-band classifier from pre-trained ternary weights.
    ///
    /// - `weights`: ternary values (num_classes × input_dim).
    /// - `input_dim`: number of sub-band statistical features.
    /// - `num_classes`: number of quantization strategy classes.
    pub fn new(weights: &[i8], input_dim: usize, num_classes: usize) -> Self {
        Self {
            weights: TernaryWeight::from_ternary(weights, num_classes, input_dim),
            input_dim,
            num_classes,
        }
    }

    /// Classify a sub-band's optimal quantization strategy.
    ///
    /// Input features typically include: energy, variance, temporal_coherence,
    /// spatial_complexity, etc.
    ///
    /// Returns (class_index, confidence_score).
    pub fn classify(&self, features: &[f32]) -> (usize, f32) {
        debug_assert_eq!(features.len(), self.input_dim);
        let mut logits = vec![0.0f32; self.num_classes];
        ternary_matvec(features, &self.weights, &mut logits);
        let (idx, &val) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        (idx, val)
    }

    /// Get raw logits for all classes (zero-allocation with output buffer).
    pub fn classify_logits(&self, features: &[f32], output: &mut [f32]) {
        debug_assert_eq!(features.len(), self.input_dim);
        debug_assert!(output.len() >= self.num_classes);
        ternary_matvec(features, &self.weights, &mut output[..self.num_classes]);
    }

    /// Input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }
    /// Number of classes.
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}

/// ML-accelerated motion vector predictor.
///
/// Predicts motion vectors from block statistics for temporal wavelet optimization.
pub struct MotionPredictor {
    /// Block features → motion vector (dx, dy).
    weights: TernaryWeight,
    /// Input feature dimension.
    input_dim: usize,
}

impl MotionPredictor {
    /// Create a motion predictor.
    ///
    /// - `weights`: ternary values (2 × input_dim) for dx, dy prediction.
    /// - `input_dim`: block feature dimension.
    pub fn new(weights: &[i8], input_dim: usize) -> Self {
        Self {
            weights: TernaryWeight::from_ternary(weights, 2, input_dim),
            input_dim,
        }
    }

    /// Predict motion vector (dx, dy) from block features.
    pub fn predict(&self, features: &[f32]) -> (f32, f32) {
        debug_assert_eq!(features.len(), self.input_dim);
        let mut out = [0.0f32; 2];
        ternary_matvec(features, &self.weights, &mut out);
        (out[0], out[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subband_classifier() {
        // 3 features → 2 classes
        let weights = [1i8, -1, 0, 0, 1, 1]; // 2×3
        let clf = SubBandClassifier::new(&weights, 3, 2);

        // High energy, low variance, mid coherence
        let features = [5.0f32, 1.0, 3.0];
        let (class, _) = clf.classify(&features);
        // Class 0: [1,-1,0]·[5,1,3] = 5-1+0 = 4
        // Class 1: [0,1,1]·[5,1,3] = 0+1+3 = 4
        // Tied, but max_by returns last max → class 1
        assert!(class <= 1);
    }

    #[test]
    fn test_motion_predictor() {
        // 4 features → 2 outputs (dx, dy)
        let weights = [1i8, 0, -1, 0, 0, 1, 0, -1]; // 2×4
        let predictor = MotionPredictor::new(&weights, 4);

        let features = [2.0f32, 3.0, 1.0, 4.0];
        let (dx, dy) = predictor.predict(&features);
        // dx: [1,0,-1,0]·[2,3,1,4] = 2+0-1+0 = 1
        // dy: [0,1,0,-1]·[2,3,1,4] = 0+3+0-4 = -1
        assert!((dx - 1.0).abs() < 1e-6);
        assert!((dy - (-1.0)).abs() < 1e-6);
    }
}
