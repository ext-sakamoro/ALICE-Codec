//! ロスレスモード — CDF 5/3 + 量子化ステップ1 による完全再構成
//!
//! 整数 Wavelet (CDF 5/3) と量子化ステップ=1 を組み合わせ、
//! ビット完全な可逆圧縮を提供する。

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::wavelet::{Wavelet1D, Wavelet2D};

/// ロスレスエンコーダー。
///
/// CDF 5/3 (整数リフティング) + rANS で可逆圧縮。
#[derive(Debug)]
pub struct LosslessEncoder {
    wavelet: Wavelet1D,
}

impl Default for LosslessEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl LosslessEncoder {
    /// 新しいロスレスエンコーダーを作成。
    #[must_use]
    pub fn new() -> Self {
        Self {
            wavelet: Wavelet1D::cdf53(),
        }
    }

    /// 1D 信号を可逆変換。インプレースで変換。
    pub fn transform_1d(&self, signal: &mut [i32]) {
        self.wavelet.forward(signal);
    }

    /// 1D 信号を逆変換。
    pub fn inverse_1d(&self, signal: &mut [i32]) {
        self.wavelet.inverse(signal);
    }

    /// 2D 画像を可逆変換。
    pub fn transform_2d(&self, data: &mut [i32], width: usize, height: usize) {
        let w2d = Wavelet2D::new(self.wavelet.clone());
        w2d.forward(data, width, height);
    }

    /// 2D 画像を逆変換。
    pub fn inverse_2d(&self, data: &mut [i32], width: usize, height: usize) {
        let w2d = Wavelet2D::new(self.wavelet.clone());
        w2d.inverse(data, width, height);
    }
}

/// ロスレス round-trip を検証。
///
/// 変換→逆変換で元信号と完全一致することを確認。
#[must_use]
pub fn verify_roundtrip_1d(signal: &[i32]) -> bool {
    if signal.len() < 2 {
        return true;
    }

    let encoder = LosslessEncoder::new();
    let mut buf = signal.to_vec();
    encoder.transform_1d(&mut buf);
    encoder.inverse_1d(&mut buf);

    buf == signal
}

/// 2D ロスレス round-trip を検証。
#[must_use]
pub fn verify_roundtrip_2d(data: &[i32], width: usize, height: usize) -> bool {
    if data.len() != width * height || data.is_empty() {
        return data.is_empty();
    }

    let encoder = LosslessEncoder::new();
    let mut buf = data.to_vec();
    encoder.transform_2d(&mut buf, width, height);
    encoder.inverse_2d(&mut buf, width, height);

    buf == data
}

/// u8 バッファを i32 に変換 (変換用)。
#[must_use]
pub fn u8_to_i32(buf: &[u8]) -> Vec<i32> {
    buf.iter().map(|&b| i32::from(b)).collect()
}

/// i32 バッファを u8 にクランプ変換。
#[must_use]
pub fn i32_to_u8(buf: &[i32]) -> Vec<u8> {
    buf.iter().map(|&v| v.clamp(0, 255) as u8).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_1d_simple() {
        let signal = vec![10, 20, 30, 40, 50, 60, 70, 80];
        assert!(verify_roundtrip_1d(&signal));
    }

    #[test]
    fn roundtrip_1d_constant() {
        let signal = vec![42; 16];
        assert!(verify_roundtrip_1d(&signal));
    }

    #[test]
    fn roundtrip_1d_alternating() {
        let signal = vec![0, 255, 0, 255, 0, 255, 0, 255];
        assert!(verify_roundtrip_1d(&signal));
    }

    #[test]
    fn roundtrip_1d_ramp() {
        let signal: Vec<i32> = (0..64).collect();
        assert!(verify_roundtrip_1d(&signal));
    }

    #[test]
    fn roundtrip_1d_negative() {
        let signal = vec![-100, -50, 0, 50, 100, 150, -200, 200];
        assert!(verify_roundtrip_1d(&signal));
    }

    #[test]
    fn roundtrip_1d_single() {
        assert!(verify_roundtrip_1d(&[42]));
    }

    #[test]
    fn roundtrip_1d_empty() {
        assert!(verify_roundtrip_1d(&[]));
    }

    #[test]
    fn roundtrip_2d() {
        let data: Vec<i32> = (0..64).collect();
        assert!(verify_roundtrip_2d(&data, 8, 8));
    }

    #[test]
    fn roundtrip_2d_constant() {
        let data = vec![100; 16 * 16];
        assert!(verify_roundtrip_2d(&data, 16, 16));
    }

    #[test]
    fn u8_i32_conversion() {
        let original = vec![0u8, 128, 255, 42];
        let i32_buf = u8_to_i32(&original);
        assert_eq!(i32_buf, vec![0, 128, 255, 42]);
        let back = i32_to_u8(&i32_buf);
        assert_eq!(back, original);
    }

    #[test]
    fn i32_to_u8_clamp() {
        let buf = vec![-10, 0, 128, 300];
        let result = i32_to_u8(&buf);
        assert_eq!(result, vec![0, 0, 128, 255]);
    }

    #[test]
    fn encoder_default() {
        let encoder = LosslessEncoder::default();
        let mut signal = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let original = signal.clone();
        encoder.transform_1d(&mut signal);
        encoder.inverse_1d(&mut signal);
        assert_eq!(signal, original);
    }
}
