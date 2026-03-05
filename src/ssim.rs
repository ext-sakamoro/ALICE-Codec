//! SSIM (Structural Similarity Index) 品質メトリクス
//!
//! 人間の知覚に近い画質評価指標。MSE/PSNR よりも
//! 主観品質との相関が高い。

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::error::CodecError;

/// SSIM 定数 (8-bit 画像用)。
const C1: f64 = 6.5025; // (0.01 * 255)^2
const C2: f64 = 58.5225; // (0.03 * 255)^2

/// 2つのブロックの SSIM を計算。
///
/// ブロックは同一サイズの f64 スライス。
fn ssim_raw(block_a: &[f64], block_b: &[f64]) -> f64 {
    let n = block_a.len() as f64;
    if n < 1.0 {
        return 1.0;
    }

    let mu_a: f64 = block_a.iter().sum::<f64>() / n;
    let mu_b: f64 = block_b.iter().sum::<f64>() / n;

    let mut sigma_a_sq = 0.0;
    let mut sigma_b_sq = 0.0;
    let mut sigma_ab = 0.0;

    for (&a, &b) in block_a.iter().zip(block_b.iter()) {
        let da = a - mu_a;
        let db = b - mu_b;
        sigma_a_sq += da * da;
        sigma_b_sq += db * db;
        sigma_ab += da * db;
    }

    // 不偏分散 (n-1 で割る) を使用
    let denom = (n - 1.0).max(1.0);
    sigma_a_sq /= denom;
    sigma_b_sq /= denom;
    sigma_ab /= denom;

    let numerator = (2.0 * mu_a).mul_add(mu_b, C1) * 2.0f64.mul_add(sigma_ab, C2);
    let denominator = (mu_a.mul_add(mu_a, mu_b * mu_b) + C1) * (sigma_a_sq + sigma_b_sq + C2);

    numerator / denominator
}

/// 画像全体の Mean SSIM を計算。
///
/// 8×8 ブロックに分割して各ブロックの SSIM を平均。
///
/// # Arguments
///
/// * `a`, `b` — 画像バッファ (u8)
/// * `width`, `height` — 画像サイズ
///
/// # Errors
///
/// バッファサイズが不一致、または `width * height` と一致しない場合。
pub fn ssim(a: &[u8], b: &[u8], width: usize, height: usize) -> Result<f64, CodecError> {
    if a.len() != b.len() {
        return Err(CodecError::InvalidBufferSize {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.len() != width * height {
        return Err(CodecError::InvalidBufferSize {
            expected: width * height,
            got: a.len(),
        });
    }
    if a.is_empty() {
        return Ok(1.0);
    }

    let block_size = 8;
    let mut total_ssim = 0.0;
    let mut block_count = 0u64;

    let mut block_a = Vec::with_capacity(block_size * block_size);
    let mut block_b = Vec::with_capacity(block_size * block_size);

    let bh = height / block_size;
    let bw = width / block_size;

    for by in 0..bh {
        for bx in 0..bw {
            block_a.clear();
            block_b.clear();

            for dy in 0..block_size {
                let y = by * block_size + dy;
                for dx in 0..block_size {
                    let x = bx * block_size + dx;
                    let idx = y * width + x;
                    block_a.push(f64::from(a[idx]));
                    block_b.push(f64::from(b[idx]));
                }
            }

            total_ssim += ssim_raw(&block_a, &block_b);
            block_count += 1;
        }
    }

    if block_count == 0 {
        return Ok(1.0);
    }

    Ok(total_ssim / block_count as f64)
}

/// Multi-Scale SSIM (MS-SSIM)。
///
/// 3 スケールで SSIM を計算し、重み付き幾何平均を返す。
/// 各スケールで 2×2 平均プーリングでダウンサンプルする。
///
/// # Errors
///
/// バッファサイズ不一致の場合。
pub fn ms_ssim(a: &[u8], b: &[u8], width: usize, height: usize) -> Result<f64, CodecError> {
    if a.len() != b.len() {
        return Err(CodecError::InvalidBufferSize {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.len() != width * height {
        return Err(CodecError::InvalidBufferSize {
            expected: width * height,
            got: a.len(),
        });
    }
    if a.is_empty() {
        return Ok(1.0);
    }

    // 重み (Wang et al. 2003 の 3-scale 版を簡略化)
    let weights = [0.3333, 0.3333, 0.3334];

    let mut cur_a: Vec<u8> = a.to_vec();
    let mut cur_b: Vec<u8> = b.to_vec();
    let mut cur_w = width;
    let mut cur_h = height;

    let mut result = 0.0_f64;

    for &weight in &weights {
        let s = ssim(&cur_a, &cur_b, cur_w, cur_h)?;
        result += weight * s.max(0.0).ln().max(-10.0);

        // 2x ダウンサンプル
        let new_w = cur_w / 2;
        let new_h = cur_h / 2;
        if new_w < 8 || new_h < 8 {
            // これ以上縮小不可 → 残りの重みを現在のスケールに割り当て
            for &w2 in weights.iter().skip(
                weights
                    .iter()
                    .position(|&x| (x - weight).abs() < 1e-10)
                    .unwrap_or(0)
                    + 1,
            ) {
                result += w2 * s.max(0.0).ln().max(-10.0);
            }
            break;
        }

        cur_a = downsample_2x(&cur_a, cur_w, cur_h);
        cur_b = downsample_2x(&cur_b, cur_w, cur_h);
        cur_w = new_w;
        cur_h = new_h;
    }

    Ok(result.exp())
}

/// 2×2 平均プーリングによるダウンサンプル。
fn downsample_2x(buf: &[u8], width: usize, height: usize) -> Vec<u8> {
    let new_w = width / 2;
    let new_h = height / 2;
    let mut out = Vec::with_capacity(new_w * new_h);

    for y in 0..new_h {
        for x in 0..new_w {
            let sy = y * 2;
            let sx = x * 2;
            let avg = (u16::from(buf[sy * width + sx])
                + u16::from(buf[sy * width + sx + 1])
                + u16::from(buf[(sy + 1) * width + sx])
                + u16::from(buf[(sy + 1) * width + sx + 1]))
                / 4;
            out.push(avg as u8);
        }
    }

    out
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssim_identical() {
        let buf = vec![128u8; 64 * 64];
        let result = ssim(&buf, &buf, 64, 64).unwrap();
        assert!(
            (result - 1.0).abs() < 1e-6,
            "Identical images should have SSIM=1.0, got {result}"
        );
    }

    #[test]
    fn ssim_different() {
        let a = vec![100u8; 64 * 64];
        let b = vec![200u8; 64 * 64];
        let result = ssim(&a, &b, 64, 64).unwrap();
        assert!(result < 1.0, "Different images should have SSIM < 1.0");
        assert!(result > 0.0, "SSIM should be positive");
    }

    #[test]
    fn ssim_slight_difference() {
        let a = vec![128u8; 64 * 64];
        let mut b = vec![128u8; 64 * 64];
        b[0] = 129;
        let result = ssim(&a, &b, 64, 64).unwrap();
        assert!(
            result > 0.99,
            "Small diff should give high SSIM, got {result}"
        );
    }

    #[test]
    fn ssim_symmetry() {
        let a: Vec<u8> = (0..64 * 64).map(|i| (i % 256) as u8).collect();
        let b: Vec<u8> = (0..64 * 64).map(|i| ((i + 10) % 256) as u8).collect();
        let ab = ssim(&a, &b, 64, 64).unwrap();
        let ba = ssim(&b, &a, 64, 64).unwrap();
        assert!(
            (ab - ba).abs() < 1e-10,
            "SSIM should be symmetric: {ab} vs {ba}"
        );
    }

    #[test]
    fn ssim_mismatched_lengths() {
        let a = vec![0u8; 100];
        let b = vec![0u8; 200];
        assert!(ssim(&a, &b, 10, 10).is_err());
    }

    #[test]
    fn ssim_wrong_dimensions() {
        let a = vec![0u8; 100];
        let b = vec![0u8; 100];
        assert!(ssim(&a, &b, 8, 8).is_err()); // 8*8=64 ≠ 100
    }

    #[test]
    fn ssim_empty() {
        let result = ssim(&[], &[], 0, 0).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ms_ssim_identical() {
        let buf = vec![128u8; 64 * 64];
        let result = ms_ssim(&buf, &buf, 64, 64).unwrap();
        assert!(
            (result - 1.0).abs() < 0.01,
            "Identical images MS-SSIM should be ~1.0, got {result}"
        );
    }

    #[test]
    fn ms_ssim_different() {
        let a = vec![50u8; 64 * 64];
        let b = vec![200u8; 64 * 64];
        let result = ms_ssim(&a, &b, 64, 64).unwrap();
        assert!(result < 1.0, "Different images should have MS-SSIM < 1.0");
    }

    #[test]
    fn ms_ssim_empty() {
        let result = ms_ssim(&[], &[], 0, 0).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn downsample_basic() {
        // 4x4 → 2x2
        let buf = vec![
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        let out = downsample_2x(&buf, 4, 4);
        assert_eq!(out.len(), 4);
        // (10+20+50+60)/4 = 35
        assert_eq!(out[0], 35);
    }

    #[test]
    fn ssim_range() {
        // SSIM should be in [-1, 1] for realistic images
        let a: Vec<u8> = (0..64 * 64).map(|i| (i * 7 % 256) as u8).collect();
        let b: Vec<u8> = (0..64 * 64).map(|i| (255 - i * 7 % 256) as u8).collect();
        let result = ssim(&a, &b, 64, 64).unwrap();
        assert!(
            (-1.0..=1.0).contains(&result),
            "SSIM should be in [-1, 1], got {result}"
        );
    }
}
