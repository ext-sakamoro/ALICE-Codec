//! レート制御 — 目標ビットレートベースの品質調整
//!
//! バッファモデルに基づき、目標ビットレートを達成するための
//! 品質パラメータ (0–100) を動的に調整する。

/// レート制御設定。
#[derive(Debug, Clone, Copy)]
pub struct RateControlConfig {
    /// 目標ビットレート (kbps)。
    pub target_bitrate_kbps: u32,
    /// フレームレート (fps)。
    pub framerate: f64,
    /// 最小品質 (0–100)。
    pub min_quality: u32,
    /// 最大品質 (0–100)。
    pub max_quality: u32,
    /// バッファサイズ (ビット)。目標ビットレートの数秒分。
    pub buffer_size_bits: u64,
}

impl Default for RateControlConfig {
    fn default() -> Self {
        Self {
            target_bitrate_kbps: 5_000,
            framerate: 30.0,
            min_quality: 10,
            max_quality: 95,
            buffer_size_bits: 5_000 * 1_000 * 2, // 2秒分
        }
    }
}

/// レート制御の状態。
#[derive(Debug)]
pub struct RateController {
    config: RateControlConfig,
    /// バッファ内のビット数。
    buffer_fullness: i64,
    /// 直近のフレームサイズ履歴 (ビット)。
    frame_history: Vec<u64>,
    /// 最大履歴長。
    max_history: usize,
    /// 現在の品質。
    current_quality: u32,
    /// エンコード済みフレーム数。
    frame_count: u64,
}

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

impl RateController {
    /// 新しいレートコントローラーを作成。
    #[must_use]
    pub const fn new(config: RateControlConfig) -> Self {
        let initial_quality = u32::midpoint(config.min_quality, config.max_quality);
        Self {
            buffer_fullness: config.buffer_size_bits as i64 / 2, // 半分からスタート
            frame_history: Vec::new(),
            max_history: 30,
            current_quality: initial_quality,
            config,
            frame_count: 0,
        }
    }

    /// デフォルト設定で作成。
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(RateControlConfig::default())
    }

    /// 1フレームあたりの目標ビット数。
    #[must_use]
    pub fn target_bits_per_frame(&self) -> u64 {
        if self.config.framerate <= 0.0 {
            return 0;
        }
        (f64::from(self.config.target_bitrate_kbps) * 1000.0 / self.config.framerate) as u64
    }

    /// フレームエンコード前に推奨品質を取得。
    #[must_use]
    pub const fn recommended_quality(&self) -> u32 {
        self.current_quality
    }

    /// フレームエンコード後にサイズを報告し、品質を調整。
    pub fn update(&mut self, frame_size_bits: u64) {
        let target = self.target_bits_per_frame() as i64;

        // バッファモデル更新
        self.buffer_fullness += target - frame_size_bits as i64;
        self.buffer_fullness = self.buffer_fullness.clamp(
            -(self.config.buffer_size_bits as i64),
            self.config.buffer_size_bits as i64,
        );

        // 履歴に追加
        self.frame_history.push(frame_size_bits);
        if self.frame_history.len() > self.max_history {
            self.frame_history.remove(0);
        }

        self.frame_count += 1;

        // 品質調整
        self.adjust_quality();
    }

    /// バッファ充填率に基づいて品質を調整。
    fn adjust_quality(&mut self) {
        let buf_ratio = self.buffer_fullness as f64 / self.config.buffer_size_bits as f64;

        // バッファが溢れそう → 品質を上げる (サイズ増加OK)
        // バッファが枯渇しそう → 品質を下げる (サイズ削減)
        let adjustment = if buf_ratio > 0.3 {
            // バッファに余裕 → 品質向上
            1_i32
        } else if buf_ratio < -0.3 {
            // バッファ不足 → 品質低下
            -2 // より積極的に品質を下げる
        } else {
            0
        };

        let new_quality = (self.current_quality as i32 + adjustment).clamp(
            self.config.min_quality as i32,
            self.config.max_quality as i32,
        ) as u32;
        self.current_quality = new_quality;
    }

    /// 現在のバッファ充填率 (-1.0 ~ 1.0)。
    #[must_use]
    pub fn buffer_ratio(&self) -> f64 {
        if self.config.buffer_size_bits == 0 {
            return 0.0;
        }
        self.buffer_fullness as f64 / self.config.buffer_size_bits as f64
    }

    /// 直近フレームの平均サイズ (ビット)。
    #[must_use]
    pub fn average_frame_size(&self) -> u64 {
        if self.frame_history.is_empty() {
            return 0;
        }
        self.frame_history.iter().sum::<u64>() / self.frame_history.len() as u64
    }

    /// エンコード済みフレーム数。
    #[must_use]
    pub const fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// 現在の品質。
    #[must_use]
    pub const fn current_quality(&self) -> u32 {
        self.current_quality
    }

    /// 目標ビットレートに対する実際のビットレート比率。
    ///
    /// 1.0 未満 = 目標以下、1.0 超 = 目標超過。
    #[must_use]
    pub fn actual_to_target_ratio(&self) -> f64 {
        let target = self.target_bits_per_frame();
        let actual = self.average_frame_size();
        if target == 0 {
            return 0.0;
        }
        actual as f64 / target as f64
    }
}

/// 目標ビットレートからの品質推定 (静的、履歴なし)。
///
/// 経験的な対応テーブルに基づく初期推定値。
/// 実際のエンコードでは `RateController` を使用して動的調整すべき。
#[must_use]
pub fn estimate_quality(target_bitrate_kbps: u32, width: u32, height: u32, fps: f64) -> u32 {
    if fps <= 0.0 || width == 0 || height == 0 {
        return 50;
    }

    let pixels_per_sec = f64::from(width) * f64::from(height) * fps;
    let bits_per_pixel = f64::from(target_bitrate_kbps) * 1000.0 / pixels_per_sec;

    // 経験的マッピング: bpp → quality
    let quality = if bits_per_pixel > 2.0 {
        95.0
    } else if bits_per_pixel > 0.5 {
        bits_per_pixel.mul_add(30.0, 35.0)
    } else if bits_per_pixel > 0.1 {
        bits_per_pixel.mul_add(75.0, 12.5)
    } else {
        bits_per_pixel * 100.0 + 5.0
    };

    (quality as u32).clamp(5, 100)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = RateControlConfig::default();
        assert_eq!(config.target_bitrate_kbps, 5_000);
        assert!((config.framerate - 30.0).abs() < 1e-10);
    }

    #[test]
    fn target_bits_per_frame() {
        let ctrl = RateController::with_defaults();
        let target = ctrl.target_bits_per_frame();
        // 5000kbps / 30fps ≈ 166666 bits
        assert!(target > 150_000 && target < 180_000, "target = {target}");
    }

    #[test]
    fn initial_quality() {
        let ctrl = RateController::with_defaults();
        let q = ctrl.recommended_quality();
        assert!((10..=95).contains(&q));
    }

    #[test]
    fn quality_decreases_on_overshoot() {
        let mut ctrl = RateController::with_defaults();
        let target = ctrl.target_bits_per_frame();
        // 目標の3倍のフレームを連続送信 → 品質低下
        for _ in 0..30 {
            ctrl.update(target * 3);
        }
        assert!(
            ctrl.current_quality() < 52,
            "Quality should decrease, got {}",
            ctrl.current_quality()
        );
    }

    #[test]
    fn quality_increases_on_undershoot() {
        let mut ctrl = RateController::with_defaults();
        let target = ctrl.target_bits_per_frame();
        // 目標の1/3のフレーム → 品質向上
        for _ in 0..30 {
            ctrl.update(target / 3);
        }
        assert!(
            ctrl.current_quality() > 52,
            "Quality should increase, got {}",
            ctrl.current_quality()
        );
    }

    #[test]
    fn quality_clamped() {
        let config = RateControlConfig {
            min_quality: 20,
            max_quality: 80,
            ..Default::default()
        };
        let mut ctrl = RateController::new(config);
        // 大量のオーバーシュート
        for _ in 0..1000 {
            ctrl.update(10_000_000);
        }
        assert!(ctrl.current_quality() >= 20);

        // 大量のアンダーシュート
        for _ in 0..1000 {
            ctrl.update(1);
        }
        assert!(ctrl.current_quality() <= 80);
    }

    #[test]
    fn buffer_ratio_range() {
        let mut ctrl = RateController::with_defaults();
        ctrl.update(0);
        let ratio = ctrl.buffer_ratio();
        assert!(
            (-1.0..=1.0).contains(&ratio),
            "Buffer ratio should be in [-1, 1], got {ratio}"
        );
    }

    #[test]
    fn average_frame_size() {
        let mut ctrl = RateController::with_defaults();
        ctrl.update(1000);
        ctrl.update(2000);
        ctrl.update(3000);
        assert_eq!(ctrl.average_frame_size(), 2000);
    }

    #[test]
    fn frame_count() {
        let mut ctrl = RateController::with_defaults();
        assert_eq!(ctrl.frame_count(), 0);
        ctrl.update(1000);
        ctrl.update(2000);
        assert_eq!(ctrl.frame_count(), 2);
    }

    #[test]
    fn actual_to_target_ratio() {
        let mut ctrl = RateController::with_defaults();
        let target = ctrl.target_bits_per_frame();
        ctrl.update(target);
        let ratio = ctrl.actual_to_target_ratio();
        assert!(
            (ratio - 1.0).abs() < 0.01,
            "Should be ~1.0 when matching target, got {ratio}"
        );
    }

    #[test]
    fn estimate_quality_high_bitrate() {
        let q = estimate_quality(50_000, 1920, 1080, 30.0);
        assert!(q > 50, "High bitrate should give high quality, got {q}");
    }

    #[test]
    fn estimate_quality_low_bitrate() {
        let q = estimate_quality(100, 1920, 1080, 30.0);
        assert!(q < 30, "Low bitrate should give low quality, got {q}");
    }

    #[test]
    fn estimate_quality_zero_dims() {
        let q = estimate_quality(5000, 0, 0, 30.0);
        assert_eq!(q, 50); // fallback
    }
}
