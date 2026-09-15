//! `no_std` 用の float 数学関数 shim
//!
//! `std` あり: `f64` の inherent method (`mul_add` / `ln` / `exp`) をそのまま使う
//! (本 module は空)
//!
//! `std` なし (`--no-default-features`): core には float の超越関数が無いため、
//! 同名 method を [`FloatExt`] trait で提供し、実装は [`libm`] (pure Rust、
//! `no_std`) に委譲する 各 module は
//! `#[cfg(not(feature = "std"))] use crate::math::FloatExt;` で取り込む
//!
//! 精度注意: `libm` と platform libm は最終 ulp で異なりうる rANS / wavelet の
//! 整数 core には影響しない (`ssim` / `rate_control` の指標計算のみ)

#[cfg(not(feature = "std"))]
pub trait FloatExt: Sized {
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn ln(self) -> Self;
    fn exp(self) -> Self;
}

#[cfg(not(feature = "std"))]
impl FloatExt for f64 {
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        libm::fma(self, a, b)
    }
    #[inline]
    fn ln(self) -> Self {
        libm::log(self)
    }
    #[inline]
    fn exp(self) -> Self {
        libm::exp(self)
    }
}
