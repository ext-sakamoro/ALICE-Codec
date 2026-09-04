//! Fuzz target: encode → `to_bytes` → `from_bytes` の roundtrip 整合性
//!
//! 攻撃者が制御可能な入力 (frame 数 / 解像度 / RGB pixel 値 / quality) で
//! `FrameEncoder::encode` が panic せず、成功時は `to_bytes()` / `from_bytes()`
//! roundtrip で header (width/height/frames/wavelet_type) が bit-exact に
//! 保存されることを保証する
//!
//! canonical CI template [[reference_alice_ci_canonical_template]] 準拠

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct Input {
    /// Frame width (u8 で 0..=255 に絞り fuzz 予算枯渇予防)
    width: u8,
    /// Frame height (同上)
    height: u8,
    /// Frame count (1..=8 に絞る、大量 frame で OOM 予防)
    frames: u8,
    /// Quality (0..=100)
    quality: u8,
    /// Wavelet type selector (0 = Cdf53, 1 = Cdf97)
    wavelet_kind: u8,
    /// RGB payload (encoder 側で size check)
    rgb: Vec<u8>,
}

fuzz_target!(|input: Input| {
    // 総 pixel 数上限 (fuzz corpus 巨大化予防)
    let w = input.width as u32;
    let h = input.height as u32;
    let f = (input.frames % 8) as u32; // 0..=7
    let quality = input.quality.min(100);

    // 入力 payload size 上限 (256 KB)
    if input.rgb.len() > 256 * 1024 {
        return;
    }

    let wavelet_type = match input.wavelet_kind % 2 {
        0 => alice_codec::pipeline::WaveletType::Cdf53,
        _ => alice_codec::pipeline::WaveletType::Cdf97,
    };

    let encoder = alice_codec::pipeline::FrameEncoder::with_wavelet(quality, wavelet_type);
    let Ok(chunk) = encoder.encode(&input.rgb, w, h, f) else {
        return; // Err は panic でない、想定内
    };

    // Roundtrip: to_bytes → from_bytes で header 一致
    let bytes = chunk.to_bytes();
    let restored = alice_codec::pipeline::EncodedChunk::from_bytes(&bytes)
        .expect("valid encoded chunk must decode from its own bytes");

    assert_eq!(restored.width, chunk.width);
    assert_eq!(restored.height, chunk.height);
    assert_eq!(restored.frames, chunk.frames);
    assert_eq!(restored.wavelet_type, chunk.wavelet_type);
});
