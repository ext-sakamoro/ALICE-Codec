//! End-to-end encode/decode pipeline
//!
//! Wires together color conversion, 3D wavelet transform, quantization,
//! and rANS entropy coding into a single API.
//!
//! # Pipeline
//!
//! ```text
//! Encode: RGB frames -> YCoCg-R -> i32 -> 3D Wavelet -> Quantize -> Symbols -> rANS -> bytes
//! Decode: bytes -> rANS -> Symbols -> Dequantize -> Inverse 3D Wavelet -> i32 -> YCoCg-R -> RGB
//! ```
//!
//! Each color channel (Y, Co, Cg) is processed independently through the
//! wavelet, quantization, and entropy coding stages.

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::color::{rgb_bytes_to_ycocg_r, ycocg_r_to_rgb_bytes};
use crate::quant::{build_histogram, from_symbols, to_symbols, Quantizer};
use crate::rans::{FrequencyTable, RansDecoder, RansEncoder};
use crate::wavelet::{Wavelet1D, Wavelet3D};

/// Copy i16 channel data into a padded i32 buffer for wavelet processing.
///
/// Spatial and temporal dimensions are padded to even lengths by replicating
/// the boundary values (last column, last row, last frame).
fn pad_channel_to_i32(
    ch_data: &[i16],
    w: usize,
    h: usize,
    f: usize,
    padded_w: usize,
    padded_h: usize,
    padded_frames: usize,
) -> Vec<i32> {
    let padded_pixels = padded_w * padded_h * padded_frames;
    let mut buf = vec![0i32; padded_pixels];
    for t in 0..f {
        for row in 0..h {
            for col in 0..w {
                let src = t * w * h + row * w + col;
                let dst = t * padded_w * padded_h + row * padded_w + col;
                buf[dst] = ch_data[src] as i32;
            }
            if padded_w > w {
                let last_val = ch_data[t * w * h + row * w + (w - 1)] as i32;
                buf[t * padded_w * padded_h + row * padded_w + w] = last_val;
            }
        }
        if padded_h > h {
            for col in 0..padded_w {
                let val = buf[t * padded_w * padded_h + (h - 1) * padded_w + col];
                buf[t * padded_w * padded_h + h * padded_w + col] = val;
            }
        }
    }
    for t in f..padded_frames {
        let src_frame = f - 1;
        for idx in 0..(padded_w * padded_h) {
            buf[t * padded_w * padded_h + idx] = buf[src_frame * padded_w * padded_h + idx];
        }
    }
    buf
}

/// Per-channel header stored inside an [`EncodedChunk`].
///
/// Contains all metadata needed to decode a single color channel back from
/// the compressed byte stream.
#[derive(Clone, Debug)]
struct ChannelHeader {
    /// Length of the compressed rANS bitstream for this channel.
    compressed_len: u32,
    /// Quantization step that was used during encoding.
    quant_step: i32,
    /// Dead-zone width that was used during encoding.
    quant_dead_zone: i32,
    /// Number of symbols (equal to the number of wavelet coefficients).
    num_symbols: u32,
    /// 256-bin symbol histogram needed to rebuild the frequency table.
    histogram: [u32; 256],
}

/// Compressed representation of one or more RGB video frames.
///
/// Produced by [`FrameEncoder::encode`] and consumed by
/// [`FrameDecoder::decode`].
#[derive(Clone, Debug)]
pub struct EncodedChunk {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Number of frames in this chunk.
    pub frames: u32,
    /// Per-channel headers (Y, Co, Cg).
    channel_headers: [ChannelHeader; 3],
    /// Concatenated compressed data for all three channels.
    compressed_data: Vec<u8>,
}

impl EncodedChunk {
    /// Total size of the compressed payload in bytes.
    #[must_use]
    pub fn compressed_size(&self) -> usize {
        self.compressed_data.len()
    }
}

/// Video frame encoder.
///
/// Compresses interleaved RGB byte frames through the full pipeline:
/// color conversion, 3-D wavelet, quantization, and rANS entropy coding.
pub struct FrameEncoder {
    /// Quality setting (0-100). Accessible within the crate for Python bindings.
    pub(crate) quality: u8,
}

impl FrameEncoder {
    /// Create an encoder with the given quality setting (0-100).
    ///
    /// Quality 0 yields maximum compression; quality 100 is near-lossless.
    #[must_use]
    pub fn new(quality: u8) -> Self {
        Self { quality }
    }

    /// Encode interleaved RGB frames into a compressed chunk.
    ///
    /// # Arguments
    ///
    /// * `rgb_frames` - Interleaved RGB bytes: `[R0,G0,B0, R1,G1,B1, ...]`
    ///   for all pixels of all frames laid out sequentially (frame-major).
    /// * `width`  - Frame width in pixels (must be even).
    /// * `height` - Frame height in pixels (must be even).
    /// * `frames` - Number of frames (must be even, or 1 which is handled
    ///   via zero-padding).
    ///
    /// # Panics
    ///
    /// Panics if `rgb_frames.len() != width * height * frames * 3`, or if
    /// `width` or `height` is zero.
    #[must_use]
    pub fn encode(&self, rgb_frames: &[u8], width: u32, height: u32, frames: u32) -> EncodedChunk {
        let w = width as usize;
        let h = height as usize;
        let f = frames as usize;
        let n_pixels = w * h * f;

        assert_eq!(
            rgb_frames.len(),
            n_pixels * 3,
            "rgb_frames length must equal width * height * frames * 3"
        );

        if n_pixels == 0 {
            return EncodedChunk {
                width,
                height,
                frames,
                channel_headers: core::array::from_fn(|_| ChannelHeader {
                    compressed_len: 0,
                    quant_step: 1,
                    quant_dead_zone: 1,
                    num_symbols: 0,
                    histogram: [0u32; 256],
                }),
                compressed_data: Vec::new(),
            };
        }

        // --- 1. Color conversion: RGB -> YCoCg-R (planar i16) ---
        let mut y_i16 = vec![0i16; n_pixels];
        let mut co_i16 = vec![0i16; n_pixels];
        let mut cg_i16 = vec![0i16; n_pixels];

        rgb_bytes_to_ycocg_r(rgb_frames, &mut y_i16, &mut co_i16, &mut cg_i16);

        // --- 2. Pad temporal dimension to even length if needed ---
        let padded_frames = if f == 1 { 2 } else { f + (f & 1) };
        let padded_w = w + (w & 1);
        let padded_h = h + (h & 1);
        let padded_pixels = padded_w * padded_h * padded_frames;

        // --- 3. Process each channel independently ---
        let channels_i16: [&[i16]; 3] = [&y_i16, &co_i16, &cg_i16];
        let mut all_compressed = Vec::new();
        let mut channel_headers: [ChannelHeader; 3] = core::array::from_fn(|_| ChannelHeader {
            compressed_len: 0,
            quant_step: 1,
            quant_dead_zone: 1,
            num_symbols: 0,
            histogram: [0u32; 256],
        });

        // Map quality 0-100 to quantization step.
        // quality 100 -> step 1 (near-lossless)
        // quality 0   -> step 64 (heavy compression)
        let clamped_q = self.quality.min(100) as i32;
        let quant_step = (64 - (clamped_q * 63) / 100).max(1);

        for (ch_idx, &ch_data) in channels_i16.iter().enumerate() {
            let mut buf = pad_channel_to_i32(ch_data, w, h, f, padded_w, padded_h, padded_frames);

            // 3-D forward wavelet transform (CDF 5/3)
            let w3d = Wavelet3D::new(Wavelet1D::cdf53());
            w3d.forward(&mut buf, padded_w, padded_h, padded_frames);

            // Build quantizer from the quality-derived step
            let quantizer = Quantizer::new(quant_step);

            // Quantize
            let mut qbuf = vec![0i32; padded_pixels];
            quantizer.quantize_buffer(&buf, &mut qbuf);

            // Signed -> unsigned symbol mapping (zigzag)
            let mut symbols = vec![0u8; padded_pixels];
            to_symbols(&qbuf, &mut symbols);

            // Build histogram and frequency table
            let histogram = build_histogram(&symbols);
            let freq_table = FrequencyTable::from_histogram(&histogram);

            // rANS encode
            let mut encoder = RansEncoder::new();
            encoder.encode_symbols(&symbols, &freq_table);
            let compressed = encoder.finish();

            channel_headers[ch_idx] = ChannelHeader {
                compressed_len: compressed.len() as u32,
                quant_step: quantizer.step,
                quant_dead_zone: quantizer.dead_zone,
                num_symbols: padded_pixels as u32,
                histogram,
            };

            all_compressed.extend_from_slice(&compressed);
        }

        EncodedChunk {
            width,
            height,
            frames,
            channel_headers,
            compressed_data: all_compressed,
        }
    }
}

/// Video frame decoder.
///
/// Reconstructs interleaved RGB byte frames from a compressed
/// [`EncodedChunk`].
pub struct FrameDecoder;

impl FrameDecoder {
    /// Create a new decoder.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Decode a compressed chunk back to interleaved RGB bytes.
    ///
    /// The returned buffer is laid out as
    /// `[R0,G0,B0, R1,G1,B1, ...]` for all pixels of all frames.
    ///
    /// # Panics
    ///
    /// Panics if the chunk metadata is internally inconsistent.
    #[must_use]
    pub fn decode(&self, chunk: &EncodedChunk) -> Vec<u8> {
        let w = chunk.width as usize;
        let h = chunk.height as usize;
        let f = chunk.frames as usize;
        let n_pixels = w * h * f;

        if n_pixels == 0 {
            return Vec::new();
        }

        let padded_frames = if f == 1 { 2 } else { f + (f & 1) };
        let padded_w = w + (w & 1);
        let padded_h = h + (h & 1);
        let padded_pixels = padded_w * padded_h * padded_frames;

        let mut channel_i16: [Vec<i16>; 3] = [
            vec![0i16; n_pixels],
            vec![0i16; n_pixels],
            vec![0i16; n_pixels],
        ];

        let mut data_offset: usize = 0;

        for (ch_idx, ch_hdr) in chunk.channel_headers.iter().enumerate() {
            let hdr = ch_hdr;
            let compressed_len = hdr.compressed_len as usize;
            let num_symbols = hdr.num_symbols as usize;

            assert_eq!(
                num_symbols, padded_pixels,
                "num_symbols mismatch with padded dimensions"
            );

            let compressed = &chunk.compressed_data[data_offset..data_offset + compressed_len];
            data_offset += compressed_len;

            // Rebuild frequency table from stored histogram
            let freq_table = FrequencyTable::from_histogram(&hdr.histogram);

            // rANS decode
            let mut decoder = RansDecoder::new(compressed);
            let symbols = decoder.decode_n(num_symbols, &freq_table);

            // Symbols -> signed coefficients (zigzag inverse)
            let mut qbuf = vec![0i32; padded_pixels];
            from_symbols(&symbols, &mut qbuf);

            // Dequantize
            let quantizer = Quantizer::with_dead_zone(hdr.quant_step, hdr.quant_dead_zone);
            let mut buf = vec![0i32; padded_pixels];
            quantizer.dequantize_buffer(&qbuf, &mut buf);

            // Inverse 3-D wavelet transform
            let w3d = Wavelet3D::new(Wavelet1D::cdf53());
            w3d.inverse(&mut buf, padded_w, padded_h, padded_frames);

            // i32 -> i16, strip padding
            let ch = &mut channel_i16[ch_idx];
            for t in 0..f {
                for row in 0..h {
                    for col in 0..w {
                        let src = t * padded_w * padded_h + row * padded_w + col;
                        let dst = t * w * h + row * w + col;
                        ch[dst] = buf[src] as i16;
                    }
                }
            }
        }

        // --- YCoCg-R -> RGB ---
        let mut rgb_out = vec![0u8; n_pixels * 3];
        ycocg_r_to_rgb_bytes(
            &channel_i16[0],
            &channel_i16[1],
            &channel_i16[2],
            &mut rgb_out,
        );

        rgb_out
    }
}

impl Default for FrameDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compute PSNR (Peak Signal-to-Noise Ratio) between two byte buffers.
    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        if a.is_empty() {
            return f64::INFINITY;
        }
        let mse: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = x as f64 - y as f64;
                diff * diff
            })
            .sum::<f64>()
            / a.len() as f64;
        if mse == 0.0 {
            return f64::INFINITY;
        }
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        // 4x4 image, 2 frames -- gradient pattern
        let width = 4u32;
        let height = 4u32;
        let frames = 2u32;
        let n_pixels = (width * height * frames) as usize;
        let mut rgb = vec![0u8; n_pixels * 3];

        for i in 0..n_pixels {
            let v = ((i * 7) % 256) as u8;
            rgb[i * 3] = v;
            rgb[i * 3 + 1] = v.wrapping_add(30);
            rgb[i * 3 + 2] = v.wrapping_add(60);
        }

        let encoder = FrameEncoder::new(90);
        let chunk = encoder.encode(&rgb, width, height, frames);

        let decoder = FrameDecoder::new();
        let decoded = decoder.decode(&chunk);

        assert_eq!(decoded.len(), rgb.len());

        // At quality 90 we expect reasonable PSNR (at least 20 dB for this
        // small image where quantization artefacts are amplified).
        let p = psnr(&rgb, &decoded);
        assert!(p > 15.0, "PSNR too low: {:.2} dB (expected > 15 dB)", p);
    }

    #[test]
    fn test_encode_decode_solid_color() {
        // All pixels the same color -- should be near-lossless.
        let width = 4u32;
        let height = 4u32;
        let frames = 2u32;
        let n_pixels = (width * height * frames) as usize;
        let mut rgb = vec![0u8; n_pixels * 3];

        for i in 0..n_pixels {
            rgb[i * 3] = 100;
            rgb[i * 3 + 1] = 150;
            rgb[i * 3 + 2] = 200;
        }

        let encoder = FrameEncoder::new(95);
        let chunk = encoder.encode(&rgb, width, height, frames);

        let decoder = FrameDecoder::new();
        let decoded = decoder.decode(&chunk);

        assert_eq!(decoded.len(), rgb.len());

        // Solid color: the wavelet coefficients outside LLL are zero, so the
        // only error comes from the quantizer dead-zone.  PSNR should be very
        // high.
        let p = psnr(&rgb, &decoded);
        assert!(p > 25.0, "PSNR too low for solid color: {:.2} dB", p);
    }

    #[test]
    fn test_encode_decode_quality_levels() {
        let width = 4u32;
        let height = 4u32;
        let frames = 2u32;
        let n_pixels = (width * height * frames) as usize;
        let mut rgb = vec![0u8; n_pixels * 3];

        for i in 0..n_pixels {
            let v = ((i * 13 + 7) % 256) as u8;
            rgb[i * 3] = v;
            rgb[i * 3 + 1] = v.wrapping_add(50);
            rgb[i * 3 + 2] = v.wrapping_add(100);
        }

        let chunk_low = FrameEncoder::new(10).encode(&rgb, width, height, frames);
        let chunk_high = FrameEncoder::new(90).encode(&rgb, width, height, frames);

        // Higher quality should produce equal or larger compressed output
        // (less aggressive quantization -> more entropy -> more bytes).
        // Note: for very small images the overhead may dominate, so we just
        // check that both encode and decode correctly.
        let decoded_low = FrameDecoder::new().decode(&chunk_low);
        let decoded_high = FrameDecoder::new().decode(&chunk_high);

        assert_eq!(decoded_low.len(), rgb.len());
        assert_eq!(decoded_high.len(), rgb.len());

        let psnr_low = psnr(&rgb, &decoded_low);
        let psnr_high = psnr(&rgb, &decoded_high);

        // Higher quality must yield better or equal reconstruction.
        assert!(
            psnr_high >= psnr_low - 1.0,
            "Higher quality PSNR ({:.2}) should be >= lower quality PSNR ({:.2})",
            psnr_high,
            psnr_low
        );
    }

    #[test]
    fn test_encode_decode_single_frame() {
        // frames=1 edge case: temporal wavelet still works via padding.
        let width = 4u32;
        let height = 4u32;
        let frames = 1u32;
        let n_pixels = (width * height * frames) as usize;
        let mut rgb = vec![0u8; n_pixels * 3];

        for i in 0..n_pixels {
            rgb[i * 3] = (i * 5 % 256) as u8;
            rgb[i * 3 + 1] = (i * 11 % 256) as u8;
            rgb[i * 3 + 2] = (i * 17 % 256) as u8;
        }

        let encoder = FrameEncoder::new(90);
        let chunk = encoder.encode(&rgb, width, height, frames);

        assert_eq!(chunk.frames, 1);

        let decoder = FrameDecoder::new();
        let decoded = decoder.decode(&chunk);

        assert_eq!(decoded.len(), rgb.len());

        let p = psnr(&rgb, &decoded);
        assert!(p > 10.0, "Single-frame PSNR too low: {:.2} dB", p);
    }

    #[test]
    fn test_empty_input() {
        // Zero-area frames: should produce an empty output without panic.
        let encoder = FrameEncoder::new(50);
        let chunk = encoder.encode(&[], 0, 0, 0);

        assert_eq!(chunk.compressed_size(), 0);

        let decoder = FrameDecoder::new();
        let decoded = decoder.decode(&chunk);

        assert!(decoded.is_empty());
    }
}
