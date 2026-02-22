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

use core::fmt;

use crate::color::{rgb_bytes_to_ycocg_r, ycocg_r_to_rgb_bytes};
use crate::quant::{build_histogram, from_symbols, to_symbols, Quantizer};
use crate::rans::{FrequencyTable, RansDecoder, RansEncoder};
use crate::wavelet::{Wavelet1D, Wavelet3D};

// ── Wavelet type selection ─────────────────────────────────────

/// Wavelet filter type used by the pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum WaveletType {
    /// Integer CDF 5/3 — lossless-capable, good quality/speed balance (default).
    Cdf53 = 0,
    /// Integer CDF 9/7 — better compression for lossy, more computation.
    Cdf97 = 1,
    /// Haar — fastest, lowest complexity, blocky at low quality.
    Haar = 2,
}

impl WaveletType {
    fn to_wavelet1d(self) -> Wavelet1D {
        match self {
            Self::Cdf53 => Wavelet1D::cdf53(),
            Self::Cdf97 => Wavelet1D::cdf97(),
            Self::Haar => Wavelet1D::haar(),
        }
    }

    fn from_u8(v: u8) -> Result<Self, CodecError> {
        match v {
            0 => Ok(Self::Cdf53),
            1 => Ok(Self::Cdf97),
            2 => Ok(Self::Haar),
            _ => Err(CodecError::InvalidBitstream(format!(
                "unknown wavelet type byte: {v}"
            ))),
        }
    }
}

// ── Error type ─────────────────────────────────────────────────

/// Errors that can occur during encoding or decoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodecError {
    /// Input buffer size does not match the declared dimensions.
    InvalidBufferSize { expected: usize, got: usize },
    /// Width or height is zero.
    InvalidDimensions { width: u32, height: u32 },
    /// Dimensions overflow `usize` when multiplied together.
    DimensionOverflow,
    /// The compressed bitstream is malformed or truncated.
    InvalidBitstream(String),
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBufferSize { expected, got } => {
                write!(f, "buffer size mismatch: expected {expected}, got {got}")
            }
            Self::InvalidDimensions { width, height } => {
                write!(f, "invalid dimensions: {width}x{height}")
            }
            Self::DimensionOverflow => write!(f, "dimensions overflow usize"),
            Self::InvalidBitstream(msg) => write!(f, "invalid bitstream: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CodecError {}

// ── Helpers ────────────────────────────────────────────────────

/// Checked `w * h * f * 3` that returns `CodecError::DimensionOverflow`.
fn checked_pixel_count(w: usize, h: usize, f: usize) -> Result<usize, CodecError> {
    w.checked_mul(h)
        .and_then(|wh| wh.checked_mul(f))
        .ok_or(CodecError::DimensionOverflow)
}

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

// ── Channel header ─────────────────────────────────────────────

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

/// Byte size of one serialized `ChannelHeader`.
const CHANNEL_HEADER_BYTES: usize = 4 + 4 + 4 + 4 + 256 * 4; // 1040

// ── EncodedChunk ───────────────────────────────────────────────

/// File magic bytes for the ALICE-Codec bitstream format.
const MAGIC: &[u8; 4] = b"ALCC";

/// Current bitstream format version.
const FORMAT_VERSION: u8 = 1;

/// Fixed portion of the header (magic + version + wavelet + width + height + frames).
const FIXED_HEADER_BYTES: usize = 4 + 1 + 1 + 4 + 4 + 4; // 18

/// Compressed representation of one or more RGB video frames.
///
/// Produced by [`FrameEncoder::encode`] and consumed by
/// [`FrameDecoder::decode`].  Can be serialized to bytes via
/// [`to_bytes`](Self::to_bytes) and deserialized via
/// [`from_bytes`](Self::from_bytes).
#[derive(Clone, Debug)]
pub struct EncodedChunk {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Number of frames in this chunk.
    pub frames: u32,
    /// Wavelet filter used during encoding.
    pub wavelet_type: WaveletType,
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

    /// Serialize the entire chunk to a self-contained byte buffer.
    ///
    /// The returned bytes can be written to a file (`.alc`) or sent over the
    /// network.  Use [`from_bytes`](Self::from_bytes) to reconstruct the
    /// chunk.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let total = FIXED_HEADER_BYTES + 3 * CHANNEL_HEADER_BYTES + self.compressed_data.len();
        let mut buf = Vec::with_capacity(total);

        // Fixed header
        buf.extend_from_slice(MAGIC);
        buf.push(FORMAT_VERSION);
        buf.push(self.wavelet_type as u8);
        buf.extend_from_slice(&self.width.to_le_bytes());
        buf.extend_from_slice(&self.height.to_le_bytes());
        buf.extend_from_slice(&self.frames.to_le_bytes());

        // Per-channel headers
        for ch in &self.channel_headers {
            buf.extend_from_slice(&ch.compressed_len.to_le_bytes());
            buf.extend_from_slice(&ch.quant_step.to_le_bytes());
            buf.extend_from_slice(&ch.quant_dead_zone.to_le_bytes());
            buf.extend_from_slice(&ch.num_symbols.to_le_bytes());
            for &count in &ch.histogram {
                buf.extend_from_slice(&count.to_le_bytes());
            }
        }

        // Payload
        buf.extend_from_slice(&self.compressed_data);
        buf
    }

    /// Reconstruct an [`EncodedChunk`] from bytes previously produced by
    /// [`to_bytes`](Self::to_bytes).
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::InvalidBitstream`] if the data is truncated,
    /// has a wrong magic number, or an unsupported version.
    pub fn from_bytes(data: &[u8]) -> Result<Self, CodecError> {
        let min_len = FIXED_HEADER_BYTES + 3 * CHANNEL_HEADER_BYTES;
        if data.len() < min_len {
            return Err(CodecError::InvalidBitstream(format!(
                "data too short: {} bytes (minimum {})",
                data.len(),
                min_len
            )));
        }

        // Validate magic
        if &data[0..4] != MAGIC {
            return Err(CodecError::InvalidBitstream(
                "bad magic (expected ALCC)".into(),
            ));
        }

        let version = data[4];
        if version != FORMAT_VERSION {
            return Err(CodecError::InvalidBitstream(format!(
                "unsupported version: {version} (expected {FORMAT_VERSION})"
            )));
        }

        let wavelet_type = WaveletType::from_u8(data[5])?;
        let width = u32::from_le_bytes([data[6], data[7], data[8], data[9]]);
        let height = u32::from_le_bytes([data[10], data[11], data[12], data[13]]);
        let frames = u32::from_le_bytes([data[14], data[15], data[16], data[17]]);

        // Parse channel headers
        let mut channel_headers: [ChannelHeader; 3] = core::array::from_fn(|_| ChannelHeader {
            compressed_len: 0,
            quant_step: 1,
            quant_dead_zone: 1,
            num_symbols: 0,
            histogram: [0u32; 256],
        });

        let mut off = FIXED_HEADER_BYTES;
        let mut total_compressed: usize = 0;
        for ch in &mut channel_headers {
            ch.compressed_len =
                u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            off += 4;
            ch.quant_step =
                i32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            off += 4;
            ch.quant_dead_zone =
                i32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            off += 4;
            ch.num_symbols =
                u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            off += 4;
            for slot in &mut ch.histogram {
                *slot =
                    u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                off += 4;
            }
            total_compressed += ch.compressed_len as usize;
        }

        if data.len() < off + total_compressed {
            return Err(CodecError::InvalidBitstream(format!(
                "truncated payload: need {} more bytes",
                (off + total_compressed) - data.len()
            )));
        }

        let compressed_data = data[off..off + total_compressed].to_vec();

        Ok(Self {
            width,
            height,
            frames,
            wavelet_type,
            channel_headers,
            compressed_data,
        })
    }
}

// ── FrameEncoder ───────────────────────────────────────────────

/// Video frame encoder.
///
/// Compresses interleaved RGB byte frames through the full pipeline:
/// color conversion, 3-D wavelet, quantization, and rANS entropy coding.
pub struct FrameEncoder {
    /// Quality setting (0-100). Accessible within the crate for Python bindings.
    pub(crate) quality: u8,
    /// Wavelet filter selection.
    pub(crate) wavelet_type: WaveletType,
}

impl FrameEncoder {
    /// Create an encoder with the given quality (0-100) and CDF 5/3 wavelet.
    ///
    /// Quality 0 yields maximum compression; quality 100 is near-lossless.
    #[must_use]
    pub fn new(quality: u8) -> Self {
        Self {
            quality,
            wavelet_type: WaveletType::Cdf53,
        }
    }

    /// Create an encoder with explicit quality and wavelet type.
    #[must_use]
    pub fn with_wavelet(quality: u8, wavelet_type: WaveletType) -> Self {
        Self {
            quality,
            wavelet_type,
        }
    }

    /// Encode interleaved RGB frames into a compressed chunk.
    ///
    /// # Arguments
    ///
    /// * `rgb_frames` - Interleaved RGB bytes: `[R0,G0,B0, R1,G1,B1, ...]`
    ///   for all pixels of all frames laid out sequentially (frame-major).
    /// * `width`  - Frame width in pixels.
    /// * `height` - Frame height in pixels.
    /// * `frames` - Number of frames (1 is supported via temporal padding).
    ///
    /// # Errors
    ///
    /// Returns [`CodecError`] if dimensions are invalid, the buffer size is
    /// wrong, or the dimensions overflow `usize`.
    pub fn encode(
        &self,
        rgb_frames: &[u8],
        width: u32,
        height: u32,
        frames: u32,
    ) -> Result<EncodedChunk, CodecError> {
        let w = width as usize;
        let h = height as usize;
        let f = frames as usize;

        let n_pixels = checked_pixel_count(w, h, f)?;

        // Allow zero-area as a special case (empty chunk)
        if n_pixels == 0 {
            if !rgb_frames.is_empty() {
                return Err(CodecError::InvalidBufferSize {
                    expected: 0,
                    got: rgb_frames.len(),
                });
            }
            return Ok(EncodedChunk {
                width,
                height,
                frames,
                wavelet_type: self.wavelet_type,
                channel_headers: core::array::from_fn(|_| ChannelHeader {
                    compressed_len: 0,
                    quant_step: 1,
                    quant_dead_zone: 1,
                    num_symbols: 0,
                    histogram: [0u32; 256],
                }),
                compressed_data: Vec::new(),
            });
        }

        // Validate that width and height are non-zero when pixels > 0
        if w == 0 || h == 0 {
            return Err(CodecError::InvalidDimensions { width, height });
        }

        let expected_len = n_pixels
            .checked_mul(3)
            .ok_or(CodecError::DimensionOverflow)?;
        if rgb_frames.len() != expected_len {
            return Err(CodecError::InvalidBufferSize {
                expected: expected_len,
                got: rgb_frames.len(),
            });
        }

        // --- 1. Color conversion: RGB -> YCoCg-R (planar i16) ---
        let mut y_i16 = vec![0i16; n_pixels];
        let mut co_i16 = vec![0i16; n_pixels];
        let mut cg_i16 = vec![0i16; n_pixels];

        rgb_bytes_to_ycocg_r(rgb_frames, &mut y_i16, &mut co_i16, &mut cg_i16);

        // --- 2. Pad dimensions to even lengths ---
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

        let w1d = self.wavelet_type.to_wavelet1d();

        for (ch_idx, &ch_data) in channels_i16.iter().enumerate() {
            let mut buf = pad_channel_to_i32(ch_data, w, h, f, padded_w, padded_h, padded_frames);

            // 3-D forward wavelet transform
            let w3d = Wavelet3D::new(w1d.clone());
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

        Ok(EncodedChunk {
            width,
            height,
            frames,
            wavelet_type: self.wavelet_type,
            channel_headers,
            compressed_data: all_compressed,
        })
    }
}

// ── FrameDecoder ───────────────────────────────────────────────

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
    /// # Errors
    ///
    /// Returns [`CodecError`] if the chunk metadata is internally
    /// inconsistent or the compressed data is malformed.
    pub fn decode(&self, chunk: &EncodedChunk) -> Result<Vec<u8>, CodecError> {
        let w = chunk.width as usize;
        let h = chunk.height as usize;
        let f = chunk.frames as usize;
        let n_pixels = checked_pixel_count(w, h, f)?;

        if n_pixels == 0 {
            return Ok(Vec::new());
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

        let w1d = chunk.wavelet_type.to_wavelet1d();

        for (ch_idx, ch_hdr) in chunk.channel_headers.iter().enumerate() {
            let compressed_len = ch_hdr.compressed_len as usize;
            let num_symbols = ch_hdr.num_symbols as usize;

            if num_symbols != padded_pixels {
                return Err(CodecError::InvalidBitstream(format!(
                    "channel {ch_idx}: num_symbols {num_symbols} != padded_pixels {padded_pixels}"
                )));
            }

            if data_offset + compressed_len > chunk.compressed_data.len() {
                return Err(CodecError::InvalidBitstream(format!(
                    "channel {ch_idx}: compressed data overrun"
                )));
            }

            let compressed = &chunk.compressed_data[data_offset..data_offset + compressed_len];
            data_offset += compressed_len;

            // Rebuild frequency table from stored histogram
            let freq_table = FrequencyTable::from_histogram(&ch_hdr.histogram);

            // rANS decode
            let mut decoder = RansDecoder::new(compressed);
            let symbols = decoder.decode_n(num_symbols, &freq_table);

            // Symbols -> signed coefficients (zigzag inverse)
            let mut qbuf = vec![0i32; padded_pixels];
            from_symbols(&symbols, &mut qbuf);

            // Dequantize
            let quantizer = Quantizer::with_dead_zone(ch_hdr.quant_step, ch_hdr.quant_dead_zone);
            let mut buf = vec![0i32; padded_pixels];
            quantizer.dequantize_buffer(&qbuf, &mut buf);

            // Inverse 3-D wavelet transform
            let w3d = Wavelet3D::new(w1d.clone());
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

        Ok(rgb_out)
    }
}

impl Default for FrameDecoder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Thread-safety compile-time assertions ──────────────────────

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    #[allow(dead_code)]
    const fn check() {
        assert_send_sync::<FrameEncoder>();
        assert_send_sync::<FrameDecoder>();
        assert_send_sync::<EncodedChunk>();
        assert_send_sync::<CodecError>();
    }
};

// ── Tests ──────────────────────────────────────────────────────

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

    fn make_gradient(w: u32, h: u32, f: u32) -> Vec<u8> {
        let n = (w * h * f) as usize;
        let mut rgb = vec![0u8; n * 3];
        for i in 0..n {
            let v = ((i * 7) % 256) as u8;
            rgb[i * 3] = v;
            rgb[i * 3 + 1] = v.wrapping_add(30);
            rgb[i * 3 + 2] = v.wrapping_add(60);
        }
        rgb
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let (w, h, f) = (4, 4, 2);
        let rgb = make_gradient(w, h, f);
        let chunk = FrameEncoder::new(90).encode(&rgb, w, h, f).unwrap();
        let decoded = FrameDecoder::new().decode(&chunk).unwrap();
        assert_eq!(decoded.len(), rgb.len());
        assert!(psnr(&rgb, &decoded) > 15.0);
    }

    #[test]
    fn test_encode_decode_solid_color() {
        let (w, h, f) = (4, 4, 2);
        let n = (w * h * f) as usize;
        let mut rgb = vec![0u8; n * 3];
        for i in 0..n {
            rgb[i * 3] = 100;
            rgb[i * 3 + 1] = 150;
            rgb[i * 3 + 2] = 200;
        }
        let chunk = FrameEncoder::new(95).encode(&rgb, w, h, f).unwrap();
        let decoded = FrameDecoder::new().decode(&chunk).unwrap();
        assert_eq!(decoded.len(), rgb.len());
        assert!(psnr(&rgb, &decoded) > 25.0);
    }

    #[test]
    fn test_encode_decode_quality_levels() {
        let (w, h, f) = (4, 4, 2);
        let rgb = make_gradient(w, h, f);
        let lo = FrameDecoder::new()
            .decode(&FrameEncoder::new(10).encode(&rgb, w, h, f).unwrap())
            .unwrap();
        let hi = FrameDecoder::new()
            .decode(&FrameEncoder::new(90).encode(&rgb, w, h, f).unwrap())
            .unwrap();
        assert_eq!(lo.len(), rgb.len());
        assert_eq!(hi.len(), rgb.len());
        assert!(psnr(&rgb, &hi) >= psnr(&rgb, &lo) - 1.0);
    }

    #[test]
    fn test_encode_decode_single_frame() {
        let (w, h, f) = (4, 4, 1);
        let rgb = make_gradient(w, h, f);
        let chunk = FrameEncoder::new(90).encode(&rgb, w, h, f).unwrap();
        assert_eq!(chunk.frames, 1);
        let decoded = FrameDecoder::new().decode(&chunk).unwrap();
        assert_eq!(decoded.len(), rgb.len());
        assert!(psnr(&rgb, &decoded) > 10.0);
    }

    #[test]
    fn test_empty_input() {
        let chunk = FrameEncoder::new(50).encode(&[], 0, 0, 0).unwrap();
        assert_eq!(chunk.compressed_size(), 0);
        let decoded = FrameDecoder::new().decode(&chunk).unwrap();
        assert!(decoded.is_empty());
    }

    // ── New: Serialization ─────────────────────────────────────

    #[test]
    fn test_serialization_roundtrip() {
        let rgb = make_gradient(4, 4, 2);
        let chunk = FrameEncoder::new(80).encode(&rgb, 4, 4, 2).unwrap();
        let bytes = chunk.to_bytes();
        let restored = EncodedChunk::from_bytes(&bytes).unwrap();
        assert_eq!(restored.width, 4);
        assert_eq!(restored.height, 4);
        assert_eq!(restored.frames, 2);
        assert_eq!(restored.wavelet_type, WaveletType::Cdf53);
        assert_eq!(restored.compressed_size(), chunk.compressed_size());
        let decoded = FrameDecoder::new().decode(&restored).unwrap();
        assert_eq!(decoded.len(), rgb.len());
        assert!(psnr(&rgb, &decoded) > 15.0);
    }

    #[test]
    fn test_serialization_empty_chunk() {
        let chunk = FrameEncoder::new(50).encode(&[], 0, 0, 0).unwrap();
        let bytes = chunk.to_bytes();
        let restored = EncodedChunk::from_bytes(&bytes).unwrap();
        assert_eq!(restored.compressed_size(), 0);
    }

    #[test]
    fn test_from_bytes_bad_magic() {
        let err = EncodedChunk::from_bytes(b"BADDxxxxxxxxxxxxxxxxx0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");
        assert!(err.is_err());
    }

    #[test]
    fn test_from_bytes_truncated() {
        let err = EncodedChunk::from_bytes(b"ALCC");
        assert!(err.is_err());
    }

    // ── New: Error handling ────────────────────────────────────

    #[test]
    fn test_invalid_buffer_size() {
        let err = FrameEncoder::new(50).encode(&[0u8; 10], 4, 4, 2);
        assert!(matches!(err, Err(CodecError::InvalidBufferSize { .. })));
    }

    #[test]
    fn test_dimension_overflow() {
        // usize::MAX / 2 * 2 overflows
        let huge = u32::MAX;
        let err = FrameEncoder::new(50).encode(&[], huge, huge, huge);
        assert!(matches!(err, Err(CodecError::DimensionOverflow)));
    }

    // ── New: Odd dimensions ────────────────────────────────────

    #[test]
    fn test_odd_width() {
        let (w, h, f) = (3, 4, 2);
        let rgb = make_gradient(w, h, f);
        let chunk = FrameEncoder::new(90).encode(&rgb, w, h, f).unwrap();
        let decoded = FrameDecoder::new().decode(&chunk).unwrap();
        assert_eq!(decoded.len(), rgb.len());
        assert!(psnr(&rgb, &decoded) > 10.0);
    }

    #[test]
    fn test_odd_height() {
        let (w, h, f) = (4, 5, 2);
        let rgb = make_gradient(w, h, f);
        let chunk = FrameEncoder::new(90).encode(&rgb, w, h, f).unwrap();
        let decoded = FrameDecoder::new().decode(&chunk).unwrap();
        assert_eq!(decoded.len(), rgb.len());
        assert!(psnr(&rgb, &decoded) > 10.0);
    }

    #[test]
    fn test_odd_both() {
        let (w, h, f) = (3, 5, 1);
        let rgb = make_gradient(w, h, f);
        let chunk = FrameEncoder::new(90).encode(&rgb, w, h, f).unwrap();
        let decoded = FrameDecoder::new().decode(&chunk).unwrap();
        assert_eq!(decoded.len(), rgb.len());
        assert!(psnr(&rgb, &decoded) > 10.0);
    }

    #[test]
    fn test_1x1_single_pixel() {
        // 1x1x1 is an extreme edge case: padded to 2x2x2 for wavelet.
        // We verify it doesn't panic and produces valid output.
        let rgb = vec![128u8, 200, 50];
        let chunk = FrameEncoder::new(100).encode(&rgb, 1, 1, 1).unwrap();
        let decoded = FrameDecoder::new().decode(&chunk).unwrap();
        assert_eq!(decoded.len(), 3);
        // Also verify serialization roundtrip for this edge case
        let bytes = chunk.to_bytes();
        let restored = EncodedChunk::from_bytes(&bytes).unwrap();
        let decoded2 = FrameDecoder::new().decode(&restored).unwrap();
        assert_eq!(decoded, decoded2);
    }

    // ── New: Wavelet type selection ────────────────────────────

    #[test]
    fn test_wavelet_cdf97() {
        let rgb = make_gradient(4, 4, 2);
        let enc = FrameEncoder::with_wavelet(90, WaveletType::Cdf97);
        let chunk = enc.encode(&rgb, 4, 4, 2).unwrap();
        assert_eq!(chunk.wavelet_type, WaveletType::Cdf97);
        let decoded = FrameDecoder::new().decode(&chunk).unwrap();
        assert_eq!(decoded.len(), rgb.len());
        assert!(psnr(&rgb, &decoded) > 10.0);
    }

    #[test]
    fn test_wavelet_haar() {
        // Haar is the coarsest wavelet — use higher quality and larger image
        // for a reasonable PSNR.
        let (w, h, f) = (8, 8, 2);
        let rgb = make_gradient(w, h, f);
        let enc = FrameEncoder::with_wavelet(100, WaveletType::Haar);
        let chunk = enc.encode(&rgb, w, h, f).unwrap();
        assert_eq!(chunk.wavelet_type, WaveletType::Haar);
        let decoded = FrameDecoder::new().decode(&chunk).unwrap();
        assert_eq!(decoded.len(), rgb.len());
        // Haar at quality 100 (step=1) should still produce reasonable output.
        // Print first few bytes for diagnosis.
        let p = psnr(&rgb, &decoded);
        assert!(
            p > 5.0,
            "Haar PSNR too low: {p:.2} dB, first 12 bytes orig={:?} dec={:?}",
            &rgb[..12],
            &decoded[..12]
        );
    }

    #[test]
    fn test_wavelet_preserved_in_serialization() {
        let rgb = make_gradient(4, 4, 2);
        let enc = FrameEncoder::with_wavelet(80, WaveletType::Cdf97);
        let chunk = enc.encode(&rgb, 4, 4, 2).unwrap();
        let bytes = chunk.to_bytes();
        let restored = EncodedChunk::from_bytes(&bytes).unwrap();
        assert_eq!(restored.wavelet_type, WaveletType::Cdf97);
        let decoded = FrameDecoder::new().decode(&restored).unwrap();
        assert_eq!(decoded.len(), rgb.len());
    }
}
