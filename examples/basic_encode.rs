//! Basic encode/decode example for ALICE-Codec.
//!
//! ```sh
//! cargo run --example basic_encode
//! ```

#![allow(
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use alice_codec::{metrics, FrameDecoder, FrameEncoder};

fn main() {
    let width = 64u32;
    let height = 64u32;
    let frames = 4u32;
    let n_pixels = (width * height * frames) as usize;

    // Generate a synthetic gradient pattern (RGB)
    let mut rgb = vec![0u8; n_pixels * 3];
    for i in 0..n_pixels {
        let v = ((i * 7) % 256) as u8;
        rgb[i * 3] = v;
        rgb[i * 3 + 1] = v.wrapping_add(30);
        rgb[i * 3 + 2] = v.wrapping_add(60);
    }

    // Encode
    let quality = 50;
    let encoder = FrameEncoder::new(quality);
    let encoded = encoder.encode(&rgb, width, height, frames).unwrap();

    let compressed_size = encoded.to_bytes().len();
    let original_size = rgb.len();
    let ratio = original_size as f64 / compressed_size as f64;

    println!("Original:   {original_size} bytes");
    println!("Compressed: {compressed_size} bytes");
    println!("Ratio:      {ratio:.2}x");

    // Decode
    let decoder = FrameDecoder::new();
    let decoded = decoder.decode(&encoded).unwrap();

    // Quality check
    let db = metrics::psnr(&rgb, &decoded).unwrap();
    println!("PSNR:       {db:.2} dB");
}
