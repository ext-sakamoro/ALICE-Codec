//! CLI for ALICE-Codec
//!
//! ```bash
//! alice-codec encode input.raw -w 1920 -h 1080 -f 2 -q 90 -o output.alc
//! alice-codec decode input.alc -o output.raw
//! alice-codec info input.alc
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless
)]

use std::fs;
use std::process;

use clap::{Parser, Subcommand};

use alice_codec::{EncodedChunk, FrameDecoder, FrameEncoder, WaveletType};

#[derive(Parser)]
#[command(
    name = "alice-codec",
    version,
    about = "ALICE-Codec: Hyper-Fast 3D Wavelet Video Codec"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Encode raw RGB frames into an .alc bitstream
    Encode {
        /// Input file (raw interleaved RGB bytes)
        input: String,
        /// Output file (.alc)
        #[arg(short, long)]
        output: String,
        /// Frame width in pixels
        #[arg(short = 'W', long)]
        width: u32,
        /// Frame height in pixels
        #[arg(short = 'H', long)]
        height: u32,
        /// Number of frames
        #[arg(short, long, default_value_t = 1)]
        frames: u32,
        /// Quality (0-100, default: 90)
        #[arg(short, long, default_value_t = 90)]
        quality: u8,
        /// Wavelet type: cdf53, cdf97, haar
        #[arg(short, long, default_value = "cdf53")]
        wavelet: String,
    },
    /// Decode an .alc bitstream back to raw RGB
    Decode {
        /// Input file (.alc)
        input: String,
        /// Output file (raw RGB bytes)
        #[arg(short, long)]
        output: String,
    },
    /// Show metadata of an .alc bitstream
    Info {
        /// Input file (.alc)
        input: String,
    },
}

fn parse_wavelet(s: &str) -> Result<WaveletType, String> {
    match s {
        "cdf53" => Ok(WaveletType::Cdf53),
        "cdf97" => Ok(WaveletType::Cdf97),
        "haar" => Ok(WaveletType::Haar),
        _ => Err(format!(
            "unknown wavelet '{s}'; expected cdf53, cdf97, or haar"
        )),
    }
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Encode {
            input,
            output,
            width,
            height,
            frames,
            quality,
            wavelet,
        } => cmd_encode(&input, &output, width, height, frames, quality, &wavelet),
        Commands::Decode { input, output } => cmd_decode(&input, &output),
        Commands::Info { input } => cmd_info(&input),
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        process::exit(1);
    }
}

fn cmd_encode(
    input: &str,
    output: &str,
    width: u32,
    height: u32,
    frames: u32,
    quality: u8,
    wavelet: &str,
) -> Result<(), String> {
    let wt = parse_wavelet(wavelet)?;
    let rgb_data = fs::read(input).map_err(|e| format!("read {input}: {e}"))?;

    let encoder = FrameEncoder::with_wavelet(quality, wt);
    let chunk = encoder
        .encode(&rgb_data, width, height, frames)
        .map_err(|e| e.to_string())?;

    let bytes = chunk.to_bytes();
    fs::write(output, &bytes).map_err(|e| format!("write {output}: {e}"))?;

    let ratio = if rgb_data.is_empty() {
        0.0
    } else {
        bytes.len() as f64 / rgb_data.len() as f64
    };

    eprintln!(
        "encoded {}x{}x{} ({} bytes) -> {} bytes ({:.1}% ratio, quality={}, wavelet={wavelet})",
        width,
        height,
        frames,
        rgb_data.len(),
        bytes.len(),
        ratio * 100.0,
        quality,
    );

    Ok(())
}

fn cmd_decode(input: &str, output: &str) -> Result<(), String> {
    let data = fs::read(input).map_err(|e| format!("read {input}: {e}"))?;
    let chunk = EncodedChunk::from_bytes(&data).map_err(|e| e.to_string())?;

    let decoder = FrameDecoder::new();
    let rgb = decoder.decode(&chunk).map_err(|e| e.to_string())?;

    fs::write(output, &rgb).map_err(|e| format!("write {output}: {e}"))?;

    eprintln!(
        "decoded {}x{}x{} -> {} bytes (raw RGB)",
        chunk.width,
        chunk.height,
        chunk.frames,
        rgb.len(),
    );

    Ok(())
}

fn cmd_info(input: &str) -> Result<(), String> {
    let data = fs::read(input).map_err(|e| format!("read {input}: {e}"))?;
    let chunk = EncodedChunk::from_bytes(&data).map_err(|e| e.to_string())?;

    let wavelet = match chunk.wavelet_type {
        WaveletType::Cdf53 => "CDF 5/3",
        WaveletType::Cdf97 => "CDF 9/7",
        WaveletType::Haar => "Haar",
    };

    let raw_size = (chunk.width as u64) * (chunk.height as u64) * (chunk.frames as u64) * 3;
    let ratio = if raw_size == 0 {
        0.0
    } else {
        chunk.compressed_size() as f64 / raw_size as f64
    };

    println!("ALICE-Codec Bitstream Info");
    println!("  File:        {input}");
    println!("  File size:   {} bytes", data.len());
    println!("  Width:       {}", chunk.width);
    println!("  Height:      {}", chunk.height);
    println!("  Frames:      {}", chunk.frames);
    println!("  Wavelet:     {wavelet}");
    println!("  Payload:     {} bytes", chunk.compressed_size());
    println!("  Raw size:    {raw_size} bytes (uncompressed RGB)");
    println!("  Ratio:       {:.1}%", ratio * 100.0);

    Ok(())
}
