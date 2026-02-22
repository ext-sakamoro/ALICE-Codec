# ALICE-Codec

**Hyper-Fast 3D Wavelet Video Codec**

> "Time is just another spatial dimension."

A radical video codec that eliminates I/P/B frames entirely. Instead, it treats video as a 3D volume $(x, y, t)$ and compresses it using **3D Integer Wavelet Transform** + **rANS entropy coding**.

Author: Moroya Sakamoto

## The Revolution

Traditional codecs (H.264, AV1, H.266):
```
┌─────────────────────────────────────────────────────┐
│  I-Frame → P-Frame → P-Frame → B-Frame → ...       │
│     ↓         ↓         ↓         ↓                │
│  [Motion Estimation] × N² block matching = SLOW    │
└─────────────────────────────────────────────────────┘
```

ALICE-Codec:
```
┌─────────────────────────────────────────────────────┐
│  64-Frame Chunk                                     │
│     ↓                                              │
│  3D Wavelet (x, y, t) - Movement = High Frequency  │
│     ↓                                              │
│  rANS Entropy Coding - Near Shannon Limit          │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### CLI

```bash
# Build CLI
cargo build --release --features cli

# Encode raw RGB frames → .alc bitstream
alice-codec encode input.raw -W 1920 -H 1080 -f 64 -q 90 -w cdf97 -o output.alc

# Show bitstream metadata
alice-codec info output.alc

# Decode .alc → raw RGB
alice-codec decode output.alc -o recovered.raw
```

### End-to-End Pipeline (Recommended)

```rust
use alice_codec::{FrameEncoder, FrameDecoder, WaveletType};

// Encode: RGB frames → compressed bytes
let encoder = FrameEncoder::with_wavelet(80, WaveletType::Cdf97);
let chunk = encoder.encode(&rgb_bytes, width, height, frames)?;

println!("Compressed: {} bytes", chunk.compressed_size());

// Serialize to file
let bytes = chunk.to_bytes();
std::fs::write("output.alc", &bytes)?;

// Deserialize from file
let loaded = EncodedChunk::from_bytes(&std::fs::read("output.alc")?)?;

// Decode: compressed bytes → RGB frames
let decoder = FrameDecoder::new();
let recovered = decoder.decode(&loaded)?;
```

Each color channel (Y, Co, Cg) is processed independently through the full pipeline:

```
RGB → YCoCg-R → 3D Wavelet (CDF 5/3 | CDF 9/7 | Haar) → Quantize → rANS → bytes
```

### Wavelet Selection

| Wavelet | Quality | Speed | Use Case |
|---------|---------|-------|----------|
| CDF 5/3 (default) | Good | Fastest | Real-time, streaming |
| CDF 9/7 | Best | Fast | Archival, broadcast |
| Haar | Moderate | Fastest | Edge detection, debug |

### Serialization Format (.alc)

```
┌──────────┬─────────┬─────────┬───────────────┬─────────────┐
│ "ALCC"   │ Version │ Wavelet │ W × H × F     │ 3× Channel  │
│ 4 bytes  │ 1 byte  │ 1 byte  │ 3×u32 (12 B)  │ Headers     │
├──────────┴─────────┴─────────┴───────────────┼─────────────┤
│                                               │ Compressed  │
│                                               │ Payload     │
└───────────────────────────────────────────────┴─────────────┘
```

### Error Handling

All encode/decode operations return `Result<T, CodecError>`:

```rust
use alice_codec::CodecError;

match encoder.encode(&data, w, h, f) {
    Ok(chunk) => { /* success */ }
    Err(CodecError::InvalidBufferSize { expected, got }) => { /* wrong size */ }
    Err(CodecError::InvalidDimensions { width, height }) => { /* zero dim */ }
    Err(CodecError::DimensionOverflow) => { /* w*h*f overflow */ }
    Err(CodecError::InvalidBitstream(msg)) => { /* corrupt data */ }
}
```

### Advanced Usage (Manual Pipeline)

For fine-grained control over each stage:

```rust
use alice_codec::{
    Wavelet3D,
    color::{rgb_to_ycocg_r, ycocg_r_to_rgb, RGB},
    quant::{FastQuantizer, AnalyticalRDO, to_symbols, from_symbols, build_histogram},
    rans::{RansEncoder, RansDecoder, FrequencyTable},
    SubBand3D,
};

// === Encode ===

// 1. Color transform
let mut y = vec![0i16; pixels.len()];
let mut co = vec![0i16; pixels.len()];
let mut cg = vec![0i16; pixels.len()];
rgb_to_ycocg_r(&rgb_pixels, &mut y, &mut co, &mut cg);

// 2. 3D Wavelet transform
let wavelet = Wavelet3D::cdf97();
let mut volume: Vec<i32> = y.iter().map(|&x| x as i32).collect();
wavelet.forward(&mut volume, width, height, depth);

// 3. Quantize (with magic number division)
let rdo = AnalyticalRDO::with_quality(75);
let quantizer = rdo.compute_quantizer(&volume, SubBand3D::LLL);
let fast_q: FastQuantizer = quantizer.into();
let mut quantized = vec![0i32; volume.len()];
fast_q.quantize_buffer(&volume, &mut quantized);

// 4. Entropy encode
let mut symbols = vec![0u8; quantized.len()];
to_symbols(&quantized, &mut symbols);
let table = FrequencyTable::from_histogram(&build_histogram(&symbols));
let mut encoder = RansEncoder::new();
encoder.encode_symbols(&symbols, &table);
let bitstream = encoder.finish();

// === Decode ===

// 1. Entropy decode
let mut decoder = RansDecoder::new(&bitstream);
let decoded_symbols = decoder.decode_n(symbols.len(), &table);

// 2. Dequantize
let mut dequantized = vec![0i32; quantized.len()];
from_symbols(&decoded_symbols, &mut dequantized);
fast_q.dequantize_buffer(&dequantized, &mut volume);

// 3. Inverse wavelet
wavelet.inverse(&mut volume, width, height, depth);

// 4. Inverse color transform
let y_out: Vec<i16> = volume.iter().map(|&x| x as i16).collect();
let mut rgb_out = vec![RGB::new(0, 0, 0); pixels.len()];
ycocg_r_to_rgb(&y_out, &co, &cg, &mut rgb_out);
```

## Core Algorithms

### 1. 3D Integer CDF 9/7 Wavelet Transform

- **Lifting Scheme**: No floating-point. Perfect reconstruction guaranteed.
- **Spatial (x, y)**: Same as JPEG2000
- **Temporal (t)**: Motion becomes high-frequency sub-bands
- **Complexity**: O(N) vs O(N²) block matching

```
        LLL  ← Static background (high compression)
       /
      L ── LLH ← Slow movement
     / \
    /   LHL ← Horizontal edges moving
   /     \
Volume    LHH ← Fast diagonal motion
   \
    \   HLL ← Temporal low, spatial high
     \ /
      H ── HLH, HHL, HHH ← High frequency noise
```

### 2. Analytical Rate-Distortion Optimization

Traditional: "Try all quantizers, count bits" → O(N × Q)

ALICE-Codec: Closed-form solution using Laplacian distribution model:
```
λ_optimal = (6 × ln(2) × σ²) / R_target
```

One-shot. No iteration.

### 3. rANS Entropy Coding

- **Asymmetric Numeral Systems**: Better than Huffman, faster than Arithmetic
- **Interleaved 4-Stream**: SIMD parallel decoding
- **Memory-bound**: Decodes at RAM bandwidth speed

### 4. YCoCg-R Color Space

```rust
// RGB → YCoCg-R (Reversible, integer-only)
Co = R - B;
t  = B + (Co >> 1);
Cg = G - t;
Y  = t + (Cg >> 1);
```

Better decorrelation than YCbCr. Lossless round-trip.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         ALICE-Codec                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                 pipeline.rs (E2E API)                       │  │
│  │                                                             │  │
│  │  FrameEncoder::encode(rgb, w, h, f) → Result<EncodedChunk>  │  │
│  │  FrameDecoder::decode(chunk)        → Result<Vec<u8>>      │  │
│  │  EncodedChunk::to_bytes() / from_bytes()  (serialization)  │  │
│  └────────────────────────────────────────────────────────────┘  │
│       │             │              │             │                │
│  ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐  ┌───▼──────┐        │
│  │color.rs  │  │wavelet.rs│  │ quant.rs │  │ rans.rs  │        │
│  │          │  │          │  │          │  │          │        │
│  │ YCoCg-R  │  │ 1D/2D/3D │  │ Dead-zone│  │ rANS     │        │
│  │ Revers.  │  │ Integer  │  │ Analytic │  │ 4-stream │        │
│  │ AVX2     │  │ Lifting  │  │ RDO      │  │ SIMD     │        │
│  │          │  │ CDF 9/7  │  │ FastQ    │  │ AVX2     │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│                                                                  │
│  ┌────────────┐  ┌──────────────────────────────────────────┐   │
│  │segment.rs  │  │ Bridges (feature-gated)                   │   │
│  │            │  │                                           │   │
│  │ Motion seg │  │ ml_bridge  → ALICE-ML  (ternary inference)│   │
│  │ Chroma-key │  │ db_bridge  → ALICE-DB  (metrics storage)  │   │
│  │ Morphology │  │ crypto_br  → ALICE-Crypto (AEAD encrypt) │   │
│  │ RLE mask   │  │ cache_br   → ALICE-Cache (frame caching) │   │
│  └────────────┘  └──────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## Modules

| Module | File | Description |
|--------|------|-------------|
| **Pipeline** | `pipeline.rs` | End-to-end API with `Result` error handling, serialization, wavelet selection |
| **Wavelet** | `wavelet.rs` | 1D/2D/3D Integer Lifting (Haar, CDF 5/3, CDF 9/7) |
| **Color** | `color.rs` | YCoCg-R reversible color transform + AVX2 SIMD |
| **Quantization** | `quant.rs` | Dead-zone quantizer, `FastQuantizer` (magic division), Analytical RDO |
| **Entropy** | `rans.rs` | 32-bit rANS, Interleaved 4-stream, `SimdRansDecoder` |
| **Segmentation** | `segment.rs` | Person segmentation (motion/chroma-key), separable morphology, RLE mask |
| **Python** | `python.rs` | PyO3 + NumPy zero-copy bindings (GIL release) |

## SIMD Optimizations

### SIMD-Interleaved rANS

4 rANS streams processed in parallel using AVX2:

```rust
// 4 states packed into __m128i
let state_vec = _mm_loadu_si128(states.as_ptr());

// freq * (x >> PROB_BITS) computed 4-wide
let term1 = _mm_mullo_epi32(freq_vec, x_shifted);
let term2 = _mm_add_epi32(term1, slots_vec);
let new_state = _mm_sub_epi32(term2, cum_freq_vec);
```

- Table lookup: scalar (gather is slow)
- State update: SIMD (4-parallel)
- Renormalization: scalar (variable byte consumption)

### Magic Number Division

Replace `idiv` instruction (~30 cycles) with multiply + shift (~3 cycles):

```rust
pub struct FastQuantizer {
    reciprocal: u64,  // Fixed-point representation of 1/step
    shift: u32,       // Precision bits
    step: i32,
}

// x / step → (x * reciprocal) >> shift
fn fast_div(&self, x: u32) -> u32 {
    ((x as u64 * self.reciprocal) >> self.shift) as u32
}
```

## Features

- **Blockless**: No 8x8/16x16 macroblocks. No blocking artifacts.
- **Scalable**: Decode at 1/2, 1/4 resolution by ignoring high-frequency sub-bands.
- **Lossless/Lossy**: Same code path. Just change quantization.
- **`no_std`**: Embeddable. No heap allocation in hot path.
- **SIMD**: AVX2/NEON optimized lifting, rANS, and quantization.

## Person Segmentation (Hybrid Streaming)

ALICE-Codec includes a high-performance person segmentation module for **ALICE Hybrid Streaming** — where SDF-rendered backgrounds replace pixel data, and only the person region is wavelet-encoded.

### Optimization Techniques

| Technique | Complexity | Effect |
|-----------|-----------|--------|
| Branchless frame diff (`saturating_sub \| saturating_sub`) | O(n) | Auto-vectorizes to VPSUBUSB + VPOR (~32 px/cycle) |
| Separable morphological dilation | O(n) vs O(n*r^2) | Distance-to-nearest forward+backward scan |
| Erosion via complement identity | O(n) | `erode(m) = NOT(dilate(NOT(m)))` |
| Row-scan bounding box | O(n) | `position()`/`rposition()` per row, cache-friendly |
| Scan-forward RLE encoding | O(n) | Bulk transition detection, auto-vectorizable |

### Python Bindings (NumPy Zero-Copy + GIL Release)

```python
import alice_codec
import numpy as np

# --- Core Pipeline ---

# Encode
encoder = alice_codec.FrameEncoder(quality=80, wavelet="cdf97")
chunk = encoder.encode(rgb_bytes, width=1920, height=1080, frames=1)

# Serialize / deserialize
data = chunk.to_bytes()
loaded = alice_codec.EncodedChunk.from_bytes(data)

# Decode
decoder = alice_codec.FrameDecoder()
recovered = decoder.decode(loaded)

# Color conversion (NumPy zero-copy)
y, co, cg = alice_codec.rgb_to_ycocg_r_numpy(rgb_array)
rgb_out = alice_codec.ycocg_r_to_rgb_numpy(y, co, cg)

# --- Person Segmentation ---

# Motion-based person segmentation
current = np.array(frame, dtype=np.uint8).reshape(height, width)
reference = np.array(bg_frame, dtype=np.uint8).reshape(height, width)

mask, bbox, fg_count = alice_codec.segment_motion_numpy(
    current, reference,
    motion_threshold=25,
    dilate_radius=2,
    erode_radius=1,
)
# mask: (H, W) uint8 NumPy array, bbox: [x, y, w, h], fg_count: int

# Chroma-key segmentation (green screen)
mask, bbox, fg_count = alice_codec.segment_chroma_numpy(
    y_channel, co_channel, cg_channel,
    green_threshold=30,
)

# Crop/paste person region
person = alice_codec.crop_bbox_numpy(frame, bbox)
alice_codec.paste_bbox_numpy(output_frame, person_data, bbox)

# RLE compress mask
rle_bytes = alice_codec.rle_encode_numpy(mask)
```

**Python Binding Optimization Layers:**

| Layer | Technique | Effect |
|-------|-----------|--------|
| L1 | GIL Release (`py.allow_threads`) | Parallel computation |
| L2 | Zero-Copy NumPy (`as_slice`/`into_pyarray`) | No memcpy |
| L3 | Batch API (whole-frame ops) | FFI amortization |
| L4 | Rust backend (segment, wavelet, rANS) | Hardware-speed |

## Cross-Crate Bridges

ALICE-Codec connects to other ALICE ecosystem crates via feature-gated bridge modules:

| Bridge | Feature | Target Crate | Description |
|--------|---------|--------------|-------------|
| `ml_bridge` | `ml` | [ALICE-ML](../ALICE-ML) | Ternary neural inference for sub-band classification and motion estimation |
| `db_bridge` | `db` | [ALICE-DB](../ALICE-DB) | Time-series storage for bitrate, PSNR, and encode-time metrics |
| `crypto_bridge` | `crypto` | [ALICE-Crypto](../ALICE-Crypto) | XChaCha20-Poly1305 AEAD encryption for compressed bitstreams |
| `cache_bridge` | `cache` | [ALICE-Cache](../ALICE-Cache) | Decoded frame caching for instant scrubbing and seeking |

### Crypto Bridge (feature: `crypto`)

Wraps compressed bitstream data with authenticated encryption for secure storage or DRM delivery.

```rust
use alice_codec::crypto_bridge::{seal_bitstream, open_bitstream, derive_key, content_hash};

// Derive a key from passphrase
let key = derive_key("alice-codec-v1", b"my-secret");

// Encrypt compressed data
let sealed = seal_bitstream(&compressed_bytes, &key)?;

// Content-addressed deduplication (no decryption needed)
let hash = content_hash(&sealed.data);

// Decrypt
let plaintext = open_bitstream(&sealed, &key)?;
```

### ML Bridge (feature: `ml`)

Uses ALICE-ML ternary (1.58-bit) neural inference for adaptive quantization and motion estimation:

```rust
use alice_codec::ml_bridge::{SubBandClassifier, MotionPredictor};

// Sub-band classifier: predict optimal quantization strategy
let clf = SubBandClassifier::new(&weights, 3, 2);
let (class, confidence) = clf.classify(&[energy, variance, coherence]);

// Motion predictor: predict motion vectors from block features
let predictor = MotionPredictor::new(&weights, 4);
let (dx, dy) = predictor.predict(&block_features);
```

### DB Bridge (feature: `db`)

Records encoding metrics into ALICE-DB for monitoring dashboards:

```rust
use alice_codec::db_bridge::{CodecMetricsSink, CodecMetrics, FrameType};

let sink = CodecMetricsSink::open("/tmp/codec_metrics")?;
sink.record(&CodecMetrics {
    timestamp_ms: 1700000000000,
    bitrate_bps: 2_500_000.0,
    psnr_db: 38.5,
    encode_time_us: 1200.0,
    frame_type: FrameType::Intra,
})?;

// Dashboard queries
let avg_bitrate = sink.average_bitrate(start, end)?;
let downsampled = sink.downsample_psnr(start, end, 1000)?;
```

### Cache Bridge (feature: `cache`)

Caches decoded frames for instant replay without re-decoding:

```rust
use alice_codec::cache_bridge::FrameCache;

let cache = FrameCache::new(64); // 64 frames, ~384 MB for 1080p
cache.put(chunk_idx, frame_offset, quality, pixels, 1920, 1080);

if let Some(frame) = cache.get(chunk_idx, frame_offset, quality) {
    // Instant access, no decode needed
}
```

## ASP Integration (ALICE-Streaming-Protocol)

ALICE-Codec is integrated into [ALICE-Streaming-Protocol](https://github.com/ext-sakamoro/ALICE-Streaming-Protocol) as the video encoding backend via the `codec` feature flag.

```toml
# In ALICE-Streaming-Protocol's Cargo.toml
libasp = { version = "1.0", features = ["codec"] }
```

### How ASP Uses ALICE-Codec

ASP wraps ALICE-Codec's 2D wavelet pipeline for single-frame video encoding within the ASP transport layer:

```
ASP Video Encode:
  RGB frame → YCoCg-R (alice-codec::color)
            → 2D CDF 9/7 Wavelet (alice-codec::Wavelet2D)
            → Quantize (alice-codec::Quantizer)
            → coeffs_to_symbols (alice-codec::quant)
            → rANS entropy coding (alice-codec::rans)
            → ASP packet framing
```

- **Rayon parallel 3-channel**: Y/Co/Cg planes are wavelet-transformed and entropy-coded in parallel via `rayon::join`
- **Quality-mapped quantization**: ASP maps quality (0-100) to quantization step for 2D single-frame use
- **Python bindings**: `libasp.encode_video_frame()` / `libasp.decode_video_frame()` with GIL release

### Standalone vs ASP Usage

| Use Case | Recommended |
|----------|-------------|
| 3D video chunks (64-frame GoP) | ALICE-Codec standalone (`FrameEncoder`) |
| Single-frame in ASP transport | ASP `media-stack` feature (`Wavelet2D`) |
| Person region in hybrid streaming | ASP hybrid + ALICE-Codec wavelet |

## Performance Targets

| Metric | Target | Optimization |
|--------|--------|--------------|
| Encode Speed | 100+ fps (1080p) | `FastQuantizer` (magic division) |
| Decode Speed | 500+ fps | `SimdRansDecoder` (4-stream AVX2) |
| Compression | Better than H.264 | Analytical RDO |
| Latency | 64 frames (tunable) | Chunk-based processing |

## Trade-offs

This codec prioritizes **throughput** and **edit-friendliness** over **low latency**.

- Excellent for: Archival, editing, streaming (with buffer)
- Not ideal for: Real-time video conferencing (64-frame latency)

## Quality Metrics

| Metric | Value |
|--------|-------|
| Tests | 111 passing |
| Clippy (default) | 0 warnings |
| Clippy (pedantic) | 0 warnings |
| Format (`cargo fmt`) | 0 violations |
| Doc warnings | 0 |
| Error handling | All APIs return `Result` |
| Thread safety | `Send + Sync` compile-time verified |
| Overflow protection | `checked_mul` for dimensions |

## Building

```bash
# Standard build (library only)
cargo build --release

# With CLI binary
cargo build --release --features cli

# With SIMD optimizations
cargo build --release --features simd

# With Python bindings (requires Python + maturin)
pip install maturin
maturin develop --release --features python

# Run tests (111 tests)
cargo test

# Pedantic lint check
cargo clippy -- -W clippy::pedantic
```

## References

- [JPEG2000 Part 1](https://www.iso.org/standard/78321.html) - CDF 9/7 Wavelet
- [rANS](https://arxiv.org/abs/1311.2540) - Asymmetric Numeral Systems
- [YCoCg-R](https://www.microsoft.com/en-us/research/publication/ycocg-r-a-color-space-with-rgb-reversibility-and-low-dynamic-range/) - Reversible Color Space
- [Magic Number Division](https://ridiculousfish.com/blog/posts/labor-of-division-episode-i.html) - Division by Invariant Integers

## License

AGPL-3.0

## Author

Moroya Sakamoto ([@ext-sakamoro](https://github.com/ext-sakamoro))
