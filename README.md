# ALICE-Codec

**Hyper-Fast 3D Wavelet Video/Audio Codec** - *Optimized Edition*

> "Time is just another spatial dimension."

A radical video/audio codec that eliminates I/P/B frames entirely. Instead, it treats video as a 3D volume $(x, y, t)$ and compresses it using **3D Integer Wavelet Transform** + **rANS entropy coding**.

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

## Core Algorithms

### 1. 3D Integer CDF 9/7 Wavelet Transform

- **Lifting Scheme**: No floating-point. Perfect reconstruction guaranteed.
- **Spatial (x, y)**: Same as JPEG2000
- **Temporal (t)**: Motion becomes high-frequency sub-bands
- **Complexity**: O(N) vs O(N²) block matching

```
        LLL ← Static background (high compression)
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

## SIMD Optimizations

> "Division is just multiplication by a magic number."

### SIMD-Interleaved rANS

4つのrANSストリームをAVX2で同時処理:

```rust
// 4状態を__m128iにパック
let state_vec = _mm_loadu_si128(states.as_ptr());

// freq * (x >> PROB_BITS) を4並列で計算
let term1 = _mm_mullo_epi32(freq_vec, x_shifted);
let term2 = _mm_add_epi32(term1, slots_vec);
let new_state = _mm_sub_epi32(term2, cum_freq_vec);
```

- テーブルルックアップ: スカラ（gatherは遅い）
- 状態更新計算: SIMD（4並列）
- 再正規化: スカラ（可変バイト消費）

### Magic Number Division

`idiv` 命令（~30 cycles）を乗算+シフト（~3 cycles）に置換:

```rust
pub struct FastQuantizer {
    reciprocal: u64,  // 1/step の固定小数点表現
    shift: u32,       // 精度ビット
    step: i32,
}

// x / step → (x * reciprocal) >> shift
fn fast_div(&self, x: u32) -> u32 {
    ((x as u64 * self.reciprocal) >> self.shift) as u32
}
```

## Features

- **Blockless**: No 8×8/16×16 macroblocks. No blocking artifacts.
- **Scalable**: Decode at 1/2, 1/4 resolution by ignoring high-frequency sub-bands.
- **Lossless/Lossy**: Same code path. Just change quantization.
- **no_std**: Embeddable. No heap allocation in hot path.
- **SIMD**: AVX2/NEON optimized lifting, rANS, and quantization.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 ALICE-Codec (Optimized Edition)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Encoder Pipeline                       │  │
│  │                                                           │  │
│  │  RGB → YCoCg-R → 3D Wavelet → FastQuantize → rANS →     │  │
│  │                                              Bitstream    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Decoder Pipeline                       │  │
│  │                                                           │  │
│  │  Bitstream → SimdRansDecoder → Dequantize →              │  │
│  │              Inverse 3D Wavelet → YCoCg-R → RGB          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐ │
│  │ wavelet.rs │  │  color.rs  │  │  quant.rs  │  │ rans.rs  │ │
│  │            │  │            │  │            │  │          │ │
│  │ 1D/2D/3D   │  │ YCoCg-R    │  │ Dead-zone  │  │ rANS     │ │
│  │ Integer    │  │ Reversible │  │ Analytical │  │ 4-stream │ │
│  │ Lifting    │  │ Transform  │  │ RDO        │  │ SIMD     │ │
│  │ CDF 9/7    │  │ AVX2 SIMD  │  │ FastQuant  │  │ AVX2     │ │
│  └────────────┘  └────────────┘  └────────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Modules

| Module | File | Description |
|--------|------|-------------|
| **Wavelet** | `wavelet.rs` | 1D/2D/3D Integer Lifting (Haar, CDF 5/3, CDF 9/7) |
| **Color** | `color.rs` | YCoCg-R reversible color transform + AVX2 SIMD |
| **Quantization** | `quant.rs` | Dead-zone quantizer, FastQuantizer (magic division), Analytical RDO |
| **Entropy** | `rans.rs` | 32-bit rANS, Interleaved 4-stream, SimdRansDecoder |
| **Segmentation** | `segment.rs` | Person segmentation (motion/chroma-key), separable morphology, RLE mask |
| **Python** | `python.rs` | PyO3 + NumPy zero-copy bindings (GIL release) |

## Quick Start

```rust
use alice_codec::{
    Wavelet3D,
    color::{rgb_to_ycocg_r, ycocg_r_to_rgb, RGB},
    quant::{FastQuantizer, AnalyticalRDO, to_symbols, from_symbols},
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
let table = FrequencyTable::from_histogram(&quant::build_histogram(&symbols));
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

## Person Segmentation (Hybrid Streaming)

ALICE-Codec includes a high-performance person segmentation module for **ALICE Hybrid Streaming** — where SDF-rendered backgrounds replace pixel data, and only the person region is wavelet-encoded.

### Optimizations (カリカリ)

| Technique | Complexity | Effect |
|-----------|-----------|--------|
| Branchless frame diff (`saturating_sub \| saturating_sub`) | O(n) | Auto-vectorizes to VPSUBUSB + VPOR (~32 px/cycle) |
| Separable morphological dilation | O(n) vs O(n×r²) | Distance-to-nearest forward+backward scan |
| Erosion via complement identity | O(n) | `erode(m) = NOT(dilate(NOT(m)))` |
| Row-scan bounding box | O(n) | `position()`/`rposition()` per row, cache-friendly |
| Scan-forward RLE encoding | O(n) | Bulk transition detection, auto-vectorizable |

### Python Bindings (NumPy Zero-Copy + GIL Release)

```python
import alice_codec
import numpy as np

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

## Performance Targets

| Metric | Target | Optimization |
|--------|--------|--------------|
| Encode Speed | 100+ fps (1080p) | FastQuantizer (magic division) |
| Decode Speed | 500+ fps | SimdRansDecoder (4-stream AVX2) |
| Compression | Better than H.264 | Analytical RDO |
| Latency | 64 frames (tunable) | Chunk-based processing |

## Trade-offs

This codec prioritizes **throughput** and **edit-friendliness** over **low latency**.

- ✅ Excellent for: Archival, editing, streaming (with buffer)
- ⚠️ Not ideal for: Real-time video conferencing (64-frame latency)

## Building

```bash
# Standard build
cargo build --release

# With SIMD optimizations
cargo build --release --features simd

# With Python bindings (requires Python + maturin)
pip install maturin
maturin develop --release --features python

# Run tests (42 tests)
cargo test

# Run benchmarks
cargo bench
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
| 3D video chunks (64-frame GoP) | ALICE-Codec standalone (`Wavelet3D`) |
| Single-frame in ASP transport | ASP `media-stack` feature (`Wavelet2D`) |
| Person region in hybrid streaming | ASP hybrid + ALICE-Codec wavelet |

## References

- [JPEG2000 Part 1](https://www.iso.org/standard/78321.html) - CDF 9/7 Wavelet
- [rANS](https://arxiv.org/abs/1311.2540) - Asymmetric Numeral Systems
- [YCoCg-R](https://www.microsoft.com/en-us/research/publication/ycocg-r-a-color-space-with-rgb-reversibility-and-low-dynamic-range/) - Reversible Color Space
- [Magic Number Division](https://ridiculousfish.com/blog/posts/labor-of-division-episode-i.html) - Division by Invariant Integers

## License

AGPL-3.0

## Author

Moroya Sakamoto ([@ext-sakamoro](https://github.com/ext-sakamoro))
