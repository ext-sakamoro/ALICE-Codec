# Changelog

All notable changes to ALICE-Codec will be documented in this file.

## [Unreleased]

## [0.1.3] - 2026-09-17

### Changed
- README / lib.rs の全称 claim を実態に限定し、各 claim 行に `<!-- claim-test: fn -->` で検証 test を紐付け (strict-eval 検査 1、2026-09-17)

### Added
- `tests/analytic_oracle.rs` — 閉形式 / 独立参照との突合 oracle 15 本 (CLAUDE.md § 解析解突合テスト規律、2026-09-17): lifting wavelet の完全再構成 (任意 i32 / 奇数長 / 2D) / JPEG2000 5/3 impulse response (T.800 タップ) / Haar pair / 定数・線形・3 次多項式の vanishing moment / f64 lifting 参照 / 2D 分離可能性、rANS の Shannon bound (≤ +2 %) + round-trip + interleaved ≡ scalar、dead-zone 量子化の bin 閉形式、YCoCg-R 可逆 + 灰 ⇒ Co = Cg = 0、PSNR / SSIM 閉形式、pipeline 既定 path (定数 frame ⇒ 定数 + DC bound、quality 単調、quality 100 ≥ 40 dB)、rate control 閉形式
- `EncodedChunk::quant_steps()` — channel 毎に実際に使った量子化 step (encoder が symbol 範囲のため広げた値)
- `CodecError::SymbolOverflow` — `to_symbols` が `|q| > 127` で fail fast (以前は u8 に silent wrap)

### Fixed (oracle 先行 red 8 → 修正、いずれも既存 test 193 本は green のまま壊れていた)
- **wavelet lifting の係数が全て半分** (`coeff · (l + r) / 2¹³`): `cdf97()` は定数 1000 の detail band に −324 が漏れ (vanishing moment 0)、`cdf53()` は update が JPEG2000 の半分、`haar()` は実際には 5/3 filter だった → Q12 係数を近傍和に掛ける法則に統一、5/3 は T.800 と bit-exact (low (−1 2 6 2 −1)/8、high (−1 2 −1)/2)、Haar は pair stencil
- **wavelet 逆変換が `−coeff` で丸め直し** → rounding tie で完全再構成が崩れる (n = 2 で ±1) → forward と同じ delta を減算
- **奇数長 signal の末尾 sample が de/interleave で消失** (0 に置換) → low band 末尾に保持
- **rANS `FrequencyTable::from_histogram`**: 不在 symbol に 1 slot ずつ + 残差を symbol 255 に押し付け → 偏った histogram (wavelet 係数の典型 = 0 が 87 %) で総和が 4096 を超え、symbol 255 の頻度が負に wrap、実 symbol の累積範囲が 4096 を跨いで **round-trip 不能** → 不在 symbol は 0、残差は最頻 symbol が吸収
- **量子化 symbol の u8 wrap**: quality 100 (step 1) で係数 > 127 が `to_symbols` で silent wrap → decode が **6.4 dB PSNR** (quality 70 の 20.9 dB より悪化) → encoder が channel 毎に `step ≥ max|c| / 127` に広げる (header に記録済)、quality 100 は 8-bit 入力で step ≤ 3 = near-lossless (実測 ≥ 40 dB)
- **dead-zone 量子化の再構成 bias**: `⌊(|v| − dz/2)/step⌋` + `q·step` は `[dz, dz/2 + step)` も 0 に落とし、非 0 係数を平均 1 step 分ゼロ側に寄せていた (誤差 ≤ step + dz/2) → bin `[dz + (|q|−1)·step, +step)` の中点再構成、誤差 ≤ step/2 `Quantizer` / `FastQuantizer` / AVX2 path 同法則、`dead_zone ≤ 0` は 1 扱い 既存 test の pin 値 (`dequantize(1) == 8` 等) は法則値に更新

### Changed

- The optional bridges (`alice-ml` / `alice-db` / `alice-crypto` / `alice-cache`)
  are plain crates.io dependencies (`path = "../ALICE-*"` removed); CI no
  longer builds manifest-only sibling stubs, so `cargo audit` / `cargo deny`
  see the real dependency tree. `alice-db` requirement `0.2.0-beta.2`.

### Fixed
- **FFI 17 関数の panic 隔離** (`src/ffi.rs`): 全 `extern "C"` (`const fn` の field 読み出し 3 本を除く) の本体を `ffi_guard(sentinel, || ..)` で包み、panic は host を落とさず sentinel (null / −1.0 / ()) + `alice_codec_last_error()` (新規、`alice_codec_clear_last_error` / `alice_codec_free_error_string` も) で通知 `[profile.release] panic = "abort"` を撤去 (abort では `catch_unwind` が機能しない) release profile で guard test 通過
- **`no_std` build が一度も通っていなかった** (`--no-default-features` で 22 error: `container` の `std::collections::HashMap` / `String` / `Vec`、`ssim` / `quant` / `rate_control` の f64 `mul_add` / `ln` / `exp`) — `src/math.rs` の `FloatExt` trait (`libm` 委譲、`std` 時は不使用) で修正、`container` module は `HashMap` metadata のため `std` 専用に gate (no_std では従来も compile 不能だったので API 影響なし) host rlib / `simd` / bare-metal `thumbv7em-none-eabihf` / `x86_64` cross で build を確認 (`crate-type` に cdylib を含むため `cargo check` では panic_handler / allocator 要求で落ちる、検証は `cargo rustc --crate-type rlib`)
- x86_64 + `simd` の未使用 import 2 件 (`quant::simd` の `use super::*`、`rans` の空 `pub use simd::*`) と `ffi` test の `vec!` 2 件 (CI に clippy が無く未検出)

### Changed
- CI: それまで fmt + actionlint のみだったところに test (default + `std,cli,simd,ffi`) / clippy `--all-targets -D warnings` 2 variant / `no_std` job (rlib + thumbv7em + clippy-driver wrapper) / `msrv` job (`rust-version = 1.87` を `cargo +1.87 check` で実 compile) / `feature-powerset` (cargo-hack、std 固定 depth 2) / doc `-D warnings` を追加、rust-cache

## [0.1.2] - 2026-03-05

### Changed
- `FastQuantizer::new` now returns `CodecError::InvalidQuantStep` instead of misusing `InvalidDimensions`
- Added `// SAFETY:` inline comments to all FFI unsafe blocks
- Updated quality metrics: 155 tests (140 unit + 15 doc-test)
- `const fn` promotion: Quantizer, FastQuantizer, AnalyticalRDO, FrequencyTable, RansEncoder, Wavelet2D/3D, FFI getters
- `mul_add` for numerically stable floating-point in RDO quality mapping
- `Self::` for all `SubBand3D` match arms (clippy `use_self`)
- `map_or` for FFI match→Result patterns (clippy `option_if_let_else`)
- proptest property-based tests: quant (3), wavelet (2), color (2)

### Fixed
- `Cargo.toml` version synced to 0.1.1 (was behind CHANGELOG)

## [0.1.1] - 2026-03-04

### Added
- `ffi` — 20 `extern "C"` FFI functions (Wavelet1D, FrameEncoder, EncodedChunk, Metrics)
- Unity C# bindings (`bindings/unity/AliceCodec.cs`) — 20 DllImport + RAII classes
- UE5 C++ header (`bindings/ue5/AliceCodec.h`) — 20 extern C + RAII wrappers

## [0.1.0] - 2026-02-23

### Added
- `Wavelet1D` / `Wavelet3D` — integer CDF 9/7 lifting (forward / inverse)
- `RansEncoder` / `RansDecoder` — rANS entropy coder with frequency tables
- `YCoCgR` colour-space conversion (RGB ↔ YCoCg-R, lossless integer lifting)
- `QuantEngine` — analytical RDO quantizer with lambda-rate model
- `SubBand3D` enum — named sub-band addressing for 3D wavelet decomposition
- `Segment` / `SegmentHeader` — bitstream framing with magic-number validation
- `MetricsCollector` — encoding statistics (PSNR, bits/voxel, timing)
- `Pipeline` — encode / decode orchestration tying all stages together
- Bridge modules: `ml_bridge`, `db_bridge`, `crypto_bridge`, `cache_bridge`
- `cli` feature — `alice-codec` command-line binary (clap)
- `python` feature — PyO3 + NumPy bindings
- `no_std` compatible core (with `alloc`)
- 121 tests (114 unit + 7 doc-test)
