# Changelog

All notable changes to ALICE-Codec will be documented in this file.

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
