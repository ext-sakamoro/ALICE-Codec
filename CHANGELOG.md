# Changelog

All notable changes to ALICE-Codec will be documented in this file.

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
