# Contributing to ALICE-Codec

## Build

```bash
cargo build
cargo build --no-default-features   # no_std check
```

## Test

```bash
cargo test
```

## Lint

```bash
cargo clippy -- -W clippy::all
cargo fmt -- --check
cargo doc --no-deps 2>&1 | grep warning
```

## Design Constraints

- **no_std core**: wavelet, rANS, quantiser must compile without `std`. Use fixed-size arrays or `alloc`.
- **Integer lifting**: wavelet transforms use only `i32` arithmetic — no floating-point in the forward/inverse path.
- **Analytical RDO**: quantisation parameters are computed in one shot (no iterative search).
- **rANS normalisation**: encoder state must stay within `[2^23, 2^31)` for 32-bit rANS.
- **3D volume model**: video is treated as (x, y, t) — no I/P/B frame distinction.
- **Reciprocal constants**: pre-compute `1.0 / N` to avoid division in hot paths.
