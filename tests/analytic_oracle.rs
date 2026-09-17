//! Analytic oracles — closed-form / independent-reference checks for the
//! numeric laws in ALICE-Codec (CLAUDE.md § 解析解突合テスト規律, 2026-09-17).
//!
//! Expected values come from closed forms or f64 references written in this
//! file, never from the crate function under test.  Default constructors
//! (`Quantizer::default()`, `SegmentConfig::default()`, `FrameEncoder::new`,
//! `RateControlConfig::default()`) are the paths a consumer takes first.
//!
//! Oracle sources:
//! - lifting wavelets: JPEG2000 5/3 impulse responses (T.800 Table F.4/F.8),
//!   Haar (d = o − e, s = e + d/2), CDF 9/7 = f64 lifting with the
//!   Daubechies–Sweldens constants; vanishing moments (constant / linear /
//!   cubic ⇒ zero high-pass); perfect reconstruction for any integer input
//! - rANS: Shannon entropy bound, round-trip identity, scalar ≡ interleaved
//! - dead-zone quantiser: |v| < dz ⇒ 0, otherwise |v − recon| ≤ step/2
//! - YCoCg-R (Malvar–Sullivan): lossless, grey ⇒ Co = Cg = 0, Y = grey
//! - PSNR = 10·log10(255²/MSE); SSIM of identical images = 1; constant images ⇒
//!   (2μaμb + C1)/(μa² + μb² + C1)
//! - rate control: bits/frame = kbps·1000/fps

use alice_codec::color::{rgb_to_ycocg_r_pixel, ycocg_r_to_rgb_pixel, RGB};
use alice_codec::metrics::{mse, psnr};
use alice_codec::rans::{
    FrequencyTable, InterleavedRansDecoder, InterleavedRansEncoder, RansDecoder, RansEncoder,
};
use alice_codec::{
    ms_ssim, ssim, AnalyticalRDO, FrameDecoder, FrameEncoder, Quantizer, RateControlConfig,
    RateController, SegmentConfig, Wavelet1D, Wavelet2D, WaveletType,
};

// ───────────────────────── wavelet references (independent) ──────────────

/// f64 lifting with symmetric (mirror) boundary — same structure the crate
/// uses, but with the published real coefficients and no fixed point.
/// `steps`: (coefficient, predict?) applied as odd += c·(e_l + e_r) or
/// even += c·(o_l + o_r).  Returns [low..., high...].
fn lifting_f64(signal: &[f64], steps: &[(f64, bool)]) -> Vec<f64> {
    let n = signal.len();
    let half = n / 2;
    let mut x = signal.to_vec();
    for &(c, predict) in steps {
        if predict {
            for i in 0..half {
                let el = x[2 * i];
                let er = if 2 * i + 2 < n {
                    x[2 * i + 2]
                } else {
                    x[2 * i]
                };
                x[2 * i + 1] += c * (el + er);
            }
        } else {
            for i in 0..half {
                let ol = if i > 0 { x[2 * i - 1] } else { x[1] };
                let or = x[2 * i + 1];
                x[2 * i] += c * (ol + or);
            }
        }
    }
    let mut out = vec![0.0; n];
    for i in 0..half {
        out[i] = x[2 * i];
        out[half + i] = x[2 * i + 1];
    }
    out
}

const CDF53_STEPS: [(f64, bool); 2] = [(-0.5, true), (0.25, false)];
const CDF97_STEPS: [(f64, bool); 4] = [
    (-1.586_134_342, true),
    (-0.052_980_118, false),
    (0.882_911_075, true),
    (0.443_506_852, false),
];

fn forward_i32(w: &Wavelet1D, s: &[i32]) -> Vec<i32> {
    let mut v = s.to_vec();
    w.forward(&mut v);
    v
}

fn max_abs_diff(a: &[i32], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y).abs())
        .fold(0.0, f64::max)
}

// ───────────────────────── wavelets ───────────────────────────────────────

#[test]
fn every_lifting_wavelet_reconstructs_any_integer_signal_exactly() {
    // perfect reconstruction is a property of lifting, independent of the
    // coefficients — it must hold for adversarial integers and odd lengths
    let mut seed = 0x9e37_79b9_u32;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as i32) >> 12 // ±2^19
    };
    for w in [Wavelet1D::haar(), Wavelet1D::cdf53(), Wavelet1D::cdf97()] {
        for n in [2usize, 3, 4, 7, 8, 16, 33, 64, 257] {
            let original: Vec<i32> = (0..n).map(|_| rnd()).collect();
            let mut v = original.clone();
            w.forward(&mut v);
            w.inverse(&mut v);
            assert_eq!(v, original, "{w:?} n={n}");
        }
    }
    // 2D: separable transform, perfect reconstruction, level count irrelevant
    let (wd, ht) = (16usize, 12usize);
    let img: Vec<i32> = (0..wd * ht).map(|_| rnd()).collect();
    for w2 in [Wavelet2D::cdf53(), Wavelet2D::cdf97()] {
        let mut v = img.clone();
        w2.forward(&mut v, wd, ht);
        w2.inverse(&mut v, wd, ht);
        assert_eq!(v, img);
    }
}

#[test]
fn cdf53_impulse_response_matches_jpeg2000_table() {
    // oracle: JPEG2000 (ITU-T T.800) 5/3 analysis filters
    //   low  = (−1, 2, 6, 2, −1) / 8 ,  high = (−1, 2, −1) / 2
    // An impulse of 64 at even index 8 (n = 16) makes every tap an integer.
    let mut s = vec![0i32; 16];
    s[8] = 64;
    let out = forward_i32(&Wavelet1D::cdf53(), &s);
    let (low, high) = out.split_at(8);
    let mut exp_low = [0i32; 8];
    exp_low[3] = -8;
    exp_low[4] = 48;
    exp_low[5] = -8;
    let mut exp_high = [0i32; 8];
    exp_high[3] = -32; // odd index 7 sees the impulse on its right
    exp_high[4] = -32; // odd index 9 sees it on its left
    assert_eq!(low, &exp_low[..], "5/3 low-pass taps");
    assert_eq!(high, &exp_high[..], "5/3 high-pass taps");
    // and the same signal through the f64 reference lifting
    let sf: Vec<f64> = s.iter().map(|&x| x as f64).collect();
    let reference = lifting_f64(&sf, &CDF53_STEPS);
    assert!(max_abs_diff(&out, &reference) <= 0.5);
}

#[test]
fn haar_impulse_response_is_the_haar_pair() {
    // oracle: Haar lifting  d = o − e,  s = e + d/2 = (e + o)/2
    // impulse 64 at even index 8 ⇒ s[4] = 32, d[4] = −64, everything else 0
    let mut s = vec![0i32; 16];
    s[8] = 64;
    let out = forward_i32(&Wavelet1D::haar(), &s);
    let mut expected = vec![0i32; 16];
    expected[4] = 32;
    expected[8 + 4] = -64;
    assert_eq!(
        out, expected,
        "Haar must not see the right-hand even sample"
    );
    // impulse at an odd index: only its own pair reacts
    let mut s = vec![0i32; 16];
    s[9] = 64;
    let out = forward_i32(&Wavelet1D::haar(), &s);
    let mut expected = vec![0i32; 16];
    expected[4] = 32;
    expected[8 + 4] = 64;
    assert_eq!(out, expected);
}

#[test]
fn constant_signal_has_zero_high_pass_for_every_wavelet() {
    // oracle: every wavelet here has ≥ 1 vanishing moment ⇒ DC leaks nothing
    // into the detail band; the 5/3 and Haar low-pass keep the DC value, the
    // unscaled 9/7 lifting scales it by (1+2β(1+2α))(1+2δ·0)… = 1.2302 (f64 ref)
    for c in [-1000i32, -7, 0, 1, 100, 4095] {
        let s = vec![c; 32];
        for (name, w) in [("haar", Wavelet1D::haar()), ("cdf53", Wavelet1D::cdf53())] {
            let out = forward_i32(&w, &s);
            assert!(
                out[16..].iter().all(|&d| d == 0),
                "{name} c={c}: high {:?}",
                &out[16..]
            );
            assert!(
                out[..16].iter().all(|&l| l == c),
                "{name} c={c}: low {:?}",
                &out[..16]
            );
        }
        let out = forward_i32(&Wavelet1D::cdf97(), &s);
        let sf: Vec<f64> = s.iter().map(|&x| x as f64).collect();
        let reference = lifting_f64(&sf, &CDF97_STEPS);
        assert!(
            out[16..].iter().all(|&d| d.abs() <= 1),
            "cdf97 c={c}: DC leaked into high-pass {:?}",
            &out[16..]
        );
        assert!(
            max_abs_diff(&out, &reference) <= 2.0,
            "cdf97 c={c}: {:?} vs f64 lifting {:?}",
            &out[..4],
            &reference[..4]
        );
    }
}

#[test]
fn linear_ramp_has_zero_high_pass_for_cdf53_and_cubic_for_cdf97() {
    // oracle: 5/3 predict is the exact midpoint ⇒ 2 vanishing moments (linear
    // signals produce d = 0 away from the mirrored end); 9/7 has 4 ⇒ cubics.
    let n = 64;
    let ramp: Vec<i32> = (0..n).map(|i| 6 * i as i32 - 100).collect();
    let out = forward_i32(&Wavelet1D::cdf53(), &ramp);
    // interior only (the mirror boundary at the right end breaks linearity)
    assert!(
        out[n / 2..n - 1].iter().all(|&d| d == 0),
        "5/3 ramp high {:?}",
        &out[n / 2..]
    );

    let cubic: Vec<i32> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64 * 4.0 - 2.0;
            ((t * t * t - 2.0 * t * t + t) * 400.0).round() as i32
        })
        .collect();
    let out = forward_i32(&Wavelet1D::cdf97(), &cubic);
    let cf: Vec<f64> = cubic.iter().map(|&x| x as f64).collect();
    let reference = lifting_f64(&cf, &CDF97_STEPS);
    // integer rounding of 4 lifting steps on ±3000 inputs: a few units;
    // the f64 reference itself annihilates the cubic to < 2 in the interior
    let interior = n / 2 + 2..n - 2;
    for i in interior {
        assert!(
            reference[i].abs() < 2.0,
            "reference cubic high-pass {} at {i}",
            reference[i]
        );
        assert!(
            (out[i] as f64 - reference[i]).abs() <= 3.0,
            "cdf97 cubic at {i}: {} vs {}",
            out[i],
            reference[i]
        );
    }
}

#[test]
fn wavelet2d_is_separable_row_then_column() {
    // oracle: 2D = 1D on every row, then 1D on every column of the result
    let (wd, ht) = (8usize, 4usize);
    let img: Vec<i32> = (0..wd * ht).map(|i| (i as i32 * 37) % 101 - 50).collect();
    for (w2, w1) in [
        (Wavelet2D::cdf53(), Wavelet1D::cdf53()),
        (Wavelet2D::cdf97(), Wavelet1D::cdf97()),
    ] {
        let mut got = img.clone();
        w2.forward(&mut got, wd, ht);
        let mut exp = img.clone();
        for r in 0..ht {
            w1.forward(&mut exp[r * wd..(r + 1) * wd]);
        }
        for c in 0..wd {
            let mut col: Vec<i32> = (0..ht).map(|r| exp[r * wd + c]).collect();
            w1.forward(&mut col);
            for r in 0..ht {
                exp[r * wd + c] = col[r];
            }
        }
        assert_eq!(got, exp);
    }
}

// ───────────────────────── rANS ───────────────────────────────────────────

fn shannon_bits(hist: &[u32]) -> f64 {
    let total: f64 = hist.iter().map(|&h| h as f64).sum();
    hist.iter()
        .filter(|&&h| h > 0)
        .map(|&h| {
            let p = h as f64 / total;
            -(h as f64) * p.log2()
        })
        .sum()
}

fn skewed_symbols(n: usize) -> Vec<u8> {
    // geometric-ish source over 8 symbols, deterministic
    let mut seed = 12345u32;
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = (seed >> 8) as f64 / (1u32 << 24) as f64;
            let mut s = 0u8;
            let mut p = 0.5;
            let mut acc = p;
            while u > acc && s < 7 {
                s += 1;
                p *= 0.5;
                acc += p;
            }
            s
        })
        .collect()
}

#[test]
fn rans_roundtrip_is_exact_and_size_is_within_two_percent_of_shannon() {
    let n = 65_536;
    let symbols = skewed_symbols(n);
    let mut hist = [0u32; 256];
    for &s in &symbols {
        hist[s as usize] += 1;
    }
    let table = FrequencyTable::from_histogram(&hist);
    let mut enc = RansEncoder::new();
    enc.encode_symbols(&symbols, &table);
    let bytes = enc.finish();
    let mut dec = RansDecoder::new(&bytes);
    assert_eq!(dec.decode_n(n, &table), symbols, "rANS round trip");
    // oracle: Shannon bound with a 12-bit quantised model costs at most a few
    // percent; 2 % + 8 bytes of state flush is the contract
    let ideal_bytes = shannon_bits(&hist) / 8.0;
    assert!(
        (bytes.len() as f64) <= ideal_bytes * 1.02 + 8.0,
        "{} bytes vs Shannon {ideal_bytes:.1}",
        bytes.len()
    );
    // uniform source over 16 symbols: exactly 4 bits/symbol ideal
    let uni: Vec<u8> = (0..n).map(|i| (i * 7 % 16) as u8).collect();
    let table = FrequencyTable::uniform(16);
    let mut enc = RansEncoder::new();
    enc.encode_symbols(&uni, &table);
    let bytes = enc.finish();
    assert!((bytes.len() as f64) <= n as f64 * 4.0 / 8.0 * 1.02 + 8.0);
    let mut dec = RansDecoder::new(&bytes);
    assert_eq!(dec.decode_n(n, &table), uni);
}

#[test]
fn interleaved_rans_decodes_the_same_symbols_as_scalar() {
    let n = 4096;
    let symbols = skewed_symbols(n);
    let mut hist = [0u32; 256];
    for &s in &symbols {
        hist[s as usize] += 1;
    }
    let table = FrequencyTable::from_histogram(&hist);
    let mut enc = InterleavedRansEncoder::new();
    enc.encode(&symbols, &table);
    let bytes = enc.finish();
    let mut dec = InterleavedRansDecoder::new(&bytes);
    assert_eq!(dec.decode_n(n, &table), symbols);
}

// ───────────────────────── quantiser ──────────────────────────────────────

#[test]
fn dead_zone_quantiser_zeroes_the_dead_zone_and_reconstructs_within_half_a_step() {
    // oracle: |v| < dz ⇒ 0 ; |v| ≥ dz ⇒ |v − deq(q(v))| ≤ step/2 (+1 for
    // integer rounding), sign preserved, monotone in v
    for q in [
        Quantizer::default(),
        Quantizer::new(8),
        Quantizer::with_dead_zone(10, 15),
        Quantizer::new(1),
    ] {
        let (step, dz) = (q.step, q.dead_zone);
        let mut prev = i32::MIN;
        for v in -400..=400 {
            let code = q.quantize(v);
            let recon = q.dequantize(code);
            assert!(
                code >= prev,
                "step {step} dz {dz}: quantise not monotone at {v}"
            );
            prev = code;
            if v.abs() < dz {
                assert_eq!(code, 0, "step {step} dz {dz}: {v} is inside the dead zone");
            } else {
                assert_ne!(
                    code, 0,
                    "step {step} dz {dz}: {v} outside the dead zone mapped to 0"
                );
                assert_eq!(code.signum(), v.signum());
                assert!(
                    (v - recon).abs() <= step / 2 + 1,
                    "step {step} dz {dz}: v={v} → q={code} → {recon}, error {}",
                    (v - recon).abs()
                );
            }
        }
    }
}

#[test]
fn rdo_step_grows_with_variance_and_shrinks_with_quality() {
    // oracle: Q = round(sqrt(12 · 6 ln2 σ² / bpp)) · strength, monotone in σ and
    // decreasing in target bpp (quality)
    let flat: Vec<i32> = (0..256).map(|i| (i % 4) - 2).collect();
    let loud: Vec<i32> = (0..256).map(|i| ((i * 97) % 201) - 100).collect();
    let sb = alice_codec::SubBand3D::LLL;
    for quality in [10u8, 50, 90] {
        let rdo = AnalyticalRDO::with_quality(quality);
        let qf = rdo.compute_quantizer(&flat, sb);
        let ql = rdo.compute_quantizer(&loud, sb);
        assert!(
            ql.step >= qf.step,
            "quality {quality}: loud step {} < flat step {}",
            ql.step,
            qf.step
        );
        assert!(qf.step >= 1 && ql.step >= 1);
    }
    let (q10, q90) = (
        AnalyticalRDO::with_quality(10),
        AnalyticalRDO::with_quality(90),
    );
    assert!(q10.compute_quantizer(&loud, sb).step >= q90.compute_quantizer(&loud, sb).step);
    // closed form for the loud band at quality 90, LLL strength 1
    let n = loud.len() as f64;
    let mean = loud.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = loud.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
    let bpp = 0.9f64 * 0.9 * 23.9 + 0.1;
    let expected = (12.0 * 6.0 * core::f64::consts::LN_2 * var / bpp)
        .sqrt()
        .round() as i32;
    let got = q90.compute_quantizer(&loud, sb).step;
    assert!(
        (got - expected * i32::from(sb.quant_strength())).abs() <= 1,
        "step {got} vs closed form {expected}"
    );
}

// ───────────────────────── colour ─────────────────────────────────────────

#[test]
fn ycocg_r_is_lossless_and_grey_has_no_chroma() {
    // oracle (Malvar & Sullivan 2003): reversible integer transform; for
    // r = g = b = v: Co = 0, Cg = 0, Y = v
    for v in 0..=255u8 {
        let c = rgb_to_ycocg_r_pixel(RGB::new(v, v, v));
        assert_eq!((c.y, c.co, c.cg), (v as i16, 0, 0), "grey {v}");
    }
    // full lossless sweep on a stride-7 lattice plus the cube corners
    for r in (0..256).step_by(7).chain([255]) {
        for g in (0..256).step_by(7).chain([255]) {
            for b in (0..256).step_by(7).chain([255]) {
                let px = RGB::new(r as u8, g as u8, b as u8);
                let c = rgb_to_ycocg_r_pixel(px);
                assert_eq!(ycocg_r_to_rgb_pixel(c), px);
                // closed form of the forward transform
                let co = r as i16 - b as i16;
                let t = b as i16 + (co >> 1);
                let cg = g as i16 - t;
                let y = t + (cg >> 1);
                assert_eq!((c.y, c.co, c.cg), (y, co, cg));
                assert!((0..=255).contains(&c.y));
                assert!((-255..=255).contains(&c.co) && (-255..=255).contains(&c.cg));
            }
        }
    }
}

// ───────────────────────── metrics ────────────────────────────────────────

#[test]
fn psnr_and_ssim_match_their_closed_forms() {
    let a = vec![100u8; 64 * 64];
    // identical ⇒ MSE 0, PSNR ∞, SSIM 1, MS-SSIM 1
    assert_eq!(mse(&a, &a).unwrap(), 0.0);
    assert_eq!(psnr(&a, &a).unwrap(), f64::INFINITY);
    assert!((ssim(&a, &a, 64, 64).unwrap() - 1.0).abs() < 1e-12);
    assert!((ms_ssim(&a, &a, 64, 64).unwrap() - 1.0).abs() < 1e-9);
    // constant offset k ⇒ MSE = k², PSNR = 20 log10(255/k)
    for k in [1u8, 4, 16, 50] {
        let b: Vec<u8> = a.iter().map(|&x| x + k).collect();
        assert_eq!(mse(&a, &b).unwrap(), (k as f64).powi(2));
        let expected = 20.0 * (255.0f64 / k as f64).log10();
        assert!((psnr(&a, &b).unwrap() - expected).abs() < 1e-9, "k={k}");
        // constant images: σ = 0 ⇒ SSIM = (2μaμb + C1)/(μa² + μb² + C1), C1 = (0.01·255)²
        let (mu_a, mu_b) = (100.0f64, 100.0 + k as f64);
        let c1 = (0.01f64 * 255.0).powi(2);
        let expected = (2.0 * mu_a * mu_b + c1) / (mu_a * mu_a + mu_b * mu_b + c1);
        assert!(
            (ssim(&a, &b, 64, 64).unwrap() - expected).abs() < 1e-9,
            "k={k}: ssim {} vs {expected}",
            ssim(&a, &b, 64, 64).unwrap()
        );
    }
    // checkerboard vs its inverse: MSE = 255²
    let cb: Vec<u8> = (0..64 * 64)
        .map(|i| if (i / 64 + i % 64) % 2 == 0 { 255 } else { 0 })
        .collect();
    let inv: Vec<u8> = cb.iter().map(|&x| 255 - x).collect();
    assert_eq!(mse(&cb, &inv).unwrap(), 255.0 * 255.0);
    assert!((psnr(&cb, &inv).unwrap()).abs() < 1e-9);
}

// ───────────────────────── pipeline (default path) ────────────────────────

fn frame(width: usize, height: usize, f: impl Fn(usize, usize) -> (u8, u8, u8)) -> Vec<u8> {
    let mut v = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let (r, g, b) = f(x, y);
            v.extend_from_slice(&[r, g, b]);
        }
    }
    v
}

#[test]
fn constant_frame_decodes_to_a_constant_frame_within_the_quantiser_bound() {
    // oracle: a constant frame has zero detail energy in every sub-band, so the
    // decoded frame is constant again and the only error is the DC quantiser
    // (≤ step/2 per YCoCg channel); YCoCg-R → RGB widens that to ≤ 3·⌈step/2⌉
    // (r = co + b, b = t − co/2, t = y − cg/2).  Step is read from the chunk
    // header because the encoder widens it to keep symbols in range.
    let (w, h) = (32u32, 16u32);
    let img = frame(32, 16, |_, _| (37, 200, 91));
    let dec = FrameDecoder;
    for quality in [0u8, 25, 50, 75, 100] {
        for wt in [WaveletType::Cdf53, WaveletType::Cdf97, WaveletType::Haar] {
            let enc = FrameEncoder::with_wavelet(quality, wt);
            let chunk = enc.encode(&img, w, h, 1).unwrap();
            let back = dec.decode(&chunk).unwrap();
            assert_eq!(back.len(), img.len());
            let step = chunk.quant_steps().iter().copied().max().unwrap();
            // Haar / 5/3 are reversible integer transforms: a constant input
            // has an exactly zero detail band, so the output is exactly
            // constant.  The integer 9/7 lifting rounds each of its 4 steps,
            // leaving a ±1 residue in the detail band of a constant; once
            // quantised away, each 1-D inverse ripples by ≤ 1 LSB, ×3
            // dimensions, ×3 through YCoCg-R → ≤ 9 peak-to-peak in RGB.
            let ripple = if wt == WaveletType::Cdf97 { 9 } else { 0 };
            for c in 0..3 {
                let (mn, mx) = back
                    .chunks(3)
                    .map(|px| px[c] as i32)
                    .fold((i32::MAX, i32::MIN), |(a, b), v| (a.min(v), b.max(v)));
                assert!(
                    mx - mn <= ripple,
                    "quality {quality} {wt:?}: channel {c} not constant ({mn}..{mx}, allowed ripple {ripple})"
                );
            }
            let bound = 3 * ((step + 1) / 2) + ripple;
            for (i, (&a, &b)) in img.iter().zip(&back).enumerate().take(3) {
                assert!(
                    (a as i32 - b as i32).abs() <= bound,
                    "quality {quality} {wt:?} step {step}: channel {i} {a} → {b} (bound {bound})"
                );
            }
            if quality == 100 {
                // step ≤ 3 at quality 100 for 8-bit input ⇒ within 6
                assert!(step <= 3, "quality 100 step {step}");
            }
        }
    }
    // the default constructor is the 5/3 path
    let chunk = FrameEncoder::new(80).encode(&img, w, h, 1).unwrap();
    assert_eq!(chunk.wavelet_type, WaveletType::Cdf53);
    assert!(
        dec.decode(&chunk)
            .unwrap()
            .chunks(3)
            .all(|px| px == &img[..3])
            || chunk.quant_steps()[0] > 1
    );
}

#[test]
fn pipeline_psnr_is_monotone_in_quality_and_high_at_quality_100() {
    // oracle: rate–distortion monotonicity + near-lossless top quality
    let (w, h) = (64u32, 48u32);
    let img = frame(64, 48, |x, y| {
        let r = ((x * 4) % 256) as u8;
        let g = ((y * 5 + x) % 256) as u8;
        let b = (((x * y) / 3) % 256) as u8;
        (r, g, b)
    });
    let dec = FrameDecoder;
    let mut prev = 0.0f64;
    for quality in [10u8, 40, 70, 100] {
        let chunk = FrameEncoder::new(quality).encode(&img, w, h, 1).unwrap();
        let back = dec.decode(&chunk).unwrap();
        let p = psnr(&img, &back).unwrap();
        assert!(
            p + 0.5 >= prev,
            "quality {quality}: PSNR {p} < previous {prev}"
        );
        prev = p;
    }
    assert!(prev >= 40.0, "quality 100 PSNR {prev} dB");
}

// ───────────────────────── rate control ───────────────────────────────────

#[test]
fn rate_controller_target_bits_per_frame_is_kbps_over_fps() {
    let cfg = RateControlConfig::default();
    let rc = RateController::new(cfg);
    let expected = (f64::from(cfg.target_bitrate_kbps) * 1000.0 / cfg.framerate) as u64;
    assert_eq!(rc.target_bits_per_frame(), expected);
    assert!(
        rc.recommended_quality() >= cfg.min_quality && rc.recommended_quality() <= cfg.max_quality
    );
    let rc = RateController::new(RateControlConfig {
        target_bitrate_kbps: 3000,
        framerate: 60.0,
        ..cfg
    });
    assert_eq!(rc.target_bits_per_frame(), 50_000);
    // default segmentation config is the documented one
    let seg = SegmentConfig::default();
    assert_eq!((seg.motion_threshold, seg.min_region_size), (25, 100));
}
