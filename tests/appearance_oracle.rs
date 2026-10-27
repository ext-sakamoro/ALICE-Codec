//! Analytic oracles for `alice_codec::appearance` — the measures a render is compared with a reference by.
//!
//! Expected values come from published tables, closed forms or f64 references written in this file, never
//! from the crate function under test.
//!
//! Oracle sources:
//! - CIEDE2000: Sharma, Wu, Dalal, "The CIEDE2000 color-difference formula: implementation notes,
//!   supplementary test data, and mathematical observations", Color Res. Appl. 30 (2005), Table 1 (34 pairs,
//!   ΔE₀₀ to 4 decimals) and its symmetry
//! - sRGB decoding: IEC 61966-2-1 piecewise transfer (closed form evaluated here in f64)
//! - CIELAB: white → (100, 0, 0); greys → a = b = 0; sRGB primary red under D65 → (53.2408, 80.0925, 67.2032)
//! - radial power spectrum: a cosine of k cycles per image has all its power at radius k, total power
//!   = Parseval's sum of squares of the mean-removed image
//! - W1 (earth mover's distance on the line): a shift by c moves every quantile by c ⇒ W1 = c; two point
//!   masses at x and y ⇒ |x − y|; identical samples ⇒ 0
//! - normal angles: orthogonal unit vectors ⇒ π/2, opposite ⇒ π, same ⇒ 0
//! - dominant period: stripes of period P along x ⇒ P

use alice_codec::appearance::{
    delta_e2000, delta_e_stats, dominant_period, lab_from_linear_srgb, linear_from_srgb8,
    normal_angle_stats, radial_power_spectrum, wasserstein1,
};
use alice_codec::error::CodecError;

/// Sharma et al. 2005, Table 1: (L1, a1, b1, L2, a2, b2, ΔE00)
const SHARMA: [[f64; 7]; 34] = [
    [50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425],
    [50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615],
    [50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412],
    [
        50.0000, -1.3802, -84.2814, 50.0000, 0.0000, -82.7485, 1.0000,
    ],
    [
        50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000,
    ],
    [
        50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000,
    ],
    [50.0000, 0.0000, 0.0000, 50.0000, -1.0000, 2.0000, 2.3669],
    [50.0000, -1.0000, 2.0000, 50.0000, 0.0000, 0.0000, 2.3669],
    [50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0009, 7.1792],
    [50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0010, 7.1792],
    [50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0011, 7.2195],
    [50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0012, 7.2195],
    [50.0000, -0.0010, 2.4900, 50.0000, 0.0009, -2.4900, 4.8045],
    [50.0000, -0.0010, 2.4900, 50.0000, 0.0010, -2.4900, 4.8045],
    [50.0000, -0.0010, 2.4900, 50.0000, 0.0011, -2.4900, 4.7461],
    [50.0000, 2.5000, 0.0000, 50.0000, 0.0000, -2.5000, 4.3065],
    [50.0000, 2.5000, 0.0000, 73.0000, 25.0000, -18.0000, 27.1492],
    [50.0000, 2.5000, 0.0000, 61.0000, -5.0000, 29.0000, 22.8977],
    [50.0000, 2.5000, 0.0000, 56.0000, -27.0000, -3.0000, 31.9030],
    [50.0000, 2.5000, 0.0000, 58.0000, 24.0000, 15.0000, 19.4535],
    [50.0000, 2.5000, 0.0000, 50.0000, 3.1736, 0.5854, 1.0000],
    [50.0000, 2.5000, 0.0000, 50.0000, 3.2972, 0.0000, 1.0000],
    [50.0000, 2.5000, 0.0000, 50.0000, 1.8634, 0.5757, 1.0000],
    [50.0000, 2.5000, 0.0000, 50.0000, 3.2592, 0.3350, 1.0000],
    [
        60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644,
    ],
    [
        63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630,
    ],
    [61.2901, 3.7196, -5.3901, 61.4292, 2.2480, -4.9620, 1.8731],
    [35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645],
    [
        22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373,
    ],
    [36.4612, 47.8580, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146],
    [90.8027, -2.0831, 1.4410, 91.1528, -1.6435, 0.0447, 1.4441],
    [90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381],
    [6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377],
    [2.0776, 0.0795, -1.1350, 0.9033, -0.0636, -0.5514, 0.9082],
];

#[test]
fn ciede2000_matches_sharma_table_1() {
    // oracle: Sharma et al. 2005 Table 1, printed to 4 decimals
    for (i, r) in SHARMA.iter().enumerate() {
        let got = delta_e2000([r[0], r[1], r[2]], [r[3], r[4], r[5]]);
        assert!(
            (got - r[6]).abs() < 5e-5,
            "pair {}: {got} vs {}",
            i + 1,
            r[6]
        );
        let back = delta_e2000([r[3], r[4], r[5]], [r[0], r[1], r[2]]);
        assert!(
            (back - got).abs() < 1e-12,
            "pair {}: not symmetric ({got} / {back})",
            i + 1
        );
    }
    assert_eq!(delta_e2000([50.0, 10.0, -10.0], [50.0, 10.0, -10.0]), 0.0);
}

/// IEC 61966-2-1 decoding written here in f64
fn srgb_ref(v: u8) -> f64 {
    let c = f64::from(v) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[test]
fn srgb_decoding_matches_the_standard() {
    for v in 0..=255u8 {
        let got = linear_from_srgb8(v);
        assert!(
            (got - srgb_ref(v)).abs() < 1e-12,
            "{v}: {got} vs {}",
            srgb_ref(v)
        );
    }
    assert_eq!(linear_from_srgb8(0), 0.0);
    assert!((linear_from_srgb8(255) - 1.0).abs() < 1e-15);
}

#[test]
fn lab_of_white_grey_and_red() {
    let w = lab_from_linear_srgb([1.0, 1.0, 1.0]);
    assert!(
        (w[0] - 100.0).abs() < 1e-3 && w[1].abs() < 1e-3 && w[2].abs() < 1e-3,
        "white {w:?}"
    );
    for g in [0.001, 0.05, 0.2, 0.5, 0.9] {
        let l = lab_from_linear_srgb([g, g, g]);
        assert!(l[1].abs() < 1e-3 && l[2].abs() < 1e-3, "grey {g}: {l:?}");
    }
    let r = lab_from_linear_srgb([1.0, 0.0, 0.0]);
    for (got, want) in r.iter().zip([53.2408, 80.0925, 67.2032]) {
        assert!((got - want).abs() < 2e-3, "red {r:?}");
    }
    assert_eq!(lab_from_linear_srgb([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
}

#[test]
fn delta_e_stats_over_an_image() {
    let a = vec![[0.2, 0.3, 0.4]; 10];
    let mut b = a.clone();
    let s = delta_e_stats(&a, &b).unwrap();
    assert_eq!((s.mean, s.p95, s.max, s.count), (0.0, 0.0, 0.0, 10));
    b[9] = [0.25, 0.3, 0.4];
    let one = delta_e2000(lab_from_linear_srgb(a[9]), lab_from_linear_srgb(b[9]));
    let s = delta_e_stats(&a, &b).unwrap();
    assert!(one > 0.5);
    assert!((s.max - one).abs() < 1e-12);
    assert!((s.mean - one / 10.0).abs() < 1e-12);
    // nearest-rank p95 of 10 values = the 10th smallest
    assert!((s.p95 - one).abs() < 1e-12);
}

#[test]
fn delta_e_stats_refuses_mismatch_and_empty() {
    assert!(matches!(
        delta_e_stats(&[[0.0; 3]; 2], &[[0.0; 3]; 3]),
        Err(CodecError::InvalidBufferSize {
            expected: 2,
            got: 3
        })
    ));
    assert!(delta_e_stats(&[], &[]).is_err());
    assert!(delta_e_stats(&[[f64::NAN, 0.0, 0.0]], &[[0.0; 3]]).is_err());
}

fn cosine_image(w: usize, h: usize, kx: usize, ky: usize) -> Vec<f64> {
    let mut v = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let t = 2.0
                * core::f64::consts::PI
                * ((kx * x) as f64 / w as f64 + (ky * y) as f64 / h as f64);
            v[y * w + x] = 0.5 + 0.25 * t.cos();
        }
    }
    v
}

#[test]
fn radial_spectrum_of_a_cosine_is_one_ring() {
    let (w, h) = (32, 32);
    for (kx, ky, ring) in [(4, 0, 4), (0, 7, 7), (3, 4, 5), (2, 3, 4)] {
        let img = cosine_image(w, h, kx, ky);
        let p = radial_power_spectrum(&img, w, h).unwrap();
        assert_eq!(p.len(), w.min(h) / 2 + 1);
        // Parseval: Σ|X|² / (w·h) = Σ (x − mean)²
        let mean = img.iter().sum::<f64>() / img.len() as f64;
        let energy: f64 = img.iter().map(|v| (v - mean) * (v - mean)).sum();
        let total: f64 = p.iter().sum();
        assert!(
            (total - energy).abs() < 1e-9 * energy,
            "({kx},{ky}): {total} vs {energy}"
        );
        for (r, v) in p.iter().enumerate() {
            if r == ring {
                assert!(
                    (v - energy).abs() < 1e-9 * energy,
                    "({kx},{ky}) ring {r}: {v}"
                );
            } else {
                assert!(
                    v.abs() < 1e-9 * energy,
                    "({kx},{ky}) ring {r} not empty: {v}"
                );
            }
        }
    }
    let flat = vec![0.3; w * h];
    assert!(radial_power_spectrum(&flat, w, h)
        .unwrap()
        .iter()
        .all(|v| v.abs() < 1e-20));
}

#[test]
fn radial_spectrum_refuses_bad_sizes() {
    assert!(radial_power_spectrum(&[0.0; 6], 2, 2).is_err());
    assert!(radial_power_spectrum(&[], 0, 0).is_err());
    assert!(radial_power_spectrum(&[0.0], 1, 1).is_err());
}

#[test]
fn wasserstein1_closed_forms() {
    let a: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.01).collect();
    let b: Vec<f64> = a.iter().map(|v| v + 0.07).collect();
    assert!((wasserstein1(&a, &b).unwrap() - 0.07).abs() < 1e-12);
    assert_eq!(wasserstein1(&a, &a).unwrap(), 0.0);
    // order of the samples does not matter
    let mut r = a.clone();
    r.reverse();
    assert_eq!(wasserstein1(&a, &r).unwrap(), 0.0);
    assert!((wasserstein1(&[0.2], &[0.9]).unwrap() - 0.7).abs() < 1e-15);
    // unequal counts: {0, 1} vs {0.5}: ∫|F_a − F_b| = 0.5·0.5 + 0.5·0.5 = 0.5
    assert!((wasserstein1(&[0.0, 1.0], &[0.5]).unwrap() - 0.5).abs() < 1e-15);
    assert!(wasserstein1(&[], &[1.0]).is_err());
    assert!(wasserstein1(&[f64::NAN], &[1.0]).is_err());
}

#[test]
fn normal_angles_closed_forms() {
    use core::f64::consts::{FRAC_PI_2, PI};
    let z = [0.0, 0.0, 1.0];
    let s = normal_angle_stats(&[z, z, z], &[[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], z]).unwrap();
    assert!((s.max - PI).abs() < 1e-9, "{s:?}");
    assert!((s.mean - (FRAC_PI_2 + PI) / 3.0).abs() < 1e-9, "{s:?}");
    // length does not matter
    let s = normal_angle_stats(&[[0.0, 0.0, 3.0]], &[[0.0, 2.0, 2.0]]).unwrap();
    assert!((s.mean - core::f64::consts::FRAC_PI_4).abs() < 1e-9);
    assert!(normal_angle_stats(&[[0.0; 3]], &[z]).is_err());
    assert!(normal_angle_stats(&[z], &[z, z]).is_err());
    assert!(normal_angle_stats(&[], &[]).is_err());
}

#[test]
fn dominant_period_of_stripes() {
    let (w, h) = (48, 8);
    for p in [2usize, 3, 4, 6, 8, 12, 16] {
        let mut img = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                img[y * w + x] = if x % p < p / 2 { 1.0 } else { 0.0 };
            }
        }
        let got = dominant_period(&img, w, h).unwrap();
        assert!((got - p as f64).abs() < 1e-12, "period {p}: {got}");
    }
    assert!(dominant_period(&vec![0.5; w * h], w, h).is_err());
    assert!(dominant_period(&[0.0; 3], 2, 2).is_err());
}
