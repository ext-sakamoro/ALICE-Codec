//! Appearance measures: what a render is compared with a reference by
//!
//! Colour difference (CIEDE2000 on CIELAB from linear sRGB, D65), the radial power spectrum of an image (how
//! its energy is spread over spatial frequency, independent of where the pattern sits), the 1-Wasserstein
//! distance between two samples of values (histograms compared without bins), the angle between normal
//! maps, and the dominant period of a pattern. Every transcendental goes through `alice-det-math`, so a
//! measure is the same bits on every platform; the inputs are linear (not sRGB-encoded) except for
//! [`linear_from_srgb8`], which decodes 8-bit sRGB.
//!
//! An input that cannot be measured (lengths that differ, nothing to compare, a non-finite value, a zero
//! normal, a flat image for a period) is an `Err`, never a number: a comparison that compared nothing must
//! not read as "no difference".

#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec, vec::Vec};

use alice_det_math::{acos64, atan2_64, cos64, exp64, powf64, sin64, sqrt64};

use crate::error::CodecError;

/// Mean, 95th percentile (nearest rank), maximum and count of a set of per-sample measures
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    /// Arithmetic mean
    pub mean: f64,
    /// 95th percentile, nearest rank: the ⌈0.95·n⌉-th smallest value
    pub p95: f64,
    /// Largest value
    pub max: f64,
    /// Number of samples measured
    pub count: usize,
}

fn stats(mut v: Vec<f64>) -> Stats {
    let count = v.len();
    let mean = v.iter().sum::<f64>() / count as f64;
    v.sort_unstable_by(f64::total_cmp);
    let rank = (count * 95).div_ceil(100).max(1);
    Stats {
        mean,
        p95: v[rank - 1],
        max: v[count - 1],
        count,
    }
}

fn invalid(msg: impl Into<String>) -> CodecError {
    CodecError::InvalidInput(msg.into())
}

fn same_len(a: usize, b: usize) -> Result<(), CodecError> {
    if a != b {
        return Err(CodecError::InvalidBufferSize {
            expected: a,
            got: b,
        });
    }
    if a == 0 {
        return Err(invalid("nothing to compare"));
    }
    Ok(())
}

fn finite3(v: &[[f64; 3]], what: &str) -> Result<(), CodecError> {
    match v.iter().position(|p| !p.iter().all(|c| c.is_finite())) {
        Some(i) => Err(invalid(format!("{what}[{i}] is not finite"))),
        None => Ok(()),
    }
}

/// Decode one 8-bit sRGB value to linear light (IEC 61966-2-1)
#[must_use]
pub fn linear_from_srgb8(v: u8) -> f64 {
    let c = f64::from(v) / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        powf64((c + 0.055) / 1.055, 2.4)
    }
}

/// linear sRGB → CIE XYZ (IEC 61966-2-1, D65)
const RGB_TO_XYZ: [[f64; 3]; 3] = [
    [0.412_456_4, 0.357_576_1, 0.180_437_5],
    [0.212_672_9, 0.715_152_2, 0.072_175_0],
    [0.019_333_9, 0.119_192_0, 0.950_304_1],
];

/// CIELAB of a linear sRGB colour, D65 white taken as the matrix's image of (1, 1, 1) so white is exactly
/// L = 100 and every grey has a = b = 0
#[must_use]
pub fn lab_from_linear_srgb(rgb: [f64; 3]) -> [f64; 3] {
    let xyz = |row: &[f64; 3], c: [f64; 3]| row[0] * c[0] + row[1] * c[1] + row[2] * c[2];
    let white = [1.0, 1.0, 1.0];
    let t = [
        xyz(&RGB_TO_XYZ[0], rgb) / xyz(&RGB_TO_XYZ[0], white),
        xyz(&RGB_TO_XYZ[1], rgb) / xyz(&RGB_TO_XYZ[1], white),
        xyz(&RGB_TO_XYZ[2], rgb) / xyz(&RGB_TO_XYZ[2], white),
    ];
    // CIE 15: ε = (6/29)³, κ = (29/3)³
    let eps = 216.0 / 24_389.0;
    let kappa = 24_389.0 / 27.0;
    let f = |t: f64| {
        if t > eps {
            powf64(t, 1.0 / 3.0)
        } else {
            (kappa * t + 16.0) / 116.0
        }
    };
    let (fx, fy, fz) = (f(t[0]), f(t[1]), f(t[2]));
    // below ε the linear branch of f makes 116·f − 16 = κ·t, so one expression serves both
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

fn deg(r: f64) -> f64 {
    r * (180.0 / core::f64::consts::PI)
}

fn rad(d: f64) -> f64 {
    d * (core::f64::consts::PI / 180.0)
}

/// CIEDE2000 colour difference of two CIELAB colours (k_L = k_C = k_H = 1), as written in Sharma, Wu and
/// Dalal (2005)
#[must_use]
pub fn delta_e2000(lab1: [f64; 3], lab2: [f64; 3]) -> f64 {
    let [l1, a1, b1] = lab1;
    let [l2, a2, b2] = lab2;
    let p7 = |x: f64| {
        let x2 = x * x;
        x2 * x2 * x2 * x
    };
    let c7_25 = p7(25.0);
    let cbar = (sqrt64(a1 * a1 + b1 * b1) + sqrt64(a2 * a2 + b2 * b2)) / 2.0;
    let g = 0.5 * (1.0 - sqrt64(p7(cbar) / (p7(cbar) + c7_25)));
    let (a1p, a2p) = ((1.0 + g) * a1, (1.0 + g) * a2);
    let (c1p, c2p) = (sqrt64(a1p * a1p + b1 * b1), sqrt64(a2p * a2p + b2 * b2));
    let hue = |b: f64, a: f64| {
        if a == 0.0 && b == 0.0 {
            0.0
        } else {
            let h = deg(atan2_64(b, a));
            if h < 0.0 {
                h + 360.0
            } else {
                h
            }
        }
    };
    let (h1p, h2p) = (hue(b1, a1p), hue(b2, a2p));
    let dl = l2 - l1;
    let dc = c2p - c1p;
    let chroma_product = c1p * c2p;
    let dh = if chroma_product == 0.0 {
        0.0
    } else {
        let d = h2p - h1p;
        if d > 180.0 {
            d - 360.0
        } else if d < -180.0 {
            d + 360.0
        } else {
            d
        }
    };
    let dh_big = 2.0 * sqrt64(chroma_product) * sin64(rad(dh) / 2.0);
    let lbar = (l1 + l2) / 2.0;
    let cbarp = (c1p + c2p) / 2.0;
    let hbar = if chroma_product == 0.0 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        (h1p + h2p) / 2.0
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) / 2.0
    } else {
        (h1p + h2p - 360.0) / 2.0
    };
    let t = 1.0 - 0.17 * cos64(rad(hbar - 30.0))
        + 0.24 * cos64(rad(2.0 * hbar))
        + 0.32 * cos64(rad(3.0 * hbar + 6.0))
        - 0.20 * cos64(rad(4.0 * hbar - 63.0));
    let q = (hbar - 275.0) / 25.0;
    let dtheta = 30.0 * exp64(-(q * q));
    let rc = 2.0 * sqrt64(p7(cbarp) / (p7(cbarp) + c7_25));
    let l50 = (lbar - 50.0) * (lbar - 50.0);
    let sl = 1.0 + 0.015 * l50 / sqrt64(20.0 + l50);
    let sc = 1.0 + 0.045 * cbarp;
    let sh = 1.0 + 0.015 * cbarp * t;
    let rt = -sin64(rad(2.0 * dtheta)) * rc;
    let (x, y, z) = (dl / sl, dc / sc, dh_big / sh);
    sqrt64(x * x + y * y + z * z + rt * y * z)
}

/// CIEDE2000 between two images of linear sRGB pixels, pixel by pixel
///
/// # Errors
///
/// [`CodecError::InvalidBufferSize`] when the lengths differ, [`CodecError::InvalidInput`] when there is
/// nothing to compare or a value is not finite.
pub fn delta_e_stats(a: &[[f64; 3]], b: &[[f64; 3]]) -> Result<Stats, CodecError> {
    same_len(a.len(), b.len())?;
    finite3(a, "a")?;
    finite3(b, "b")?;
    Ok(stats(
        a.iter()
            .zip(b)
            .map(|(p, q)| delta_e2000(lab_from_linear_srgb(*p), lab_from_linear_srgb(*q)))
            .collect(),
    ))
}

fn check_image(img: &[f64], w: usize, h: usize) -> Result<(), CodecError> {
    if w < 2 || h < 2 {
        return Err(CodecError::InvalidDimensions {
            width: w as u32,
            height: h as u32,
        });
    }
    let n = w.checked_mul(h).ok_or(CodecError::DimensionOverflow)?;
    if img.len() != n {
        return Err(CodecError::InvalidBufferSize {
            expected: n,
            got: img.len(),
        });
    }
    if let Some(i) = img.iter().position(|v| !v.is_finite()) {
        return Err(invalid(format!("pixel {i} is not finite")));
    }
    Ok(())
}

/// The DFT of each row of a w×h complex image, in place: X[k] = Σₓ x[x]·e^(−2πi·kx/w)
fn dft_rows(re: &mut [f64], im: &mut [f64], w: usize, h: usize) {
    let tw: Vec<(f64, f64)> = (0..w)
        .map(|k| {
            let a = -2.0 * core::f64::consts::PI * k as f64 / w as f64;
            (cos64(a), sin64(a))
        })
        .collect();
    let mut out_re = vec![0.0; w];
    let mut out_im = vec![0.0; w];
    for y in 0..h {
        let row = y * w;
        for k in 0..w {
            let (mut sr, mut si) = (0.0, 0.0);
            for x in 0..w {
                let (c, s) = tw[(k * x) % w];
                let (xr, xi) = (re[row + x], im[row + x]);
                sr += xr * c - xi * s;
                si += xr * s + xi * c;
            }
            out_re[k] = sr;
            out_im[k] = si;
        }
        re[row..row + w].copy_from_slice(&out_re);
        im[row..row + w].copy_from_slice(&out_im);
    }
}

fn transpose(v: &[f64], w: usize, h: usize) -> Vec<f64> {
    let mut t = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            t[x * h + y] = v[y * w + x];
        }
    }
    t
}

/// The 2-D power spectrum |X|²/(w·h) of the mean-removed image, row-major w×h
fn power_2d(img: &[f64], w: usize, h: usize) -> Vec<f64> {
    let mean = img.iter().sum::<f64>() / img.len() as f64;
    let mut re: Vec<f64> = img.iter().map(|v| v - mean).collect();
    let mut im = vec![0.0; w * h];
    dft_rows(&mut re, &mut im, w, h);
    let mut re_t = transpose(&re, w, h);
    let mut im_t = transpose(&im, w, h);
    dft_rows(&mut re_t, &mut im_t, h, w);
    let norm = (w * h) as f64;
    let p_t: Vec<f64> = re_t
        .iter()
        .zip(&im_t)
        .map(|(r, i)| (r * r + i * i) / norm)
        .collect();
    transpose(&p_t, h, w)
}

fn signed_freq(k: usize, n: usize) -> f64 {
    if k <= n / 2 {
        k as f64
    } else {
        k as f64 - n as f64
    }
}

/// Radial power spectrum: the power |X|²/(w·h) of the mean-removed image summed over rings of integer
/// radius r = round(√(fx² + fy²)) cycles per image, r = 0 ..= min(w, h)/2
///
/// By Parseval the rings sum to Σ(x − mean)² when no power lies beyond radius min(w, h)/2 (the corners of
/// the frequency plane are not reported). Two images of the same material but a shifted pattern have the
/// same spectrum.
///
/// # Errors
///
/// [`CodecError::InvalidDimensions`] below 2×2, [`CodecError::InvalidBufferSize`] when the length is not
/// w·h, [`CodecError::InvalidInput`] for a non-finite pixel.
pub fn radial_power_spectrum(img: &[f64], w: usize, h: usize) -> Result<Vec<f64>, CodecError> {
    check_image(img, w, h)?;
    let p = power_2d(img, w, h);
    let rmax = w.min(h) / 2;
    let mut rings = vec![0.0; rmax + 1];
    for ky in 0..h {
        let fy = signed_freq(ky, h);
        for kx in 0..w {
            let fx = signed_freq(kx, w);
            let r = sqrt64(fx * fx + fy * fy);
            let ri = (r + 0.5) as usize;
            if ri <= rmax {
                rings[ri] += p[ky * w + kx];
            }
        }
    }
    Ok(rings)
}

/// 1-Wasserstein (earth mover's) distance between two samples of values: ∫ |F_a(x) − F_b(x)| dx over their
/// empirical distribution functions (the counts may differ, the order of the samples does not matter)
///
/// # Errors
///
/// [`CodecError::InvalidInput`] when a sample is empty or holds a non-finite value.
pub fn wasserstein1(a: &[f64], b: &[f64]) -> Result<f64, CodecError> {
    if a.is_empty() || b.is_empty() {
        return Err(invalid("an empty sample"));
    }
    if a.iter().chain(b).any(|v| !v.is_finite()) {
        return Err(invalid("a value is not finite"));
    }
    let mut sa = a.to_vec();
    let mut sb = b.to_vec();
    sa.sort_unstable_by(f64::total_cmp);
    sb.sort_unstable_by(f64::total_cmp);
    let (na, nb) = (sa.len() as f64, sb.len() as f64);
    let (mut ia, mut ib) = (0usize, 0usize);
    let mut total = 0.0;
    let mut x = sa[0].min(sb[0]);
    while ia < sa.len() || ib < sb.len() {
        while ia < sa.len() && sa[ia] <= x {
            ia += 1;
        }
        while ib < sb.len() && sb[ib] <= x {
            ib += 1;
        }
        let next = match (sa.get(ia), sb.get(ib)) {
            (Some(p), Some(q)) => p.min(*q),
            (Some(p), None) => *p,
            (None, Some(q)) => *q,
            (None, None) => break,
        };
        total += (ia as f64 / na - ib as f64 / nb).abs() * (next - x);
        x = next;
    }
    Ok(total)
}

/// Angle (radians) between corresponding normals of two normal maps; the vectors need not be unit length
///
/// # Errors
///
/// [`CodecError::InvalidBufferSize`] when the lengths differ, [`CodecError::InvalidInput`] when there is
/// nothing to compare, a value is not finite or a vector has zero length.
pub fn normal_angle_stats(a: &[[f64; 3]], b: &[[f64; 3]]) -> Result<Stats, CodecError> {
    same_len(a.len(), b.len())?;
    finite3(a, "a")?;
    finite3(b, "b")?;
    let mut angles = Vec::with_capacity(a.len());
    for (i, (p, q)) in a.iter().zip(b).enumerate() {
        let np = sqrt64(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        let nq = sqrt64(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
        if np == 0.0 || nq == 0.0 {
            return Err(invalid(format!("normal {i} has zero length")));
        }
        let c = (p[0] * q[0] + p[1] * q[1] + p[2] * q[2]) / (np * nq);
        angles.push(acos64(c.clamp(-1.0, 1.0)));
    }
    Ok(stats(angles))
}

/// Dominant period (pixels) of a pattern along x: w / k for the horizontal frequency k = 1 ..= w/2 that
/// holds the most power, summed over the rows (the lowest such k on a tie)
///
/// # Errors
///
/// As [`radial_power_spectrum`], and [`CodecError::InvalidInput`] for an image with no variation along x.
pub fn dominant_period(img: &[f64], w: usize, h: usize) -> Result<f64, CodecError> {
    check_image(img, w, h)?;
    let mean = img.iter().sum::<f64>() / img.len() as f64;
    let mut re: Vec<f64> = img.iter().map(|v| v - mean).collect();
    let mut im = vec![0.0; w * h];
    dft_rows(&mut re, &mut im, w, h);
    let mut best = (0usize, 0.0f64);
    for k in 1..=w / 2 {
        let p: f64 = (0..h)
            .map(|y| {
                let (r, i) = (re[y * w + k], im[y * w + k]);
                r * r + i * i
            })
            .sum();
        if p > best.1 {
            best = (k, p);
        }
    }
    let scale: f64 = re.iter().map(|v| v.abs()).fold(0.0, f64::max);
    if best.0 == 0 || best.1 <= 1e-24 * scale.max(1.0) {
        return Err(invalid("no variation along x"));
    }
    Ok(w as f64 / best.0 as f64)
}
