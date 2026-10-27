//! The appearance measures on two generated patterns: a striped "reference" and a "render" with a slightly
//! different period and colour
//!
//! cargo run --example appearance

use alice_codec::appearance::{
    delta_e_stats, dominant_period, radial_power_spectrum, wasserstein1,
};
use alice_codec::error::CodecError;

fn stripes(w: usize, h: usize, period: usize, lo: f64, hi: f64) -> Vec<f64> {
    (0..w * h)
        .map(|i| {
            if (i % w) % period < period / 2 {
                hi
            } else {
                lo
            }
        })
        .collect()
}

fn main() -> Result<(), CodecError> {
    let (w, h) = (64, 32);
    let reference = stripes(w, h, 8, 0.10, 0.60);
    let render = stripes(w, h, 16, 0.12, 0.58);

    let to_rgb = |v: &[f64]| v.iter().map(|&g| [g, g * 0.9, g * 0.8]).collect::<Vec<_>>();
    let de = delta_e_stats(&to_rgb(&reference), &to_rgb(&render))?;
    println!(
        "dE2000: mean {:.3} p95 {:.3} max {:.3} over {} pixels",
        de.mean, de.p95, de.max, de.count
    );

    let pr = radial_power_spectrum(&reference, w, h)?;
    let pq = radial_power_spectrum(&render, w, h)?;
    let peak = |p: &[f64]| {
        (0..p.len())
            .max_by(|&a, &b| p[a].total_cmp(&p[b]))
            .unwrap_or(0)
    };
    println!(
        "spectrum peak ring: reference {} / render {}",
        peak(&pr),
        peak(&pq)
    );
    println!("histogram W1: {:.4}", wasserstein1(&reference, &render)?);
    println!(
        "period: reference {} px / render {} px",
        dominant_period(&reference, w, h)?,
        dominant_period(&render, w, h)?
    );
    Ok(())
}
