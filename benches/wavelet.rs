use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_haar_1d(c: &mut Criterion) {
    let wavelet = alice_codec::Wavelet1D::haar();
    let original: Vec<i32> = (0..1024).map(|i| (i * 7 + 13) % 256).collect();

    c.bench_function("haar_1d_forward_1024", |b| {
        let mut signal = original.clone();
        b.iter(|| {
            signal.copy_from_slice(&original);
            wavelet.forward(black_box(&mut signal));
        });
    });

    c.bench_function("haar_1d_inverse_1024", |b| {
        let mut signal = original.clone();
        wavelet.forward(&mut signal);
        let transformed = signal.clone();
        b.iter(|| {
            signal.copy_from_slice(&transformed);
            wavelet.inverse(black_box(&mut signal));
        });
    });
}

fn bench_cdf53_1d(c: &mut Criterion) {
    let wavelet = alice_codec::Wavelet1D::cdf53();
    let original: Vec<i32> = (0..1024).map(|i| (i * 7 + 13) % 256).collect();

    c.bench_function("cdf53_1d_forward_1024", |b| {
        let mut signal = original.clone();
        b.iter(|| {
            signal.copy_from_slice(&original);
            wavelet.forward(black_box(&mut signal));
        });
    });
}

fn bench_cdf53_2d(c: &mut Criterion) {
    let wavelet = alice_codec::Wavelet2D::cdf53();
    let original: Vec<i32> = (0..64 * 64).map(|i| (i * 3 + 10) % 256).collect();

    c.bench_function("cdf53_2d_forward_64x64", |b| {
        let mut image = original.clone();
        b.iter(|| {
            image.copy_from_slice(&original);
            wavelet.forward(black_box(&mut image), 64, 64);
        });
    });
}

fn bench_cdf53_3d(c: &mut Criterion) {
    let wavelet = alice_codec::Wavelet3D::cdf53();
    let original: Vec<i32> = (0..32 * 32 * 8).map(|i| (i * 3 + 10) % 256).collect();

    c.bench_function("cdf53_3d_forward_32x32x8", |b| {
        let mut volume = original.clone();
        b.iter(|| {
            volume.copy_from_slice(&original);
            wavelet.forward(black_box(&mut volume), 32, 32, 8);
        });
    });
}

fn bench_quantizer(c: &mut Criterion) {
    let quantizer = alice_codec::Quantizer::new(16);
    let input: Vec<i32> = (0..4096).map(|i: i32| (i - 2048) * 3).collect();
    let mut output = vec![0i32; 4096];

    c.bench_function("quantize_4096", |b| {
        b.iter(|| {
            quantizer
                .quantize_buffer(black_box(&input), &mut output)
                .unwrap();
        });
    });
}

criterion_group!(
    benches,
    bench_haar_1d,
    bench_cdf53_1d,
    bench_cdf53_2d,
    bench_cdf53_3d,
    bench_quantizer,
);
criterion_main!(benches);
