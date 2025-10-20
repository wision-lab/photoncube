use criterion::{criterion_group, criterion_main, Criterion};
use fastrand;
use ndarray::{Array2, Array3};
use photoncube::transforms::{interpolate_where_mask, unpack_single};
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

pub fn benchmark_interpolate_where_mask(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(0x4d595df4d0f33173);

    let data = Array3::from_shape_fn((255, 255, 1), |(i, j, _k)| i.max(j) as u8).into_dyn();
    let mask = Array2::from_shape_simple_fn((255, 255), || rng.f32() < 0.1);

    c.bench_function("interpolate_where_mask", |b| {
        b.iter(|| {
            let _ = interpolate_where_mask(&data, &mask, false);
        })
    });
}

pub fn benchmark_unpack_single(c: &mut Criterion) {
    let data = Array2::from_shape_fn((1024, 1024 / 8), |(i, j)| i.max(j) as u8).into_dyn();

    c.bench_function("unpack_single", |b| {
        b.iter(|| {
            let _ = unpack_single::<u8>(&data.view(), 1);
        })
    });
}

#[cfg(target_os = "linux")]
criterion_group! {
    name = benches;
    config = Criterion::default()
        .with_profiler(
            PProfProfiler::new(100, Output::Flamegraph(None))
        );
    targets =
        benchmark_interpolate_where_mask,
        benchmark_unpack_single
}

#[cfg(not(target_os = "linux"))]
criterion_group! {
    benches,
    benchmark_interpolate_where_mask,
    benchmark_unpack_single
}

criterion_main!(benches);
