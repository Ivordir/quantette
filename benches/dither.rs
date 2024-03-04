#[path = "../util/util.rs"]
mod util;

use util::benchmark_images;

use std::time::Duration;

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, Bencher, BenchmarkId, Criterion,
    SamplingMode,
};
use palette::{IntoColor, Oklab};
use quantette::{wu, ColorSlice, ColorSpace, FloydSteinberg, IndexedColorCounts, PaletteSize};

fn bench<ColorCount>(
    c: &mut Criterion,
    group: &str,
    counts: &[(String, ColorCount)],
    mut f: impl FnMut(&mut Bencher<WallTime>, &(PaletteSize, &ColorCount)),
) {
    let mut group = c.benchmark_group(group);
    group
        .sample_size(30)
        .noise_threshold(0.05)
        .sampling_mode(SamplingMode::Flat)
        .warm_up_time(Duration::from_secs(2));

    for k in [PaletteSize::MAX, 64.into(), 16.into()] {
        for (path, counts) in counts {
            group.bench_with_input(BenchmarkId::new(k.to_string(), path), &(k, counts), &mut f);
        }
    }
}

fn dither_oklab_single(c: &mut Criterion) {
    let counts = benchmark_images()
        .iter()
        .map(|(path, image)| {
            (
                path.clone(),
                (
                    image,
                    IndexedColorCounts::<Oklab, _, 3>::try_from_rgbimage_par(image, |srgb| {
                        srgb.into_linear().into_color()
                    })
                    .unwrap(),
                ),
            )
        })
        .collect::<Vec<_>>();

    bench(
        c,
        "dither_oklab_single",
        &counts,
        |b, &(k, (image, color_counts))| {
            let (width, height) = image.dimensions();
            let result =
                wu::indexed_palette_par(color_counts, k, &ColorSpace::default_binner_oklab_f32());

            b.iter(|| {
                let mut indices = result.indices.clone();
                FloydSteinberg::new().dither_indexed(
                    &result.palette,
                    &mut indices,
                    color_counts.colors(),
                    color_counts.indices(),
                    width,
                    height,
                )
            })
        },
    )
}

fn dither_srgb_single(c: &mut Criterion) {
    bench(
        c,
        "dither_srgb_single",
        benchmark_images(),
        |b, &(k, image)| {
            let (width, height) = image.dimensions();
            let colors = ColorSlice::try_from(image).unwrap();
            let result = wu::indexed_palette_par(&colors, k, &ColorSpace::default_binner_srgb_u8());

            b.iter(|| {
                let mut indices = result.indices.clone();
                FloydSteinberg::new().dither(&result.palette, &mut indices, &colors, width, height)
            })
        },
    )
}

fn dither_oklab_par(c: &mut Criterion) {
    let counts = benchmark_images()
        .iter()
        .map(|(path, image)| {
            (
                path.clone(),
                (
                    image,
                    IndexedColorCounts::<Oklab, _, 3>::try_from_rgbimage_par(image, |srgb| {
                        srgb.into_linear().into_color()
                    })
                    .unwrap(),
                ),
            )
        })
        .collect::<Vec<_>>();

    bench(
        c,
        "dither_oklab_par",
        &counts,
        |b, &(k, (image, color_counts))| {
            let (width, height) = image.dimensions();
            let result =
                wu::indexed_palette_par(color_counts, k, &ColorSpace::default_binner_oklab_f32());

            b.iter(|| {
                let mut indices = result.indices.clone();
                FloydSteinberg::new().dither_indexed_par(
                    &result.palette,
                    &mut indices,
                    color_counts.colors(),
                    color_counts.indices(),
                    width,
                    height,
                )
            })
        },
    )
}

fn dither_srgb_par(c: &mut Criterion) {
    bench(
        c,
        "dither_srgb_par",
        benchmark_images(),
        |b, &(k, image)| {
            let (width, height) = image.dimensions();
            let colors = ColorSlice::try_from(image).unwrap();
            let result = wu::indexed_palette_par(&colors, k, &ColorSpace::default_binner_srgb_u8());

            b.iter(|| {
                let mut indices = result.indices.clone();
                FloydSteinberg::new().dither_par(
                    &result.palette,
                    &mut indices,
                    &colors,
                    width,
                    height,
                )
            })
        },
    )
}

criterion_group!(
    benches,
    dither_oklab_single,
    dither_srgb_single,
    dither_oklab_par,
    dither_srgb_par
);
criterion_main!(benches);
