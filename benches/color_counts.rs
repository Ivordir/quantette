#[path = "../util/util.rs"]
mod util;

use util::benchmark_images;

use std::time::Duration;

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, Bencher, BenchmarkId, Criterion,
    SamplingMode,
};
use image::RgbImage;
use palette::{IntoColor, Oklab};
use quantette::{IndexedColorCounts, UniqueColorCounts};

fn bench(c: &mut Criterion, group: &str, mut f: impl FnMut(&mut Bencher<WallTime>, &RgbImage)) {
    let mut group = c.benchmark_group(group);
    group
        .sample_size(30)
        .noise_threshold(0.05)
        .sampling_mode(SamplingMode::Flat)
        .warm_up_time(Duration::from_millis(500));

    for (path, image) in benchmark_images() {
        group.bench_with_input(BenchmarkId::from_parameter(path), image, &mut f);
    }
}

fn color_counts_srgb_palette_single(c: &mut Criterion) {
    bench(c, "color_counts_srgb_palette_single", |b, image| {
        b.iter(|| UniqueColorCounts::<_, _, 3>::try_from_rgbimage(image, |srgb| srgb).unwrap())
    });
}

fn color_counts_oklab_palette_single(c: &mut Criterion) {
    bench(c, "color_counts_oklab_palette_single", |b, image| {
        b.iter(|| {
            UniqueColorCounts::<Oklab, _, 3>::try_from_rgbimage(image, |srgb| {
                srgb.into_linear().into_color()
            })
            .unwrap()
        })
    });
}
fn color_counts_srgb_remap_single(c: &mut Criterion) {
    bench(c, "color_counts_srgb_remap_single", |b, image| {
        b.iter(|| IndexedColorCounts::<_, _, 3>::try_from_rgbimage(image, |srgb| srgb).unwrap())
    });
}

fn color_counts_oklab_remap_single(c: &mut Criterion) {
    bench(c, "color_counts_oklab_remap_single", |b, image| {
        b.iter(|| {
            IndexedColorCounts::<Oklab, _, 3>::try_from_rgbimage(image, |srgb| {
                srgb.into_linear().into_color()
            })
            .unwrap()
        })
    });
}

fn color_counts_srgb_palette_par(c: &mut Criterion) {
    bench(c, "color_counts_srgb_palette_par", |b, image| {
        b.iter(|| UniqueColorCounts::<_, _, 3>::try_from_rgbimage_par(image, |srgb| srgb).unwrap())
    });
}

fn color_counts_oklab_palette_par(c: &mut Criterion) {
    bench(c, "color_counts_oklab_palette_par", |b, image| {
        b.iter(|| {
            UniqueColorCounts::<Oklab, _, 3>::try_from_rgbimage_par(image, |srgb| {
                srgb.into_linear().into_color()
            })
            .unwrap()
        })
    });
}

fn color_counts_srgb_remap_par(c: &mut Criterion) {
    bench(c, "color_counts_srgb_remap_par", |b, image| {
        b.iter(|| IndexedColorCounts::<_, _, 3>::try_from_rgbimage_par(image, |srgb| srgb).unwrap())
    });
}

fn color_counts_oklab_remap_par(c: &mut Criterion) {
    bench(c, "color_counts_oklab_remap_par", |b, image| {
        b.iter(|| {
            IndexedColorCounts::<Oklab, _, 3>::try_from_rgbimage_par(image, |srgb| {
                srgb.into_linear().into_color()
            })
            .unwrap()
        })
    });
}

criterion_group!(
    benches,
    color_counts_srgb_palette_single,
    color_counts_oklab_palette_single,
    color_counts_srgb_remap_single,
    color_counts_oklab_remap_single,
    color_counts_srgb_palette_par,
    color_counts_oklab_palette_par,
    color_counts_srgb_remap_par,
    color_counts_oklab_remap_par,
);
criterion_main!(benches);
