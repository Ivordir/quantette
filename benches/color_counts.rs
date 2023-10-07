#[path = "../util/util.rs"]
mod util;

use util::unsplash_images;

use std::time::Duration;

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, Bencher, BenchmarkId, Criterion,
    SamplingMode,
};
use image::RgbImage;
use palette::{IntoColor, Oklab};
use quantette::{RemappableColorCounts, UnmappableColorCounts};

fn bench(c: &mut Criterion, group: &str, mut f: impl FnMut(&mut Bencher<WallTime>, &RgbImage)) {
    let mut group = c.benchmark_group(group);
    group
        .sample_size(30)
        .noise_threshold(0.05)
        .sampling_mode(SamplingMode::Flat)
        .warm_up_time(Duration::from_millis(500));

    for (path, image) in unsplash_images() {
        let (width, height) = image.dimensions();
        let pixels = width * height;
        let mut w = 480;
        let mut h = 270;
        let mut next_width = w * 2;
        let mut next_height = h * 2;
        while next_width * next_height < pixels {
            let image = image::imageops::thumbnail(image, w, h);
            group.bench_with_input(BenchmarkId::new(path, format!("{w}x{h}")), &image, &mut f);
            w = next_width;
            h = next_height;
            next_width *= 2;
            next_height *= 2;
        }
        group.bench_with_input(
            BenchmarkId::new(path, format!("{width}x{height}")),
            image,
            &mut f,
        );
    }
}

fn color_counts_srgb_palette_single(c: &mut Criterion) {
    bench(c, "color_counts_srgb_palette_single", |b, image| {
        b.iter(|| {
            UnmappableColorCounts::<_, _, 3>::unmappable_try_from_rgbimage(image, |srgb| srgb)
                .unwrap()
        })
    });
}

fn color_counts_oklab_palette_single(c: &mut Criterion) {
    bench(c, "color_counts_oklab_palette_single", |b, image| {
        b.iter(|| {
            UnmappableColorCounts::<Oklab, _, 3>::unmappable_try_from_rgbimage(image, |srgb| {
                srgb.into_linear().into_color()
            })
            .unwrap()
        })
    });
}
fn color_counts_srgb_remap_single(c: &mut Criterion) {
    bench(c, "color_counts_srgb_remap_single", |b, image| {
        b.iter(|| {
            RemappableColorCounts::<_, _, 3>::remappable_try_from_rgbimage(image, |srgb| srgb)
                .unwrap()
        })
    });
}

fn color_counts_oklab_remap_single(c: &mut Criterion) {
    bench(c, "color_counts_oklab_remap_single", |b, image| {
        b.iter(|| {
            RemappableColorCounts::<Oklab, _, 3>::remappable_try_from_rgbimage(image, |srgb| {
                srgb.into_linear().into_color()
            })
            .unwrap()
        })
    });
}

fn color_counts_srgb_palette_par(c: &mut Criterion) {
    bench(c, "color_counts_srgb_palette_par", |b, image| {
        b.iter(|| {
            UnmappableColorCounts::<_, _, 3>::unmappable_try_from_rgbimage_par(image, |srgb| srgb)
                .unwrap()
        })
    });
}

fn color_counts_oklab_palette_par(c: &mut Criterion) {
    bench(c, "color_counts_oklab_palette_par", |b, image| {
        b.iter(|| {
            UnmappableColorCounts::<Oklab, _, 3>::unmappable_try_from_rgbimage_par(image, |srgb| {
                srgb.into_linear().into_color()
            })
            .unwrap()
        })
    });
}

fn color_counts_srgb_remap_par(c: &mut Criterion) {
    bench(c, "color_counts_srgb_remap_par", |b, image| {
        b.iter(|| {
            RemappableColorCounts::<_, _, 3>::remappable_try_from_rgbimage_par(image, |srgb| srgb)
                .unwrap()
        })
    });
}

fn color_counts_oklab_remap_par(c: &mut Criterion) {
    bench(c, "color_counts_oklab_remap_par", |b, image| {
        b.iter(|| {
            RemappableColorCounts::<Oklab, _, 3>::remappable_try_from_rgbimage_par(image, |srgb| {
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
