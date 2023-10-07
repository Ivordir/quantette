#[path = "../util/util.rs"]
mod util;

use util::{to_oklab_counts, unsplash_images};

use std::time::Duration;

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, Bencher, BenchmarkId, Criterion,
    SamplingMode,
};
use quantette::{wu, ColorSlice, ColorSpace, PaletteSize};

fn bench<ColorFreq>(
    c: &mut Criterion,
    group: &str,
    counts: &[(String, ColorFreq)],
    mut f: impl FnMut(&mut Bencher<WallTime>, &(PaletteSize, &ColorFreq)),
) {
    let mut group = c.benchmark_group(group);
    group
        .sample_size(30)
        .noise_threshold(0.05)
        .sampling_mode(SamplingMode::Flat)
        .warm_up_time(Duration::from_millis(500));

    for (k, secs) in [(64.into(), 3), (16.into(), 2), (PaletteSize::MAX, 4)] {
        group.measurement_time(Duration::from_secs(secs));
        for (path, counts) in counts {
            group.bench_with_input(BenchmarkId::new(k.to_string(), path), &(k, counts), &mut f);
        }
    }
}

fn wu_oklab_palette_single(c: &mut Criterion) {
    let counts = to_oklab_counts(unsplash_images());
    bench(c, "wu_oklab_palette_single", &counts, |b, &(k, counts)| {
        b.iter(|| wu::palette(counts, k, &ColorSpace::default_binner_oklab_f32()))
    })
}

fn wu_srgb_palette_single(c: &mut Criterion) {
    let counts = unsplash_images();
    bench(c, "wu_srgb_palette_single", counts, |b, &(k, image)| {
        b.iter(|| {
            wu::palette(
                &ColorSlice::try_from(image).unwrap(),
                k,
                &ColorSpace::default_binner_srgb_u8(),
            )
        })
    })
}

fn wu_oklab_remap_single(c: &mut Criterion) {
    let counts = to_oklab_counts(unsplash_images());
    bench(c, "wu_oklab_remap_single", &counts, |b, &(k, counts)| {
        b.iter(|| wu::quantize(counts, k, &ColorSpace::default_binner_oklab_f32()))
    })
}

fn wu_srgb_remap_single(c: &mut Criterion) {
    let counts = unsplash_images();
    bench(c, "wu_srgb_remap_single", counts, |b, &(k, image)| {
        b.iter(|| {
            wu::quantize(
                &ColorSlice::try_from(image).unwrap(),
                k,
                &ColorSpace::default_binner_srgb_u8(),
            )
        })
    })
}

fn wu_oklab_palette_par(c: &mut Criterion) {
    let counts = to_oklab_counts(unsplash_images());
    bench(c, "wu_oklab_palette_par", &counts, |b, &(k, counts)| {
        b.iter(|| wu::palette_par(counts, k, &ColorSpace::default_binner_oklab_f32()))
    })
}

fn wu_srgb_palette_par(c: &mut Criterion) {
    let counts = unsplash_images();
    bench(c, "wu_srgb_palette_par", counts, |b, &(k, image)| {
        b.iter(|| {
            wu::palette_par(
                &ColorSlice::try_from(image).unwrap(),
                k,
                &ColorSpace::default_binner_srgb_u8(),
            )
        })
    })
}

fn wu_oklab_remap_par(c: &mut Criterion) {
    let counts = to_oklab_counts(unsplash_images());
    bench(c, "wu_oklab_remap_par", &counts, |b, &(k, counts)| {
        b.iter(|| wu::quantize_par(counts, k, &ColorSpace::default_binner_oklab_f32()))
    })
}

fn wu_srgb_remap_par(c: &mut Criterion) {
    let counts = unsplash_images();
    bench(c, "wu_srgb_remap_par", counts, |b, &(k, image)| {
        b.iter(|| {
            wu::quantize_par(
                &ColorSlice::try_from(image).unwrap(),
                k,
                &ColorSpace::default_binner_srgb_u8(),
            )
        })
    })
}

criterion_group!(
    benches,
    wu_oklab_palette_single,
    wu_srgb_palette_single,
    wu_oklab_remap_single,
    wu_srgb_remap_single,
    wu_oklab_palette_par,
    wu_srgb_palette_par,
    wu_oklab_remap_par,
    wu_srgb_remap_par,
);
criterion_main!(benches);
