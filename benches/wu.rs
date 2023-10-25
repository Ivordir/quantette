#[path = "../util/util.rs"]
mod util;

use util::{benchmark_counts, benchmark_images};

use std::time::Duration;

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, Bencher, BenchmarkId, Criterion,
    SamplingMode,
};
use quantette::{wu, ColorSlice, ColorSpace, PaletteSize};

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
        .warm_up_time(Duration::from_millis(500));

    for k in [PaletteSize::MAX, 64.into(), 16.into()] {
        for (path, counts) in counts {
            group.bench_with_input(BenchmarkId::new(k.to_string(), path), &(k, counts), &mut f);
        }
    }
}

fn wu_oklab_palette_single(c: &mut Criterion) {
    bench(
        c,
        "wu_oklab_palette_single",
        benchmark_counts(),
        |b, &(k, counts)| {
            b.iter(|| wu::palette(counts, k, &ColorSpace::default_binner_oklab_f32()))
        },
    )
}

fn wu_srgb_palette_single(c: &mut Criterion) {
    bench(
        c,
        "wu_srgb_palette_single",
        benchmark_images(),
        |b, &(k, image)| {
            b.iter(|| {
                wu::palette(
                    &ColorSlice::try_from(image).unwrap(),
                    k,
                    &ColorSpace::default_binner_srgb_u8(),
                )
            })
        },
    )
}

fn wu_oklab_remap_single(c: &mut Criterion) {
    bench(
        c,
        "wu_oklab_remap_single",
        benchmark_counts(),
        |b, &(k, counts)| {
            b.iter(|| wu::indexed_palette(counts, k, &ColorSpace::default_binner_oklab_f32()))
        },
    )
}

fn wu_srgb_remap_single(c: &mut Criterion) {
    bench(
        c,
        "wu_srgb_remap_single",
        benchmark_images(),
        |b, &(k, image)| {
            b.iter(|| {
                wu::indexed_palette(
                    &ColorSlice::try_from(image).unwrap(),
                    k,
                    &ColorSpace::default_binner_srgb_u8(),
                )
            })
        },
    )
}

fn wu_oklab_palette_par(c: &mut Criterion) {
    bench(
        c,
        "wu_oklab_palette_par",
        benchmark_counts(),
        |b, &(k, counts)| {
            b.iter(|| wu::palette_par(counts, k, &ColorSpace::default_binner_oklab_f32()))
        },
    )
}

fn wu_srgb_palette_par(c: &mut Criterion) {
    bench(
        c,
        "wu_srgb_palette_par",
        benchmark_images(),
        |b, &(k, image)| {
            b.iter(|| {
                wu::palette_par(
                    &ColorSlice::try_from(image).unwrap(),
                    k,
                    &ColorSpace::default_binner_srgb_u8(),
                )
            })
        },
    )
}

fn wu_oklab_remap_par(c: &mut Criterion) {
    bench(
        c,
        "wu_oklab_remap_par",
        benchmark_counts(),
        |b, &(k, counts)| {
            b.iter(|| wu::indexed_palette_par(counts, k, &ColorSpace::default_binner_oklab_f32()))
        },
    )
}

fn wu_srgb_remap_par(c: &mut Criterion) {
    bench(
        c,
        "wu_srgb_remap_par",
        benchmark_images(),
        |b, &(k, image)| {
            b.iter(|| {
                wu::indexed_palette_par(
                    &ColorSlice::try_from(image).unwrap(),
                    k,
                    &ColorSpace::default_binner_srgb_u8(),
                )
            })
        },
    )
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
