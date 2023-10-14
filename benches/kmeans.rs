#[path = "../util/util.rs"]
mod util;

use util::{to_oklab_counts, to_srgb_counts, unsplash_images};

use std::time::Duration;

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, Bencher, BenchmarkId, Criterion,
    SamplingMode,
};
use quantette::{
    kmeans::{self, Centroids},
    wu, ColorComponents, ColorCounts, ColorSpace, PaletteSize,
};

const BATCH_SIZE: u32 = 4096;

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

    for (k, secs) in [
        (PaletteSize::MAX, 4),
        (128.into(), 4),
        (64.into(), 3),
        (32.into(), 2),
        (16.into(), 2),
    ] {
        group.measurement_time(Duration::from_secs(secs));
        for (path, counts) in counts {
            group.bench_with_input(BenchmarkId::new(k.to_string(), path), &(k, counts), &mut f);
        }
    }
}

fn num_samples<Color, Component, const N: usize>(
    color_counts: &impl ColorCounts<Color, Component, N>,
) -> u32
where
    Color: ColorComponents<Component, N>,
{
    color_counts.num_colors() / 2
}

fn kmeans_oklab_palette_single(c: &mut Criterion) {
    let counts = to_oklab_counts(unsplash_images());
    bench(
        c,
        "online_oklab_palette_single",
        &counts,
        |b, &(k, counts)| {
            let initial_centroids: Centroids<_> =
                wu::palette(counts, k, &ColorSpace::default_binner_oklab_f32())
                    .palette
                    .try_into()
                    .unwrap();

            b.iter(|| {
                kmeans::palette::<_, _, 3>(
                    counts,
                    num_samples(counts),
                    initial_centroids.clone(),
                    0,
                )
            })
        },
    )
}

fn kmeans_srgb_palette_single(c: &mut Criterion) {
    let counts = to_srgb_counts(unsplash_images());
    bench(
        c,
        "online_srgb_palette_single",
        &counts,
        |b, &(k, counts)| {
            let initial_centroids: Centroids<_> =
                wu::palette(counts, k, &ColorSpace::default_binner_srgb_u8())
                    .palette
                    .try_into()
                    .unwrap();

            b.iter(|| {
                kmeans::palette::<_, _, 3>(
                    counts,
                    num_samples(counts),
                    initial_centroids.clone(),
                    0,
                )
            })
        },
    )
}

fn kmeans_oklab_remap_single(c: &mut Criterion) {
    let counts = to_oklab_counts(unsplash_images());
    bench(
        c,
        "online_oklab_remap_single",
        &counts,
        |b, &(k, counts)| {
            let initial_centroids: Centroids<_> =
                wu::palette(counts, k, &ColorSpace::default_binner_oklab_f32())
                    .palette
                    .try_into()
                    .unwrap();

            b.iter(|| {
                kmeans::indexed_palette::<_, _, 3>(
                    counts,
                    num_samples(counts),
                    initial_centroids.clone(),
                    0,
                )
            })
        },
    )
}

fn kmeans_srgb_remap_single(c: &mut Criterion) {
    let counts = to_srgb_counts(unsplash_images());
    bench(c, "online_srgb_remap_single", &counts, |b, &(k, counts)| {
        let initial_centroids: Centroids<_> =
            wu::palette(counts, k, &ColorSpace::default_binner_srgb_u8())
                .palette
                .try_into()
                .unwrap();

        b.iter(|| {
            kmeans::indexed_palette::<_, _, 3>(
                counts,
                num_samples(counts),
                initial_centroids.clone(),
                0,
            )
        })
    })
}

fn kmeans_oklab_palette_par(c: &mut Criterion) {
    let counts = to_oklab_counts(unsplash_images());
    bench(
        c,
        "minibatch_oklab_palette_par",
        &counts,
        |b, &(k, counts)| {
            let initial_centroids: Centroids<_> =
                wu::palette(counts, k, &ColorSpace::default_binner_oklab_f32())
                    .palette
                    .try_into()
                    .unwrap();

            b.iter(|| {
                kmeans::palette_par::<_, _, 3>(
                    counts,
                    num_samples(counts),
                    BATCH_SIZE,
                    initial_centroids.clone(),
                    0,
                )
            })
        },
    )
}

fn kmeans_srgb_palette_par(c: &mut Criterion) {
    let counts = to_srgb_counts(unsplash_images());
    bench(
        c,
        "minibatch_srgb_palette_par",
        &counts,
        |b, &(k, counts)| {
            let initial_centroids: Centroids<_> =
                wu::palette(counts, k, &ColorSpace::default_binner_srgb_u8())
                    .palette
                    .try_into()
                    .unwrap();

            b.iter(|| {
                kmeans::palette_par::<_, _, 3>(
                    counts,
                    num_samples(counts),
                    BATCH_SIZE,
                    initial_centroids.clone(),
                    0,
                )
            })
        },
    )
}

fn kmeans_oklab_remap_par(c: &mut Criterion) {
    let counts = to_oklab_counts(unsplash_images());
    bench(
        c,
        "minibatch_oklab_remap_par",
        &counts,
        |b, &(k, counts)| {
            let initial_centroids: Centroids<_> =
                wu::palette(counts, k, &ColorSpace::default_binner_oklab_f32())
                    .palette
                    .try_into()
                    .unwrap();

            b.iter(|| {
                kmeans::indexed_palette_par::<_, _, 3>(
                    counts,
                    num_samples(counts),
                    BATCH_SIZE,
                    initial_centroids.clone(),
                    0,
                )
            })
        },
    )
}

fn kmeans_srgb_remap_par(c: &mut Criterion) {
    let counts = to_srgb_counts(unsplash_images());
    bench(c, "minibatch_srgb_remap_par", &counts, |b, &(k, counts)| {
        let initial_centroids: Centroids<_> =
            wu::palette(counts, k, &ColorSpace::default_binner_srgb_u8())
                .palette
                .try_into()
                .unwrap();

        b.iter(|| {
            kmeans::indexed_palette_par::<_, _, 3>(
                counts,
                num_samples(counts),
                BATCH_SIZE,
                initial_centroids.clone(),
                0,
            )
        })
    })
}

criterion_group!(
    benches,
    kmeans_oklab_palette_single,
    kmeans_srgb_palette_single,
    kmeans_oklab_remap_single,
    kmeans_srgb_remap_single,
    kmeans_oklab_palette_par,
    kmeans_srgb_palette_par,
    kmeans_oklab_remap_par,
    kmeans_srgb_remap_par,
);
criterion_main!(benches);
