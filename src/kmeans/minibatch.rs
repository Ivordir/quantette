use super::{Centroids, KdTree};

use crate::{ColorAndFrequency, ColorComponents, ParallelColorRemap, QuantizeOutput};

use std::marker::PhantomData;

use num_traits::AsPrimitive;
use palette::cast::{self, AsArrays};
use rand::{prelude::Distribution, SeedableRng};
use rand_distr::{weighted_alias::WeightedAliasIndex, Uniform};
use rand_xoshiro::Xoroshiro128PlusPlus;
use rayon::prelude::*;

struct State<'a, Color, Component, const N: usize, ColorFreq>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f64> + 'static + Send + Sync,
    ColorFreq: ColorAndFrequency<Color, Component, N>,
    f64: AsPrimitive<Component>,
{
    _phatom: PhantomData<Component>,
    color_counts: &'a ColorFreq,
    tree: KdTree<N>,
    counts: Vec<u32>,
    centroids: Vec<[f64; N]>,
    output: Vec<Color>,
}

impl<'a, Color, Component, const N: usize, ColorCount> State<'a, Color, Component, N, ColorCount>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f64> + 'static + Send + Sync,
    ColorCount: ColorAndFrequency<Color, Component, N>,
    f64: AsPrimitive<Component>,
{
    fn new(color_counts: &'a ColorCount, centroids: Vec<Color>) -> Self {
        let output = centroids;

        let centroids = output
            .as_arrays()
            .iter()
            .map(|c| c.map(Into::into))
            .collect::<Vec<_>>();

        Self {
            _phatom: PhantomData,
            color_counts,
            tree: KdTree::new(&centroids),
            counts: vec![0; centroids.len()],
            centroids,
            output,
        }
    }

    fn kmeans_inner(
        &mut self,
        max_samples: u32,
        batch_size: u32,
        seed: u64,
        distribution: &(impl Distribution<usize> + Sync),
    ) {
        let Self {
            tree, counts, centroids, color_counts, ..
        } = self;

        let colors = color_counts.color_components();

        let threads = rayon::current_num_threads();
        let chunk_size = (batch_size as usize + threads - 1) / threads;

        let mut rng = (0..threads)
            .map(|i| Xoroshiro128PlusPlus::seed_from_u64(seed ^ i as u64))
            .collect::<Vec<_>>();

        let mut batch = vec![[0.0.as_(); N]; batch_size as usize];
        let mut assignments = vec![0; batch_size as usize];

        for _ in 0..(max_samples / batch_size) {
            batch
                .par_chunks_mut(chunk_size)
                .zip(assignments.par_chunks_mut(chunk_size))
                .zip(&mut rng)
                .for_each(|((batch, assignments), rng)| {
                    for (chunk, assignments) in batch
                        .chunks_exact_mut(4)
                        .zip(assignments.chunks_exact_mut(4))
                    {
                        for color in &mut *chunk {
                            *color = colors[distribution.sample(rng)];
                        }

                        for (center, color) in assignments.iter_mut().zip(&*chunk) {
                            let color = color.map(Into::into);
                            *center = tree.nearest_neighbor_entry(color).0.key;
                        }
                    }
                    for (color, center) in batch
                        .chunks_exact_mut(4)
                        .into_remainder()
                        .iter_mut()
                        .zip(assignments.chunks_exact_mut(4).into_remainder())
                    {
                        *color = colors[distribution.sample(rng)];
                        let color = color.map(Into::into);
                        *center = tree.nearest_neighbor_entry(color).0.key;
                    }
                });

            for (color, &i) in batch.iter().zip(&assignments) {
                let color = color.map(Into::into);
                let i = usize::from(i);

                let count = counts[i] + 1;
                let rate = 1.0 / f64::from(count).sqrt(); // learning rate of 0.5 => count^(-0.5)

                let centroid = &mut centroids[i];
                for c in 0..N {
                    centroid[c] += rate * (color[c] - centroid[c]);
                }

                counts[i] = count;
            }

            tree.update_batch(&*centroids);
        }
    }

    fn minibatch_kmeans(&mut self, samples: u32, batch_size: u32, seed: u64) {
        if let Some(counts) = self.color_counts.counts() {
            // WeightedAliasIndex::new fails if:
            // - The vector is empty => should be handled by caller
            // - The vector is longer than u32::MAX =>
            //      ColorAndFrequency implementors guarantee length <= MAX_PIXELS = u32::MAX
            // - For any weight w: w < 0 or w > max where max = W::MAX / weights.len() =>
            //      max count and max length are u32::MAX, so converting all counts to u64 will prevent this
            // - The sum of weights is zero =>
            //      ColorAndFrequency implementors guarantee every count is > 0
            #[allow(clippy::unwrap_used)]
            let distribution =
                WeightedAliasIndex::new(counts.iter().copied().map(u64::from).collect()).unwrap();
            self.kmeans_inner(samples, batch_size, seed, &distribution);
        } else {
            let distribution = Uniform::new(0, self.color_counts.len());
            self.kmeans_inner(samples, batch_size, seed, &distribution);
        }
    }

    fn into_summary(self, indices: Vec<u8>) -> QuantizeOutput<Color> {
        let Self { counts, centroids, output, .. } = self;

        let mut palette = output;
        for (output, centroid) in palette.iter_mut().zip(centroids) {
            *output = cast::from_array(centroid.map(AsPrimitive::as_));
        }

        QuantizeOutput { palette, counts, indices }
    }
}

impl<'a, Color, Component, const N: usize, ColorCount> State<'a, Color, Component, N, ColorCount>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f64> + 'static + Send + Sync,
    ColorCount: ColorAndFrequency<Color, Component, N> + ParallelColorRemap,
    f64: AsPrimitive<Component>,
{
    fn indices(&mut self) -> Vec<u8> {
        self.color_counts
            .color_components()
            .par_iter()
            .map(|color| {
                let color = color.map(Into::into);
                self.tree.nearest_neighbor_entry(color).0.key
            })
            .collect()
    }
}

#[must_use]
pub fn palette_par<Color, Component, const N: usize>(
    color_counts: &impl ColorAndFrequency<Color, Component, N>,
    num_samples: u32,
    batch_size: u32,
    initial_centroids: Centroids<Color>,
    seed: u64,
) -> QuantizeOutput<Color>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f64> + 'static + Send + Sync,
    f64: AsPrimitive<Component>,
{
    if color_counts.num_colors() <= u32::from(initial_centroids.num_colors()) {
        QuantizeOutput::trivial_palette(color_counts)
    } else {
        let mut state = State::new(color_counts, initial_centroids.into());
        state.minibatch_kmeans(num_samples, batch_size, seed);
        state.into_summary(Vec::new())
    }
}

#[must_use]
pub fn indexed_palette_par<Color, Component, const N: usize>(
    color_counts: &(impl ColorAndFrequency<Color, Component, N> + ParallelColorRemap),
    num_samples: u32,
    batch_size: u32,
    initial_centroids: Centroids<Color>,
    seed: u64,
) -> QuantizeOutput<Color>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f64> + 'static + Send + Sync,
    f64: AsPrimitive<Component>,
{
    if color_counts.num_colors() <= u32::from(initial_centroids.num_colors()) {
        QuantizeOutput::trivial_quantize_par(color_counts)
    } else {
        let mut state = State::new(color_counts, initial_centroids.into());
        state.minibatch_kmeans(num_samples, batch_size, seed);
        let indices = state.indices();
        state.into_summary(color_counts.map_indices_par(indices))
    }
}
