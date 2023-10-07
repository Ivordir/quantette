use super::{Centroids, KdTree};

use crate::{ColorAndFrequency, ColorComponents, ColorRemap, QuantizeOutput};

use std::marker::PhantomData;

use num_traits::AsPrimitive;
use palette::cast::{self, AsArrays};
use rand::{prelude::Distribution, SeedableRng};
use rand_distr::{weighted_alias::WeightedAliasIndex, Uniform};
use rand_xoshiro::Xoroshiro128PlusPlus;

struct State<'a, Color, Component, const N: usize, ColorFreq>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f64> + 'static,
    f64: AsPrimitive<Component>,
    ColorFreq: ColorAndFrequency<Color, Component, N>,
{
    _phatom: PhantomData<Component>,
    color_counts: &'a ColorFreq,
    tree: KdTree<N>,
    counts: Vec<u32>,
    output: Vec<Color>,
}

impl<'a, Color, Component, const N: usize, ColorCount> State<'a, Color, Component, N, ColorCount>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f64> + 'static,
    ColorCount: ColorAndFrequency<Color, Component, N>,
    f64: AsPrimitive<Component>,
{
    fn new(color_counts: &'a ColorCount, centroids: Vec<Color>) -> Self {
        let centroids_float = centroids
            .as_arrays()
            .iter()
            .map(|c| c.map(Into::into))
            .collect::<Vec<_>>();

        Self {
            _phatom: PhantomData,
            color_counts,
            tree: KdTree::new(&centroids_float),
            counts: vec![0; centroids.len()],
            output: centroids,
        }
    }

    #[inline]
    fn add_sample(&mut self, color: [Component; N]) {
        let Self { tree, counts, .. } = self;
        let color = color.map(Into::into);

        let (entry, mut center) = tree.nearest_neighbor_entry(color);
        let i = usize::from(entry.key);

        let count = counts[i] + 1;
        let rate = 1.0 / f64::from(count).sqrt(); // learning rate of 0.5 => count^(-0.5)

        for c in 0..N {
            center[c] += rate * (color[c] - center[c]);
        }

        tree.update_entry(entry, center);
        counts[i] = count;
    }

    fn kmeans_inner(&mut self, samples: u32, seed: u64, distribution: &impl Distribution<usize>) {
        let rng = &mut Xoroshiro128PlusPlus::seed_from_u64(seed);
        let colors = self.color_counts.color_components();

        for _ in 0..(samples / 4) {
            let c1 = colors[distribution.sample(rng)];
            let c2 = colors[distribution.sample(rng)];
            let c3 = colors[distribution.sample(rng)];
            let c4 = colors[distribution.sample(rng)];

            for color in [c1, c2, c3, c4] {
                self.add_sample(color);
            }
        }

        for _ in 0..(samples % 4) {
            self.add_sample(colors[distribution.sample(rng)]);
        }
    }

    fn online_kmeans(&mut self, samples: u32, seed: u64) {
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
            self.kmeans_inner(samples, seed, &distribution);
        } else {
            let distribution = Uniform::new(0, self.color_counts.len());
            self.kmeans_inner(samples, seed, &distribution);
        }
    }

    fn into_summary(self, indices: Vec<u8>) -> QuantizeOutput<Color> {
        let Self { tree, counts, output, .. } = self;

        let mut palette = output;
        debug_assert_eq!(palette.len(), usize::from(tree.num_points()));
        for (i, point) in tree.iter() {
            palette[usize::from(i)] = cast::from_array(point.map(AsPrimitive::as_));
        }

        QuantizeOutput { palette, counts, indices }
    }
}

impl<'a, Color, Component, const N: usize, ColorCount> State<'a, Color, Component, N, ColorCount>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f64> + 'static,
    ColorCount: ColorAndFrequency<Color, Component, N> + ColorRemap,
    f64: AsPrimitive<Component>,
{
    fn indices(&mut self) -> Vec<u8> {
        self.color_counts
            .color_components()
            .iter()
            .map(|color| {
                let color = color.map(Into::into);
                self.tree.nearest_neighbor_entry(color).0.key
            })
            .collect()
    }
}

#[must_use]
pub fn palette<Color, Component, const N: usize>(
    color_counts: &impl ColorAndFrequency<Color, Component, N>,
    num_samples: u32,
    initial_centroids: Centroids<Color>,
    seed: u64,
) -> QuantizeOutput<Color>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f64> + 'static,
    f64: AsPrimitive<Component>,
{
    if color_counts.num_colors() <= u32::from(initial_centroids.num_colors()) {
        QuantizeOutput::trivial_palette(color_counts)
    } else {
        let mut state = State::new(color_counts, initial_centroids.into());
        state.online_kmeans(num_samples, seed);
        state.into_summary(Vec::new())
    }
}

#[must_use]
pub fn quantize<Color, Component, const N: usize>(
    color_counts: &(impl ColorAndFrequency<Color, Component, N> + ColorRemap),
    num_samples: u32,
    initial_centroids: Centroids<Color>,
    seed: u64,
) -> QuantizeOutput<Color>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f64> + 'static,
    f64: AsPrimitive<Component>,
{
    if color_counts.num_colors() <= u32::from(initial_centroids.num_colors()) {
        QuantizeOutput::trivial_quantize(color_counts)
    } else {
        let mut state = State::new(color_counts, initial_centroids.into());
        state.online_kmeans(num_samples, seed);
        let indices = state.indices();
        state.into_summary(color_counts.map_indices(indices))
    }
}
