use super::Centroids;

use crate::{ColorAndFrequency, ColorComponents, ColorRemap, QuantizeOutput};

use std::{array, marker::PhantomData};

use num_traits::AsPrimitive;
use palette::cast::{self, AsArrays};
use rand::{prelude::Distribution, SeedableRng};
use rand_distr::{weighted_alias::WeightedAliasIndex, Uniform};
use rand_xoshiro::Xoroshiro128PlusPlus;
use wide::{f32x8, u32x8, CmpLe};

#[inline]
fn simd_min<const N: usize>(points: &[[f32x8; N]], query: [f32; N]) -> (u8, u8) {
    let incr = u32x8::ONE;
    let mut cur_chunk = u32x8::ZERO;
    let mut min_chunk = cur_chunk;
    let mut min_distance = f32x8::splat(f32::INFINITY);

    let query = query.map(f32x8::splat);

    for chunk in points {
        #[allow(clippy::unwrap_used)]
        let distance = array::from_fn::<_, N, _>(|i| {
            let diff = query[i] - chunk[i];
            diff * diff
        })
        .into_iter()
        .reduce(|a, b| a + b)
        .unwrap();

        #[allow(unsafe_code)]
        let mask: u32x8 = unsafe { std::mem::transmute(distance.cmp_le(min_distance)) };
        min_chunk = mask.blend(cur_chunk, min_chunk);
        min_distance = min_distance.fast_min(distance);
        cur_chunk += incr;
    }

    let mut min_lane = 0;
    let mut min_dist = f32::INFINITY;
    for (i, &v) in min_distance.as_array_ref().iter().enumerate() {
        if v < min_dist {
            min_dist = v;
            min_lane = i;
        }
    }

    let min_chunk = min_chunk.as_array_ref()[min_lane] as usize;

    #[allow(clippy::cast_possible_truncation)]
    {
        (min_chunk as u8, min_lane as u8)
    }
}

struct State<'a, Color, Component, const N: usize, ColorFreq>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32> + 'static,
    ColorFreq: ColorAndFrequency<Color, Component, N>,
    f32: AsPrimitive<Component>,
{
    _phatom: PhantomData<Component>,
    color_counts: &'a ColorFreq,
    components: Vec<[f32x8; N]>,
    counts: Vec<u32>,
    output: Vec<Color>,
}

impl<'a, Color, Component, const N: usize, ColorCount> State<'a, Color, Component, N, ColorCount>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32> + 'static,
    ColorCount: ColorAndFrequency<Color, Component, N>,
    f32: AsPrimitive<Component>,
{
    fn new(color_counts: &'a ColorCount, centroids: Vec<Color>) -> Self {
        let mut components = Vec::with_capacity(centroids.len().next_multiple_of(8));
        let chunks = centroids.as_arrays().chunks_exact(8);
        components.extend(
            chunks.clone().map(|chunk| {
                array::from_fn(|i| f32x8::new(array::from_fn(|j| chunk[j][i].into())))
            }),
        );

        if !chunks.remainder().is_empty() {
            let mut arr = [[f32::INFINITY; 8]; N];
            for (i, &color) in chunks.remainder().iter().enumerate() {
                for (arr, c) in arr.iter_mut().zip(color) {
                    arr[i] = c.into();
                }
            }
            components.push(arr.map(f32x8::new));
        }

        Self {
            _phatom: PhantomData,
            color_counts,
            components,
            counts: vec![0; centroids.len()],
            output: centroids,
        }
    }

    #[inline]
    fn add_sample(&mut self, color: [Component; N]) {
        let Self { components, counts, .. } = self;
        let color = color.map(Into::into);

        let (chunk, lane) = simd_min(components, color);

        let chunk = usize::from(chunk);
        let lane = usize::from(lane);
        let i = chunk * 8 + lane;

        let count = counts[i] + 1;
        #[allow(clippy::cast_possible_truncation)]
        let rate = (1.0 / f64::from(count).sqrt()) as f32; // learning rate of 0.5 => count^(-0.5)

        let mut center = components[chunk].map(|v| v.as_array_ref()[lane]);
        for c in 0..N {
            center[c] += rate * (color[c] - center[c]);
        }

        for (d, s) in components[chunk].iter_mut().zip(center) {
            #[allow(unsafe_code)]
            let d = unsafe { &mut *(d as *mut f32x8).cast::<[f32; 8]>() };
            d[lane] = s;
        }

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
        let Self { components, counts, output, .. } = self;

        let mut palette = output;

        let len = palette.len();
        palette.clear();
        palette.extend(components.into_iter().flat_map(|x| {
            array::from_fn::<Color, 8, _>(|i| {
                cast::from_array(x.map(|y| y.as_array_ref()[i].as_()))
            })
        }));
        palette.truncate(len);

        QuantizeOutput { palette, counts, indices }
    }
}

impl<'a, Color, Component, const N: usize, ColorCount> State<'a, Color, Component, N, ColorCount>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32> + 'static,
    ColorCount: ColorAndFrequency<Color, Component, N> + ColorRemap,
    f32: AsPrimitive<Component>,
{
    fn indices(&mut self) -> Vec<u8> {
        self.color_counts
            .color_components()
            .iter()
            .map(|color| {
                let color = color.map(Into::into);
                let (chunk, lane) = simd_min(&self.components, color);
                chunk * 8 + lane
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
    Component: Copy + Into<f32> + 'static,
    f32: AsPrimitive<Component>,
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
pub fn indexed_palette<Color, Component, const N: usize>(
    color_counts: &(impl ColorAndFrequency<Color, Component, N> + ColorRemap),
    num_samples: u32,
    initial_centroids: Centroids<Color>,
    seed: u64,
) -> QuantizeOutput<Color>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32> + 'static,
    f32: AsPrimitive<Component>,
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
