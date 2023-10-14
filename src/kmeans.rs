// The k-means implementations here are based upon the following paper:
//
// Thompson, S., Celebi, M.E. & Buck, K.H. Fast color quantization using MacQueen’s k-means algorithm.
// Journal of Real-Time Image Processing, vol. 17, 1609–1624, 2020.
// https://doi.org/10.1007/s11554-019-00914-6
//
// Accessed from https://faculty.uca.edu/ecelebi/documents/JRTIP_2020a.pdf

use crate::{
    AboveMaxLen, ColorComponents, ColorCounts, ColorRemap, QuantizeOutput, MAX_COLORS, MAX_K,
};

#[cfg(feature = "threads")]
use crate::ParallelColorRemap;

use std::{array, marker::PhantomData};

use num_traits::AsPrimitive;
use palette::cast::{self, AsArrays};
use rand::{prelude::Distribution, SeedableRng};
use rand_distr::{weighted_alias::WeightedAliasIndex, Uniform};
use rand_xoshiro::Xoroshiro128PlusPlus;
use wide::{f32x8, u32x8, CmpLe};

#[cfg(feature = "threads")]
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct Centroids<Color>(Vec<Color>);

impl<Color> Centroids<Color> {
    #[must_use]
    pub fn into_inner(self) -> Vec<Color> {
        self.0
    }

    #[must_use]
    pub fn from_truncated(mut centroids: Vec<Color>) -> Self {
        centroids.truncate(MAX_K);
        Self(centroids)
    }

    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub fn num_colors(&self) -> u16 {
        self.0.len() as u16
    }
}

impl<Color> From<Centroids<Color>> for Vec<Color> {
    fn from(value: Centroids<Color>) -> Self {
        value.into_inner()
    }
}

impl<Color> TryFrom<Vec<Color>> for Centroids<Color> {
    type Error = AboveMaxLen<u16>;

    fn try_from(colors: Vec<Color>) -> Result<Self, Self::Error> {
        if colors.len() <= MAX_K {
            Ok(Self(colors))
        } else {
            Err(AboveMaxLen(MAX_COLORS))
        }
    }
}

#[inline]
fn simd_argmin<const N: usize>(points: &[[f32x8; N]], query: [f32; N]) -> (u8, u8) {
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

    let min_chunk = min_chunk.as_array_ref()[min_lane];

    #[allow(clippy::cast_possible_truncation)]
    {
        (min_chunk as u8, min_lane as u8)
    }
}

struct State<'a, Color, Component, const N: usize, ColorCount>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32> + 'static,
    ColorCount: ColorCounts<Color, Component, N>,
    f32: AsPrimitive<Component>,
{
    _phatom: PhantomData<Component>,
    color_counts: &'a ColorCount,
    components: Vec<[f32x8; N]>,
    counts: Vec<u32>,
    output: Vec<Color>,
}

impl<'a, Color, Component, const N: usize, ColorCount> State<'a, Color, Component, N, ColorCount>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32> + 'static,
    ColorCount: ColorCounts<Color, Component, N>,
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

        let (chunk, lane) = simd_argmin(components, color);

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

    fn online_kmeans_inner(
        &mut self,
        samples: u32,
        seed: u64,
        distribution: &impl Distribution<usize>,
    ) {
        const BATCH: u32 = 256;

        let rng = &mut Xoroshiro128PlusPlus::seed_from_u64(seed);
        let colors = self.color_counts.color_components();

        let mut batch = Vec::with_capacity(BATCH as usize);

        for _ in 0..(samples / BATCH) {
            batch.extend((0..BATCH).map(|_| colors[distribution.sample(rng)]));

            for &color in &batch {
                self.add_sample(color);
            }

            batch.clear();
        }

        batch.extend((0..(samples % BATCH)).map(|_| colors[distribution.sample(rng)]));

        for color in batch {
            self.add_sample(color);
        }
    }

    #[allow(clippy::unwrap_used)]
    fn alias_index(counts: &[u32]) -> WeightedAliasIndex<u64> {
        // WeightedAliasIndex::new fails if:
        // - The vector is empty => should be handled by caller
        // - The vector is longer than u32::MAX =>
        //      ColorCounts implementors guarantee length <= MAX_PIXELS = u32::MAX
        // - For any weight w: w < 0 or w > max where max = W::MAX / weights.len() =>
        //      max count and max length are u32::MAX, so converting all counts to u64 will prevent this
        // - The sum of weights is zero =>
        //      ColorCounts implementors guarantee every count is > 0
        WeightedAliasIndex::new(counts.iter().copied().map(u64::from).collect()).unwrap()
    }

    fn online_kmeans(&mut self, samples: u32, seed: u64) {
        if let Some(counts) = self.color_counts.counts() {
            let distribution = Self::alias_index(counts);
            self.online_kmeans_inner(samples, seed, &distribution);
        } else {
            let distribution = Uniform::new(0, self.color_counts.len());
            self.online_kmeans_inner(samples, seed, &distribution);
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
    ColorCount: ColorCounts<Color, Component, N> + ColorRemap,
    f32: AsPrimitive<Component>,
{
    fn indices(&mut self) -> Vec<u8> {
        self.color_counts
            .color_components()
            .iter()
            .map(|color| {
                let color = color.map(Into::into);
                let (chunk, lane) = simd_argmin(&self.components, color);
                chunk * 8 + lane
            })
            .collect()
    }
}

#[must_use]
pub fn palette<Color, Component, const N: usize>(
    color_counts: &impl ColorCounts<Color, Component, N>,
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
    color_counts: &(impl ColorCounts<Color, Component, N> + ColorRemap),
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

#[cfg(feature = "threads")]
impl<'a, Color, Component, const N: usize, ColorCount> State<'a, Color, Component, N, ColorCount>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32> + 'static + Send + Sync,
    ColorCount: ColorCounts<Color, Component, N>,
    f32: AsPrimitive<Component>,
{
    fn minibatch_kmeans_inner(
        &mut self,
        max_samples: u32,
        batch_size: u32,
        seed: u64,
        distribution: &(impl Distribution<usize> + Sync),
    ) {
        let Self { components, counts, color_counts, .. } = self;

        let colors = color_counts.color_components();

        let threads = rayon::current_num_threads();
        let chunk_size = (batch_size as usize).div_ceil(threads);

        let mut rng = (0..threads)
            .map(|i| Xoroshiro128PlusPlus::seed_from_u64(seed ^ i as u64))
            .collect::<Vec<_>>();

        let mut batch = vec![[0.0.as_(); N]; batch_size as usize];
        let mut assignments = vec![(0, 0); batch_size as usize];

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
                            *center = simd_argmin(components, color);
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
                        *center = simd_argmin(components, color);
                    }
                });

            for (color, &(chunk, lane)) in batch.iter().zip(&assignments) {
                let color = color.map(Into::into);
                let chunk = usize::from(chunk);
                let lane = usize::from(lane);
                let i = chunk * 8 + lane;

                let count = counts[i] + 1;
                #[allow(clippy::cast_possible_truncation)]
                let rate = (1.0 / f64::from(count).sqrt()) as f32; // learning rate of 0.5 => count^(-0.5)

                #[allow(unsafe_code)]
                let centroid = unsafe {
                    &mut *std::ptr::addr_of_mut!(components[chunk]).cast::<[[f32; 8]; N]>()
                };
                for c in 0..N {
                    centroid[c][lane] += rate * (color[c] - centroid[c][lane]);
                }

                counts[i] = count;
            }
        }
    }

    fn minibatch_kmeans(&mut self, samples: u32, batch_size: u32, seed: u64) {
        if let Some(counts) = self.color_counts.counts() {
            let distribution = Self::alias_index(counts);
            self.minibatch_kmeans_inner(samples, batch_size, seed, &distribution);
        } else {
            let distribution = Uniform::new(0, self.color_counts.len());
            self.minibatch_kmeans_inner(samples, batch_size, seed, &distribution);
        }
    }
}

#[cfg(feature = "threads")]
impl<'a, Color, Component, const N: usize, ColorCount> State<'a, Color, Component, N, ColorCount>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32> + 'static + Send + Sync,
    ColorCount: ColorCounts<Color, Component, N> + ParallelColorRemap,
    f32: AsPrimitive<Component>,
{
    fn indices_par(&mut self) -> Vec<u8> {
        self.color_counts
            .color_components()
            .par_iter()
            .map(|color| {
                let color = color.map(Into::into);
                let (chunk, lane) = simd_argmin(&self.components, color);
                chunk * 8 + lane
            })
            .collect()
    }
}

#[cfg(feature = "threads")]
#[must_use]
pub fn palette_par<Color, Component, const N: usize>(
    color_counts: &impl ColorCounts<Color, Component, N>,
    num_samples: u32,
    batch_size: u32,
    initial_centroids: Centroids<Color>,
    seed: u64,
) -> QuantizeOutput<Color>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32> + 'static + Send + Sync,
    f32: AsPrimitive<Component>,
{
    if color_counts.num_colors() <= u32::from(initial_centroids.num_colors()) {
        QuantizeOutput::trivial_palette(color_counts)
    } else {
        let mut state = State::new(color_counts, initial_centroids.into());
        state.minibatch_kmeans(num_samples, batch_size, seed);
        state.into_summary(Vec::new())
    }
}

#[cfg(feature = "threads")]
#[must_use]
pub fn indexed_palette_par<Color, Component, const N: usize>(
    color_counts: &(impl ColorCounts<Color, Component, N> + ParallelColorRemap),
    num_samples: u32,
    batch_size: u32,
    initial_centroids: Centroids<Color>,
    seed: u64,
) -> QuantizeOutput<Color>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32> + 'static + Send + Sync,
    f32: AsPrimitive<Component>,
{
    if color_counts.num_colors() <= u32::from(initial_centroids.num_colors()) {
        QuantizeOutput::trivial_quantize_par(color_counts)
    } else {
        let mut state = State::new(color_counts, initial_centroids.into());
        state.minibatch_kmeans(num_samples, batch_size, seed);
        let indices = state.indices_par();
        state.into_summary(color_counts.map_indices_par(indices))
    }
}
