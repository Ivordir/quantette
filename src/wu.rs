//! Wu's color quantizer (Greedy Orthogonal Bipartitioning).
//!
//! This preclustering method recursively splits the histogram box with the greatest variance
//! along the dimension and bin that results in the greatest decrease in variance.
//! It should give much better results than median cut
//! while having nearly the same computational cost.
//!
//! A binner with `32` bins (`B = 32`) is a good compromise between speed and accuracy.
//! If the [`PaletteSize`] is small, less bins can be used without sacrificing accuracy too much.
//! Otherwise, use more bins (e.g., `64`) if you want greater accuracy.

// Referenced code: https://www.ece.mcmaster.ca/~xwu/cq.c
// and relevant paper (free access):
// Xiaolin Wu, Color quantization by dynamic programming and principal analysis,
// ACM Transactions on Graphics, vol. 11, no. 4, 348–372, 1992.
// https://doi.org/10.1145/146443.146475

use crate::{
    ColorComponents, ColorCounts, ColorCountsRemap, PaletteSize, QuantizeOutput, SumPromotion,
    ZeroedIsZero,
};
use num_traits::{AsPrimitive, Float, Zero};
use ordered_float::OrderedFloat;
use palette::cast;
use std::{
    array,
    collections::BinaryHeap,
    marker::PhantomData,
    ops::{Add, AddAssign, Index, IndexMut, Sub},
};
#[cfg(feature = "threads")]
use {crate::ColorCountsParallelRemap, rayon::prelude::*};

/// The number of components in the color types. Only 3 components are supported for now.
const N: usize = 3;

/// A hypercube over a multi-dimensional range of histogram bins.
#[derive(Clone, Copy, Default)]
struct Cube {
    /// The lower bin indices (inclusive).
    min: [u8; N],
    /// The upper bin indices (exclusive).
    max: [u8; N],
}

impl Cube {
    /// Whether or not this cube contains a single bin.
    fn is_single_bin(self) -> bool {
        let Self { min, max } = self;
        (0..N).all(|c| max[c] - min[c] == 1)
    }
}

/// A new type wrapper around a 3-dimensional array.
#[repr(transparent)]
#[derive(Clone, Copy)]
struct Histogram3<T, const B: usize>([[[T; B]; B]; B]);

#[allow(unsafe_code)] // Histogram3 is repr(transparent) and inner types are bounded
unsafe impl<T, const B: usize> ZeroedIsZero for Histogram3<T, B>
where
    T: Copy,
    [[[T; B]; B]; B]: ZeroedIsZero,
{
}

impl<T, const B: usize> Index<[usize; N]> for Histogram3<T, B> {
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; N]) -> &Self::Output {
        &self.0[index[0]][index[1]][index[2]]
    }
}

impl<T, const B: usize> IndexMut<[usize; N]> for Histogram3<T, B> {
    #[inline]
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        &mut self.0[index[0]][index[1]][index[2]]
    }
}

impl<T, const B: usize> Index<[u8; N]> for Histogram3<T, B> {
    type Output = T;

    #[inline]
    fn index(&self, index: [u8; N]) -> &Self::Output {
        &self[index.map(usize::from)]
    }
}

impl<T, const B: usize> IndexMut<[u8; N]> for Histogram3<T, B> {
    #[inline]
    fn index_mut(&mut self, index: [u8; N]) -> &mut Self::Output {
        &mut self[index.map(usize::from)]
    }
}

/// Statistics for a histogram bin.
#[repr(align(64))]
#[derive(Clone, Copy)]
struct Stats<T, const N: usize> {
    /// The number of pixels/colors assigned to the bin.
    count: u32,
    /// The component-wise sum of the colors assigned to the bin.
    components: [T; N],
    /// The sum of the squared components of the colors assigned to the bin.
    sum_squared: f64,
}

#[allow(unsafe_code)] // all inner types impl ZeroedIsZero
unsafe impl<T, const N: usize> ZeroedIsZero for Stats<T, N>
where
    u32: ZeroedIsZero,
    T: Copy,
    [T; N]: ZeroedIsZero,
    f64: ZeroedIsZero,
{
}

impl<T: Copy + Add<Output = T>, const N: usize> Add for Stats<T, N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            count: self.count + rhs.count,
            components: array::from_fn(|i| self.components[i] + rhs.components[i]),
            sum_squared: self.sum_squared + rhs.sum_squared,
        }
    }
}

impl<T: Copy + Sub<Output = T>, const N: usize> Sub for Stats<T, N> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            count: self.count - rhs.count,
            components: array::from_fn(|i| self.components[i] - rhs.components[i]),
            sum_squared: self.sum_squared - rhs.sum_squared,
        }
    }
}

impl<T: Copy + AddAssign, const N: usize> AddAssign for Stats<T, N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.count += rhs.count;
        for i in 0..N {
            self.components[i] += rhs.components[i];
        }
        self.sum_squared += rhs.sum_squared;
    }
}

impl<T: Copy + Zero, const N: usize> Zero for Stats<T, N> {
    fn zero() -> Self {
        Self {
            count: 0,
            components: [T::zero(); N],
            sum_squared: 0.0,
        }
    }

    fn is_zero(&self) -> bool {
        self.count == 0 && self.sum_squared == 0.0 && self.components.iter().all(Zero::is_zero)
    }
}

/// This macro generates code for a fixed number of recursive calls to a volume function.
macro_rules! ndvolume {
    ($self: ident, $min: ident, $max: ident, $index: ident; $n: literal $(, $ns: literal)* $(,)?) => {{
        $index[$n] = $max[$n] - 1;
        let upper = ndvolume!($self, $min, $max, $index; $($ns,)*);

        let lower = if $min[$n] == 0 {
            T::zero()
        } else {
            $index[$n] = $min[$n] - 1;
            ndvolume!($self, $min, $max, $index; $($ns,)*)
        };

        upper - lower
    }};
    ($self: ident, $min: ident, $max: ident, $index: ident;) => {
        $self[$index]
    };
}

impl<T, const B: usize> Histogram3<T, B>
where
    T: Copy + Zero + Add<Output = T> + Sub<Output = T>,
{
    /// Returns the sum of the histogram bins specified by the given cube.
    fn volume(&self, Cube { min, max }: Cube) -> T {
        let mut index = [0u8; N];
        ndvolume!(self, min, max, index; 0, 1, 2)
    }

    /// Returns the sum of the histogram bins specified by the given cube
    /// but with one of the dimensions fixed to the given bin.
    fn volume_at(&self, Cube { min, max }: Cube, dim: u8, bin: u8) -> T {
        if bin == 0 {
            T::zero()
        } else {
            let bin = bin - 1;
            let mut index = [0u8; N];
            match dim {
                0 => {
                    index[0] = bin;
                    ndvolume!(self, min, max, index; 1, 2)
                }
                1 => {
                    index[1] = bin;
                    ndvolume!(self, min, max, index; 0, 2)
                }
                2 => {
                    index[2] = bin;
                    ndvolume!(self, min, max, index; 0, 1)
                }
                _ => unreachable!("dim < {N}"),
            }
        }
    }
}

/// An interface for taking colors with 3 components and returning a 3-dimensional histogram bin index.
///
/// The returned indices should be less than `B`.
pub trait Binner3<T, const B: usize> {
    /// Returns the 3-dimensional histogram bin index for the given color components.
    ///
    /// Each index must be less than `B`. It is recommended to mark implementations of this function as `inline`.
    fn bin(&self, components: [T; N]) -> [u8; N];
}

/// The struct holding the data for Wu's color quantization method (for 3 dimensions).
struct Wu3<'a, Color, Component, Binner, const B: usize, ColorCount>
where
    Color: ColorComponents<Component, N>,
    Component: SumPromotion<u32>,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64>,
    Binner: Binner3<Component, B>,
    ColorCount: ColorCounts<Color, Component, N>,
    u32: Into<Component::Sum>,
{
    /// The color type must remain the same for each [`Wu3`].
    _phantom: PhantomData<Color>,
    /// The color and count data.
    color_counts: &'a ColorCount,
    /// The histogram binner to use.
    binner: &'a Binner,
    /// The histogram data.
    hist: Box<Histogram3<Stats<Component::Sum, N>, B>>,
}

impl<'a, Color, Component, Binner, const B: usize, ColorCount>
    Wu3<'a, Color, Component, Binner, B, ColorCount>
where
    Color: ColorComponents<Component, N>,
    Component: SumPromotion<u32>,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64>,
    Binner: Binner3<Component, B>,
    ColorCount: ColorCounts<Color, Component, N>,
    u32: Into<Component::Sum>,
{
    /// Creates a new [`Wu3`] with zeored histogram data.
    fn new_zero(color_counts: &'a ColorCount, binner: &'a Binner) -> Self {
        assert!((1..=256).contains(&B));

        Self {
            _phantom: PhantomData,
            color_counts,
            binner,
            hist: ZeroedIsZero::box_zeroed(),
        }
    }

    /// Adds the given color with the given count to the histogram.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn add_color(&mut self, color: [Component; N], n: u32) {
        let Stats { count, components, sum_squared } = &mut self.hist[self.binner.bin(color)];
        let color = color.map(Into::into);

        *count += n;

        let w: Component::Sum = n.into();
        for (c, v) in components.iter_mut().zip(color) {
            *c += w * v;
        }

        let w = f64::from(n);
        *sum_squared += w * Self::sum_of_squares(color);
    }

    /// Adds the given colors to the histogram.
    fn add_colors(&mut self, colors: &[[Component; N]]) {
        for colors in colors.chunks_exact(4) {
            for &color in colors {
                self.add_color(color, 1);
            }
        }
        for &color in colors.chunks_exact(4).remainder() {
            self.add_color(color, 1);
        }
    }

    /// Adds the given colors and their counts to the histogram.
    fn add_color_counts(&mut self, colors: &[[Component; N]], counts: &[u32]) {
        for (colors, counts) in colors.chunks_exact(4).zip(counts.chunks_exact(4)) {
            for (&color, &n) in colors.iter().zip(counts) {
                self.add_color(color, n);
            }
        }
        for (&color, &n) in colors
            .chunks_exact(4)
            .remainder()
            .iter()
            .zip(counts.chunks_exact(4).remainder())
        {
            self.add_color(color, n);
        }
    }

    /// Creates a new [`Wu3`] with histogram data filled by the given `color_counts`.
    fn new(color_counts: &'a ColorCount, binner: &'a Binner) -> Self {
        let mut data = Self::new_zero(color_counts, binner);

        if let Some(counts) = data.color_counts.counts() {
            data.add_color_counts(data.color_counts.color_components(), counts);
        } else {
            data.add_colors(data.color_counts.color_components());
        }

        data.calc_cumulative_moments();

        data
    }

    /// Creates moments from the histogram bins to allow inclusion-excluison lookups/calculations.
    fn calc_cumulative_moments(&mut self) {
        let hist = &mut self.hist;

        for r in 0..B {
            let mut area = [Stats::zero(); B];

            for g in 0..B {
                let mut line = Stats::zero();

                for b in 0..B {
                    line += hist[[r, g, b]];
                    area[b] += line;

                    // compiler should hoist/remove the following if statement
                    if r == 0 {
                        hist[[r, g, b]] = area[b];
                    } else {
                        hist[[r, g, b]] = hist[[r - 1, g, b]] + area[b];
                    }
                }
            }
        }
    }

    /// Returns the sum of the squares of the given components.
    #[inline]
    fn sum_of_squares(components: [Component::Sum; N]) -> f64 {
        let mut square = 0.0;
        for c in components {
            let c = c.as_();
            square += c * c;
        }
        square
    }

    /// Computes the variance of the given cube.
    fn variance(&self, cube: Cube) -> f64 {
        if cube.is_single_bin() {
            0.0
        } else {
            let Stats { count, components, sum_squared } = self.hist.volume(cube);
            sum_squared - Self::sum_of_squares(components) / f64::from(count)
        }
    }

    /// Finds the index of the bin to cut along for the given dimension in order to minimize variance.
    fn minimize(&self, cube: Cube, dim: u8, sum: Stats<Component::Sum, N>) -> Option<(u8, f64)> {
        let d = usize::from(dim);
        let bottom = cube.min[d];
        let top = cube.max[d];

        let base = self.hist.volume_at(cube, dim, bottom);

        ((bottom + 1)..top)
            .filter_map(|bin| {
                let upper = self.hist.volume_at(cube, dim, bin) - base;
                let lower = sum - upper;
                if upper.count == 0 || lower.count == 0 {
                    None
                } else {
                    let upper2 = Self::sum_of_squares(upper.components) / f64::from(upper.count);
                    let lower2 = Self::sum_of_squares(lower.components) / f64::from(lower.count);
                    Some((bin, -(upper2 + lower2)))
                }
            })
            .min_by_key(|&(_, v)| OrderedFloat(v))
    }

    /// Attempts to cut the given cube to give a lower variance.
    fn cut(&self, cube: &mut Cube) -> Option<Cube> {
        let sum = self.hist.volume(*cube);

        #[allow(clippy::cast_possible_truncation)]
        let cut = (0..(N as u8))
            .filter_map(|c| {
                self.minimize(*cube, c, sum)
                    .map(|(x, v)| ((usize::from(c), x), v))
            })
            .min_by_key(|&(_, v)| OrderedFloat(v));

        if let Some(((i, cut), _)) = cut {
            let mut new_cube = *cube;
            cube.max[i] = cut;
            new_cube.min[i] = cut;
            Some(new_cube)
        } else {
            None
        }
    }

    /// Returns the disjoint cubes resulting from Wu's color quantization method.
    fn cubes(&self, k: PaletteSize) -> impl Iterator<Item = Cube> {
        /// A cube and it's variance.
        struct CubeVar(Cube, f64);

        impl PartialOrd for CubeVar {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for CubeVar {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                OrderedFloat(self.1).cmp(&OrderedFloat(other.1))
            }
        }

        impl Eq for CubeVar {}

        impl PartialEq for CubeVar {
            fn eq(&self, other: &Self) -> bool {
                self.1 == other.1
            }
        }

        let k = usize::from(k.into_inner());

        let mut queue = BinaryHeap::with_capacity(k);
        #[allow(clippy::cast_possible_truncation)]
        queue.push(CubeVar(
            Cube { min: [0; N], max: [B as u8; N] },
            f64::INFINITY,
        ));

        while queue.len() < k {
            // there should always be one cube, since at least one cube is added back for each popped
            #[allow(clippy::expect_used)]
            let CubeVar(mut cube1, variance) = queue.pop().expect("at least one cube");

            if variance <= 0.0 {
                // all cubes cannot be cut further
                queue.push(CubeVar(cube1, 0.0));
                break;
            }

            if let Some(cube2) = self.cut(&mut cube1) {
                queue.push(CubeVar(cube1, self.variance(cube1)));
                queue.push(CubeVar(cube2, self.variance(cube2)));
            } else {
                queue.push(CubeVar(cube1, 0.0));
            }
        }

        queue.into_iter().map(|x| x.0)
    }

    /// Returns the average color of and the number of colors in the given cube.
    fn cube_color_and_count(&self, cube: Cube) -> (Color, u32) {
        let Stats { count, components, .. } = self.hist.volume(cube);
        debug_assert!(count > 0);
        let n = count.into();
        let color = cast::from_array(
            components.map(|c| <Component::Sum as AsPrimitive<Component>>::as_(c / n)),
        );
        (color, count)
    }

    /// Computes the color palette.
    fn palette(&self, k: PaletteSize) -> QuantizeOutput<Color> {
        let (palette, counts) = self
            .cubes(k)
            .map(|cube| self.cube_color_and_count(cube))
            .unzip();

        QuantizeOutput { palette, counts, indices: Vec::new() }
    }

    /// Computes the color palette and the color for each histogram bin.
    fn palette_and_lookup(&self, k: PaletteSize) -> (Vec<Color>, Vec<u32>, Box<Histogram3<u8, B>>) {
        let mut lookup = Histogram3::box_zeroed();

        let (colors, counts) = self
            .cubes(k)
            .enumerate()
            .map(|(i, cube)| {
                let Cube { min, max } = cube;
                #[allow(clippy::cast_possible_truncation)]
                let i = i as u8;
                for r in min[0]..max[0] {
                    for g in min[1]..max[1] {
                        for b in min[2]..max[2] {
                            lookup[[r, g, b]] = i;
                        }
                    }
                }

                self.cube_color_and_count(cube)
            })
            .unzip();

        (colors, counts, lookup)
    }
}

impl<'a, Color, Component, Binner, const B: usize, ColorCount>
    Wu3<'a, Color, Component, Binner, B, ColorCount>
where
    Color: ColorComponents<Component, N>,
    Component: SumPromotion<u32>,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64>,
    Binner: Binner3<Component, B>,
    ColorCount: ColorCountsRemap<Color, Component, N>,
    u32: Into<Component::Sum>,
{
    /// Computes the color palette and indices into it.
    fn indexed_palette(&self, k: PaletteSize) -> QuantizeOutput<Color> {
        let (palette, counts, lookup) = self.palette_and_lookup(k);

        let indices = self
            .color_counts
            .color_components()
            .iter()
            .map(|&color| lookup[self.binner.bin(color)])
            .collect();

        let indices = self.color_counts.map_indices(indices);

        QuantizeOutput { palette, counts, indices }
    }
}

#[cfg(feature = "threads")]
impl<'a, Color, Component, Binner, const B: usize, ColorCount>
    Wu3<'a, Color, Component, Binner, B, ColorCount>
where
    Color: ColorComponents<Component, N> + Send,
    Component: SumPromotion<u32> + Sync,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64> + Send,
    Binner: Binner3<Component, B> + Sync,
    ColorCount: ColorCounts<Color, Component, N> + Send + Sync,
    u32: Into<Component::Sum>,
{
    /// Creates a new [`Wu3`] in parallel with histogram data filled by the given `color_counts`.
    fn new_par(color_counts: &'a ColorCount, binner: &'a Binner) -> Self {
        let chunk_size = color_counts.len().div_ceil(rayon::current_num_threads());
        let mut data = if let Some(counts) = color_counts.counts() {
            color_counts
                .color_components()
                .par_chunks(chunk_size)
                .zip(counts.par_chunks(chunk_size))
                .map(|(colors, counts)| {
                    let mut data = Self::new_zero(color_counts, binner);
                    data.add_color_counts(colors, counts);
                    data
                })
                .reduce_with(Self::merge_partial)
        } else {
            color_counts
                .color_components()
                .par_chunks(chunk_size)
                .map(|colors| {
                    let mut data = Self::new_zero(color_counts, binner);
                    data.add_colors(colors);
                    data
                })
                .reduce_with(Self::merge_partial)
        }
        .unwrap_or_else(|| Self::new_zero(color_counts, binner));

        data.calc_cumulative_moments();

        data
    }

    /// Merges multiple [`Wu3`]s together by element-wise summing their histogram bins together
    /// and then computing cumulative moments on the final histogram.
    #[allow(clippy::needless_pass_by_value)]
    fn merge_partial(mut self, other: Self) -> Self {
        for x in 0..B {
            for y in 0..B {
                for z in 0..B {
                    self.hist[[x, y, z]] += other.hist[[x, y, z]];
                }
            }
        }
        self
    }
}

#[cfg(feature = "threads")]
impl<'a, Color, Component, Binner, const B: usize, ColorCount>
    Wu3<'a, Color, Component, Binner, B, ColorCount>
where
    Color: ColorComponents<Component, N>,
    Component: SumPromotion<u32> + Sync,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64> + Send,
    Binner: Binner3<Component, B> + Sync,
    ColorCount: ColorCountsParallelRemap<Color, Component, N> + Send + Sync,
    u32: Into<Component::Sum>,
{
    /// Computes the color palette and indices into it in parallel.
    fn indexed_palette_par(&self, k: PaletteSize) -> QuantizeOutput<Color> {
        let (palette, counts, lookup) = self.palette_and_lookup(k);

        let indices = self
            .color_counts
            .color_components()
            .par_iter()
            .map(|&color| lookup[self.binner.bin(color)])
            .collect();

        let indices = self.color_counts.map_indices_par(indices);

        QuantizeOutput { palette, counts, indices }
    }
}

/// Computes a color palette from the given `color_counts`
/// with at most `palette_size` entries and using the given `binner`.
pub fn palette<Color, Component, const B: usize>(
    color_counts: &impl ColorCounts<Color, Component, N>,
    palette_size: PaletteSize,
    binner: &impl Binner3<Component, B>,
) -> QuantizeOutput<Color>
where
    Color: ColorComponents<Component, N>,
    Component: SumPromotion<u32>,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64>,
    u32: Into<Component::Sum>,
{
    if palette_size.into_inner() == 0 || color_counts.is_empty() {
        QuantizeOutput::default()
    } else {
        Wu3::new(color_counts, binner).palette(palette_size)
    }
}

/// Computes a color palette from the given `color_counts`
/// with at most `palette_size` entries and using the given `binner`.
/// The returned [`QuantizeOutput`] will have its `indices` populated.
pub fn indexed_palette<Color, Component, const B: usize>(
    color_counts: &impl ColorCountsRemap<Color, Component, N>,
    palette_size: PaletteSize,
    binner: &impl Binner3<Component, B>,
) -> QuantizeOutput<Color>
where
    Color: ColorComponents<Component, N>,
    Component: SumPromotion<u32>,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64>,
    u32: Into<Component::Sum>,
{
    if palette_size.into_inner() == 0 || color_counts.is_empty() {
        QuantizeOutput::default()
    } else {
        Wu3::new(color_counts, binner).indexed_palette(palette_size)
    }
}

/// Computes a color palette in parallel from the given `color_counts`
/// with at most `palette_size` entries and using the given `binner`.
#[cfg(feature = "threads")]
pub fn palette_par<Color, Component, const B: usize>(
    color_counts: &(impl ColorCounts<Color, Component, N> + Send + Sync),
    palette_size: PaletteSize,
    binner: &(impl Binner3<Component, B> + Sync),
) -> QuantizeOutput<Color>
where
    Color: ColorComponents<Component, N> + Send,
    Component: SumPromotion<u32> + Sync,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64> + Send,
    u32: Into<Component::Sum>,
{
    if palette_size.into_inner() == 0 || color_counts.is_empty() {
        QuantizeOutput::default()
    } else {
        Wu3::new_par(color_counts, binner).palette(palette_size)
    }
}

/// Computes a color palette in parallel from the given `color_counts`
/// with at most `palette_size` entries and using the given `binner`.
/// The returned [`QuantizeOutput`] will have its `indices` populated.
#[cfg(feature = "threads")]
pub fn indexed_palette_par<Color, Component, const B: usize>(
    color_counts: &(impl ColorCountsParallelRemap<Color, Component, N> + Send + Sync),
    palette_size: PaletteSize,
    binner: &(impl Binner3<Component, B> + Sync),
) -> QuantizeOutput<Color>
where
    Color: ColorComponents<Component, N> + Send,
    Component: SumPromotion<u32> + Sync,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64> + Send,
    u32: Into<Component::Sum>,
{
    if palette_size.into_inner() == 0 || color_counts.is_empty() {
        QuantizeOutput::default()
    } else {
        Wu3::new_par(color_counts, binner).indexed_palette_par(palette_size)
    }
}

/// A binner for colors with components that are unsigned integer types.
///
/// `B` is the number of bins to have in each dimension
/// and must be a power of 2 less than or equal to `256`.
#[derive(Debug, Clone, Copy, Default)]
pub struct UIntBinner<const B: usize>;

impl<const B: usize> Binner3<u8, B> for UIntBinner<B> {
    #[inline]
    fn bin(&self, components: [u8; N]) -> [u8; N] {
        assert!(B.is_power_of_two());
        let bits: u32 = B.ilog2();
        assert!(bits <= u8::BITS);
        components.map(|c| c >> (u8::BITS - bits))
    }
}

impl<const B: usize> Binner3<u16, B> for UIntBinner<B> {
    #[inline]
    fn bin(&self, components: [u16; N]) -> [u8; N] {
        assert!(B.is_power_of_two());
        let bits: u32 = B.ilog2();
        assert!(bits <= u8::BITS);
        #[allow(clippy::cast_possible_truncation)]
        components.map(|c| (c >> (u16::BITS - bits)) as u8)
    }
}

/// A binner for colors with components that are floating point types.
///
/// `B` is the number of bins to have in each dimension.
#[derive(Debug, Clone, Copy)]
pub struct FloatBinner<F: Float, const B: usize> {
    /// The minimum values in each dimension.
    mins: [F; N],
    /// The widths of the bins in each dimension.
    steps: [F; N],
}

impl<F, const B: usize> FloatBinner<F, B>
where
    F: Float + AsPrimitive<u8> + 'static,
    usize: AsPrimitive<F>,
{
    /// Creates a new [`FloatBinner`] from the given ranges of values for each component.
    ///
    /// Each range should be of the form `(min_value, max_value)`.
    pub fn new(ranges: [(F, F); N]) -> Self {
        Self {
            mins: ranges.map(|(low, _)| low),
            steps: ranges.map(|(low, high)| (high - low) / B.as_()),
        }
    }
}

impl<F, const B: usize> Binner3<F, B> for FloatBinner<F, B>
where
    F: Float + AsPrimitive<u8> + 'static,
{
    #[inline]
    fn bin(&self, components: [F; N]) -> [u8; N] {
        let Self { mins, steps } = self;

        let mut index = [0; N];
        #[allow(clippy::cast_possible_truncation)]
        for c in 0..N {
            index[c] = ((components[c] - mins[c]) / steps[c])
                .as_()
                .min(B as u8 - 1);
        }

        index
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{tests::*, ColorSlice};
    use palette::Srgb;

    fn assert_indices_count(output: &QuantizeOutput<Srgb<u8>>) {
        let mut counts = vec![0; output.palette.len()];
        for &i in &output.indices {
            counts[usize::from(i)] += 1;
        }
        assert_eq!(counts, output.counts);
    }

    #[test]
    fn empty_input() {
        let expected = QuantizeOutput::default();

        let colors = ColorSlice::<Srgb<u8>>::new_unchecked(&[]);
        let palette_size = PaletteSize::MAX;
        let binner = UIntBinner::<32>;

        let actual = palette(&colors, palette_size, &binner);
        assert_eq!(actual, expected);

        let actual = indexed_palette(&colors, palette_size, &binner);
        assert_eq!(actual, expected);

        #[cfg(feature = "threads")]
        {
            let actual = palette_par(&colors, palette_size, &binner);
            assert_eq!(actual, expected);

            let actual = indexed_palette_par(&colors, palette_size, &binner);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn not_enough_colors() {
        let len = 64;
        let colors = &test_data_1024()[..len];
        let colors = ColorSlice::try_from(colors).unwrap();
        let palette_size = PaletteSize::MAX;
        let binner = UIntBinner::<32>;

        let result = palette(&colors, palette_size, &binner);
        assert_eq!(len, result.palette.len());
        assert_eq!(result.counts.into_iter().sum::<u32>(), colors.num_colors());

        let result = indexed_palette(&colors, palette_size, &binner);
        assert_eq!(len, result.palette.len());
        assert_indices_count(&result);
        assert_eq!(result.counts.into_iter().sum::<u32>(), colors.num_colors());

        #[cfg(feature = "threads")]
        {
            let result = palette_par(&colors, palette_size, &binner);
            assert_eq!(len, result.palette.len());
            assert_eq!(result.counts.into_iter().sum::<u32>(), colors.num_colors());

            let result = indexed_palette_par(&colors, palette_size, &binner);
            assert_eq!(len, result.palette.len());
            assert_indices_count(&result);
            assert_eq!(result.counts.into_iter().sum::<u32>(), colors.num_colors());
        }
    }

    #[test]
    fn exact_match_image_unaffected() {
        const COUNT: u32 = 4;

        fn reorder_output(mut output: QuantizeOutput<Srgb<u8>>) -> QuantizeOutput<Srgb<u8>> {
            let mut palette_reorder = output
                .palette
                .iter()
                .copied()
                .enumerate()
                .collect::<Vec<_>>();

            palette_reorder.sort_by_key(|(_, srgb)| srgb.into_components());
            let (reorder, palette): (Vec<_>, Vec<_>) = palette_reorder.into_iter().unzip();

            output.palette = palette;

            let mut remap = vec![0; output.palette.len()];
            for (new_index, old_index) in reorder.into_iter().enumerate() {
                remap[old_index] = new_index;
            }

            #[allow(clippy::cast_possible_truncation)]
            for i in &mut output.indices {
                *i = remap[usize::from(*i)] as u8;
            }

            output
        }

        let expected_palette = {
            let mut palette = test_data_256();
            palette.sort_by_key(|srgb| srgb.into_components());
            palette
        };

        let indices = {
            #[allow(clippy::cast_possible_truncation)]
            let indices = (0..expected_palette.len())
                .map(|i| i as u8)
                .collect::<Vec<_>>();
            let mut indices = [indices.as_slice(); COUNT as usize].concat();
            indices.rotate_right(7);
            indices
        };

        let colors = indices
            .iter()
            .map(|&i| expected_palette[usize::from(i)])
            .collect::<Vec<_>>();

        let colors = &ColorSlice::try_from(colors.as_slice()).unwrap();
        let palette_size = PaletteSize::MAX;
        let binner = UIntBinner::<32>;

        let actual = {
            let mut result = palette(colors, palette_size, &binner);
            result.palette.sort_by_key(|srgb| srgb.into_components());
            result
        };
        let expected = QuantizeOutput {
            palette: expected_palette.clone(),
            counts: vec![COUNT; expected_palette.len()],
            indices: Vec::new(),
        };
        assert_eq!(actual, expected);
        assert_eq!(actual.counts.into_iter().sum::<u32>(), colors.num_colors());

        let actual = reorder_output(indexed_palette(colors, palette_size, &binner));
        let expected = QuantizeOutput { indices: indices.clone(), ..expected };
        assert_eq!(actual, expected);
        assert_indices_count(&actual);
        assert_eq!(actual.counts.into_iter().sum::<u32>(), colors.num_colors());

        #[cfg(feature = "threads")]
        {
            let actual = {
                let mut result = palette_par(colors, palette_size, &binner);
                result.palette.sort_by_key(|srgb| srgb.into_components());
                result
            };
            let expected = QuantizeOutput {
                palette: expected_palette.clone(),
                counts: vec![COUNT; expected_palette.len()],
                indices: Vec::new(),
            };
            assert_eq!(actual, expected);
            assert_eq!(actual.counts.into_iter().sum::<u32>(), colors.num_colors());

            let actual = reorder_output(indexed_palette_par(colors, palette_size, &binner));
            let expected = QuantizeOutput { indices, ..expected };
            assert_eq!(actual, expected);
            assert_indices_count(&actual);
            assert_eq!(actual.counts.into_iter().sum::<u32>(), colors.num_colors());
        }
    }

    #[test]
    #[cfg(feature = "threads")]
    fn single_and_multi_threaded_match() {
        let colors = test_data_1024();
        let colors = ColorSlice::try_from(colors.as_slice()).unwrap();
        let palette_size = PaletteSize::MAX;
        let binner = UIntBinner::<32>;

        let wu_single = Wu3::new(&colors, &binner);
        let wu_par = Wu3::new_par(&colors, &binner);

        for (a, b) in wu_single.hist.0.iter().zip(&wu_par.hist.0) {
            for (a, b) in a.iter().zip(b) {
                for (a, b) in a.iter().zip(b) {
                    assert_eq!(a.count, b.count);
                    assert_eq!(a.components, b.components);
                    #[allow(clippy::float_cmp)]
                    {
                        assert_eq!(a.sum_squared, b.sum_squared);
                    }
                }
            }
        }

        let single = palette(&colors, palette_size, &binner);
        let par = palette_par(&colors, palette_size, &binner);
        assert_eq!(single, par);
        assert_eq!(single.counts.into_iter().sum::<u32>(), colors.num_colors());

        let single = indexed_palette(&colors, palette_size, &binner);
        let par = indexed_palette_par(&colors, palette_size, &binner);
        assert_eq!(single, par);
        assert_indices_count(&single);
        assert_eq!(single.counts.into_iter().sum::<u32>(), colors.num_colors());
    }
}
