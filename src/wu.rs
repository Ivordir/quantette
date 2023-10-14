// Referenced code: https://www.ece.mcmaster.ca/~xwu/cq.c
// and relevant paper (free access):
// Xiaolin Wu, Color quantization by dynamic programming and principal analysis,
// ACM Transactions on Graphics, vol. 11, no. 4, 348–372, 1992.
// https://doi.org/10.1145/146443.146475

use crate::{
    ColorAndFrequency, ColorComponents, ColorRemap, PaletteSize, QuantizeOutput, SumPromotion,
    ZeroedIsZero,
};

use std::{
    array,
    collections::BinaryHeap,
    marker::PhantomData,
    ops::{Add, AddAssign, Index, IndexMut, Sub},
};

use num_traits::{AsPrimitive, Float, Zero};
use ordered_float::OrderedFloat;
use palette::cast;

#[cfg(feature = "threads")]
use crate::ParallelColorRemap;
#[cfg(feature = "threads")]
use rayon::prelude::*;

const N: usize = 3;

#[derive(Clone, Copy, Default)]
struct Cube {
    min: [u8; N],
    max: [u8; N],
}

impl Cube {
    fn is_single_bin(self) -> bool {
        let Self { min, max } = self;
        (0..N).all(|c| max[c] - min[c] == 1)
    }
}

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

#[derive(Clone, Copy)]
#[repr(align(64))]
struct Stats<T, const N: usize> {
    count: u32,
    components: [T; N],
    squared: f64,
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
            squared: self.squared + rhs.squared,
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
            squared: self.squared - rhs.squared,
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
        self.squared += rhs.squared;
    }
}

impl<T: Copy + Zero, const N: usize> Zero for Stats<T, N> {
    fn zero() -> Self {
        Self {
            count: 0,
            components: [T::zero(); N],
            squared: 0.0,
        }
    }

    fn is_zero(&self) -> bool {
        self.count == 0 && self.squared == 0.0 && self.components.iter().all(Zero::is_zero)
    }
}

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

impl<T: Copy + Zero + Add<Output = T> + Sub<Output = T>, const B: usize> Histogram3<T, B> {
    fn volume(&self, Cube { min, max }: Cube) -> T {
        let mut index = [0u8; N];
        ndvolume!(self, min, max, index; 0, 1, 2)
    }

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

pub trait Binner3<T, const B: usize> {
    fn bin(&self, components: [T; N]) -> [u8; N];
}

struct Wu3<'a, Color, Component, Binner, const B: usize, ColorFreq>
where
    Color: ColorComponents<Component, N>,
    Component: SumPromotion<u32>,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64>,
    Binner: Binner3<Component, B>,
    ColorFreq: ColorAndFrequency<Color, Component, N>,
    u32: Into<Component::Sum>,
{
    _phantom: PhantomData<Color>,
    color_counts: &'a ColorFreq,
    binner: &'a Binner,
    hist: Box<Histogram3<Stats<Component::Sum, N>, B>>,
}

impl<'a, Color, Component, Binner, const B: usize, ColorFreq>
    Wu3<'a, Color, Component, Binner, B, ColorFreq>
where
    Color: ColorComponents<Component, N>,
    Component: SumPromotion<u32>,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64>,
    Binner: Binner3<Component, B>,
    ColorFreq: ColorAndFrequency<Color, Component, N>,
    u32: Into<Component::Sum>,
{
    fn new(color_counts: &'a ColorFreq, binner: &'a Binner) -> Self {
        assert!((1..=256).contains(&B));

        Self {
            _phantom: PhantomData,
            color_counts,
            binner,
            hist: ZeroedIsZero::box_zeroed(),
        }
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn add_color(&mut self, color: [Component; N], n: u32) {
        let Stats { count, components, squared } = &mut self.hist[self.binner.bin(color)];
        let color = color.map(Into::into);

        *count += n;

        let w: Component::Sum = n.into();
        for (c, v) in components.iter_mut().zip(color) {
            *c += w * v;
        }

        let w = f64::from(n);
        *squared += w * Self::sum_of_squares(color);
    }

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

    fn from_color_counts(color_counts: &'a ColorFreq, binner: &'a Binner) -> Self {
        let mut data = Self::new(color_counts, binner);

        if let Some(counts) = data.color_counts.counts() {
            data.add_color_counts(data.color_counts.color_components(), counts);
        } else {
            data.add_colors(data.color_counts.color_components());
        }

        data.calc_cumulative_moments();

        data
    }

    fn calc_cumulative_moments(&mut self) {
        let hist = &mut self.hist;

        for r in 0..B {
            let mut area = [Stats::zero(); B];

            for g in 0..B {
                let mut line = Stats::zero();

                #[allow(clippy::needless_range_loop)]
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

    #[inline]
    fn sum_of_squares(components: [Component::Sum; N]) -> f64 {
        let mut square = 0.0;
        for c in components {
            let c = c.as_();
            square += c * c;
        }
        square
    }

    fn variance(&self, cube: Cube) -> f64 {
        if cube.is_single_bin() {
            0.0
        } else {
            let Stats { count, components, squared } = self.hist.volume(cube);
            squared - Self::sum_of_squares(components) / f64::from(count)
        }
    }

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

    fn cubes(&self, k: PaletteSize) -> impl Iterator<Item = Cube> {
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
            #[allow(clippy::unwrap_used)]
            let CubeVar(mut cube1, variance) = queue.pop().unwrap();

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

    fn cube_color_and_count(&self, cube: Cube) -> (Color, u32) {
        let Stats { count, components, .. } = self.hist.volume(cube);
        debug_assert!(count > 0);
        let n = count.into();
        let color = cast::from_array(
            components.map(|c| <Component::Sum as AsPrimitive<Component>>::as_(c / n)),
        );
        (color, count)
    }

    fn palette(&self, k: PaletteSize) -> QuantizeOutput<Color> {
        let (palette, counts) = self
            .cubes(k)
            .map(|cube| self.cube_color_and_count(cube))
            .unzip();

        QuantizeOutput { palette, counts, indices: Vec::new() }
    }

    fn quantize_and_lookup(
        &self,
        k: PaletteSize,
    ) -> (Vec<Color>, Vec<u32>, Box<Histogram3<u8, B>>) {
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

impl<'a, Color, Component, Binner, const B: usize, ColorFreq>
    Wu3<'a, Color, Component, Binner, B, ColorFreq>
where
    Color: ColorComponents<Component, N>,
    Component: SumPromotion<u32>,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64>,
    Binner: Binner3<Component, B>,
    ColorFreq: ColorAndFrequency<Color, Component, N> + ColorRemap,
    u32: Into<Component::Sum>,
{
    fn indexed_palette(&self, k: PaletteSize) -> QuantizeOutput<Color> {
        let (palette, counts, lookup) = self.quantize_and_lookup(k);

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
impl<'a, Color, Component, Binner, const B: usize, ColorFreq>
    Wu3<'a, Color, Component, Binner, B, ColorFreq>
where
    Color: ColorComponents<Component, N> + Send,
    Component: SumPromotion<u32> + Sync,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64> + Send,
    Binner: Binner3<Component, B> + Sync,
    ColorFreq: ColorAndFrequency<Color, Component, N> + Send + Sync,
    u32: Into<Component::Sum>,
{
    fn from_color_counts_par(color_counts: &'a ColorFreq, binner: &'a Binner) -> Self {
        let chunk_size = color_counts.len().div_ceil(rayon::current_num_threads());
        let partials = if let Some(counts) = color_counts.counts() {
            color_counts
                .color_components()
                .par_chunks(chunk_size)
                .zip(counts.par_chunks(chunk_size))
                .map(|(colors, counts)| {
                    let mut data = Self::new(color_counts, binner);
                    data.add_color_counts(colors, counts);
                    data
                })
                .collect()
        } else {
            color_counts
                .color_components()
                .par_chunks(chunk_size)
                .map(|colors| {
                    let mut data = Self::new(color_counts, binner);
                    data.add_colors(colors);
                    data
                })
                .collect()
        };

        Self::merge_partials(partials, color_counts, binner)
    }

    fn merge_partials(
        mut partials: Vec<Self>,
        color_counts: &'a ColorFreq,
        binner: &'a Binner,
    ) -> Self {
        let mut data = partials
            .pop()
            .unwrap_or_else(|| Self::new(color_counts, binner));

        for other in partials {
            for x in 0..B {
                for y in 0..B {
                    for z in 0..B {
                        data.hist[[x, y, z]] += other.hist[[x, y, z]];
                    }
                }
            }
        }

        data.calc_cumulative_moments();

        data
    }
}

#[cfg(feature = "threads")]
impl<'a, Color, Component, Binner, const B: usize, ColorFreq>
    Wu3<'a, Color, Component, Binner, B, ColorFreq>
where
    Color: ColorComponents<Component, N>,
    Component: SumPromotion<u32> + Sync,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64> + Send,
    Binner: Binner3<Component, B> + Sync,
    ColorFreq: ColorAndFrequency<Color, Component, N> + ParallelColorRemap + Send + Sync,
    u32: Into<Component::Sum>,
{
    fn quantize_par(&self, k: PaletteSize) -> QuantizeOutput<Color> {
        let (palette, counts, lookup) = self.quantize_and_lookup(k);

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

pub fn palette<Color, Component, const B: usize>(
    color_counts: &impl ColorAndFrequency<Color, Component, N>,
    k: PaletteSize,
    binner: &impl Binner3<Component, B>,
) -> QuantizeOutput<Color>
where
    Color: ColorComponents<Component, N>,
    Component: SumPromotion<u32>,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64>,
    u32: Into<Component::Sum>,
{
    if color_counts.num_colors() <= u32::from(k.into_inner()) {
        QuantizeOutput::trivial_palette(color_counts)
    } else {
        Wu3::from_color_counts(color_counts, binner).palette(k)
    }
}

pub fn indexed_palette<Color, Component, const B: usize>(
    color_counts: &(impl ColorAndFrequency<Color, Component, N> + ColorRemap),
    k: PaletteSize,
    binner: &impl Binner3<Component, B>,
) -> QuantizeOutput<Color>
where
    Color: ColorComponents<Component, N>,
    Component: SumPromotion<u32>,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64>,
    u32: Into<Component::Sum>,
{
    if color_counts.num_colors() <= u32::from(k.into_inner()) {
        QuantizeOutput::trivial_quantize(color_counts)
    } else {
        Wu3::from_color_counts(color_counts, binner).indexed_palette(k)
    }
}

#[cfg(feature = "threads")]
pub fn palette_par<Color, Component, const B: usize>(
    color_counts: &(impl ColorAndFrequency<Color, Component, N> + Send + Sync),
    k: PaletteSize,
    binner: &(impl Binner3<Component, B> + Sync),
) -> QuantizeOutput<Color>
where
    Color: ColorComponents<Component, N> + Send,
    Component: SumPromotion<u32> + Sync,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64> + Send,
    u32: Into<Component::Sum>,
{
    if color_counts.num_colors() <= u32::from(k.into_inner()) {
        QuantizeOutput::trivial_palette(color_counts)
    } else {
        Wu3::from_color_counts_par(color_counts, binner).palette(k)
    }
}

#[cfg(feature = "threads")]
pub fn indexed_palette_par<Color, Component, const B: usize>(
    color_counts: &(impl ColorAndFrequency<Color, Component, N> + ParallelColorRemap + Send + Sync),
    k: PaletteSize,
    binner: &(impl Binner3<Component, B> + Sync),
) -> QuantizeOutput<Color>
where
    Color: ColorComponents<Component, N> + Send,
    Component: SumPromotion<u32> + Sync,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64> + Send,
    u32: Into<Component::Sum>,
{
    if color_counts.num_colors() <= u32::from(k.into_inner()) {
        QuantizeOutput::trivial_quantize_par(color_counts)
    } else {
        Wu3::from_color_counts_par(color_counts, binner).quantize_par(k)
    }
}

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
        assert!((u8::BITS..=u16::BITS).contains(&bits));
        #[allow(clippy::cast_possible_truncation)]
        components.map(|c| (c >> (u16::BITS - bits)) as u8)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FloatBinner<F: Float, const B: usize> {
    mins: [F; N],
    steps: [F; N],
}

impl<F, const B: usize> FloatBinner<F, B>
where
    F: Float + AsPrimitive<u8> + 'static,
    usize: AsPrimitive<F>,
{
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
