//! Contains the code for color/pixel deduplication and associated traits and types.

use crate::{ColorComponents, ColorSlice, ZeroedIsZero};
use bitvec::vec::BitVec;
use palette::cast::{self, AsArrays};
#[cfg(feature = "threads")]
use rayon::prelude::*;
use std::{marker::PhantomData, ops::Range};
#[cfg(feature = "image")]
use {crate::AboveMaxLen, image::RgbImage, palette::Srgb};

/// A generalization trait over regular [`ColorSlice`]s and deduplicated pixels like [`UniqueColorCounts`].
pub trait ColorCounts<Color, Component, const N: usize>
where
    Color: ColorComponents<Component, N>,
{
    /// The slice of colors to quantize.
    ///
    /// The colors need not be unique,
    /// but the length of this slice must not be greater than [`MAX_PIXELS`](crate::MAX_PIXELS).
    fn colors(&self) -> &[Color];

    /// The total number of pixels/colors in the (original) color slice.
    ///
    /// For [`ColorSlice`]s, this is simply the length of the slice.
    /// For deduplicated pixels like [`UniqueColorCounts`],
    /// this is the length of the input [`ColorSlice`] before deduplication.
    /// This must be equal to the sum of `counts` (or `num_colors` if `counts` is `None`).
    fn total_count(&self) -> u32;

    /// The number of pixels corresponding to each `Color` in the slice returned by `colors`.
    ///
    /// For [`ColorSlice`]s, this returns `None`, indicating each `Color` has a count of `1`.
    /// For deduplicated pixels like [`UniqueColorCounts`],
    /// each count indicates the number times each unique color was present in the original color slice.
    ///
    /// Each count must be nonzero, and the length of the returned slice (if any)
    /// must have the same length as the slice returned by `colors`.
    /// As such, the length of this slice must also not be greater than [`MAX_PIXELS`](crate::MAX_PIXELS).
    fn counts(&self) -> Option<&[u32]>;

    /// A slice of indices into the color slice returned by `colors`.
    /// This is used to retain the original color slice/image after deduplication.
    ///
    /// The length of this slice must not be greater than [`MAX_PIXELS`](crate::MAX_PIXELS),
    /// and each index must be valid index into the slice returned by `colors`.
    fn indices(&self) -> Option<&[u32]>;

    /// The slice returned by `colors` casted to a slice of component arrays.
    fn color_components(&self) -> &[[Component; N]] {
        self.colors().as_arrays()
    }

    /// The length of the slice returned by `colors` as a `u32`.
    #[allow(clippy::cast_possible_truncation)]
    fn num_colors(&self) -> u32 {
        self.len() as u32
    }

    /// The length of the slice returned by `colors`.
    fn len(&self) -> usize {
        self.colors().len()
    }

    /// Whether or not the slice returned by `colors` is empty.
    fn is_empty(&self) -> bool {
        self.colors().is_empty()
    }
}

impl<'a, Color, Component, const N: usize> ColorCounts<Color, Component, N>
    for ColorSlice<'a, Color>
where
    Color: ColorComponents<Component, N>,
{
    fn colors(&self) -> &[Color] {
        self
    }

    fn total_count(&self) -> u32 {
        self.num_colors()
    }

    fn counts(&self) -> Option<&[u32]> {
        None
    }

    fn indices(&self) -> Option<&[u32]> {
        None
    }

    fn num_colors(&self) -> u32 {
        self.num_colors()
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn is_empty(&self) -> bool {
        self.as_ref().is_empty()
    }
}

impl<Color, Component, const N: usize> ColorCounts<Color, Component, N>
    for UniqueColorCounts<Color, Component, N>
where
    Color: ColorComponents<Component, N>,
{
    fn colors(&self) -> &[Color] {
        &self.colors
    }

    fn counts(&self) -> Option<&[u32]> {
        Some(&self.counts)
    }

    fn indices(&self) -> Option<&[u32]> {
        None
    }

    fn total_count(&self) -> u32 {
        self.total_count
    }
}

impl<Color, Component, const N: usize> ColorCounts<Color, Component, N>
    for IndexedColorCounts<Color, Component, N>
where
    Color: ColorComponents<Component, N>,
{
    fn colors(&self) -> &[Color] {
        &self.colors
    }

    fn counts(&self) -> Option<&[u32]> {
        Some(&self.counts)
    }

    fn indices(&self) -> Option<&[u32]> {
        Some(&self.indices)
    }

    fn total_count(&self) -> u32 {
        self.total_count
    }
}

/// Types that allow reconstructing the original image/color slice from a `Vec` of indices
/// into a color palette.
pub trait ColorCountsRemap<Color, Component, const N: usize>:
    ColorCounts<Color, Component, N>
where
    Color: ColorComponents<Component, N>,
{
    /// Uses the given `Vec` of indices to create indices for each pixel/color
    /// in the original image/color slice.
    ///
    /// The given `Vec` will have the same length as the color slice returned by `colors`.
    ///
    /// The length of the returned `Vec` must not be greater than [`MAX_PIXELS`](crate::MAX_PIXELS).
    fn map_indices(&self, indices: Vec<u8>) -> Vec<u8>;
}

impl<'a, Color, Component, const N: usize> ColorCountsRemap<Color, Component, N>
    for ColorSlice<'a, Color>
where
    Color: ColorComponents<Component, N>,
{
    fn map_indices(&self, indices: Vec<u8>) -> Vec<u8> {
        indices
    }
}

impl<Color, Component, const N: usize> ColorCountsRemap<Color, Component, N>
    for IndexedColorCounts<Color, Component, N>
where
    Color: ColorComponents<Component, N>,
{
    fn map_indices(&self, indices: Vec<u8>) -> Vec<u8> {
        let indices = indices.as_slice(); // faster for some reason???
        self.indices.iter().map(|&i| indices[i as usize]).collect()
    }
}

/// Types that allow reconstructing the original image/color slice in parallel
/// from a `Vec` of indices into a color palette.
#[cfg(feature = "threads")]
pub trait ColorCountsParallelRemap<Color, Component, const N: usize>:
    ColorCounts<Color, Component, N>
where
    Color: ColorComponents<Component, N>,
{
    /// Uses the given `Vec` of indices to create indices, in parallel, for each pixel/color
    /// in the original image/color slice.
    ///
    /// The given `Vec` will have the same length as the color slice returned by `colors`.
    ///
    /// The length of the returned `Vec` must not be greater than [`MAX_PIXELS`](crate::MAX_PIXELS).
    fn map_indices_par(&self, indices: Vec<u8>) -> Vec<u8>;
}

#[cfg(feature = "threads")]
impl<'a, Color, Component, const N: usize> ColorCountsParallelRemap<Color, Component, N>
    for ColorSlice<'a, Color>
where
    Color: ColorComponents<Component, N>,
{
    fn map_indices_par(&self, indices: Vec<u8>) -> Vec<u8> {
        indices
    }
}

#[cfg(feature = "threads")]
impl<Color, Component, const N: usize> ColorCountsParallelRemap<Color, Component, N>
    for IndexedColorCounts<Color, Component, N>
where
    Color: ColorComponents<Component, N>,
{
    fn map_indices_par(&self, indices: Vec<u8>) -> Vec<u8> {
        let indices = indices.as_slice(); // faster for some reason???
        self.indices
            .par_iter()
            .map(|&i| indices[i as usize])
            .collect()
    }
}

/// A byte-sized Radix
const RADIX: usize = u8::MAX as usize + 1;

/// Returns the range associated with the `i`-th chunk.
#[inline]
fn chunk_range(chunks: &[u32], i: usize) -> Range<usize> {
    (chunks[i] as usize)..(chunks[i + 1] as usize)
}

/// Count the first byte component in the given color slice.
fn u8_counts<Color, const CHUNKS: usize>(pixels: &[Color], counts: &mut [[u32; RADIX + 1]; CHUNKS])
where
    Color: ColorComponents<u8, 3>,
{
    for color in pixels.as_arrays().chunks_exact(CHUNKS) {
        for (counts, &[r, ..]) in counts.iter_mut().zip(color) {
            counts[usize::from(r)] += 1;
        }
    }

    for &[r, ..] in pixels.as_arrays().chunks_exact(CHUNKS).remainder() {
        counts[0][usize::from(r)] += 1;
    }

    #[allow(clippy::expect_used)]
    let (counts, partial_counts) = counts.split_first_mut().expect("CHUNKS != 0");
    for i in 0..RADIX {
        for partial in &*partial_counts {
            counts[i] += partial[i];
        }
    }
}

/// Computes the prefix sum of the array in place.
#[inline]
fn prefix_sum<const M: usize>(counts: &mut [u32; M]) {
    for i in 1..M {
        counts[i] += counts[i - 1];
    }
}

/// Deduplicated colors and their frequency counts.
///
/// Currently, only colors that implement [`ColorComponents<u8, 3>`] are supported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniqueColorCounts<Color, Component, const N: usize>
where
    Color: ColorComponents<Component, N>,
{
    /// The component type must remain the same for each [`UniqueColorCounts`].
    _phantom: PhantomData<Component>,
    /// The unique colors.
    colors: Vec<Color>,
    /// The number of times each color was present in the original color slice/image.
    counts: Vec<u32>,
    /// The total number of pixels/colors in the original color slice/image.
    total_count: u32,
}

impl<Color, Component, const N: usize> Default for UniqueColorCounts<Color, Component, N>
where
    Color: ColorComponents<Component, N>,
{
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
            colors: Vec::new(),
            counts: Vec::new(),
            total_count: 0,
        }
    }
}

impl<Color, Component, const N: usize> UniqueColorCounts<Color, Component, N>
where
    Color: ColorComponents<Component, N>,
{
    /// Returns the slice of unique colors.
    #[must_use]
    pub fn colors(&self) -> &[Color] {
        &self.colors
    }

    /// Returns a slice for the number of times each unique color was present in the original color slice/image.
    #[must_use]
    pub fn counts(&self) -> &[u32] {
        &self.counts
    }

    /// Returns the number of original pixels/colors.
    ///
    /// This is equal to the sum of [`UniqueColorCounts::counts`].
    #[must_use]
    pub fn total_count(&self) -> u32 {
        self.total_count
    }

    /// Returns the number of unique colors.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn num_colors(&self) -> u32 {
        self.colors.len() as u32
    }

    /// Creates a new [`UniqueColorCounts`] from a [`ColorSlice`].
    ///
    /// The `convert_color` function can be used to convert the color space of the resulting colors.
    /// Use the identity function (i.e., `|color| color`) to skip color space conversion.
    pub fn new<InputColor>(
        pixels: ColorSlice<InputColor>,
        convert_color: impl Fn(InputColor) -> Color,
    ) -> Self
    where
        InputColor: ColorComponents<u8, 3>,
    {
        if pixels.is_empty() {
            Self::default()
        } else {
            let total_count = pixels.num_colors();

            let mut colors = Vec::new();
            let mut counts = Vec::new();
            let mut green_blue = vec![[0; 2]; pixels.len()];

            let mut lower_counts = <[[u32; RADIX]; RADIX]>::box_zeroed();
            let mut bitmask: BitVec = BitVec::repeat(false, RADIX * RADIX);

            let mut red_prefix = ZeroedIsZero::box_zeroed();
            u8_counts::<_, 4>(&pixels, &mut red_prefix);
            let red_prefix = &mut red_prefix[0];
            prefix_sum(red_prefix);

            for &[r, g, b] in pixels.as_arrays() {
                let r = usize::from(r);
                let j = red_prefix[r] - 1;
                green_blue[j as usize] = [g, b];
                red_prefix[r] = j;
            }
            red_prefix[RADIX] = total_count;

            for r in 0..RADIX {
                let chunk = chunk_range(red_prefix, r);

                if !chunk.is_empty() {
                    let green_blue = &green_blue[chunk.clone()];

                    if chunk.len() < RADIX * RADIX / 4 {
                        for gb in green_blue {
                            let [g, b] = gb.map(usize::from);
                            lower_counts[g][b] += 1;
                            bitmask.set(g * RADIX + b, true);
                        }

                        for i in bitmask.iter_ones() {
                            let g = i / RADIX;
                            let b = i % RADIX;
                            #[allow(clippy::cast_possible_truncation)]
                            let color = cast::from_array([r as u8, g as u8, b as u8]);
                            colors.push(convert_color(color));
                            counts.push(lower_counts[g][b]);
                            lower_counts[g][b] = 0;
                        }

                        bitmask.fill(false);
                    } else {
                        for &[g, b] in green_blue {
                            lower_counts[usize::from(g)][usize::from(b)] += 1;
                        }

                        for (g, count) in lower_counts.iter().enumerate() {
                            for (b, &count) in count.iter().enumerate() {
                                if count > 0 {
                                    #[allow(clippy::cast_possible_truncation)]
                                    let color = cast::from_array([r as u8, g as u8, b as u8]);
                                    colors.push(convert_color(color));
                                    counts.push(count);
                                }
                            }
                        }

                        lower_counts.fill_zero();
                    }
                }
            }

            Self {
                _phantom: PhantomData,
                colors,
                counts,
                total_count,
            }
        }
    }

    /// Tries to create a new [`UniqueColorCounts`] from a [`RgbImage`].
    ///
    /// The `convert_color` function can be used to convert the color space of the resulting colors.
    /// Use the identity function (i.e., `|srgb| srgb`) to skip color space conversion.
    ///
    /// # Errors
    /// Return an error if the number of pixels in the image are above [`MAX_PIXELS`](crate::MAX_PIXELS).
    #[cfg(feature = "image")]
    pub fn try_from_rgbimage(
        image: &RgbImage,
        convert_color: impl Fn(Srgb<u8>) -> Color,
    ) -> Result<Self, AboveMaxLen<u32>> {
        image
            .try_into()
            .map(|slice| Self::new(slice, convert_color))
    }
}

/// Deduplicated colors and their frequency counts.
///
/// Unlike [`UniqueColorCounts`], this struct also holds indices for the original colors
/// to be able to reconstruct a quantized image.
///
/// Currently, only colors that implement [`ColorComponents<u8, 3>`] are supported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedColorCounts<Color, Component, const N: usize>
where
    Color: ColorComponents<Component, N>,
{
    /// The component type must remain the same for each [`IndexedColorCounts`].
    _phantom: PhantomData<Component>,
    /// The unique colors.
    colors: Vec<Color>,
    /// The number of times each color was present in the original color slice/image.
    counts: Vec<u32>,
    /// The total number of pixels/colors in the original color slice/image.
    total_count: u32,
    /// The indices into `colors` for each of the original pixels/colors.
    indices: Vec<u32>,
}

impl<Color, Component, const N: usize> Default for IndexedColorCounts<Color, Component, N>
where
    Color: ColorComponents<Component, N>,
{
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
            colors: Vec::new(),
            counts: Vec::new(),
            total_count: 0,
            indices: Vec::new(),
        }
    }
}

impl<Color, Component, const N: usize> IndexedColorCounts<Color, Component, N>
where
    Color: ColorComponents<Component, N>,
{
    /// Returns the slice of unique colors.
    #[must_use]
    pub fn colors(&self) -> &[Color] {
        &self.colors
    }

    /// Returns a slice for the number of times each unique color was present in the original color slice/image.
    #[must_use]
    pub fn counts(&self) -> &[u32] {
        &self.counts
    }

    /// Returns the number of original pixels/colors.
    ///
    /// This is equal to the sum of [`IndexedColorCounts::counts`].
    #[must_use]
    pub fn total_count(&self) -> u32 {
        self.total_count
    }

    /// Returns the number of unique colors.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn num_colors(&self) -> u32 {
        self.colors.len() as u32
    }

    /// Returns the indices into [`IndexedColorCounts::colors`] for each of the original pixels/colors.
    #[must_use]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Creates a new [`IndexedColorCounts`] from a [`ColorSlice`].
    ///
    /// The `convert_color` function can be used to convert the color space of the resulting colors.
    /// Use the identity function (i.e., `|color| color`) to skip color space conversion.
    pub fn new<InputColor>(
        pixels: ColorSlice<InputColor>,
        convert_color: impl Fn(InputColor) -> Color,
    ) -> Self
    where
        InputColor: ColorComponents<u8, 3>,
    {
        if pixels.is_empty() {
            Self::default()
        } else {
            let total_count = pixels.num_colors();

            let mut colors = Vec::new();
            let mut counts = Vec::new();
            let mut indices = vec![0; pixels.len()];

            let mut orig_index = vec![0; pixels.len()];
            let mut green_blue = vec![[0; 2]; pixels.len()];

            let mut lower_counts = <[[u32; RADIX]; RADIX]>::box_zeroed();
            let mut bitmask: BitVec = BitVec::repeat(false, RADIX * RADIX);

            let mut red_prefix = ZeroedIsZero::box_zeroed();
            u8_counts::<_, 4>(&pixels, &mut red_prefix);
            let red_prefix = &mut red_prefix[0];
            prefix_sum(red_prefix);

            for (i, &[r, g, b]) in pixels.as_arrays().iter().enumerate() {
                let r = usize::from(r);
                let j = red_prefix[r] - 1;
                green_blue[j as usize] = [g, b];
                #[allow(clippy::cast_possible_truncation)]
                {
                    orig_index[j as usize] = i as u32;
                }
                red_prefix[r] = j;
            }
            red_prefix[RADIX] = total_count;

            for r in 0..RADIX {
                let chunk = chunk_range(red_prefix, r);

                if !chunk.is_empty() {
                    let green_blue = &green_blue[chunk.clone()];
                    let orig_index = &orig_index[chunk.clone()];

                    let sparse = chunk.len() < RADIX * RADIX / 4;
                    if sparse {
                        for gb in green_blue {
                            let [g, b] = gb.map(usize::from);
                            lower_counts[g][b] += 1;
                            bitmask.set(g * RADIX + b, true);
                        }

                        for i in bitmask.iter_ones() {
                            let g = i / RADIX;
                            let b = i % RADIX;
                            let count = lower_counts[g][b];
                            #[allow(clippy::cast_possible_truncation)]
                            {
                                lower_counts[g][b] = colors.len() as u32;
                            }
                            #[allow(clippy::cast_possible_truncation)]
                            let color = cast::from_array([r as u8, g as u8, b as u8]);
                            colors.push(convert_color(color));
                            counts.push(count);
                        }
                    } else {
                        for &[g, b] in green_blue {
                            lower_counts[usize::from(g)][usize::from(b)] += 1;
                        }

                        for (g, count) in lower_counts.iter_mut().enumerate() {
                            for (b, count) in count.iter_mut().enumerate() {
                                let n = *count;
                                if n > 0 {
                                    #[allow(clippy::cast_possible_truncation)]
                                    {
                                        *count = colors.len() as u32;
                                    }
                                    #[allow(clippy::cast_possible_truncation)]
                                    let color = cast::from_array([r as u8, g as u8, b as u8]);
                                    colors.push(convert_color(color));
                                    counts.push(n);
                                }
                            }
                        }
                    }

                    for (&i, &[g, b]) in orig_index.iter().zip(green_blue) {
                        indices[i as usize] = lower_counts[usize::from(g)][usize::from(b)];
                    }

                    if sparse {
                        for i in bitmask.iter_ones() {
                            let g = i / RADIX;
                            let b = i % RADIX;
                            lower_counts[g][b] = 0;
                        }
                        bitmask.fill(false);
                    } else {
                        lower_counts.fill_zero();
                    }
                }
            }

            Self {
                _phantom: PhantomData,
                colors,
                counts,
                total_count,
                indices,
            }
        }
    }

    /// Tries to create a new [`IndexedColorCounts`] from a [`RgbImage`].
    ///
    /// The `convert_color` function can be used to convert the color space of the resulting colors.
    /// Use the identity function (i.e., `|srgb| srgb`) to skip color space conversion.
    ///
    /// # Errors
    /// Return an error if the number of pixels in the image are above [`MAX_PIXELS`](crate::MAX_PIXELS).
    #[cfg(feature = "image")]
    pub fn try_from_rgbimage(
        image: &RgbImage,
        convert_color: impl Fn(Srgb<u8>) -> Color,
    ) -> Result<Self, AboveMaxLen<u32>> {
        image
            .try_into()
            .map(|slice| Self::new(slice, convert_color))
    }
}

/// Unsafe utilities for sharing data across multiple threads.
#[cfg(feature = "threads")]
#[allow(unsafe_code)]
mod sync_unsafe {
    use std::{cell::UnsafeCell, ops::Range};

    #[cfg(test)]
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Unsafely share a mutable slice across multiple threads.
    pub struct SyncUnsafeSlice<'a, T> {
        /// The inner [`UnsafeCell`] containing the mutable slice.
        cell: UnsafeCell<&'a mut [T]>,
        /// Check each index is written to only once during tests.
        #[cfg(test)]
        written: Vec<AtomicBool>,
    }

    unsafe impl<'a, T: Send + Sync> Send for SyncUnsafeSlice<'a, T> {}
    unsafe impl<'a, T: Send + Sync> Sync for SyncUnsafeSlice<'a, T> {}

    impl<'a, T> SyncUnsafeSlice<'a, T> {
        /// Creates a new [`SyncUnsafeSlice`] with the given slice.
        pub fn new(slice: &'a mut [T]) -> Self {
            Self {
                #[cfg(test)]
                written: slice.iter().map(|_| AtomicBool::new(false)).collect(),
                cell: UnsafeCell::new(slice),
            }
        }

        /// Unsafely get the inner mutable reference.
        unsafe fn get(&self) -> &'a mut [T] {
            unsafe { *self.cell.get() }
        }

        /// Unsafely write the given value to the given index in the slice.
        ///
        /// # Safety
        /// It is undefined behaviour if two threads write to the same index without synchronization.
        #[inline]
        pub unsafe fn write(&self, index: usize, value: T) {
            #[cfg(test)]
            {
                assert!(!self.written[index].swap(true, Ordering::SeqCst));
            }
            unsafe { self.get()[index] = value };
        }
    }

    impl<'a, T: Copy> SyncUnsafeSlice<'a, T> {
        /// Unsafely write the given slice to the given range.
        ///
        /// # Safety
        /// It is undefined behaviour if two threads write to the same range/indices without synchronization.
        #[inline]
        pub unsafe fn write_slice(&self, range: Range<usize>, slice: &[T]) {
            #[cfg(test)]
            {
                assert!(!self.written[range.clone()]
                    .iter()
                    .any(|b| b.swap(true, Ordering::SeqCst)));
            }
            unsafe { self.get()[range].copy_from_slice(slice) };
        }
    }
}

#[cfg(feature = "threads")]
use sync_unsafe::SyncUnsafeSlice;

#[cfg(feature = "threads")]
impl<Color, Component, const N: usize> UniqueColorCounts<Color, Component, N>
where
    Color: ColorComponents<Component, N> + Send,
{
    /// Creates a new [`UniqueColorCounts`] in parallel from a [`ColorSlice`].
    ///
    /// The `convert_color` function can be used to convert the color space of the resulting colors.
    /// Use the identity function (i.e., `|color| color`) to skip color space conversion.
    #[allow(clippy::too_many_lines)]
    pub fn new_par<InputColor>(
        pixels: ColorSlice<InputColor>,
        convert_color: impl Fn(InputColor) -> Color + Sync,
    ) -> Self
    where
        InputColor: ColorComponents<u8, 3> + Sync,
    {
        if pixels.is_empty() {
            Self::default()
        } else {
            let total_count = pixels.num_colors();

            let chunk_size = pixels.len().div_ceil(rayon::current_num_threads());
            let red_prefixes = {
                let mut red_prefixes = pixels
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        let mut red_prefix = ZeroedIsZero::box_zeroed();
                        u8_counts::<_, 4>(chunk, &mut red_prefix);
                        red_prefix[0]
                    })
                    .collect::<Vec<_>>();

                let mut carry = 0;
                for i in 0..RADIX {
                    red_prefixes[0][i] += carry;
                    for j in 1..red_prefixes.len() {
                        red_prefixes[j][i] += red_prefixes[j - 1][i];
                    }
                    carry = red_prefixes[red_prefixes.len() - 1][i];
                }

                debug_assert_eq!(carry, total_count);

                red_prefixes
            };

            let red_prefix = {
                let mut prefix = [0; RADIX + 1];
                prefix[1..].copy_from_slice(&red_prefixes[red_prefixes.len() - 1][..RADIX]);
                prefix
            };

            let mut green_blue = vec![[0; 2]; pixels.len()];
            {
                let green_blue = SyncUnsafeSlice::new(&mut green_blue);

                pixels
                    .as_arrays()
                    .par_chunks(chunk_size)
                    .zip(red_prefixes)
                    .for_each(|(chunk, mut red_prefix)| {
                        /// Buffer length
                        const BUF_LEN: u8 = 128;

                        let mut buffer = <[[[u8; 2]; BUF_LEN as usize]; RADIX]>::box_zeroed();
                        let mut lengths = [0u8; RADIX];

                        // Prefix sums ensure that each location in green_blue is written to only once
                        // and is therefore safe to write to without any form of synchronization.
                        #[allow(unsafe_code)]
                        for &[r, g, b] in chunk {
                            let r = usize::from(r);
                            let len = lengths[r];
                            let len = if len >= BUF_LEN {
                                let i = red_prefix[r] - u32::from(BUF_LEN);
                                let j = i as usize;
                                unsafe {
                                    green_blue
                                        .write_slice(j..(j + usize::from(BUF_LEN)), &buffer[r]);
                                }
                                red_prefix[r] = i;
                                0
                            } else {
                                len
                            };
                            buffer[r][usize::from(len)] = [g, b];
                            lengths[r] = len + 1;
                        }
                        #[allow(unsafe_code)]
                        for (r, buf) in buffer.iter().enumerate() {
                            let len = lengths[r];
                            let i = red_prefix[r] - u32::from(len);
                            let len = usize::from(len);
                            let j = i as usize;
                            unsafe { green_blue.write_slice(j..(j + len), &buf[..len]) };
                        }
                    });
            }

            let (colors, counts): (Vec<_>, Vec<_>) = (0..RADIX)
                .into_par_iter()
                .map(|r| {
                    let chunk = chunk_range(&red_prefix, r);

                    let mut colors = Vec::new();
                    let mut counts = Vec::new();

                    if !chunk.is_empty() {
                        let green_blue = &green_blue[chunk.clone()];
                        let mut lower_counts = <[[u32; RADIX]; RADIX]>::box_zeroed();

                        if chunk.len() < RADIX * RADIX / 4 {
                            let mut bitmask: BitVec = BitVec::repeat(false, RADIX * RADIX);

                            for gb in green_blue {
                                let [g, b] = gb.map(usize::from);
                                lower_counts[g][b] += 1;
                                bitmask.set(g * RADIX + b, true);
                            }

                            for i in bitmask.iter_ones() {
                                let g = i / RADIX;
                                let b = i % RADIX;
                                #[allow(clippy::cast_possible_truncation)]
                                let color = cast::from_array([r as u8, g as u8, b as u8]);
                                colors.push(convert_color(color));
                                counts.push(lower_counts[g][b]);
                            }
                        } else {
                            for &[g, b] in green_blue {
                                lower_counts[usize::from(g)][usize::from(b)] += 1;
                            }

                            for (g, count) in lower_counts.iter().enumerate() {
                                for (b, &count) in count.iter().enumerate() {
                                    if count > 0 {
                                        #[allow(clippy::cast_possible_truncation)]
                                        let color = cast::from_array([r as u8, g as u8, b as u8]);
                                        colors.push(convert_color(color));
                                        counts.push(count);
                                    }
                                }
                            }
                        }
                    };

                    (colors, counts)
                })
                .unzip();

            Self {
                _phantom: PhantomData,
                colors: colors.concat(),
                counts: counts.concat(),
                total_count,
            }
        }
    }

    /// Tries to create a new [`UniqueColorCounts`] in parallel from a [`RgbImage`].
    ///
    /// The `convert_color` function can be used to convert the color space of the resulting colors.
    /// Use the identity function (i.e., `|srgb| srgb`) to skip color space conversion.
    ///
    /// # Errors
    /// Return an error if the number of pixels in the image are above [`MAX_PIXELS`](crate::MAX_PIXELS).
    #[cfg(feature = "image")]
    pub fn try_from_rgbimage_par(
        image: &RgbImage,
        convert_color: impl Fn(Srgb<u8>) -> Color + Sync,
    ) -> Result<Self, AboveMaxLen<u32>> {
        image
            .try_into()
            .map(|slice| Self::new_par(slice, convert_color))
    }
}

#[cfg(feature = "threads")]
impl<Color, Component, const N: usize> IndexedColorCounts<Color, Component, N>
where
    Color: ColorComponents<Component, N> + Send,
{
    /// Creates a new [`IndexedColorCounts`] in parallel from a [`ColorSlice`].
    ///
    /// The `convert_color` function can be used to convert the color space of the resulting colors.
    /// Use the identity function (i.e., `|color| color`) to skip color space conversion.
    #[allow(clippy::too_many_lines)]
    pub fn new_par<InputColor>(
        pixels: ColorSlice<InputColor>,
        convert_color: impl Fn(InputColor) -> Color + Sync,
    ) -> Self
    where
        InputColor: ColorComponents<u8, 3> + Sync,
    {
        if pixels.is_empty() {
            Self::default()
        } else {
            let total_count = pixels.num_colors();

            let chunk_size = pixels.len().div_ceil(rayon::current_num_threads());
            let red_prefixes = {
                let mut red_prefixes = pixels
                    .as_arrays()
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        let mut counts = [0; RADIX];
                        for &[r, ..] in chunk {
                            counts[usize::from(r)] += 1;
                        }
                        counts
                    })
                    .collect::<Vec<_>>();

                let mut carry = 0;
                for i in 0..RADIX {
                    red_prefixes[0][i] += carry;
                    for j in 1..red_prefixes.len() {
                        red_prefixes[j][i] += red_prefixes[j - 1][i];
                    }
                    carry = red_prefixes[red_prefixes.len() - 1][i];
                }

                red_prefixes
            };

            let red_prefix = {
                let mut prefix = [0; RADIX + 1];
                prefix[1..].copy_from_slice(&red_prefixes[red_prefixes.len() - 1]);
                prefix
            };

            let mut green_blue = vec![[0; 2]; pixels.len()];
            let mut orig_index = vec![0; pixels.len()];
            {
                let green_blue = SyncUnsafeSlice::new(&mut green_blue);
                let orig_index = SyncUnsafeSlice::new(&mut orig_index);

                // Prefix sums ensure that each location in green_blue is written to only once
                // and is therefore safe to write to without any form of synchronization.
                #[allow(unsafe_code)]
                pixels
                    .as_arrays()
                    .par_chunks(chunk_size)
                    .zip(red_prefixes)
                    .enumerate()
                    .for_each(|(chunk_i, (chunk, mut red_prefix))| {
                        let offset = chunk_i * chunk_size;
                        for (i, &[r, g, b]) in chunk.iter().enumerate() {
                            let i = i + offset;
                            let r = usize::from(r);
                            let j = red_prefix[r] - 1;
                            unsafe { green_blue.write(j as usize, [g, b]) };
                            #[allow(clippy::cast_possible_truncation)]
                            {
                                unsafe { orig_index.write(j as usize, i as u32) };
                            }
                            red_prefix[r] = j;
                        }
                    });
            }

            let mut indices = vec![0; pixels.len()];

            let (colors, counts): (Vec<_>, Vec<_>) = {
                let indices = SyncUnsafeSlice::new(&mut indices);

                (0..RADIX)
                    .into_par_iter()
                    .map(|r| {
                        let chunk = chunk_range(&red_prefix, r);

                        let mut colors = Vec::new();
                        let mut counts = Vec::new();

                        if !chunk.is_empty() {
                            let green_blue = &green_blue[chunk.clone()];
                            let orig_index = &orig_index[chunk.clone()];
                            let mut lower_counts = <[[u32; RADIX]; RADIX]>::box_zeroed();

                            if chunk.len() < RADIX * RADIX / 4 {
                                let mut bitmask: BitVec = BitVec::repeat(false, RADIX * RADIX);

                                for gb in green_blue {
                                    let [g, b] = gb.map(usize::from);
                                    lower_counts[g][b] += 1;
                                    bitmask.set(g * RADIX + b, true);
                                }

                                for i in bitmask.iter_ones() {
                                    let g = i / RADIX;
                                    let b = i % RADIX;
                                    let count = lower_counts[g][b];
                                    #[allow(clippy::cast_possible_truncation)]
                                    {
                                        lower_counts[g][b] = colors.len() as u32;
                                    }
                                    #[allow(clippy::cast_possible_truncation)]
                                    let color = cast::from_array([r as u8, g as u8, b as u8]);
                                    colors.push(convert_color(color));
                                    counts.push(count);
                                }
                            } else {
                                for &[g, b] in green_blue {
                                    lower_counts[usize::from(g)][usize::from(b)] += 1;
                                }

                                for (g, count) in lower_counts.iter_mut().enumerate() {
                                    for (b, count) in count.iter_mut().enumerate() {
                                        let c = *count;
                                        if c > 0 {
                                            #[allow(clippy::cast_possible_truncation)]
                                            {
                                                *count = colors.len() as u32;
                                            }
                                            #[allow(clippy::cast_possible_truncation)]
                                            let color =
                                                cast::from_array([r as u8, g as u8, b as u8]);
                                            colors.push(convert_color(color));
                                            counts.push(c);
                                        }
                                    }
                                }
                            }

                            #[allow(unsafe_code)] // each index in orig_index is unique
                            for (&i, &[g, b]) in orig_index.iter().zip(green_blue) {
                                unsafe {
                                    indices.write(
                                        i as usize,
                                        lower_counts[usize::from(g)][usize::from(b)],
                                    );
                                }
                            }
                        }

                        (colors, counts)
                    })
                    .unzip()
            };

            debug_assert_eq!(colors.len(), RADIX);

            let mut indices_prefix = [0; RADIX];
            #[allow(clippy::cast_possible_truncation)]
            for (index, colors) in indices_prefix[1..].iter_mut().zip(&colors) {
                *index = colors.len() as u32;
            }

            prefix_sum(&mut indices_prefix);

            indices
                .par_iter_mut()
                .zip(pixels.as_arrays())
                .for_each(|(index, &[r, ..])| {
                    *index += indices_prefix[usize::from(r)];
                });

            Self {
                _phantom: PhantomData,
                colors: colors.concat(),
                counts: counts.concat(),
                total_count,
                indices,
            }
        }
    }

    /// Tries to create a new [`IndexedColorCounts`] in parallel from a [`RgbImage`].
    ///
    /// The `convert_color` function can be used to convert the color space of the resulting colors.
    /// Use the identity function (i.e., `|srgb| srgb`) to skip color space conversion.
    ///
    /// # Errors
    /// Return an error if the number of pixels in the image are above [`MAX_PIXELS`](crate::MAX_PIXELS).
    #[cfg(feature = "image")]
    pub fn try_from_rgbimage_par(
        image: &RgbImage,
        convert_color: impl Fn(Srgb<u8>) -> Color + Sync,
    ) -> Result<Self, AboveMaxLen<u32>> {
        image
            .try_into()
            .map(|slice| Self::new_par(slice, convert_color))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::tests::*;
    use palette::Srgb;
    use rand::{seq::SliceRandom, SeedableRng};
    use rand_xoshiro::Xoroshiro128PlusPlus;

    fn assert_valid_unique(
        unique: &UniqueColorCounts<Srgb<u8>, u8, 3>,
        colors: ColorSlice<Srgb<u8>>,
    ) {
        assert_eq!(unique.total_count(), colors.num_colors());

        let unique = unique.colors();
        for i in 1..unique.len() {
            assert!(unique[i - 1].into_components() < unique[i].into_components());
        }
    }

    fn assert_valid_indexed(
        indexed: &IndexedColorCounts<Srgb<u8>, u8, 3>,
        colors: ColorSlice<Srgb<u8>>,
    ) {
        assert_eq!(indexed.total_count(), colors.num_colors());

        let indexed_colors = indexed.colors();
        let mut counts = vec![0; indexed.len()];
        for (&i, &color) in indexed.indices().iter().zip(colors.as_ref()) {
            let i = i as usize;
            assert_eq!(indexed_colors[i], color);
            counts[i] += 1;
        }
        assert_eq!(counts, indexed.counts());

        for i in 1..indexed_colors.len() {
            assert!(indexed_colors[i - 1].into_components() < indexed_colors[i].into_components());
        }
    }

    #[test]
    fn empty_input() {
        let empty_input = ColorSlice::<Srgb<u8>>::new_unchecked(&[]);

        let unique = UniqueColorCounts::new(empty_input, |srgb| srgb);
        assert!(unique.is_empty() && unique.colors().is_empty() && unique.counts().is_empty());
        assert_eq!(unique.total_count(), 0);

        let indexed = IndexedColorCounts::new(empty_input, |srgb| srgb);
        assert!(
            indexed.is_empty()
                && indexed.colors().is_empty()
                && indexed.counts().is_empty()
                && indexed.indices().is_empty()
        );
        assert_eq!(indexed.total_count(), 0);

        #[cfg(feature = "threads")]
        {
            let unique = UniqueColorCounts::new_par(empty_input, |srgb| srgb);
            assert!(unique.is_empty() && unique.colors().is_empty() && unique.counts().is_empty());
            assert_eq!(unique.total_count(), 0);

            let indexed = IndexedColorCounts::new_par(empty_input, |srgb| srgb);
            assert!(
                indexed.is_empty()
                    && indexed.colors().is_empty()
                    && indexed.counts().is_empty()
                    && indexed.indices().is_empty()
            );
            assert_eq!(indexed.total_count(), 0);
        }
    }

    fn add_duplicate_color_with_data(colors: Vec<Srgb<u8>>) {
        fn index_of(colors: &[Srgb<u8>], color: Srgb<u8>) -> usize {
            colors.iter().position(|&c| c == color).unwrap()
        }

        let colors = {
            let mut colors = colors;
            let len = colors.len();
            colors[len - 1] = colors[0];
            colors
        };

        let duplicate = colors[0];
        let without_duplicate = ColorSlice::try_from(&colors[..(colors.len() - 1)]).unwrap();
        let with_duplicate = ColorSlice::try_from(colors.as_slice()).unwrap();

        let expected = {
            let mut unique = UniqueColorCounts::new(without_duplicate, |srgb| srgb);
            let i = index_of(unique.colors(), duplicate);
            unique.counts[i] += 1;
            unique.total_count += 1;
            unique
        };
        let actual = UniqueColorCounts::new(with_duplicate, |srgb| srgb);
        assert_valid_unique(&actual, with_duplicate);
        assert_eq!(actual, expected);

        let expected = {
            let mut indexed = IndexedColorCounts::new(without_duplicate, |srgb| srgb);
            let i = index_of(indexed.colors(), duplicate);
            indexed.counts[i] += 1;
            #[allow(clippy::cast_possible_truncation)]
            {
                indexed.indices.push(i as u32);
            }
            indexed.total_count += 1;
            indexed
        };
        let actual = IndexedColorCounts::new(with_duplicate, |srgb| srgb);
        assert_valid_indexed(&actual, with_duplicate);
        assert_eq!(actual, expected);

        #[cfg(feature = "threads")]
        {
            let expected = {
                let mut unique = UniqueColorCounts::new_par(without_duplicate, |srgb| srgb);
                let i = index_of(unique.colors(), duplicate);
                unique.counts[i] += 1;
                unique.total_count += 1;
                unique
            };
            let actual = UniqueColorCounts::new_par(with_duplicate, |srgb| srgb);
            assert_valid_unique(&actual, with_duplicate);
            assert_eq!(actual, expected);

            let expected = {
                let mut indexed = IndexedColorCounts::new_par(without_duplicate, |srgb| srgb);
                let i = index_of(indexed.colors(), duplicate);
                indexed.counts[i] += 1;
                indexed.total_count += 1;
                #[allow(clippy::cast_possible_truncation)]
                {
                    indexed.indices.push(i as u32);
                }
                indexed
            };
            let actual = IndexedColorCounts::new_par(with_duplicate, |srgb| srgb);
            assert_valid_indexed(&actual, with_duplicate);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn add_duplicate_color() {
        let colors = test_data_1024();
        add_duplicate_color_with_data(colors.clone());

        // for testing non-bitvec branches
        add_duplicate_color_with_data([colors.as_slice(); 256].concat());
    }

    fn reordered_input_with_data(colors: &[Srgb<u8>]) {
        let mut reordered = colors.to_vec();
        let reordered = {
            let mut rng = Xoroshiro128PlusPlus::seed_from_u64(0);
            reordered.shuffle(&mut rng);
            ColorSlice::try_from(reordered.as_slice()).unwrap()
        };

        let colors = ColorSlice::try_from(colors).unwrap();

        let expected = UniqueColorCounts::new(colors, |srgb| srgb);
        let actual = UniqueColorCounts::new(reordered, |srgb| srgb);
        assert_valid_unique(&actual, reordered);
        assert_eq!(actual, expected);

        let expected = IndexedColorCounts::new(colors, |srgb| srgb);
        let actual = IndexedColorCounts::new(reordered, |srgb| srgb);
        assert_valid_indexed(&actual, reordered);
        assert_eq!(actual.colors(), expected.colors());
        assert_eq!(actual.counts(), expected.counts());
        assert_eq!(actual.indices().len(), expected.indices.len());

        #[cfg(feature = "threads")]
        {
            let expected = UniqueColorCounts::new_par(colors, |srgb| srgb);
            let actual = UniqueColorCounts::new_par(reordered, |srgb| srgb);
            assert_valid_unique(&actual, reordered);
            assert_eq!(actual, expected);

            let expected = IndexedColorCounts::new_par(colors, |srgb| srgb);
            let actual = IndexedColorCounts::new_par(reordered, |srgb| srgb);
            assert_valid_indexed(&actual, reordered);
            assert_eq!(actual.colors(), expected.colors());
            assert_eq!(actual.counts(), expected.counts());
            assert_eq!(actual.indices().len(), expected.indices.len());
        }
    }

    #[test]
    fn reordered_input() {
        let colors = test_data_1024();
        reordered_input_with_data(&colors);

        // for testing non-bitvec branches
        let colors = [colors.as_slice(); 256].concat();
        reordered_input_with_data(&colors);
    }

    #[cfg(feature = "threads")]
    fn single_and_multi_threaded_match_with_data(colors: &[Srgb<u8>]) {
        let colors = ColorSlice::try_from(colors).unwrap();

        let single = UniqueColorCounts::new(colors, |srgb| srgb);
        let par = UniqueColorCounts::new_par(colors, |srgb| srgb);
        assert_valid_unique(&single, colors);
        assert_eq!(single, par);

        let single = IndexedColorCounts::new(colors, |srgb| srgb);
        let par = IndexedColorCounts::new_par(colors, |srgb| srgb);
        assert_valid_indexed(&single, colors);
        assert_eq!(single, par);
    }

    #[test]
    #[cfg(feature = "threads")]
    fn single_and_multi_threaded_match() {
        let colors = test_data_1024();
        single_and_multi_threaded_match_with_data(&colors);

        // for testing non-bitvec branches
        let colors = [colors.as_slice(); 256].concat();
        single_and_multi_threaded_match_with_data(&colors);
    }
}
