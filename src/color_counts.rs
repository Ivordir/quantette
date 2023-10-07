use crate::{AboveMaxLen, ColorComponents, ColorSlice, ZeroedIsZero};

use std::{marker::PhantomData, ops::Range};

use bitvec::vec::BitVec;
use palette::cast::{self, AsArrays};

#[cfg(feature = "image")]
use image::RgbImage;
#[cfg(any(feature = "image", feature = "colorspaces"))]
use palette::Srgb;
#[cfg(feature = "threads")]
use rayon::prelude::*;

pub trait ColorAndFrequency<Color, Component, const N: usize>
where
    Color: ColorComponents<Component, N>,
{
    fn colors(&self) -> &[Color];

    fn color_components(&self) -> &[[Component; N]] {
        self.colors().as_arrays()
    }

    fn num_colors(&self) -> u32;

    fn counts(&self) -> Option<&[u32]>;

    fn total_count(&self) -> u32;

    fn indices(&self) -> Option<&[u32]>;

    fn len(&self) -> usize {
        self.num_colors() as usize
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, Color, Component, const N: usize> ColorAndFrequency<Color, Component, N>
    for ColorSlice<'a, Color>
where
    Color: ColorComponents<Component, N>,
{
    fn colors(&self) -> &[Color] {
        self.as_slice()
    }

    fn counts(&self) -> Option<&[u32]> {
        None
    }

    fn num_colors(&self) -> u32 {
        self.num_colors()
    }

    fn total_count(&self) -> u32 {
        self.num_colors()
    }

    fn indices(&self) -> Option<&[u32]> {
        None
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

impl<Color, Component, const N: usize> ColorAndFrequency<Color, Component, N>
    for RemappableColorCounts<Color, Component, N>
where
    Color: ColorComponents<Component, N>,
{
    fn colors(&self) -> &[Color] {
        &self.colors
    }

    fn counts(&self) -> Option<&[u32]> {
        Some(&self.counts)
    }

    fn num_colors(&self) -> u32 {
        self.num_colors()
    }

    fn total_count(&self) -> u32 {
        self.total_count
    }

    fn indices(&self) -> Option<&[u32]> {
        Some(&self.indices)
    }
}

impl<Color, Component, const N: usize> ColorAndFrequency<Color, Component, N>
    for UnmappableColorCounts<Color, Component, N>
where
    Color: ColorComponents<Component, N>,
{
    fn colors(&self) -> &[Color] {
        &self.colors
    }

    fn counts(&self) -> Option<&[u32]> {
        Some(&self.counts)
    }

    fn num_colors(&self) -> u32 {
        self.num_colors()
    }

    fn total_count(&self) -> u32 {
        self.total_count
    }

    fn indices(&self) -> Option<&[u32]> {
        None
    }
}

pub trait ColorRemap {
    fn map_indices(&self, indices: Vec<u8>) -> Vec<u8>;
}

impl<'a, Color> ColorRemap for ColorSlice<'a, Color> {
    fn map_indices(&self, indices: Vec<u8>) -> Vec<u8> {
        indices
    }
}

impl<Color, Component, const N: usize> ColorRemap for RemappableColorCounts<Color, Component, N>
where
    Color: ColorComponents<Component, N>,
{
    fn map_indices(&self, indices: Vec<u8>) -> Vec<u8> {
        let indices = indices.as_slice(); // faster for some reason???
        self.indices.iter().map(|&i| indices[i as usize]).collect()
    }
}

#[cfg(feature = "threads")]
pub trait ParallelColorRemap {
    fn map_indices_par(&self, indices: Vec<u8>) -> Vec<u8>;
}

#[cfg(feature = "threads")]
impl<'a, Color> ParallelColorRemap for ColorSlice<'a, Color> {
    fn map_indices_par(&self, indices: Vec<u8>) -> Vec<u8> {
        indices
    }
}

#[cfg(feature = "threads")]
impl<Color, Component, const N: usize> ParallelColorRemap
    for RemappableColorCounts<Color, Component, N>
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

#[derive(Debug, Clone)]
pub struct ColorCounts<Color, Component, const N: usize, Indices>
where
    Color: ColorComponents<Component, N>,
{
    _phantom: PhantomData<Component>,
    colors: Vec<Color>,
    counts: Vec<u32>,
    total_count: u32,
    indices: Indices,
}

impl<Color, Component, const N: usize, Indices> Default
    for ColorCounts<Color, Component, N, Indices>
where
    Color: ColorComponents<Component, N>,
    Indices: Default,
{
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
            colors: Vec::new(),
            counts: Vec::new(),
            total_count: 0,
            indices: Default::default(),
        }
    }
}

pub type RemappableColorCounts<Color, Component, const N: usize> =
    ColorCounts<Color, Component, N, Vec<u32>>;

pub type UnmappableColorCounts<Color, Component, const N: usize> =
    ColorCounts<Color, Component, N, ()>;

/// A byte-sized Radix
const RADIX: usize = u8::MAX as usize + 1;

impl<Color, Component, const N: usize, Indices> ColorCounts<Color, Component, N, Indices>
where
    Color: ColorComponents<Component, N>,
{
    #[must_use]
    pub fn total_count(&self) -> u32 {
        self.total_count
    }

    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn num_colors(&self) -> u32 {
        self.colors.len() as u32
    }

    #[must_use]
    pub fn colors(&self) -> &[Color] {
        &self.colors
    }

    #[must_use]
    pub fn counts(&self) -> &[u32] {
        &self.counts
    }

    #[inline]
    fn prefix_sum<const M: usize>(counts: &mut [u32; M]) {
        for i in 1..M {
            counts[i] += counts[i - 1];
        }
    }

    #[inline]
    fn chunk_range(chunks: &[u32], i: usize) -> Range<usize> {
        (chunks[i] as usize)..(chunks[i + 1] as usize)
    }

    fn u8_counts<InputColor, const CHUNKS: usize>(
        pixels: &[InputColor],
        counts: &mut [[u32; RADIX + 1]; CHUNKS],
    ) where
        InputColor: ColorComponents<u8, 3>,
    {
        for color in pixels.as_arrays().chunks_exact(CHUNKS) {
            for (counts, &[r, ..]) in counts.iter_mut().zip(color) {
                counts[usize::from(r)] += 1;
            }
        }

        for &[r, ..] in pixels.as_arrays().chunks_exact(CHUNKS).remainder() {
            counts[0][usize::from(r)] += 1;
        }

        #[allow(clippy::unwrap_used)]
        let (counts, partial_counts) = counts.split_first_mut().unwrap();
        for i in 0..RADIX {
            for partial in &*partial_counts {
                counts[i] += partial[i];
            }
        }
    }
}

impl<Color, Component, const N: usize> RemappableColorCounts<Color, Component, N>
where
    Color: ColorComponents<Component, N>,
{
    #[must_use]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn remappable_u8_3_colors<InputColor>(
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
            let pixels = pixels.as_slice();

            let mut colors = Vec::new();
            let mut counts = Vec::new();
            let mut indices = vec![0; pixels.len()];

            let mut orig_index = vec![0; pixels.len()];
            let mut green_blue = vec![[0; 2]; pixels.len()];

            let mut lower_counts = <[[u32; RADIX]; RADIX]>::box_zeroed();
            let mut bitmask: BitVec = BitVec::repeat(false, RADIX * RADIX);

            let mut red_prefix = ZeroedIsZero::box_zeroed();
            Self::u8_counts::<_, 4>(pixels, &mut red_prefix);
            let red_prefix = &mut red_prefix[0];
            Self::prefix_sum(red_prefix);

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
                let chunk = Self::chunk_range(red_prefix, r);

                if !chunk.is_empty() {
                    let green_blue = &green_blue[chunk.clone()];
                    let orig_index = &orig_index[chunk.clone()];

                    if chunk.len() < RADIX * RADIX / 4 {
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

                        bitmask.fill(false);
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

                    lower_counts.fill_zero();
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

    #[cfg(feature = "image")]
    pub fn remappable_try_from_rgbimage(
        image: &RgbImage,
        convert_color: impl Fn(Srgb<u8>) -> Color,
    ) -> Result<Self, AboveMaxLen<u32>> {
        image
            .try_into()
            .map(|slice| Self::remappable_u8_3_colors(slice, convert_color))
    }
}

impl<Color, Component, const N: usize> UnmappableColorCounts<Color, Component, N>
where
    Color: ColorComponents<Component, N>,
{
    pub fn unmappable_u8_3_colors<InputColor>(
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
            let pixels = pixels.as_slice();

            let mut colors = Vec::new();
            let mut counts = Vec::new();
            let mut green_blue = vec![[0; 2]; pixels.len()];

            let mut lower_counts = <[[u32; RADIX]; RADIX]>::box_zeroed();
            let mut bitmask: BitVec = BitVec::repeat(false, RADIX * RADIX);

            let mut red_prefix = ZeroedIsZero::box_zeroed();
            Self::u8_counts::<_, 4>(pixels, &mut red_prefix);
            let red_prefix = &mut red_prefix[0];
            Self::prefix_sum(red_prefix);

            for &[r, g, b] in pixels.as_arrays() {
                let r = usize::from(r);
                let j = red_prefix[r] - 1;
                green_blue[j as usize] = [g, b];
                red_prefix[r] = j;
            }
            red_prefix[RADIX] = total_count;

            for r in 0..RADIX {
                let chunk = Self::chunk_range(red_prefix, r);

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
                    }

                    lower_counts.fill_zero();
                }
            }

            Self {
                _phantom: PhantomData,
                colors,
                counts,
                total_count,
                indices: (),
            }
        }
    }

    #[cfg(feature = "image")]
    pub fn unmappable_try_from_rgbimage(
        image: &RgbImage,
        convert_color: impl Fn(Srgb<u8>) -> Color,
    ) -> Result<Self, AboveMaxLen<u32>> {
        image
            .try_into()
            .map(|slice| Self::unmappable_u8_3_colors(slice, convert_color))
    }
}

/// Unsafe utilities for sharing data across multiple threads
#[cfg(feature = "threads")]
#[allow(unsafe_code)]
mod sync_unsafe {
    use std::{cell::UnsafeCell, ops::Range};

    /// Unsafely share a mutable slice across multiple threads
    pub struct SyncUnsafeSlice<'a, T>(UnsafeCell<&'a mut [T]>);

    unsafe impl<'a, T: Send + Sync> Send for SyncUnsafeSlice<'a, T> {}
    unsafe impl<'a, T: Send + Sync> Sync for SyncUnsafeSlice<'a, T> {}

    impl<'a, T> SyncUnsafeSlice<'a, T> {
        /// Create a new [`SyncUnsafeSlice`] with the given slice
        pub fn new(slice: &'a mut [T]) -> Self {
            Self(UnsafeCell::new(slice))
        }

        unsafe fn get(&self) -> &'a mut [T] {
            unsafe { *self.0.get() }
        }

        /// Unsafely write the given value to the given index in the slice
        ///
        /// # Safety
        /// It is undefined behaviour if two threads write to the same index without synchronization.
        #[inline]
        pub unsafe fn write(&self, index: usize, value: T) {
            unsafe { self.get()[index] = value };
        }
    }

    impl<'a, T: Copy> SyncUnsafeSlice<'a, T> {
        /// Unsafely write the given slice to the given range
        ///
        /// # Safety
        /// It is undefined behaviour if two threads write to the same range/indices without synchronization.
        #[inline]
        pub unsafe fn write_slice(&self, range: Range<usize>, slice: &[T]) {
            unsafe { self.get()[range].copy_from_slice(slice) };
        }
    }
}

#[cfg(feature = "threads")]
use sync_unsafe::SyncUnsafeSlice;

#[cfg(feature = "threads")]
impl<Color, Component, const N: usize> RemappableColorCounts<Color, Component, N>
where
    Color: ColorComponents<Component, N> + Send,
{
    #[allow(clippy::too_many_lines)]
    pub fn remappable_u8_3_colors_par<InputColor>(
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
            let pixels = pixels.as_slice();

            let threads = rayon::current_num_threads();
            let chunk_size = (pixels.len() + threads - 1) / threads;
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
                        let chunk = Self::chunk_range(&red_prefix, r);

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

            Self::prefix_sum(&mut indices_prefix);

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

    #[cfg(feature = "image")]
    pub fn remappable_try_from_rgbimage_par(
        image: &RgbImage,
        convert_color: impl Fn(Srgb<u8>) -> Color + Sync,
    ) -> Result<Self, AboveMaxLen<u32>> {
        image
            .try_into()
            .map(|slice| Self::remappable_u8_3_colors_par(slice, convert_color))
    }
}

#[cfg(feature = "threads")]
impl<Color, Component, const N: usize> UnmappableColorCounts<Color, Component, N>
where
    Color: ColorComponents<Component, N> + Send,
{
    #[allow(clippy::too_many_lines)]
    pub fn unmappable_u8_3_colors_par<InputColor>(
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
            let pixels = pixels.as_slice();

            let threads = rayon::current_num_threads();
            let chunk_size = (pixels.len() + threads - 1) / threads;

            let red_prefixes = {
                let mut red_prefixes = pixels
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        let mut red_prefix = ZeroedIsZero::box_zeroed();
                        Self::u8_counts::<_, 4>(chunk, &mut red_prefix);
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

            let mut green_blue = vec![[0, 2]; pixels.len()];
            {
                let green_blue = SyncUnsafeSlice::new(&mut green_blue);

                pixels
                    .as_arrays()
                    .par_chunks(chunk_size)
                    .zip(red_prefixes)
                    .for_each(|(chunk, mut red_prefix)| {
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
                    let chunk = Self::chunk_range(&red_prefix, r);

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
                indices: (),
            }
        }
    }

    #[cfg(feature = "image")]
    pub fn unmappable_try_from_rgbimage_par(
        image: &RgbImage,
        convert_color: impl Fn(Srgb<u8>) -> Color + Sync,
    ) -> Result<Self, AboveMaxLen<u32>> {
        image
            .try_into()
            .map(|slice| Self::unmappable_u8_3_colors_par(slice, convert_color))
    }
}
