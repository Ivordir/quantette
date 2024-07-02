//! Contains dither implementation(s).

use crate::ColorComponents;
use ordered_float::OrderedFloat;
use palette::cast::AsArrays;
#[cfg(feature = "threads")]
use rayon::prelude::*;
use std::array;
use wide::{f32x8, u32x8, CmpLe};

/// Floyd–Steinberg dithering.
#[derive(Debug, Clone, Copy)]
pub struct FloydSteinberg(f32);

impl FloydSteinberg {
    /// The default error diffusion factor.
    pub const DEFAULT_ERROR_DIFFUSION: f32 = 7.0 / 8.0;

    /// Creates a new [`FloydSteinberg`] with the default error diffusion factor.
    #[must_use]
    pub const fn new() -> Self {
        Self(Self::DEFAULT_ERROR_DIFFUSION)
    }

    /// Creates a new [`FloydSteinberg`] with the given error diffusion factor.
    ///
    /// For example, a factor of `1.0` diffuses all of the error to the neighboring pixels.
    ///
    /// This will return `None` if `error_diffusion` is not in the range `0.0..=1.0`.
    #[must_use]
    pub fn with_error_diffusion(error_diffusion: f32) -> Option<Self> {
        if (0.0..=1.0).contains(&error_diffusion) {
            Some(Self(error_diffusion))
        } else {
            None
        }
    }

    /// Gets the error diffusion factor for this [`FloydSteinberg`].
    #[must_use]
    pub const fn error_diffusion(&self) -> f32 {
        self.0
    }
}

impl Default for FloydSteinberg {
    fn default() -> Self {
        Self::new()
    }
}

/// Squared euclidean distance between two points.
fn squared_euclidean_distance<const N: usize>(x: [f32; N], y: [f32; N]) -> f32 {
    let mut dist = 0.0;
    for c in 0..N {
        let d = x[c] - y[c];
        dist += d * d;
    }
    dist
}

/// Provides quick nearest neighbor lookups by storing, for each palette color,
/// the other palette colors sorted by increasing distance.
///
/// When the ditherer applies error correction to a pixel, we expect the change to not be that drastic.
/// Using the triangle inequality and the sorted palette colors,
/// we can stop the nearest neighbor search early once the distance becomes too large.
struct DistanceTable<const N: usize> {
    /// The palette colors as arrays of `f32`.
    palette: Vec<[f32; N]>,
    /// A table where each i-th row corresponds to the i-th color in `palette`,
    /// and the row consists of other palette colors and their indices in `palette`
    /// sorted by increasing distance to the row's color.
    components: Vec<(u32x8, [f32x8; N])>,
    /// The distances between each row's color and the first color of each 8-item chunk of `components`.
    distances: Vec<f32>,
}

impl<const N: usize> DistanceTable<N> {
    /// Constructs a nearest neighbor distance table for the given palette.
    fn new<Color, Component>(palette: &[Color]) -> Self
    where
        Color: ColorComponents<Component, N>,
        Component: Copy + Into<f32>,
    {
        let k = palette.len();

        let palette = palette
            .as_arrays()
            .iter()
            .map(|c| c.map(Into::into))
            .collect::<Vec<_>>();

        let mut distances = vec![(0, 0.0); k * k];
        #[allow(clippy::cast_possible_truncation)]
        for i in 0..k {
            let pi = palette[i];
            distances[i * k + i] = (i as u8, 0.0);
            for j in (i + 1)..k {
                let pj = palette[j];
                let dist = squared_euclidean_distance(pi, pj) / 4.0;
                distances[j * k + i] = (i as u8, dist);
                distances[i * k + j] = (j as u8, dist);
            }
        }

        for row in distances.chunks_exact_mut(k) {
            row.sort_by_key(|&(_, d)| OrderedFloat(d));
        }

        let (neighbors, distances): (Vec<_>, Vec<_>) = distances.into_iter().unzip();

        let mut components = Vec::with_capacity(k * k.div_ceil(8));
        for row in neighbors.chunks_exact(k) {
            let chunks = row.chunks_exact(8);

            components.extend(chunks.clone().map(|chunk| {
                let neighbors = u32x8::new(array::from_fn(|i| chunk[i].into()));
                let chunk = array::from_fn::<_, 8, _>(|i| palette[usize::from(chunk[i])]);
                let components = array::from_fn(|i| f32x8::new(array::from_fn(|j| chunk[j][i])));
                (neighbors, components)
            }));

            if !chunks.remainder().is_empty() {
                let mut neigh = [0; 8];
                let mut comp = [[f32::INFINITY; 8]; N];
                for (i, &j) in chunks.remainder().iter().enumerate() {
                    neigh[i] = j.into();
                    for (d, s) in comp.iter_mut().zip(palette[usize::from(j)]) {
                        d[i] = s;
                    }
                }
                components.push((u32x8::new(neigh), comp.map(f32x8::new)));
            }
        }

        let distances = distances
            .chunks_exact(k)
            .flat_map(|row| row.iter().copied().step_by(8))
            .collect();

        Self { palette, components, distances }
    }

    /// Given a point and a guess for its nearest palette index,
    /// this returns the nearest palette entry and its index.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn nearest_neighbor(&self, i: u8, point: [f32; N]) -> (u8, [f32; N]) {
        let Self { palette, distances, components } = self;
        let k = palette.len();
        let i = usize::from(i);

        let p = point.map(f32x8::splat);

        let row_start = i * k.div_ceil(8);
        let row_end = (i + 1) * k.div_ceil(8);

        let (mut min_neighbor, start_components) = components[row_start];

        #[allow(clippy::expect_used)]
        let mut min_distance = array::from_fn::<_, N, _>(|i| {
            let diff = p[i] - start_components[i];
            diff * diff
        })
        .into_iter()
        .reduce(|a, b| a + b)
        .expect("N != 0");

        let dist = min_distance.as_array_ref()[0];

        let row = (row_start + 1)..row_end;
        for (&(neighbor, chunk), &half_dist) in components[row.clone()].iter().zip(&distances[row])
        {
            if dist < half_dist {
                break;
            }

            #[allow(clippy::expect_used)]
            let distance = array::from_fn::<_, N, _>(|i| {
                let diff = p[i] - chunk[i];
                diff * diff
            })
            .into_iter()
            .reduce(|a, b| a + b)
            .expect("N != 0");

            let mask = u32x8::new(distance.cmp_le(min_distance).to_array().map(f32::to_bits));
            min_neighbor = mask.blend(neighbor, min_neighbor);
            min_distance = min_distance.fast_min(distance);
        }

        let mut min_lane = 0;
        let mut min_dist = f32::INFINITY;
        for (i, &v) in min_distance.as_array_ref().iter().enumerate() {
            if v < min_dist {
                min_dist = v;
                min_lane = i;
            }
        }

        #[allow(clippy::cast_possible_truncation)]
        let min_index = min_neighbor.as_array_ref()[min_lane] as u8;
        (min_index, palette[usize::from(min_index)])
    }
}

/// Multiplies `other` by a scalar, `alpha`, and adds the result to `arr`.
#[inline]
fn arr_mul_add_assign<const N: usize>(arr: &mut [f32; N], alpha: f32, other: [f32; N]) {
    for i in 0..N {
        arr[i] += alpha * other[i];
    }
}

/// Multiplies `other` by a scalar, `alpha`, and assigns the result to `arr`.
#[inline]
fn arr_mul_assign<const N: usize>(arr: &mut [f32; N], alpha: f32, other: [f32; N]) {
    for i in 0..N {
        arr[i] = alpha * other[i];
    }
}

/// Propagates, stores, and applies the dither error to the pixels.
struct ErrorBuf<'a, const N: usize> {
    /// The width of a row of pixels.
    width: usize,
    /// The propagated error for the current row of pixels.
    this_err: &'a mut [[f32; N]],
    /// The propagated error for the next row of pixels.
    next_err: &'a mut [[f32; N]],
}

impl<'a, const N: usize> ErrorBuf<'a, N> {
    /// Create the backing buffer for a new `ErrorBuf`.
    fn new_buf(width: usize) -> Vec<[f32; N]> {
        vec![[0.0; N]; 2 * (width + 2)]
    }

    /// Create a new `ErrorBuf` using the given `buf`
    fn new(width: usize, buf: &'a mut [[f32; N]]) -> Self {
        let (this_err, next_err) = buf.split_at_mut(width + 2);
        Self { width, this_err, next_err }
    }

    /// Propagate error using floyd steinberg dithering, going from left to right.
    #[inline]
    fn propagate_ltr(&mut self, i: usize, err: [f32; N]) {
        arr_mul_add_assign(&mut self.this_err[i + 2], 7.0 / 16.0, err);
        arr_mul_add_assign(&mut self.next_err[i], 3.0 / 16.0, err);
        arr_mul_add_assign(&mut self.next_err[i + 1], 5.0 / 16.0, err);
        arr_mul_assign(&mut self.next_err[i + 2], 1.0 / 16.0, err);
    }

    /// Propagate error using floyd steinberg dithering, going from right to left.
    #[inline]
    fn propagate_rtl(&mut self, i: usize, err: [f32; N]) {
        arr_mul_add_assign(&mut self.this_err[i], 7.0 / 16.0, err);
        arr_mul_add_assign(&mut self.next_err[i + 2], 3.0 / 16.0, err);
        arr_mul_add_assign(&mut self.next_err[i + 1], 5.0 / 16.0, err);
        arr_mul_assign(&mut self.next_err[i], 1.0 / 16.0, err);
    }

    /// Apply the accumulated error to this pixel.
    #[inline]
    fn apply(&self, i: usize, point: &mut [f32; N]) {
        let err = self.this_err[i + 1];
        for i in 0..N {
            point[i] += err[i];
        }
    }

    /// Reset and swap the error buffers for the next row of pixels.
    #[inline]
    fn next_row(&mut self) {
        std::mem::swap(&mut self.this_err, &mut self.next_err);
        self.next_err[1] = [0.0; N];
        self.next_err[self.width] = [0.0; N];
    }
}

/// Dither a single pixel, returning the error.
#[inline]
fn dither_pixel<const N: usize>(
    i: usize,
    index: &mut u8,
    mut point: [f32; N],
    table: &DistanceTable<N>,
    error: &mut ErrorBuf<N>,
    diffusion: f32,
) -> [f32; N] {
    error.apply(i, &mut point);
    let (nearest_index, nearest_point) = table.nearest_neighbor(*index, point);
    *index = nearest_index;
    array::from_fn(|i| diffusion * (point[i] - nearest_point[i]))
}

/// Performs dithering on the given indices.
///
/// The original input/image is taken in the form of an indexed palette.
#[allow(clippy::too_many_arguments)]
#[inline]
fn dither_indexed<Color, Component, const N: usize>(
    width: usize,
    indices: &mut [u8],
    original_colors: &[Color],
    original_indices: &[u32],
    table: &DistanceTable<N>,
    mut error: ErrorBuf<N>,
    diffusion: f32,
    mut pixel_row: Vec<[f32; N]>,
) where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32>,
{
    let original_colors = original_colors.as_arrays();

    for (row, (indices, colors)) in indices
        .chunks_exact_mut(width)
        .zip(original_indices.chunks_exact(width))
        .enumerate()
    {
        for (d, &s) in pixel_row.iter_mut().zip(colors) {
            *d = original_colors[s as usize].map(Into::into);
        }

        if row % 2 == 0 {
            for (i, (index, &point)) in indices.iter_mut().zip(&pixel_row).enumerate() {
                let err = dither_pixel(i, index, point, table, &mut error, diffusion);
                error.propagate_ltr(i, err);
            }
        } else {
            for (i, (index, &point)) in indices.iter_mut().zip(&pixel_row).enumerate().rev() {
                let err = dither_pixel(i, index, point, table, &mut error, diffusion);
                error.propagate_rtl(i, err);
            }
        }

        error.next_row();
    }
}

/// Performs dithering on the given indices.
#[inline]
fn dither<Color, Component, const N: usize>(
    width: usize,
    indices: &mut [u8],
    original_colors: &[Color],
    table: &DistanceTable<N>,
    mut error: ErrorBuf<N>,
    diffusion: f32,
) where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32>,
{
    for (row, (indices, colors)) in indices
        .chunks_exact_mut(width)
        .zip(original_colors.as_arrays().chunks_exact(width))
        .enumerate()
    {
        if row % 2 == 0 {
            for (i, (index, &og)) in indices.iter_mut().zip(colors).enumerate() {
                let err = dither_pixel(i, index, og.map(Into::into), table, &mut error, diffusion);
                error.propagate_ltr(i, err);
            }
        } else {
            for (i, (index, &og)) in indices.iter_mut().zip(colors).enumerate().rev() {
                let err = dither_pixel(i, index, og.map(Into::into), table, &mut error, diffusion);
                error.propagate_rtl(i, err);
            }
        }

        error.next_row();
    }
}

impl FloydSteinberg {
    /// Performs dithering on the given indices.
    ///
    /// The original input/image is taken in the form of an indexed palette.
    pub fn dither_indexed<Color, Component, const N: usize>(
        &self,
        palette: &[Color],
        indices: &mut [u8],
        original_colors: &[Color],
        original_indices: &[u32],
        width: u32,
        height: u32,
    ) where
        Color: ColorComponents<Component, N>,
        Component: Copy + Into<f32>,
    {
        let &FloydSteinberg(diffusion) = self;

        if palette.is_empty() || diffusion == 0.0 || width * height == 0 {
            return;
        }

        let width = width as usize;
        let table = DistanceTable::new(palette);
        let mut error = ErrorBuf::new_buf(width);
        let error = ErrorBuf::new(width, &mut error);
        let pixel_row = vec![[0.0; N]; width];
        dither_indexed(
            width,
            indices,
            original_colors,
            original_indices,
            &table,
            error,
            diffusion,
            pixel_row,
        );
    }

    /// Performs dithering on the given indices.
    pub fn dither<Color, Component, const N: usize>(
        &self,
        palette: &[Color],
        indices: &mut [u8],
        original_colors: &[Color],
        width: u32,
        height: u32,
    ) where
        Color: ColorComponents<Component, N>,
        Component: Copy + Into<f32>,
    {
        let &FloydSteinberg(diffusion) = self;

        if palette.is_empty() || diffusion == 0.0 || width * height == 0 {
            return;
        }

        let width = width as usize;
        let table = DistanceTable::new(palette);
        let mut error = ErrorBuf::new_buf(width);
        let error = ErrorBuf::new(width, &mut error);
        dither(width, indices, original_colors, &table, error, diffusion);
    }
}

#[cfg(feature = "threads")]
impl FloydSteinberg {
    /// Performs dithering on the given indices in parallel.
    ///
    /// The original input/image is taken in the form of an indexed palette.
    pub fn dither_indexed_par<Color, Component, const N: usize>(
        &self,
        palette: &[Color],
        indices: &mut [u8],
        original_colors: &[Color],
        original_indices: &[u32],
        width: u32,
        height: u32,
    ) where
        Color: ColorComponents<Component, N> + Sync,
        Component: Copy + Into<f32> + Sync,
    {
        let &FloydSteinberg(diffusion) = self;

        if palette.is_empty() || diffusion == 0.0 || width * height == 0 {
            return;
        }

        let table = DistanceTable::new(palette);

        let width = width as usize;
        let height = height as usize;
        let chunk_size = chunk_size(width, height);

        let mut indices_prev_chunk_last_row =
            indices_prev_chunk_last_row(width, chunk_size, indices);

        indices
            .par_chunks_mut(chunk_size)
            .zip(indices_prev_chunk_last_row.par_chunks_mut(width))
            .enumerate()
            .for_each(|(chunk_i, (indices, prev_row))| {
                let chunk_start = chunk_i * chunk_size;

                let mut error = ErrorBuf::new_buf(width);
                let mut error = ErrorBuf::new(width, &mut error);
                let mut pixel_row = vec![[0.0; N]; width];

                if chunk_i > 0 {
                    let colors = &original_indices[(chunk_start - width)..chunk_start];
                    let original_colors = original_colors.as_arrays();

                    for (d, &s) in pixel_row.iter_mut().zip(colors) {
                        *d = original_colors[s as usize].map(Into::into);
                    }

                    for (i, (index, &og)) in prev_row.iter_mut().zip(&pixel_row).enumerate().rev() {
                        let err = dither_pixel(
                            i,
                            index,
                            og.map(Into::into),
                            &table,
                            &mut error,
                            diffusion,
                        );
                        error.propagate_rtl(i, err);
                    }

                    error.next_row();
                }

                let original_indices = &original_indices
                    [chunk_start..usize::min(chunk_start + chunk_size, original_indices.len())];

                dither_indexed(
                    width,
                    indices,
                    original_colors,
                    original_indices,
                    &table,
                    error,
                    diffusion,
                    pixel_row,
                );
            });
    }

    /// Performs dithering on the given indices in parallel.
    pub fn dither_par<Color, Component, const N: usize>(
        &self,
        palette: &[Color],
        indices: &mut [u8],
        original_colors: &[Color],
        width: u32,
        height: u32,
    ) where
        Color: ColorComponents<Component, N> + Sync,
        Component: Copy + Into<f32>,
    {
        let &FloydSteinberg(diffusion) = self;

        if palette.is_empty() || diffusion == 0.0 || width * height == 0 {
            return;
        }

        let table = DistanceTable::new(palette);

        let width = width as usize;
        let height = height as usize;
        let chunk_size = chunk_size(width, height);

        let mut indices_prev_chunk_last_row =
            indices_prev_chunk_last_row(width, chunk_size, indices);

        indices
            .par_chunks_mut(chunk_size)
            .zip(indices_prev_chunk_last_row.par_chunks_mut(width))
            .enumerate()
            .for_each(|(chunk_i, (indices, prev_row))| {
                let chunk_start = chunk_i * chunk_size;

                let mut error = ErrorBuf::new_buf(width);
                let mut error = ErrorBuf::new(width, &mut error);

                if chunk_i > 0 {
                    let colors = original_colors[(chunk_start - width)..chunk_start].as_arrays();

                    for (i, (index, &og)) in prev_row.iter_mut().zip(colors).enumerate().rev() {
                        let err = dither_pixel(
                            i,
                            index,
                            og.map(Into::into),
                            &table,
                            &mut error,
                            diffusion,
                        );
                        error.propagate_rtl(i, err);
                    }

                    error.next_row();
                }

                let original_colors = &original_colors
                    [chunk_start..usize::min(chunk_start + chunk_size, original_colors.len())];

                dither(width, indices, original_colors, &table, error, diffusion);
            });
    }
}

/// Returns the `indices` chunk size for parallel dithering.
#[cfg(feature = "threads")]
fn chunk_size(width: usize, height: usize) -> usize {
    let num_chunks = usize::min(rayon::current_num_threads(), height.div_ceil(256));
    let rows_per_chunk = height.div_ceil(num_chunks);
    width * rows_per_chunk
}

/// Returns concatenated copies of the last row for each `indices` chunk.
#[cfg(feature = "threads")]
fn indices_prev_chunk_last_row(width: usize, chunk_size: usize, indices: &[u8]) -> Vec<u8> {
    let mut prev_rows = indices
        .chunks(chunk_size)
        .map(|chunk| &chunk[(chunk.len() - width)..])
        .collect::<Vec<_>>();

    prev_rows.rotate_right(1);
    prev_rows.concat()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;
    use ordered_float::OrderedFloat;
    use palette::Srgb;

    #[test]
    fn components_match_indices() {
        let k = 249; // use non-multiple of 8 to test remainder handling
        let centroids = &test_data_256()[..k];
        let table = DistanceTable::new(centroids);
        let centroids = centroids.as_arrays();

        let mut expected = table
            .components
            .iter()
            .flat_map(|&(indices, _)| {
                indices
                    .as_array_ref()
                    .map(|i| centroids[i as usize].map(Into::into))
            })
            .collect::<Vec<_>>();

        for row in expected.chunks_exact_mut(k.next_multiple_of(8)) {
            row[k..].fill([f32::INFINITY; 3]);
        }

        let actual = table
            .components
            .into_iter()
            .flat_map(|(_, components)| {
                array::from_fn::<_, 8, _>(|i| components.map(|c| c.as_array_ref()[i]))
            })
            .collect::<Vec<_>>();

        assert_eq!(expected, actual);
    }

    #[test]
    fn distances_match_components() {
        let k = 249; // use non-multiple of 8 to test remainder handling
        let centroids = &test_data_256()[..k];
        let table = DistanceTable::new(centroids);

        let expected = table
            .components
            .chunks_exact(k.div_ceil(8))
            .zip(centroids.as_arrays())
            .flat_map(|(row, &centroid)| {
                row.iter().map(move |(_, components)| {
                    squared_euclidean_distance(
                        centroid.map(Into::into),
                        components.map(|c| c.as_array_ref()[0]),
                    ) / 4.0
                })
            })
            .collect::<Vec<_>>();

        assert_eq!(expected, table.distances);
    }

    #[test]
    fn distances_are_ascending() {
        let k = 249; // use non-multiple of 8 to test remainder handling
        let centroids = &test_data_256()[..k];
        let table = DistanceTable::new(centroids);

        let row_len = k.div_ceil(8);
        for row in table.distances.chunks_exact(row_len) {
            for i in 1..row_len {
                assert!(row[i - 1] <= row[i]);
            }
        }
    }

    #[test]
    fn naive_nearest_neighbor_oracle() {
        let k = 249; // use non-multiple of 8 to test remainder handling
        let centroids = &test_data_256()[..k];
        let points = &test_data_1024();
        let table = DistanceTable::new(centroids);
        let centroids = centroids.as_arrays();

        for (i, color) in points.as_arrays().iter().enumerate() {
            let color = color.map(Into::into);

            #[allow(clippy::unwrap_used)]
            let expected = centroids
                .iter()
                .map(|&centroid| {
                    OrderedFloat(squared_euclidean_distance(centroid.map(Into::into), color))
                })
                .min()
                .unwrap()
                .0;

            // should give correct results regardless of guess
            #[allow(clippy::cast_possible_truncation)]
            let guess = (i % k) as u8;

            let actual = squared_euclidean_distance(color, table.nearest_neighbor(guess, color).1);

            #[allow(clippy::float_cmp)]
            {
                assert_eq!(expected, actual);
            }
        }
    }

    #[test]
    fn empty_inputs() {
        let ditherer = FloydSteinberg::new();

        // empty image and palette
        ditherer.dither::<Srgb<u8>, u8, 3>(&[], &mut [], &[], 0, 0);
        ditherer.dither_indexed::<Srgb<u8>, u8, 3>(&[], &mut [], &[], &[], 0, 0);

        // only empty palette
        let colors = test_data_1024();
        #[allow(clippy::cast_possible_truncation)]
        let len = colors.len() as u32;
        ditherer.dither::<Srgb<u8>, u8, 3>(&[], &mut [], &colors, len, 1);
        let indices = (0..len).collect::<Vec<_>>();
        ditherer.dither_indexed::<Srgb<u8>, u8, 3>(&[], &mut [], &colors, &indices, len, 1);

        #[cfg(feature = "threads")]
        {
            // empty image and palette
            ditherer.dither_par::<Srgb<u8>, u8, 3>(&[], &mut [], &[], 0, 0);
            ditherer.dither_indexed_par::<Srgb<u8>, u8, 3>(&[], &mut [], &[], &[], 0, 0);

            // only empty palette
            ditherer.dither_par::<Srgb<u8>, u8, 3>(&[], &mut [], &colors, len, 1);
            ditherer.dither_indexed_par::<Srgb<u8>, u8, 3>(&[], &mut [], &colors, &indices, len, 1);
        }
    }

    #[test]
    fn exact_match_image_unaffected() {
        let ditherer = FloydSteinberg::new();

        let palette = test_data_256();
        let indices = {
            #[allow(clippy::cast_possible_truncation)]
            let indices = (0..palette.len()).map(|i| i as u8).collect::<Vec<_>>();
            let mut indices = [indices.as_slice(); 4].concat();
            indices.rotate_right(7);
            indices
        };

        let width = 32;
        let height = 32;
        assert_eq!(width as usize * height as usize, indices.len());

        let mut new_indices = indices.clone();
        let original_colors = indices
            .iter()
            .map(|&i| palette[usize::from(i)])
            .collect::<Vec<_>>();
        ditherer.dither(&palette, &mut new_indices, &original_colors, width, height);
        assert_eq!(indices, new_indices);

        let mut new_indices = indices.clone();
        let original_indices = indices.iter().copied().map(u32::from).collect::<Vec<_>>();
        ditherer.dither_indexed(
            &palette,
            &mut new_indices,
            &palette,
            &original_indices,
            width,
            height,
        );
        assert_eq!(indices, new_indices);

        #[cfg(feature = "threads")]
        {
            let mut new_indices = indices.clone();
            ditherer.dither_par(&palette, &mut new_indices, &original_colors, width, height);
            assert_eq!(indices, new_indices);

            let mut new_indices = indices.clone();
            ditherer.dither_indexed_par(
                &palette,
                &mut new_indices,
                &palette,
                &original_indices,
                width,
                height,
            );
            assert_eq!(indices, new_indices);
        }
    }
}
