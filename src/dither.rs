//! Contains dither implementation(s).

use crate::ColorComponents;

use std::array;

use ordered_float::OrderedFloat;
use palette::cast::AsArrays;
use wide::{f32x8, u32x8, CmpLe};

/// Floyd–Steinberg dithering.
///
/// The inner `f32` value denotes the error diffusion factor.
/// E.g, a factor of `1.0` diffuses all of the error to the neighboring pixels.
#[derive(Debug, Clone, Copy)]
pub struct FloydSteinberg(pub f32);

impl FloydSteinberg {
    /// The default error diffusion factor.
    pub const DEFAULT_ERROR_DIFFUSION: f32 = 7.0 / 8.0;

    /// Creates a new [`FloydSteinberg`] with the default error diffusion factor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for FloydSteinberg {
    fn default() -> Self {
        Self(Self::DEFAULT_ERROR_DIFFUSION)
    }
}

/// Constructs a nearest neighbor distance table for the given palette.
fn distance_table<const N: usize>(palette: &[[f32; N]]) -> (Vec<f32>, Vec<(u32x8, [f32x8; N])>) {
    /// Squared euclidean distance between two points.
    fn squared_euclidean_distance<const N: usize>(x: [f32; N], y: [f32; N]) -> f32 {
        let mut dist = 0.0;
        for c in 0..N {
            let d = x[c] - y[c];
            dist += d * d;
        }
        dist
    }

    let k = palette.len();
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

    let distances = distances.into_iter().step_by(8).collect();

    (distances, components)
}

/// Given a point and a guess for its nearest palette index,
/// this returns the nearest palette entry and its index.
#[allow(clippy::inline_always)]
#[inline(always)]
fn nearest_neighbor<const N: usize>(
    i: u8,
    point: [f32; N],
    palette: &[[f32; N]],
    distances: &[f32],
    components: &[(u32x8, [f32x8; N])],
) -> (u8, [f32; N]) {
    let k = palette.len();
    let i = usize::from(i);

    let p = point.map(f32x8::splat);

    let row_start = i * k.div_ceil(8);
    let row_end = (i + 1) * k.div_ceil(8);

    let (mut min_neighbor, start_components) = components[row_start];

    #[allow(clippy::unwrap_used)]
    let mut min_distance = array::from_fn::<_, N, _>(|i| {
        let diff = p[i] - start_components[i];
        diff * diff
    })
    .into_iter()
    .reduce(|a, b| a + b)
    .unwrap();

    let row = (row_start + 1)..row_end;
    for (&(neighbor, chunk), &half_dist) in components[row.clone()].iter().zip(&distances[row]) {
        if min_distance.cmp_le(half_dist).any() {
            break;
        }

        #[allow(clippy::unwrap_used)]
        let distance = array::from_fn::<_, N, _>(|i| {
            let diff = p[i] - chunk[i];
            diff * diff
        })
        .into_iter()
        .reduce(|a, b| a + b)
        .unwrap();

        #[allow(unsafe_code)]
        let mask: u32x8 = unsafe { std::mem::transmute(distance.cmp_le(min_distance)) };
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

/// Multiplies `other` by a scalar, `alpha`, and adds the result to `arr`.
#[inline]
fn arr_mul_add_assign<const N: usize>(arr: &mut [f32; N], alpha: f32, other: &[f32; N]) {
    for i in 0..N {
        arr[i] += alpha * other[i];
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
        let FloydSteinberg(diffusion) = *self;
        let width = width as usize;
        let _ = height; // not needed for now

        let palette = palette
            .as_arrays()
            .iter()
            .map(|c| c.map(Into::into))
            .collect::<Vec<_>>();

        let (distances, components) = distance_table(&palette);

        let original_colors = original_colors.as_arrays();

        let mut error = vec![[0.0; N]; 2 * (width + 2)];
        let (mut error1, mut error2) = error.split_at_mut(width + 2);
        let mut pixel_row = vec![[0.0; N]; width];

        for (row, (indices, colors)) in indices
            .chunks_exact_mut(width)
            .zip(original_indices.chunks_exact(width))
            .enumerate()
        {
            for (d, &s) in pixel_row.iter_mut().zip(colors) {
                *d = original_colors[s as usize].map(Into::into);
            }

            for (x, (i, &point)) in indices.iter_mut().zip(&pixel_row).enumerate() {
                let mut point = point;
                let err = error1[x + 1];
                for i in 0..N {
                    point[i] += err[i];
                }

                let (j, nearest) = nearest_neighbor(*i, point, &palette, &distances, &components);

                *i = j;
                let err = array::from_fn(|i| diffusion * (point[i] - nearest[i]));

                if row % 2 == 0 {
                    arr_mul_add_assign(&mut error1[x + 2], 7.0 / 16.0, &err);
                    arr_mul_add_assign(&mut error2[x], 3.0 / 16.0, &err);
                    arr_mul_add_assign(&mut error2[x + 1], 5.0 / 16.0, &err);

                    let e = &mut error2[x + 2];
                    for i in 0..N {
                        e[i] = (1.0 / 16.0) * err[i];
                    }
                } else {
                    arr_mul_add_assign(&mut error1[x], 7.0 / 16.0, &err);
                    arr_mul_add_assign(&mut error2[x + 2], 3.0 / 16.0, &err);
                    arr_mul_add_assign(&mut error2[x + 1], 5.0 / 16.0, &err);

                    let e = &mut error2[x];
                    for i in 0..N {
                        e[i] = (1.0 / 16.0) * err[i];
                    }
                }
            }

            std::mem::swap(&mut error1, &mut error2);
            error2[1] = [0.0; N];
            error2[width] = [0.0; N];
        }
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
        let FloydSteinberg(diffusion) = *self;
        let width = width as usize;
        let _ = height; // not needed for now

        let palette = palette
            .as_arrays()
            .iter()
            .map(|c| c.map(Into::into))
            .collect::<Vec<_>>();

        let (distances, components) = distance_table(&palette);

        let mut error = vec![[0.0; N]; 2 * (width + 2)];
        let (mut error1, mut error2) = error.split_at_mut(width + 2);

        for (row, (indices, colors)) in indices
            .chunks_exact_mut(width)
            .zip(original_colors.as_arrays().chunks_exact(width))
            .enumerate()
        {
            for (x, (i, &og)) in indices.iter_mut().zip(colors).enumerate() {
                let mut point = og.map(Into::into);
                let err = error1[x + 1];
                for i in 0..N {
                    point[i] += err[i];
                }

                let (j, nearest) = nearest_neighbor(*i, point, &palette, &distances, &components);

                *i = j;
                let err = array::from_fn(|i| diffusion * (point[i] - nearest[i]));

                if row % 2 == 0 {
                    arr_mul_add_assign(&mut error1[x + 2], 7.0 / 16.0, &err);
                    arr_mul_add_assign(&mut error2[x], 3.0 / 16.0, &err);
                    arr_mul_add_assign(&mut error2[x + 1], 5.0 / 16.0, &err);

                    let e = &mut error2[x + 2];
                    for i in 0..N {
                        e[i] = (1.0 / 16.0) * err[i];
                    }
                } else {
                    arr_mul_add_assign(&mut error1[x], 7.0 / 16.0, &err);
                    arr_mul_add_assign(&mut error2[x + 2], 3.0 / 16.0, &err);
                    arr_mul_add_assign(&mut error2[x + 1], 5.0 / 16.0, &err);

                    let e = &mut error2[x];
                    for i in 0..N {
                        e[i] = (1.0 / 16.0) * err[i];
                    }
                }
            }
        }

        std::mem::swap(&mut error1, &mut error2);
        error2[1] = [0.0; N];
        error2[width] = [0.0; N];
    }
}
