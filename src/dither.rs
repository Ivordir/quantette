use crate::{squared_euclidean_distance, ColorComponents};

use std::array;

use ordered_float::OrderedFloat;
use palette::cast::AsArrays;

pub trait Ditherer<T> {
    fn dither_indexed<Color, const N: usize>(
        &self,
        palette: &[Color],
        indices: &mut [u8],
        original_palette: &[Color],
        original_indices: &[u32],
        width: u32,
        height: u32,
    ) where
        Color: ColorComponents<T, N>;

    fn dither<Color, const N: usize>(
        &self,
        palette: &[Color],
        indices: &mut [u8],
        original_colors: &[Color],
        width: u32,
        height: u32,
    ) where
        Color: ColorComponents<T, N>;
}

#[derive(Debug, Clone, Copy)]
pub struct FloydSteinberg(pub f64);

impl FloydSteinberg {
    pub const DEFAULT_STRENGTH: f64 = 7.0 / 8.0;

    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for FloydSteinberg {
    fn default() -> Self {
        Self(Self::DEFAULT_STRENGTH)
    }
}

fn distance_table<const N: usize>(palette: &[[f64; N]]) -> (Vec<u8>, Vec<f64>) {
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

    distances.into_iter().unzip()
}

#[inline]

fn nearest_neighbor<const N: usize>(
    i: u8,
    point: [f64; N],
    palette: &[[f64; N]],
    neighbors: &[u8],
    distances: &[f64],
) -> (u8, [f64; N]) {
    let mut min_index = i;
    let i = usize::from(i);
    let mut nearest = palette[i];
    let dist = squared_euclidean_distance(point, nearest);
    let mut min_dist = dist;

    let k = palette.len();
    let row = (i * k + 1)..((i + 1) * k);
    for (&j, &half_dist) in neighbors[row.clone()].iter().zip(&distances[row]) {
        if dist < half_dist {
            break;
        }
        let other = palette[usize::from(j)];
        let distance = squared_euclidean_distance(point, other);
        if distance < min_dist {
            min_dist = distance;
            min_index = j;
            nearest = other;
        }
    }

    (min_index, nearest)
}

impl<T> Ditherer<T> for FloydSteinberg
where
    T: Copy + Into<f64>,
{
    fn dither_indexed<Color, const N: usize>(
        &self,
        palette: &[Color],
        indices: &mut [u8],
        original_colors: &[Color],
        original_indices: &[u32],
        width: u32,
        _height: u32,
    ) where
        Color: ColorComponents<T, N>,
    {
        let FloydSteinberg(strength) = *self;
        let width = width as usize;
        let palette = palette
            .as_arrays()
            .iter()
            .map(|c| c.map(Into::into))
            .collect::<Vec<_>>();

        let original_colors = original_colors.as_arrays();

        let (neighbors, distances) = distance_table(&palette);

        let mut error = vec![[0.0; N]; 2 * (width + 2)];
        let (mut error1, mut error2) = error.split_at_mut(width + 2);

        for (row, (indices, colors)) in indices
            .chunks_exact_mut(width)
            .zip(original_indices.chunks_exact(width))
            .enumerate()
        {
            #[inline]
            fn arr_mul_add_assign<const N: usize>(
                arr: &mut [f64; N],
                alpha: f64,
                other: &[f64; N],
            ) {
                for i in 0..N {
                    arr[i] += alpha * other[i];
                }
            }

            if row % 2 == 0 {
                for (x, (i, &og_i)) in indices.iter_mut().zip(colors).enumerate() {
                    let mut point = original_colors[og_i as usize].map(Into::into);
                    let err = error1[x + 1];
                    for i in 0..N {
                        point[i] += err[i];
                    }

                    let (j, nearest) =
                        nearest_neighbor(*i, point, &palette, &neighbors, &distances);

                    *i = j;
                    let err = array::from_fn(|i| point[i] - nearest[i]);

                    arr_mul_add_assign(&mut error1[x + 2], strength * (7.0 / 16.0), &err);
                    arr_mul_add_assign(&mut error2[x], strength * (3.0 / 16.0), &err);
                    arr_mul_add_assign(&mut error2[x + 1], strength * (5.0 / 16.0), &err);

                    let e = &mut error2[x + 2];
                    for i in 0..N {
                        e[i] = strength * (1.0 / 16.0) * err[i];
                    }
                }
            } else {
                for (x, (i, &og_i)) in indices.iter_mut().zip(colors).enumerate().rev() {
                    let mut point = original_colors[og_i as usize].map(Into::into);
                    let err = error1[x + 1];
                    for i in 0..N {
                        point[i] += err[i];
                    }

                    let (j, nearest) =
                        nearest_neighbor(*i, point, &palette, &neighbors, &distances);

                    *i = j;
                    let err = array::from_fn(|i| point[i] - nearest[i]);

                    arr_mul_add_assign(&mut error1[x], strength * (7.0 / 16.0), &err);
                    arr_mul_add_assign(&mut error2[x + 2], strength * (3.0 / 16.0), &err);
                    arr_mul_add_assign(&mut error2[x + 1], strength * (5.0 / 16.0), &err);

                    let e = &mut error2[x];
                    for i in 0..N {
                        e[i] = strength * (1.0 / 16.0) * err[i];
                    }
                }
            }

            std::mem::swap(&mut error1, &mut error2);
            error2[1] = [0.0; N];
            error2[width] = [0.0; N];
        }
    }

    fn dither<Color, const N: usize>(
        &self,
        palette: &[Color],
        indices: &mut [u8],
        original_colors: &[Color],
        width: u32,
        _height: u32,
    ) where
        Color: ColorComponents<T, N>,
    {
        let FloydSteinberg(strength) = *self;
        let width = width as usize;
        let palette = palette
            .as_arrays()
            .iter()
            .map(|c| c.map(Into::into))
            .collect::<Vec<_>>();

        let (neighbors, distances) = distance_table(&palette);

        let mut error = vec![[0.0; N]; 2 * (width + 2)];
        let (mut error1, mut error2) = error.split_at_mut(width + 2);

        for (row, (indices, colors)) in indices
            .chunks_exact_mut(width)
            .zip(original_colors.as_arrays().chunks_exact(width))
            .enumerate()
        {
            #[inline]
            fn arr_mul_add_assign<const N: usize>(
                arr: &mut [f64; N],
                alpha: f64,
                other: &[f64; N],
            ) {
                for i in 0..N {
                    arr[i] += alpha * other[i];
                }
            }

            if row % 2 == 0 {
                for (x, (i, &og)) in indices.iter_mut().zip(colors).enumerate() {
                    let mut point = og.map(Into::into);
                    let err = error1[x + 1];
                    for i in 0..N {
                        point[i] += err[i];
                    }

                    let (j, nearest) =
                        nearest_neighbor(*i, point, &palette, &neighbors, &distances);

                    *i = j;
                    let err = array::from_fn(|i| point[i] - nearest[i]);

                    arr_mul_add_assign(&mut error1[x + 2], strength * (7.0 / 16.0), &err);
                    arr_mul_add_assign(&mut error2[x], strength * (3.0 / 16.0), &err);
                    arr_mul_add_assign(&mut error2[x + 1], strength * (5.0 / 16.0), &err);

                    let e = &mut error2[x + 2];
                    for i in 0..N {
                        e[i] = strength * (1.0 / 16.0) * err[i];
                    }
                }
            } else {
                for (x, (i, &og)) in indices.iter_mut().zip(colors).enumerate().rev() {
                    let mut point = og.map(Into::into);
                    let err = error1[x + 1];
                    for i in 0..N {
                        point[i] += err[i];
                    }

                    let (j, nearest) =
                        nearest_neighbor(*i, point, &palette, &neighbors, &distances);

                    *i = j;
                    let err = array::from_fn(|i| point[i] - nearest[i]);

                    arr_mul_add_assign(&mut error1[x], strength * (7.0 / 16.0), &err);
                    arr_mul_add_assign(&mut error2[x + 2], strength * (3.0 / 16.0), &err);
                    arr_mul_add_assign(&mut error2[x + 1], strength * (5.0 / 16.0), &err);

                    let e = &mut error2[x];
                    for i in 0..N {
                        e[i] = strength * (1.0 / 16.0) * err[i];
                    }
                }
            }

            std::mem::swap(&mut error1, &mut error2);
            error2[1] = [0.0; N];
            error2[width] = [0.0; N];
        }
    }
}
