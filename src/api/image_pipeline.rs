#[cfg(feature = "kmeans")]
use super::num_samples;

use crate::{
    dither::{Ditherer, FloydSteinberg},
    wu, ColorAndFrequency, ColorComponents, ColorRemap, ColorSlice, ColorSpace, PalettePipeline,
    PaletteSize, QuantizeMethod, RemappableColorCounts,
};

#[cfg(all(feature = "colorspaces", feature = "threads"))]
use crate::colorspace::convert_color_space_par;
#[cfg(feature = "image")]
use crate::AboveMaxLen;
#[cfg(feature = "threads")]
use crate::ParallelColorRemap;
#[cfg(feature = "colorspaces")]
use crate::{
    colorspace::{convert_color_space, from_srgb, to_srgb},
    Binner3, SumPromotion, ZeroedIsZero,
};
#[cfg(feature = "kmeans")]
use crate::{
    kmeans::{self, Centroids},
    KmeansOptions,
};

use palette::Srgb;

#[cfg(feature = "image")]
use image::RgbImage;
#[cfg(feature = "colorspaces")]
use num_traits::AsPrimitive;
#[cfg(feature = "image")]
use palette::cast::IntoComponents;
#[cfg(feature = "colorspaces")]
use palette::{Lab, Oklab};
#[cfg(feature = "threads")]
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct ImagePipeline<'a, Color, const N: usize>
where
    Color: ColorComponents<u8, N>,
{
    pub(crate) colors: ColorSlice<'a, Color>,
    pub(crate) dimensions: (u32, u32),
    pub(crate) k: PaletteSize,
    pub(crate) colorspace: ColorSpace,
    pub(crate) quantize_method: QuantizeMethod<Color>,
    pub(crate) dither: bool,
    pub(crate) dither_strength: f64,
    pub(crate) dedup_pixels: bool,
}

impl<'a, Color, const N: usize> ImagePipeline<'a, Color, N>
where
    Color: ColorComponents<u8, N>,
{
    fn new_unchecked(colors: ColorSlice<'a, Color>, width: u32, height: u32) -> Self {
        Self {
            colors,
            dimensions: (width, height),
            k: PaletteSize::default(),
            colorspace: ColorSpace::Srgb,
            quantize_method: QuantizeMethod::Wu,
            dither: true,
            dither_strength: FloydSteinberg::DEFAULT_STRENGTH,
            dedup_pixels: true,
        }
    }

    #[must_use]
    pub fn new(colors: ColorSlice<'a, Color>, width: u32, height: u32) -> Option<Self> {
        if colors.len() == width as usize * height as usize {
            Some(Self::new_unchecked(colors, width, height))
        } else {
            None
        }
    }

    #[must_use]
    pub fn palette_size(mut self, size: PaletteSize) -> Self {
        self.k = size;
        self
    }

    #[must_use]
    #[cfg(feature = "colorspaces")]
    pub fn colorspace(mut self, colorspace: ColorSpace) -> Self {
        self.colorspace = colorspace;
        self
    }

    #[must_use]
    #[cfg(any(feature = "colorspaces", feature = "kmeans"))]
    pub fn dedup_pixels(mut self, dedup_pixels: bool) -> Self {
        self.dedup_pixels = dedup_pixels;
        self
    }

    #[must_use]
    #[cfg(feature = "kmeans")]
    pub fn quantize_method(mut self, quantize_method: QuantizeMethod<Color>) -> Self {
        self.quantize_method = quantize_method;
        self
    }

    #[must_use]
    pub fn dither(mut self, dither: bool) -> Self {
        self.dither = dither;
        self
    }

    #[must_use]
    pub fn dither_strength(mut self, strength: f64) -> Self {
        self.dither_strength = strength;
        self
    }
}

#[cfg(feature = "image")]
impl<'a> TryFrom<&'a RgbImage> for ImagePipeline<'a, Srgb<u8>, 3> {
    type Error = AboveMaxLen<u32>;

    fn try_from(image: &'a RgbImage) -> Result<Self, Self::Error> {
        Ok(Self::new_unchecked(
            image.try_into()?,
            image.width(),
            image.height(),
        ))
    }
}

impl<'a, Color> ImagePipeline<'a, Color, 3>
where
    Color: ColorComponents<u8, 3>,
{
    #[must_use]
    pub fn palette(self) -> Vec<Color> {
        PalettePipeline::from(self).palette()
    }

    fn ditherer(&self) -> Option<FloydSteinberg> {
        if self.dither {
            Some(FloydSteinberg(self.dither_strength))
        } else {
            None
        }
    }

    #[must_use]
    pub fn indexed_palette(self) -> (Vec<Color>, Vec<u8>) {
        match self.colorspace {
            ColorSpace::Srgb => {
                let ditherer = self.ditherer();
                let Self {
                    colors,
                    k,
                    quantize_method,
                    dedup_pixels,
                    dimensions,
                    ..
                } = self;

                let (width, height) = dimensions;

                match quantize_method {
                    #[cfg(feature = "kmeans")]
                    QuantizeMethod::Kmeans(options) if dedup_pixels => {
                        let color_counts =
                            RemappableColorCounts::remappable_u8_3_colors(colors, |c| c);

                        indexed_palette(
                            &color_counts,
                            width,
                            height,
                            k,
                            QuantizeMethod::Kmeans(options),
                            ditherer,
                            &ColorSpace::default_binner_srgb_u8(),
                        )
                    }
                    quantize_method => indexed_palette(
                        &colors,
                        width,
                        height,
                        k,
                        quantize_method,
                        ditherer,
                        &ColorSpace::default_binner_srgb_u8(),
                    ),
                }
            }
            #[cfg(feature = "colorspaces")]
            ColorSpace::Lab => self.indexed_palette_convert(
                &ColorSpace::default_binner_lab_f32(),
                from_srgb::<_, Lab>,
                to_srgb,
            ),
            #[cfg(feature = "colorspaces")]
            ColorSpace::Oklab => self.indexed_palette_convert(
                &ColorSpace::default_binner_oklab_f32(),
                from_srgb::<_, Oklab>,
                to_srgb,
            ),
        }
    }

    #[cfg(feature = "colorspaces")]
    fn convert_quantize_method<QuantColor>(
        quantize_method: QuantizeMethod<Color>,
        convert_to: impl Fn(Color) -> QuantColor,
    ) -> QuantizeMethod<QuantColor> {
        match quantize_method {
            QuantizeMethod::Wu => QuantizeMethod::Wu,
            #[cfg(feature = "kmeans")]
            QuantizeMethod::Kmeans(KmeansOptions {
                sampling_factor,
                initial_centroids,
                seed,
                #[cfg(feature = "threads")]
                batch_size,
            }) => QuantizeMethod::Kmeans(KmeansOptions {
                initial_centroids: initial_centroids.map(|c| {
                    Centroids::from_truncated(c.into_inner().into_iter().map(&convert_to).collect())
                }),
                sampling_factor,
                seed,
                #[cfg(feature = "threads")]
                batch_size,
            }),
        }
    }

    #[cfg(feature = "colorspaces")]
    fn indexed_palette_convert<QuantColor, Component, const B: usize>(
        self,
        binner: &impl Binner3<Component, B>,
        convert_to: impl Fn(Color) -> QuantColor,
        convert_back: impl Fn(QuantColor) -> Color,
    ) -> (Vec<Color>, Vec<u8>)
    where
        QuantColor: ColorComponents<Component, 3>,
        Component: SumPromotion<u32> + Into<f64>,
        Component::Sum: ZeroedIsZero + AsPrimitive<f64>,
        FloydSteinberg: Ditherer<Component>,
        u32: Into<Component::Sum>,
        f64: AsPrimitive<Component>,
    {
        let ditherer = self.ditherer();
        let Self {
            colors,
            k,
            quantize_method,
            dedup_pixels,
            dimensions,
            ..
        } = self;

        let (width, height) = dimensions;
        let quantize_method = Self::convert_quantize_method(quantize_method, &convert_to);

        let (palette, indices) = if dedup_pixels {
            let color_counts = RemappableColorCounts::remappable_u8_3_colors(colors, convert_to);
            indexed_palette(
                &color_counts,
                width,
                height,
                k,
                quantize_method,
                ditherer,
                binner,
            )
        } else {
            let colors = convert_color_space(colors, convert_to);
            let colors = ColorSlice::from_truncated(&colors);
            indexed_palette(&colors, width, height, k, quantize_method, ditherer, binner)
        };

        let palette = palette.into_iter().map(convert_back).collect();
        (palette, indices)
    }
}

#[cfg(feature = "image")]
impl<'a> ImagePipeline<'a, Srgb<u8>, 3> {
    #[must_use]
    pub fn quantized_rgbimage(self) -> RgbImage {
        let (width, height) = self.dimensions;
        let (palette, indices) = self.indexed_palette();

        let palette = palette.as_slice(); // faster for some reason
        let buf = indices
            .into_iter()
            .map(|i| palette[usize::from(i)])
            .collect::<Vec<_>>()
            .into_components();

        #[allow(clippy::unwrap_used)]
        {
            // indices.len() will be equal to width * height,
            // so buf should be large enough by nature of its construction
            RgbImage::from_vec(width, height, buf).unwrap()
        }
    }
}

#[cfg(feature = "threads")]
impl<'a, Color> ImagePipeline<'a, Color, 3>
where
    Color: ColorComponents<u8, 3> + Send + Sync,
{
    #[must_use]
    pub fn palette_par(self) -> Vec<Color> {
        PalettePipeline::from(self).palette_par()
    }

    #[must_use]
    pub fn indexed_palette_par(self) -> (Vec<Color>, Vec<u8>) {
        match self.colorspace {
            ColorSpace::Srgb => {
                let ditherer = self.ditherer();
                let Self {
                    colors,
                    k,
                    quantize_method,
                    dedup_pixels,
                    dimensions,
                    ..
                } = self;

                let (width, height) = dimensions;

                match quantize_method {
                    #[cfg(feature = "kmeans")]
                    QuantizeMethod::Kmeans(options) if dedup_pixels => {
                        let color_counts =
                            RemappableColorCounts::remappable_u8_3_colors_par(colors, |c| c);

                        indexed_palette_par(
                            &color_counts,
                            width,
                            height,
                            k,
                            QuantizeMethod::Kmeans(options),
                            ditherer,
                            &ColorSpace::default_binner_srgb_u8(),
                        )
                    }
                    quantize_method => indexed_palette_par(
                        &colors,
                        width,
                        height,
                        k,
                        quantize_method,
                        ditherer,
                        &ColorSpace::default_binner_srgb_u8(),
                    ),
                }
            }
            #[cfg(feature = "colorspaces")]
            ColorSpace::Lab => self.indexed_palette_convert_par(
                &ColorSpace::default_binner_lab_f32(),
                from_srgb::<_, Lab>,
                to_srgb,
            ),
            #[cfg(feature = "colorspaces")]
            ColorSpace::Oklab => self.indexed_palette_convert_par(
                &ColorSpace::default_binner_oklab_f32(),
                from_srgb::<_, Oklab>,
                to_srgb,
            ),
        }
    }

    #[cfg(feature = "colorspaces")]
    fn indexed_palette_convert_par<QuantColor, Component, const B: usize>(
        self,
        binner: &(impl Binner3<Component, B> + Sync),
        convert_to: impl Fn(Color) -> QuantColor + Send + Sync,
        convert_back: impl Fn(QuantColor) -> Color,
    ) -> (Vec<Color>, Vec<u8>)
    where
        QuantColor: ColorComponents<Component, 3> + Send + Sync,
        Component: SumPromotion<u32> + Into<f64> + Send + Sync,
        Component::Sum: ZeroedIsZero + AsPrimitive<f64> + Send,
        u32: Into<Component::Sum>,
        f64: AsPrimitive<Component>,
    {
        let ditherer = self.ditherer();
        let Self {
            colors,
            k,
            quantize_method,
            dedup_pixels,
            dimensions,
            ..
        } = self;

        let (width, height) = dimensions;
        let quantize_method = Self::convert_quantize_method(quantize_method, &convert_to);

        let (palette, indices) = if dedup_pixels {
            let color_counts =
                RemappableColorCounts::remappable_u8_3_colors_par(colors, convert_to);

            indexed_palette_par(
                &color_counts,
                width,
                height,
                k,
                quantize_method,
                ditherer,
                binner,
            )
        } else {
            let colors = convert_color_space_par(colors, convert_to);
            let colors = ColorSlice::from_truncated(&colors);
            indexed_palette_par(&colors, width, height, k, quantize_method, ditherer, binner)
        };

        let palette = palette.into_iter().map(convert_back).collect();
        (palette, indices)
    }
}

#[cfg(all(feature = "threads", feature = "image"))]
impl<'a> ImagePipeline<'a, Srgb<u8>, 3> {
    #[must_use]
    pub fn quantized_rgbimage_par(self) -> RgbImage {
        let (width, height) = self.dimensions;
        let (palette, indices) = self.indexed_palette_par();

        let palette = palette.as_slice(); // faster for some reason
        let buf = indices
            .par_iter()
            .map(|&i| palette[usize::from(i)])
            .collect::<Vec<_>>()
            .into_components();

        #[allow(clippy::unwrap_used)]
        {
            // indices.len() will be equal to width * height,
            // so buf should be large enough by nature of its construction
            RgbImage::from_vec(width, height, buf).unwrap()
        }
    }
}

fn indexed_palette<Color, Component, const B: usize>(
    color_counts: &(impl ColorAndFrequency<Color, Component, 3> + ColorRemap),
    width: u32,
    height: u32,
    k: PaletteSize,
    method: QuantizeMethod<Color>,
    ditherer: Option<FloydSteinberg>,
    binner: &impl Binner3<Component, B>,
) -> (Vec<Color>, Vec<u8>)
where
    Color: ColorComponents<Component, 3>,
    Component: SumPromotion<u32> + Into<f64>,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64>,
    FloydSteinberg: Ditherer<Component>,
    u32: Into<Component::Sum>,
    f64: AsPrimitive<Component>,
{
    let (palette, mut indices) = match method {
        QuantizeMethod::Wu => {
            let res = wu::indexed_palette(color_counts, k, binner);
            (res.palette, res.indices)
        }
        #[cfg(feature = "kmeans")]
        QuantizeMethod::Kmeans(KmeansOptions {
            sampling_factor, initial_centroids, seed, ..
        }) => {
            let initial_centroids = initial_centroids.unwrap_or_else(|| {
                Centroids::from_truncated(wu::palette(color_counts, k, binner).palette)
            });

            let num_samples = num_samples(sampling_factor, color_counts);

            let res = kmeans::indexed_palette(color_counts, num_samples, initial_centroids, seed);
            (res.palette, res.indices)
        }
    };

    if let Some(ditherer) = ditherer {
        if let Some(original_indices) = color_counts.indices() {
            ditherer.dither_indexed(
                &palette,
                &mut indices,
                color_counts.colors(),
                original_indices,
                width,
                height,
            );
        } else {
            ditherer.dither(&palette, &mut indices, color_counts.colors(), width, height);
        }
    }

    (palette, indices)
}

#[cfg(feature = "threads")]
fn indexed_palette_par<Color, Component, const B: usize>(
    color_counts: &(impl ColorAndFrequency<Color, Component, 3> + ParallelColorRemap + Send + Sync),
    width: u32,
    height: u32,
    k: PaletteSize,
    method: QuantizeMethod<Color>,
    ditherer: Option<FloydSteinberg>,
    binner: &(impl Binner3<Component, B> + Sync),
) -> (Vec<Color>, Vec<u8>)
where
    Color: ColorComponents<Component, 3> + Send,
    Component: SumPromotion<u32> + Into<f64> + Send + Sync,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64> + Send,
    FloydSteinberg: Ditherer<Component>,
    u32: Into<Component::Sum>,
    f64: AsPrimitive<Component>,
{
    let (palette, mut indices) = match method {
        QuantizeMethod::Wu => {
            let res = wu::indexed_palette_par(color_counts, k, binner);
            (res.palette, res.indices)
        }
        #[cfg(feature = "kmeans")]
        QuantizeMethod::Kmeans(KmeansOptions {
            sampling_factor,
            initial_centroids,
            seed,
            batch_size,
        }) => {
            let initial_centroids = initial_centroids.unwrap_or_else(|| {
                Centroids::from_truncated(wu::palette_par(color_counts, k, binner).palette)
            });

            let num_samples = num_samples(sampling_factor, color_counts);

            let res = kmeans::indexed_palette_par(
                color_counts,
                num_samples,
                batch_size,
                initial_centroids,
                seed,
            );
            (res.palette, res.indices)
        }
    };

    if let Some(ditherer) = ditherer {
        if let Some(original_indices) = color_counts.indices() {
            ditherer.dither_indexed(
                &palette,
                &mut indices,
                color_counts.colors(),
                original_indices,
                width,
                height,
            );
        } else {
            ditherer.dither(&palette, &mut indices, color_counts.colors(), width, height);
        }
    }

    (palette, indices)
}
