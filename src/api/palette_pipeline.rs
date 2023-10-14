#[cfg(feature = "kmeans")]
use super::num_samples;

use crate::{
    wu, ColorComponents, ColorCounts, ColorSlice, ColorSpace, ImagePipeline, PaletteSize,
    QuantizeMethod, UniqueColorCounts,
};

#[cfg(all(feature = "colorspaces", feature = "threads"))]
use crate::colorspace::convert_color_space_par;
#[cfg(feature = "image")]
use crate::AboveMaxLen;
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
#[cfg(feature = "colorspaces")]
use palette::{Lab, Oklab};

#[derive(Debug, Clone)]
pub struct PalettePipeline<'a, Color, const N: usize>
where
    Color: ColorComponents<u8, N>,
{
    pub(crate) colors: ColorSlice<'a, Color>,
    pub(crate) k: PaletteSize,
    pub(crate) colorspace: ColorSpace,
    pub(crate) quantize_method: QuantizeMethod<Color>,
    pub(crate) dedup_pixels: bool,
}

impl<'a, Color, const N: usize> PalettePipeline<'a, Color, N>
where
    Color: ColorComponents<u8, N>,
{
    #[must_use]
    pub fn new(colors: ColorSlice<'a, Color>) -> Self {
        Self {
            colors,
            k: PaletteSize::default(),
            colorspace: ColorSpace::Srgb,
            quantize_method: QuantizeMethod::Wu,
            dedup_pixels: true,
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
}

#[cfg(feature = "image")]
impl<'a> TryFrom<&'a RgbImage> for PalettePipeline<'a, Srgb<u8>, 3> {
    type Error = AboveMaxLen<u32>;

    fn try_from(image: &'a RgbImage) -> Result<Self, Self::Error> {
        Ok(Self::new(image.try_into()?))
    }
}

impl<'a, Color, const N: usize> From<ImagePipeline<'a, Color, N>> for PalettePipeline<'a, Color, N>
where
    Color: ColorComponents<u8, N>,
{
    fn from(value: ImagePipeline<'a, Color, N>) -> Self {
        let ImagePipeline {
            colors,
            k,
            colorspace,
            quantize_method,
            dedup_pixels,
            ..
        } = value;

        Self {
            colors,
            k,
            colorspace,
            quantize_method,
            dedup_pixels,
        }
    }
}

impl<'a, Color> PalettePipeline<'a, Color, 3>
where
    Color: ColorComponents<u8, 3>,
{
    #[must_use]
    pub fn palette(self) -> Vec<Color> {
        match self.colorspace {
            ColorSpace::Srgb => {
                let Self {
                    colors, k, quantize_method, dedup_pixels, ..
                } = self;

                match quantize_method {
                    #[cfg(feature = "kmeans")]
                    QuantizeMethod::Kmeans(options) if dedup_pixels => {
                        let color_counts = UniqueColorCounts::new(colors, |c| c);

                        palette(
                            &color_counts,
                            k,
                            QuantizeMethod::Kmeans(options),
                            &ColorSpace::default_binner_srgb_u8(),
                        )
                    }
                    quantize_method => palette(
                        &colors,
                        k,
                        quantize_method,
                        &ColorSpace::default_binner_srgb_u8(),
                    ),
                }
            }
            #[cfg(feature = "colorspaces")]
            ColorSpace::Lab => self.palette_convert(
                &ColorSpace::default_binner_lab_f32(),
                from_srgb::<_, Lab>,
                to_srgb,
            ),
            #[cfg(feature = "colorspaces")]
            ColorSpace::Oklab => self.palette_convert(
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
    fn palette_convert<QuantColor, Component, const B: usize>(
        self,
        binner: &impl Binner3<Component, B>,
        convert_to: impl Fn(Color) -> QuantColor,
        convert_back: impl Fn(QuantColor) -> Color,
    ) -> Vec<Color>
    where
        QuantColor: ColorComponents<Component, 3>,
        Component: SumPromotion<u32> + Into<f32>,
        Component::Sum: ZeroedIsZero + AsPrimitive<f64>,
        u32: Into<Component::Sum>,
        f32: AsPrimitive<Component>,
    {
        let Self {
            colors, k, quantize_method, dedup_pixels, ..
        } = self;

        let quantize_method = Self::convert_quantize_method(quantize_method, &convert_to);

        let palette = if dedup_pixels {
            let color_counts = UniqueColorCounts::new(colors, convert_to);
            palette(&color_counts, k, quantize_method, binner)
        } else {
            let colors = convert_color_space(colors, convert_to);
            let colors = ColorSlice::from_truncated(&colors);
            palette(&colors, k, quantize_method, binner)
        };

        palette.into_iter().map(convert_back).collect()
    }
}

#[cfg(feature = "threads")]
impl<'a, Color> PalettePipeline<'a, Color, 3>
where
    Color: ColorComponents<u8, 3> + Send + Sync,
{
    #[must_use]
    pub fn palette_par(self) -> Vec<Color> {
        match self.colorspace {
            ColorSpace::Srgb => {
                let Self {
                    colors, k, quantize_method, dedup_pixels, ..
                } = self;

                match quantize_method {
                    #[cfg(feature = "kmeans")]
                    QuantizeMethod::Kmeans(options) if dedup_pixels => {
                        let color_counts = UniqueColorCounts::new_par(colors, |c| c);

                        palette_par(
                            &color_counts,
                            k,
                            QuantizeMethod::Kmeans(options),
                            &ColorSpace::default_binner_srgb_u8(),
                        )
                    }
                    quantize_method => palette_par(
                        &colors,
                        k,
                        quantize_method,
                        &ColorSpace::default_binner_srgb_u8(),
                    ),
                }
            }
            #[cfg(feature = "colorspaces")]
            ColorSpace::Lab => self.palette_convert_par(
                &ColorSpace::default_binner_lab_f32(),
                from_srgb::<_, Lab>,
                to_srgb,
            ),
            #[cfg(feature = "colorspaces")]
            ColorSpace::Oklab => self.palette_convert_par(
                &ColorSpace::default_binner_oklab_f32(),
                from_srgb::<_, Oklab>,
                to_srgb,
            ),
        }
    }

    #[cfg(feature = "colorspaces")]
    fn palette_convert_par<QuantColor, Component, const B: usize>(
        self,
        binner: &(impl Binner3<Component, B> + Sync),
        convert_to: impl Fn(Color) -> QuantColor + Send + Sync,
        convert_back: impl Fn(QuantColor) -> Color,
    ) -> Vec<Color>
    where
        QuantColor: ColorComponents<Component, 3> + Send + Sync,
        Component: SumPromotion<u32> + Into<f32> + Send + Sync,
        Component::Sum: ZeroedIsZero + AsPrimitive<f64> + Send,
        u32: Into<Component::Sum>,
        f32: AsPrimitive<Component>,
    {
        let Self {
            colors, k, quantize_method, dedup_pixels, ..
        } = self;

        let quantize_method = Self::convert_quantize_method(quantize_method, &convert_to);

        let palette = if dedup_pixels {
            let color_counts = UniqueColorCounts::new_par(colors, convert_to);

            palette_par(&color_counts, k, quantize_method, binner)
        } else {
            let colors = convert_color_space_par(colors, convert_to);
            let colors = ColorSlice::from_truncated(&colors);
            palette_par(&colors, k, quantize_method, binner)
        };

        palette.into_iter().map(convert_back).collect()
    }
}

fn palette<Color, Component, const B: usize>(
    color_counts: &impl ColorCounts<Color, Component, 3>,
    k: PaletteSize,
    method: QuantizeMethod<Color>,
    binner: &impl Binner3<Component, B>,
) -> Vec<Color>
where
    Color: ColorComponents<Component, 3>,
    Component: SumPromotion<u32> + Into<f32>,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64>,
    u32: Into<Component::Sum>,
    f32: AsPrimitive<Component>,
{
    match method {
        QuantizeMethod::Wu => wu::palette(color_counts, k, binner).palette,
        #[cfg(feature = "kmeans")]
        QuantizeMethod::Kmeans(KmeansOptions {
            sampling_factor, initial_centroids, seed, ..
        }) => {
            let initial_centroids = initial_centroids.unwrap_or_else(|| {
                Centroids::from_truncated(wu::palette(color_counts, k, binner).palette)
            });

            let num_samples = num_samples(sampling_factor, color_counts);

            kmeans::palette(color_counts, num_samples, initial_centroids, seed).palette
        }
    }
}

#[cfg(feature = "threads")]
fn palette_par<Color, Component, const B: usize>(
    color_counts: &(impl ColorCounts<Color, Component, 3> + Send + Sync),
    k: PaletteSize,
    method: QuantizeMethod<Color>,
    binner: &(impl Binner3<Component, B> + Sync),
) -> Vec<Color>
where
    Color: ColorComponents<Component, 3> + Send,
    Component: SumPromotion<u32> + Into<f32> + Sync + Send,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64> + Send,
    u32: Into<Component::Sum>,
    f32: AsPrimitive<Component>,
{
    match method {
        QuantizeMethod::Wu => wu::palette_par(color_counts, k, binner).palette,
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

            kmeans::palette_par(
                color_counts,
                num_samples,
                batch_size,
                initial_centroids,
                seed,
            )
            .palette
        }
    }
}
