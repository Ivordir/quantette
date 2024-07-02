//! Contains the [`PalettePipeline`] builder struct for the high level API.

#[cfg(all(feature = "colorspaces", feature = "threads"))]
use crate::colorspace::convert_color_slice_par;
#[cfg(any(feature = "colorspaces", feature = "kmeans"))]
use crate::UniqueColorCounts;
use crate::{
    wu::{self, Binner3},
    AboveMaxLen, ColorComponents, ColorCounts, ColorSlice, ColorSpace, ImagePipeline, PaletteSize,
    QuantizeMethod, SumPromotion, ZeroedIsZero,
};
#[cfg(feature = "image")]
use image::RgbImage;
use num_traits::AsPrimitive;
use palette::Srgb;
#[cfg(feature = "kmeans")]
use {
    super::num_samples,
    crate::{
        kmeans::{self, Centroids},
        KmeansOptions,
    },
};
#[cfg(feature = "colorspaces")]
use {
    crate::colorspace::{convert_color_slice, from_srgb, to_srgb},
    palette::{Lab, Oklab},
};

/// A builder struct to specify options to create a color palette for an image or slice of colors.
///
/// # Examples
/// To start, create a [`PalettePipeline`] from a [`RgbImage`] (note that the `image` feature is needed):
/// ```no_run
/// # use quantette::PalettePipeline;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let img = image::open("some image")?.into_rgb8();
/// let pipeline = PalettePipeline::try_from(&img)?;
/// # Ok(())
/// # }
/// ```
///
/// A slice of colors can be used instead:
/// ```
/// # use quantette::{PalettePipeline, AboveMaxLen};
/// # use palette::Srgb;
/// # fn main() -> Result<(), AboveMaxLen<u32>> {
/// let srgb = vec![Srgb::new(0, 0, 0)];
/// let pipeline = PalettePipeline::new(srgb.as_slice().try_into()?);
/// # Ok(())
/// # }
/// ```
///
/// Then, you can change the various options:
/// ```
/// # use quantette::{PalettePipeline, AboveMaxLen, ColorSpace, QuantizeMethod};
/// # use palette::Srgb;
/// # fn main() -> Result<(), AboveMaxLen<u32>> {
/// # let srgb = vec![Srgb::new(0, 0, 0)];
/// # let mut pipeline = PalettePipeline::new(srgb.as_slice().try_into()?);
/// pipeline
///     .palette_size(192) // the number of colors in the palette
///     .colorspace(ColorSpace::Oklab) // use a more accurate color space (needs the `colorspaces` feature)
///     .quantize_method(QuantizeMethod::kmeans()); // use a more accurate quantization method (needs the `kmeans` feature)
/// # Ok(())
/// # }
/// ```
///
/// After changing all the options you want, run the pipeline using
/// [`palette`](PalettePipeline::palette()):
/// ```no_run
/// # use quantette::{PalettePipeline, AboveMaxLen, ColorSpace, QuantizeMethod};
/// # use palette::Srgb;
/// # fn main() -> Result<(), AboveMaxLen<u32>> {
/// # let srgb = vec![Srgb::new(0, 0, 0)];
/// # let mut pipeline = PalettePipeline::new(srgb.as_slice().try_into()?);
/// let palette = pipeline
///     .palette_size(192)
///     .colorspace(ColorSpace::Oklab)
///     .quantize_method(QuantizeMethod::kmeans())
///     .palette();
/// # Ok(())
/// # }
/// ```
///
/// Or, in parallel across multiple threads using
/// [`palette_par`](PalettePipeline::palette_par()) (needs the `threads` feature):
/// ```no_run
/// # use quantette::{PalettePipeline, AboveMaxLen, ColorSpace, KmeansOptions};
/// # use palette::Srgb;
/// # fn main() -> Result<(), AboveMaxLen<u32>> {
/// # let srgb = vec![Srgb::new(0, 0, 0)];
/// # let mut pipeline = PalettePipeline::new(srgb.as_slice().try_into()?);
/// let palette = pipeline
///     .palette_size(192)
///     .colorspace(ColorSpace::Oklab)
///     .quantize_method(KmeansOptions::new())
///     .palette_par();
/// # Ok(())
/// # }
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct PalettePipeline<'a> {
    /// The input image/slice of colors.
    pub(crate) colors: ColorSlice<'a, Srgb<u8>>,
    /// The number of colors to put in the palette.
    pub(crate) k: PaletteSize,
    /// The color space to perform color quantization in.
    pub(crate) colorspace: ColorSpace,
    /// The color quantization method to use.
    pub(crate) quantize_method: QuantizeMethod,
    /// Whether or not to deduplicate the input pixels/colors.
    #[cfg(any(feature = "kmeans", feature = "colorspaces"))]
    pub(crate) dedup_pixels: bool,
}

impl<'a> PalettePipeline<'a> {
    /// Creates a new [`PalettePipeline`] with default options.
    pub fn new(colors: ColorSlice<'a, Srgb<u8>>) -> Self {
        Self {
            colors,
            k: PaletteSize::default(),
            colorspace: ColorSpace::Srgb,
            quantize_method: QuantizeMethod::wu(),
            #[cfg(any(feature = "kmeans", feature = "colorspaces"))]
            dedup_pixels: true,
        }
    }

    /// Sets the palette size which determines the (maximum) number of colors to have in the palette.
    ///
    /// The default palette size is [`PaletteSize::MAX`].
    pub fn palette_size(&mut self, size: impl Into<PaletteSize>) -> &mut Self {
        self.k = size.into();
        self
    }

    /// Sets the color space to perform color quantization in.
    ///
    /// See [`ColorSpace`] for more details.
    ///
    /// The default color space is [`ColorSpace::Srgb`].
    #[cfg(feature = "colorspaces")]
    pub fn colorspace(&mut self, colorspace: ColorSpace) -> &mut Self {
        self.colorspace = colorspace;
        self
    }

    /// Sets the color quantization method to use.
    ///
    /// See [`QuantizeMethod`] for more details.
    ///
    /// The default quantization method is [`QuantizeMethod::Wu`].
    #[cfg(feature = "kmeans")]
    pub fn quantize_method(&mut self, quantize_method: impl Into<QuantizeMethod>) -> &mut Self {
        self.quantize_method = quantize_method.into();
        self
    }

    /// Sets whether or not to deduplicate pixels in the image.
    ///
    /// It is recommended to keep this option as default, unless the image is very small or
    /// you have reason to believe that the image contains very little redundancy
    /// (i.e., most pixels are their own unique color).
    /// `quantette` will only deduplicate pixels if it is worth doing so
    /// (i.e., the k-means quantization method is chosen or a color space converision is needed).
    ///
    /// The default value is `true`.
    #[cfg(any(feature = "colorspaces", feature = "kmeans"))]
    pub fn dedup_pixels(&mut self, dedup_pixels: bool) -> &mut Self {
        self.dedup_pixels = dedup_pixels;
        self
    }
}

impl<'a> TryFrom<&'a [Srgb<u8>]> for PalettePipeline<'a> {
    type Error = AboveMaxLen<u32>;

    fn try_from(slice: &'a [Srgb<u8>]) -> Result<Self, Self::Error> {
        Ok(Self::new(slice.try_into()?))
    }
}

#[cfg(feature = "image")]
impl<'a> TryFrom<&'a RgbImage> for PalettePipeline<'a> {
    type Error = AboveMaxLen<u32>;

    fn try_from(image: &'a RgbImage) -> Result<Self, Self::Error> {
        Ok(Self::new(image.try_into()?))
    }
}

impl<'a> From<ImagePipeline<'a>> for PalettePipeline<'a> {
    fn from(pipeline: ImagePipeline<'a>) -> Self {
        let ImagePipeline {
            colors,
            k,
            colorspace,
            quantize_method,
            #[cfg(any(feature = "kmeans", feature = "colorspaces"))]
            dedup_pixels,
            ..
        } = pipeline;

        Self {
            colors,
            k,
            colorspace,
            quantize_method,
            #[cfg(any(feature = "kmeans", feature = "colorspaces"))]
            dedup_pixels,
        }
    }
}

impl<'a> PalettePipeline<'a> {
    /// Runs the pipeline and returns the computed color palette.
    #[must_use]
    pub fn palette(&self) -> Vec<Srgb<u8>> {
        match self.colorspace {
            ColorSpace::Srgb => {
                let Self {
                    colors,
                    k,
                    quantize_method,
                    #[cfg(feature = "kmeans")]
                    dedup_pixels,
                    ..
                } = *self;

                let binner = ColorSpace::default_binner_srgb_u8();

                match quantize_method {
                    #[cfg(feature = "kmeans")]
                    QuantizeMethod::Kmeans(options) if dedup_pixels => {
                        let color_counts = UniqueColorCounts::new(colors, |c| c);
                        palette(&color_counts, k, QuantizeMethod::Kmeans(options), &binner)
                    }
                    quantize_method => palette(&colors, k, quantize_method, &binner),
                }
            }
            #[cfg(feature = "colorspaces")]
            ColorSpace::Lab => self.palette_convert(
                &ColorSpace::default_binner_lab_f32(),
                from_srgb::<Lab>,
                to_srgb,
            ),
            #[cfg(feature = "colorspaces")]
            ColorSpace::Oklab => self.palette_convert(
                &ColorSpace::default_binner_oklab_f32(),
                from_srgb::<Oklab>,
                to_srgb,
            ),
        }
    }

    /// Computes a color palette, converting to a different color space to perform the quantization.
    #[cfg(feature = "colorspaces")]
    fn palette_convert<Color, Component, const B: usize>(
        &self,
        binner: &impl Binner3<Component, B>,
        convert_to: impl Fn(Srgb<u8>) -> Color,
        convert_back: impl Fn(Color) -> Srgb<u8>,
    ) -> Vec<Srgb<u8>>
    where
        Color: ColorComponents<Component, 3>,
        Component: SumPromotion<u32> + Into<f32>,
        Component::Sum: ZeroedIsZero + AsPrimitive<f64>,
        u32: Into<Component::Sum>,
        f32: AsPrimitive<Component>,
    {
        let Self {
            colors, k, quantize_method, dedup_pixels, ..
        } = *self;

        let palette = if dedup_pixels {
            let color_counts = UniqueColorCounts::new(colors, convert_to);
            palette(&color_counts, k, quantize_method, binner)
        } else {
            let colors = convert_color_slice(colors, convert_to);
            let colors = ColorSlice::new_unchecked(&colors);
            palette(&colors, k, quantize_method, binner)
        };

        palette.into_iter().map(convert_back).collect()
    }
}

#[cfg(feature = "threads")]
impl<'a> PalettePipeline<'a> {
    /// Runs the pipeline in parallel and returns the computed color palette.
    #[must_use]
    pub fn palette_par(&self) -> Vec<Srgb<u8>> {
        match self.colorspace {
            ColorSpace::Srgb => {
                let Self {
                    colors,
                    k,
                    quantize_method,
                    #[cfg(feature = "kmeans")]
                    dedup_pixels,
                    ..
                } = *self;

                let binner = ColorSpace::default_binner_srgb_u8();

                match quantize_method {
                    #[cfg(feature = "kmeans")]
                    QuantizeMethod::Kmeans(options) if dedup_pixels => {
                        let color_counts = UniqueColorCounts::new_par(colors, |c| c);
                        palette_par(&color_counts, k, QuantizeMethod::Kmeans(options), &binner)
                    }
                    quantize_method => palette_par(&colors, k, quantize_method, &binner),
                }
            }
            #[cfg(feature = "colorspaces")]
            ColorSpace::Lab => self.palette_convert_par(
                &ColorSpace::default_binner_lab_f32(),
                from_srgb::<Lab>,
                to_srgb,
            ),
            #[cfg(feature = "colorspaces")]
            ColorSpace::Oklab => self.palette_convert_par(
                &ColorSpace::default_binner_oklab_f32(),
                from_srgb::<Oklab>,
                to_srgb,
            ),
        }
    }

    /// Computes a color palette in parallel, converting to a different color space
    /// to perform the quantization.
    #[cfg(feature = "colorspaces")]
    fn palette_convert_par<Color, Component, const B: usize>(
        &self,
        binner: &(impl Binner3<Component, B> + Sync),
        convert_to: impl Fn(Srgb<u8>) -> Color + Send + Sync,
        convert_back: impl Fn(Color) -> Srgb<u8>,
    ) -> Vec<Srgb<u8>>
    where
        Color: ColorComponents<Component, 3> + Send + Sync,
        Component: SumPromotion<u32> + Into<f32> + Send + Sync,
        Component::Sum: ZeroedIsZero + AsPrimitive<f64> + Send,
        u32: Into<Component::Sum>,
        f32: AsPrimitive<Component>,
    {
        let Self {
            colors, k, quantize_method, dedup_pixels, ..
        } = *self;

        let palette = if dedup_pixels {
            let color_counts = UniqueColorCounts::new_par(colors, convert_to);
            palette_par(&color_counts, k, quantize_method, binner)
        } else {
            let colors = convert_color_slice_par(colors, convert_to);
            let colors = ColorSlice::new_unchecked(&colors);
            palette_par(&colors, k, quantize_method, binner)
        };

        palette.into_iter().map(convert_back).collect()
    }
}

/// Computes a color palette.
#[allow(clippy::needless_pass_by_value)]
fn palette<Color, Component, const B: usize>(
    color_counts: &impl ColorCounts<Color, Component, 3>,
    k: PaletteSize,
    method: QuantizeMethod,
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
        QuantizeMethod::Wu(_) => wu::palette(color_counts, k, binner).palette,
        #[cfg(feature = "kmeans")]
        QuantizeMethod::Kmeans(KmeansOptions { sampling_factor, seed, .. }) => {
            let initial_centroids =
                Centroids::new_unchecked(wu::palette(color_counts, k, binner).palette);
            let num_samples = num_samples(sampling_factor, color_counts);
            kmeans::palette(color_counts, num_samples, initial_centroids, seed).palette
        }
    }
}

/// Computes a color palette in parallel.
#[cfg(feature = "threads")]
#[allow(clippy::needless_pass_by_value)]
fn palette_par<Color, Component, const B: usize>(
    color_counts: &(impl ColorCounts<Color, Component, 3> + Send + Sync),
    k: PaletteSize,
    method: QuantizeMethod,
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
        QuantizeMethod::Wu(_) => wu::palette_par(color_counts, k, binner).palette,
        #[cfg(feature = "kmeans")]
        QuantizeMethod::Kmeans(KmeansOptions { sampling_factor, seed, batch_size }) => {
            let initial_centroids =
                Centroids::new_unchecked(wu::palette_par(color_counts, k, binner).palette);
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
