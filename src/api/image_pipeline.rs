//! Contains the [`ImagePipeline`] builder struct for the high level API.

#[cfg(all(feature = "colorspaces", feature = "threads"))]
use crate::colorspace::convert_color_slice_par;
#[cfg(feature = "threads")]
use crate::ColorCountsParallelRemap;
#[cfg(any(feature = "colorspaces", feature = "kmeans"))]
use crate::IndexedColorCounts;
use crate::{
    dither::FloydSteinberg,
    wu::{self, Binner3},
    ColorComponents, ColorCounts, ColorCountsRemap, ColorSlice, ColorSpace, PalettePipeline,
    PaletteSize, QuantizeMethod, SumPromotion, ZeroedIsZero,
};
use num_traits::AsPrimitive;
use palette::Srgb;
#[cfg(all(feature = "threads", feature = "image"))]
use rayon::prelude::*;
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
#[cfg(feature = "image")]
use {crate::AboveMaxLen, image::RgbImage, palette::cast::IntoComponents};

/// A builder struct to specify options to create a quantized image or an indexed palette from an image.
///
/// # Examples
/// To start, create a [`ImagePipeline`] from a [`RgbImage`] (note that the `image` feature is needed):
/// ```no_run
/// # use quantette::ImagePipeline;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let img = image::open("some image")?.into_rgb8();
/// let mut pipeline = ImagePipeline::try_from(&img)?;
/// # Ok(())
/// # }
/// ```
///
/// Then, you can change different options like the number of colors in the palette:
/// ```
/// # use quantette::{ImagePipeline, AboveMaxLen, ColorSpace, QuantizeMethod};
/// # use palette::Srgb;
/// # fn main() -> Result<(), AboveMaxLen<u32>> {
/// # let srgb = vec![Srgb::new(0, 0, 0)];
/// # let mut pipeline = ImagePipeline::new(srgb.as_slice().try_into()?, 1, 1).unwrap();
/// let pipeline = pipeline
///     .palette_size(192)
///     .colorspace(ColorSpace::Oklab)
///     .quantize_method(QuantizeMethod::kmeans());
/// # Ok(())
/// # }
/// ```
///
/// [`ImagePipeline`] has all options that [`PalettePipeline`] does,
/// so you can check its documentation example for more information.
/// In addition, [`ImagePipeline`] has options to control the dither behavior:
/// ```
/// # use quantette::{ImagePipeline, AboveMaxLen, ColorSpace, QuantizeMethod};
/// # use palette::Srgb;
/// # fn main() -> Result<(), AboveMaxLen<u32>> {
/// # let srgb = vec![Srgb::new(0, 0, 0)];
/// # let mut pipeline = ImagePipeline::new(srgb.as_slice().try_into()?, 1, 1).unwrap();
/// let pipeline = pipeline
///     .palette_size(192)
///     .colorspace(ColorSpace::Oklab)
///     .quantize_method(QuantizeMethod::kmeans())
///     .dither_error_diffusion(0.8);
/// # Ok(())
/// # }
/// ```
///
/// Finally, run the pipeline:
/// ```no_run
/// # use quantette::{ImagePipeline, AboveMaxLen};
/// # use palette::Srgb;
/// # fn main() -> Result<(), AboveMaxLen<u32>> {
/// # let srgb = vec![Srgb::new(0, 0, 0)];
/// # let pipeline = ImagePipeline::new(srgb.as_slice().try_into()?, 1, 1).unwrap();
/// let image = pipeline.quantized_rgbimage();
/// # Ok(())
/// # }
/// ```
///
/// Or, in parallel across multiple threads (needs the `threads` feature):
/// ```no_run
/// # use quantette::{ImagePipeline, AboveMaxLen};
/// # use palette::Srgb;
/// # fn main() -> Result<(), AboveMaxLen<u32>> {
/// # let srgb = vec![Srgb::new(0, 0, 0)];
/// # let pipeline = ImagePipeline::new(srgb.as_slice().try_into()?, 1, 1).unwrap();
/// let image = pipeline.quantized_rgbimage_par();
/// # Ok(())
/// # }
/// ```
///
/// Instead of an [`RgbImage`] you can also get an indexed image
/// (a palette and a list of indices into the palette):
/// ```no_run
/// # use quantette::{ImagePipeline, AboveMaxLen};
/// # use palette::Srgb;
/// # fn main() -> Result<(), AboveMaxLen<u32>> {
/// # let srgb = vec![Srgb::new(0, 0, 0)];
/// # let pipeline = ImagePipeline::new(srgb.as_slice().try_into()?, 1, 1).unwrap();
/// let (palette, indices) = pipeline.indexed_palette();
/// let (palette, indices) = pipeline.indexed_palette_par();
/// # Ok(())
/// # }
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct ImagePipeline<'a> {
    /// The input image as a flat slice of pixels.
    pub(crate) colors: ColorSlice<'a, Srgb<u8>>,
    /// The dimensions of the image.
    pub(crate) dimensions: (u32, u32),
    /// The number of colors to put in the palette.
    pub(crate) k: PaletteSize,
    /// The color space to perform color quantization in.
    pub(crate) colorspace: ColorSpace,
    /// The color quantization method to use.
    pub(crate) quantize_method: QuantizeMethod,
    /// Whether or not to perform dithering on the image.
    pub(crate) dither: bool,
    /// The error diffusion factor to use when dithering.
    pub(crate) dither_error_diffusion: f32,
    /// Whether or not to deduplicate the input pixels/colors.
    #[cfg(any(feature = "kmeans", feature = "colorspaces"))]
    pub(crate) dedup_pixels: bool,
}

impl<'a> ImagePipeline<'a> {
    /// Creates a new [`ImagePipeline`] with default options
    /// and does not validate the size of the input image/slice.
    fn new_unchecked(colors: ColorSlice<'a, Srgb<u8>>, width: u32, height: u32) -> Self {
        Self {
            colors,
            dimensions: (width, height),
            k: PaletteSize::default(),
            colorspace: ColorSpace::Srgb,
            quantize_method: QuantizeMethod::wu(),
            dither: true,
            dither_error_diffusion: FloydSteinberg::DEFAULT_ERROR_DIFFUSION,
            #[cfg(any(feature = "kmeans", feature = "colorspaces"))]
            dedup_pixels: true,
        }
    }

    /// Creates a new [`ImagePipeline`] with default options.
    /// Returns `None` if the length of `colors` is not equal to `width * height`.
    #[must_use]
    pub fn new(colors: ColorSlice<'a, Srgb<u8>>, width: u32, height: u32) -> Option<Self> {
        if colors.len() == width as usize * height as usize {
            Some(Self::new_unchecked(colors, width, height))
        } else {
            None
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

    /// Sets whether or not to apply dithering to the image.
    ///
    /// The current implementation uses Floyd–Steinberg dithering.
    /// The dithering strength/aggresiveness can be controlled via
    /// [`ImagePipeline::dither_error_diffusion`].
    ///
    /// The default value is `true`.
    pub fn dither(&mut self, dither: bool) -> &mut Self {
        self.dither = dither;
        self
    }

    /// Sets the error diffusion factor for the dither.
    ///
    /// A value of `1.0` diffuses all error to the neighboring pixels and may result in color bleed.
    /// Values less than `1.0` only diffuse part of the error for a more tamed dither.
    /// The given `diffusion` should be in the range `0.0..=1.0`,
    /// otherwise the default error diffusion will be used as a fallback.
    ///
    /// The default value is [`FloydSteinberg::DEFAULT_ERROR_DIFFUSION`].
    pub fn dither_error_diffusion(&mut self, diffusion: f32) -> &mut Self {
        self.dither_error_diffusion = diffusion;
        self
    }
}

#[cfg(feature = "image")]
impl<'a> TryFrom<&'a RgbImage> for ImagePipeline<'a> {
    type Error = AboveMaxLen<u32>;

    fn try_from(image: &'a RgbImage) -> Result<Self, Self::Error> {
        Ok(Self::new_unchecked(
            image.try_into()?,
            image.width(),
            image.height(),
        ))
    }
}

impl<'a> ImagePipeline<'a> {
    /// Runs the pipeline and returns the computed color palette.
    #[must_use]
    pub fn palette(&self) -> Vec<Srgb<u8>> {
        PalettePipeline::from(self.clone()).palette()
    }

    /// Creates the ditherer specified by the current options.
    fn ditherer(&self) -> Option<FloydSteinberg> {
        if self.dither {
            Some(
                FloydSteinberg::with_error_diffusion(self.dither_error_diffusion)
                    .unwrap_or_default(),
            )
        } else {
            None
        }
    }

    /// Runs the pipeline and returns the quantized image as a list of indices into a palette.
    #[must_use]
    pub fn indexed_palette(&self) -> (Vec<Srgb<u8>>, Vec<u8>) {
        match self.colorspace {
            ColorSpace::Srgb => {
                let ditherer = self.ditherer();
                let Self {
                    colors,
                    k,
                    quantize_method,
                    #[cfg(feature = "kmeans")]
                    dedup_pixels,
                    dimensions,
                    ..
                } = *self;

                let (width, height) = dimensions;
                let binner = ColorSpace::default_binner_srgb_u8();

                match quantize_method {
                    #[cfg(feature = "kmeans")]
                    QuantizeMethod::Kmeans(options) if dedup_pixels => {
                        let color_counts = IndexedColorCounts::new(colors, |c| c);
                        indexed_palette(
                            &color_counts,
                            width,
                            height,
                            k,
                            QuantizeMethod::Kmeans(options),
                            ditherer,
                            &binner,
                        )
                    }
                    quantize_method => indexed_palette(
                        &colors,
                        width,
                        height,
                        k,
                        quantize_method,
                        ditherer,
                        &binner,
                    ),
                }
            }
            #[cfg(feature = "colorspaces")]
            ColorSpace::Lab => self.indexed_palette_convert(
                &ColorSpace::default_binner_lab_f32(),
                from_srgb::<Lab>,
                to_srgb,
            ),
            #[cfg(feature = "colorspaces")]
            ColorSpace::Oklab => self.indexed_palette_convert(
                &ColorSpace::default_binner_oklab_f32(),
                from_srgb::<Oklab>,
                to_srgb,
            ),
        }
    }

    /// Computes an indexed palette, converting to a different color space to perform the quantization.
    #[cfg(feature = "colorspaces")]
    fn indexed_palette_convert<Color, Component, const B: usize>(
        &self,
        binner: &impl Binner3<Component, B>,
        convert_to: impl Fn(Srgb<u8>) -> Color,
        convert_back: impl Fn(Color) -> Srgb<u8>,
    ) -> (Vec<Srgb<u8>>, Vec<u8>)
    where
        Color: ColorComponents<Component, 3>,
        Component: SumPromotion<u32> + Into<f32>,
        Component::Sum: ZeroedIsZero + AsPrimitive<f64>,
        u32: Into<Component::Sum>,
        f32: AsPrimitive<Component>,
    {
        let ditherer = self.ditherer();
        let Self {
            colors,
            k,
            quantize_method,
            dedup_pixels,
            dimensions,
            ..
        } = *self;

        let (width, height) = dimensions;

        let (palette, indices) = if dedup_pixels {
            let color_counts = IndexedColorCounts::new(colors, convert_to);
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
            let colors = convert_color_slice(colors, convert_to);
            let colors = ColorSlice::new_unchecked(&colors);
            indexed_palette(&colors, width, height, k, quantize_method, ditherer, binner)
        };

        let palette = palette.into_iter().map(convert_back).collect();
        (palette, indices)
    }
}

#[cfg(feature = "image")]
impl<'a> ImagePipeline<'a> {
    /// Runs the pipeline and returns the quantized image.
    #[must_use]
    pub fn quantized_rgbimage(&self) -> RgbImage {
        let (width, height) = self.dimensions;
        let (palette, indices) = self.indexed_palette();

        let palette = palette.as_slice(); // faster for some reason
        let buf = indices
            .into_iter()
            .map(|i| palette[usize::from(i)])
            .collect::<Vec<_>>()
            .into_components();

        #[allow(clippy::expect_used)]
        {
            // indices.len() will be equal to width * height,
            // so buf should be large enough by nature of its construction
            RgbImage::from_vec(width, height, buf).expect("large enough buffer")
        }
    }
}

#[cfg(feature = "threads")]
impl<'a> ImagePipeline<'a> {
    /// Runs the pipeline in parallel and returns the computed color palette.
    #[must_use]
    pub fn palette_par(&self) -> Vec<Srgb<u8>> {
        PalettePipeline::from(self.clone()).palette_par()
    }

    /// Runs the pipeline in parallel and returns the quantized image as a
    /// list of indices into a palette.
    #[must_use]
    pub fn indexed_palette_par(&self) -> (Vec<Srgb<u8>>, Vec<u8>) {
        match self.colorspace {
            ColorSpace::Srgb => {
                let ditherer = self.ditherer();
                let Self {
                    colors,
                    k,
                    quantize_method,
                    #[cfg(feature = "kmeans")]
                    dedup_pixels,
                    dimensions,
                    ..
                } = *self;

                let (width, height) = dimensions;
                let binner = ColorSpace::default_binner_srgb_u8();

                match quantize_method {
                    #[cfg(feature = "kmeans")]
                    QuantizeMethod::Kmeans(options) if dedup_pixels => {
                        let color_counts = IndexedColorCounts::new_par(colors, |c| c);
                        indexed_palette_par(
                            &color_counts,
                            width,
                            height,
                            k,
                            QuantizeMethod::Kmeans(options),
                            ditherer,
                            &binner,
                        )
                    }
                    quantize_method => indexed_palette_par(
                        &colors,
                        width,
                        height,
                        k,
                        quantize_method,
                        ditherer,
                        &binner,
                    ),
                }
            }
            #[cfg(feature = "colorspaces")]
            ColorSpace::Lab => self.indexed_palette_convert_par(
                &ColorSpace::default_binner_lab_f32(),
                from_srgb::<Lab>,
                to_srgb,
            ),
            #[cfg(feature = "colorspaces")]
            ColorSpace::Oklab => self.indexed_palette_convert_par(
                &ColorSpace::default_binner_oklab_f32(),
                from_srgb::<Oklab>,
                to_srgb,
            ),
        }
    }

    /// Computes an indexed palette in parallel, converting to a different color space
    /// to perform the quantization.
    #[cfg(feature = "colorspaces")]
    fn indexed_palette_convert_par<Color, Component, const B: usize>(
        &self,
        binner: &(impl Binner3<Component, B> + Sync),
        convert_to: impl Fn(Srgb<u8>) -> Color + Send + Sync,
        convert_back: impl Fn(Color) -> Srgb<u8>,
    ) -> (Vec<Srgb<u8>>, Vec<u8>)
    where
        Color: ColorComponents<Component, 3> + Send + Sync,
        Component: SumPromotion<u32> + Into<f32> + Send + Sync,
        Component::Sum: ZeroedIsZero + AsPrimitive<f64> + Send,
        u32: Into<Component::Sum>,
        f32: AsPrimitive<Component>,
    {
        let ditherer = self.ditherer();
        let Self {
            colors,
            k,
            quantize_method,
            dedup_pixels,
            dimensions,
            ..
        } = *self;

        let (width, height) = dimensions;

        let (palette, indices) = if dedup_pixels {
            let color_counts = IndexedColorCounts::new_par(colors, convert_to);
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
            let colors = convert_color_slice_par(colors, convert_to);
            let colors = ColorSlice::new_unchecked(&colors);
            indexed_palette_par(&colors, width, height, k, quantize_method, ditherer, binner)
        };

        let palette = palette.into_iter().map(convert_back).collect();
        (palette, indices)
    }
}

#[cfg(all(feature = "threads", feature = "image"))]
impl<'a> ImagePipeline<'a> {
    /// Runs the pipeline in parallel and returns the quantized image.
    #[must_use]
    pub fn quantized_rgbimage_par(&self) -> RgbImage {
        let (width, height) = self.dimensions;
        let (palette, indices) = self.indexed_palette_par();

        let palette = palette.as_slice(); // faster for some reason
        let buf = indices
            .par_iter()
            .map(|&i| palette[usize::from(i)])
            .collect::<Vec<_>>()
            .into_components();

        #[allow(clippy::expect_used)]
        {
            // indices.len() will be equal to width * height,
            // so buf should be large enough by nature of its construction
            RgbImage::from_vec(width, height, buf).expect("large enough buffer")
        }
    }
}

/// Computes a color palette and the indices into it.
#[allow(clippy::needless_pass_by_value)]
fn indexed_palette<Color, Component, const B: usize>(
    color_counts: &impl ColorCountsRemap<Color, Component, 3>,
    width: u32,
    height: u32,
    k: PaletteSize,
    method: QuantizeMethod,
    ditherer: Option<FloydSteinberg>,
    binner: &impl Binner3<Component, B>,
) -> (Vec<Color>, Vec<u8>)
where
    Color: ColorComponents<Component, 3>,
    Component: SumPromotion<u32> + Into<f32>,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64>,
    u32: Into<Component::Sum>,
    f32: AsPrimitive<Component>,
{
    let (palette, mut indices) = match method {
        QuantizeMethod::Wu(_) => {
            let res = wu::indexed_palette(color_counts, k, binner);
            (res.palette, res.indices)
        }
        #[cfg(feature = "kmeans")]
        QuantizeMethod::Kmeans(KmeansOptions { sampling_factor, seed, .. }) => {
            let initial_centroids =
                Centroids::new_unchecked(wu::palette(color_counts, k, binner).palette);
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

/// Computes a color palette and the indices into it in parallel.
#[cfg(feature = "threads")]
#[allow(clippy::needless_pass_by_value)]
fn indexed_palette_par<Color, Component, const B: usize>(
    color_counts: &(impl ColorCountsParallelRemap<Color, Component, 3> + Send + Sync),
    width: u32,
    height: u32,
    k: PaletteSize,
    method: QuantizeMethod,
    ditherer: Option<FloydSteinberg>,
    binner: &(impl Binner3<Component, B> + Sync),
) -> (Vec<Color>, Vec<u8>)
where
    Color: ColorComponents<Component, 3> + Send + Sync,
    Component: SumPromotion<u32> + Into<f32> + Send + Sync,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64> + Send,
    u32: Into<Component::Sum>,
    f32: AsPrimitive<Component>,
{
    let (palette, mut indices) = match method {
        QuantizeMethod::Wu(_) => {
            let res = wu::indexed_palette_par(color_counts, k, binner);
            (res.palette, res.indices)
        }
        #[cfg(feature = "kmeans")]
        QuantizeMethod::Kmeans(KmeansOptions { sampling_factor, seed, batch_size }) => {
            let initial_centroids =
                Centroids::new_unchecked(wu::palette_par(color_counts, k, binner).palette);
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
            ditherer.dither_indexed_par(
                &palette,
                &mut indices,
                color_counts.colors(),
                original_indices,
                width,
                height,
            );
        } else {
            ditherer.dither_par(&palette, &mut indices, color_counts.colors(), width, height);
        }
    }

    (palette, indices)
}
