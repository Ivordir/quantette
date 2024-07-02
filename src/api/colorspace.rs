//! Contains the supported color spaces and utility functions for converting between them.

use crate::wu::{Binner3, UIntBinner};
#[cfg(all(feature = "threads", feature = "colorspaces"))]
use rayon::prelude::*;
#[cfg(feature = "colorspaces")]
use {
    crate::{wu::FloatBinner, ColorSlice},
    ::palette::{IntoColor, LinSrgb, Srgb},
};

/// The set of supported color spaces that can be used when performing color quantization.
///
/// If the `colorspaces` feature is enabled, then this will add support for
/// the CIELAB and Oklab color spaces. Otherwise, only sRGB is supported.
///
/// See the descriptions on each enum variant for more information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// The sRGB color space.
    ///
    /// Using this color space avoids color space conversion and gives the quickest results.
    /// However, the sRGB color space is not perceptually uniform,
    /// so it can give potentially lackluster results.
    Srgb,
    /// The CIELAB color space.
    ///
    /// This color space is mostly perceptually uniform and should give better results than sRGB.
    /// However, it is recommended to use the Oklab color space instead, since it is
    /// more perceptually uniform compared to CIELAB while having the same computational cost.
    #[cfg(feature = "colorspaces")]
    Lab,
    /// The Oklab color space.
    ///
    /// This color space is perceptually uniform and should give the most accurate results
    /// compared to the other supported color spaces.
    #[cfg(feature = "colorspaces")]
    Oklab,
}

impl ColorSpace {
    /// The valid range of values for `f32` components of a [`Srgb`] color.
    pub const SRGB_F32_COMPONENT_RANGES: [(f32, f32); 3] = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];

    /// The range of possible values for `f32` components of a [`Lab`](palette::Lab) color,
    /// provided that it was converted from a [`Srgb<u8>`] color.
    #[cfg(feature = "colorspaces")]
    pub const LAB_F32_COMPONENT_RANGES_FROM_SRGB: [(f32, f32); 3] = [
        (0.0, 100.0),
        (-86.182686, 98.23433),
        (-107.86016, 94.477974),
    ];

    /// The range of possible values for `f32` components of an [`Oklab`](palette::Oklab) color,
    /// provided that it was converted from a [`Srgb<u8>`] color.
    #[cfg(feature = "colorspaces")]
    pub const OKLAB_F32_COMPONENT_RANGES_FROM_SRGB: [(f32, f32); 3] = [
        (0.0, 1.0),
        (-0.2338874, 0.2762164),
        (-0.31152815, 0.19856972),
    ];

    /// Returns the range of possible values for `f32` components of the given [`ColorSpace`],
    /// provided that the color is or was converted from a [`Srgb<u8>`] color.
    #[must_use]
    pub const fn f32_component_ranges_from_srgb(self) -> [(f32, f32); 3] {
        match self {
            ColorSpace::Srgb => Self::SRGB_F32_COMPONENT_RANGES,
            #[cfg(feature = "colorspaces")]
            ColorSpace::Lab => Self::LAB_F32_COMPONENT_RANGES_FROM_SRGB,
            #[cfg(feature = "colorspaces")]
            ColorSpace::Oklab => Self::OKLAB_F32_COMPONENT_RANGES_FROM_SRGB,
        }
    }

    /// Returns the default binner used to create histograms for [`Srgb<u8>`] colors.
    #[must_use]
    pub fn default_binner_srgb_u8() -> impl Binner3<u8, 32> {
        UIntBinner
    }

    /// Returns the default binner used to create histograms for [`Lab`](palette::Lab) colors
    /// that were converted from [`Srgb<u8>`] colors.
    #[must_use]
    #[cfg(feature = "colorspaces")]
    pub fn default_binner_lab_f32() -> impl Binner3<f32, 32> {
        FloatBinner::new(Self::LAB_F32_COMPONENT_RANGES_FROM_SRGB)
    }

    /// Returns the default binner used to create histograms for [`Oklab`](palette::Oklab) colors
    /// that were converted from [`Srgb<u8>`] colors.
    #[must_use]
    #[cfg(feature = "colorspaces")]
    pub fn default_binner_oklab_f32() -> impl Binner3<f32, 32> {
        FloatBinner::new(Self::OKLAB_F32_COMPONENT_RANGES_FROM_SRGB)
    }
}

/// Convert from [`Srgb<u8>`] to another color type
#[cfg(feature = "colorspaces")]
pub(crate) fn from_srgb<Color>(color: Srgb<u8>) -> Color
where
    LinSrgb: IntoColor<Color>,
{
    color.into_linear().into_color()
}

/// Convert from a color type to [`Srgb<u8>`]
#[cfg(feature = "colorspaces")]
pub(crate) fn to_srgb<Color>(color: Color) -> Srgb<u8>
where
    Color: IntoColor<LinSrgb>,
{
    color.into_color().into_encoding()
}

/// Convert a color slice to a different color type
#[cfg(feature = "colorspaces")]
pub(crate) fn convert_color_slice<FromColor, ToColor>(
    colors: ColorSlice<FromColor>,
    convert: impl Fn(FromColor) -> ToColor,
) -> Vec<ToColor>
where
    FromColor: Copy,
    ToColor: Copy,
{
    colors.iter().copied().map(convert).collect()
}

/// Convert a color slice to a different color type in parallel
#[cfg(all(feature = "colorspaces", feature = "threads"))]
pub(crate) fn convert_color_slice_par<FromColor, ToColor>(
    colors: ColorSlice<FromColor>,
    convert: impl Fn(FromColor) -> ToColor + Send + Sync,
) -> Vec<ToColor>
where
    FromColor: Copy + Send + Sync,
    ToColor: Copy + Send,
{
    colors.par_iter().copied().map(convert).collect()
}
