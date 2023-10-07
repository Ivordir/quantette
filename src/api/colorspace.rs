use crate::{Binner3, UIntBinner};

#[cfg(feature = "colorspaces")]
use crate::{ColorComponents, ColorSlice, FloatBinner};

#[cfg(feature = "colorspaces")]
use ::palette::{cast, IntoColor, LinSrgb, Srgb};
#[cfg(all(feature = "threads", feature = "colorspaces"))]
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
pub enum ColorSpace {
    Srgb,
    #[cfg(feature = "colorspaces")]
    Lab,
    #[cfg(feature = "colorspaces")]
    Oklab,
}

impl ColorSpace {
    pub const SRGB_F32_COMPONENT_RANGES: [(f32, f32); 3] = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];

    #[cfg(feature = "colorspaces")]
    pub const OKLAB_F32_COMPONENT_RANGES_FROM_SRGB: [(f32, f32); 3] = [
        (0.0, 1.0),
        (-0.2338874, 0.2762164),
        (-0.31152815, 0.19856972),
    ];

    #[cfg(feature = "colorspaces")]
    pub const LAB_F32_COMPONENT_RANGES_FROM_SRGB: [(f32, f32); 3] = [
        (0.0, 100.0),
        (-86.182686, 98.23433),
        (-107.86016, 94.477974),
    ];

    #[must_use]
    pub fn f32_component_ranges_from_srgb(self) -> [(f32, f32); 3] {
        match self {
            ColorSpace::Srgb => Self::SRGB_F32_COMPONENT_RANGES,
            #[cfg(feature = "colorspaces")]
            ColorSpace::Lab => Self::LAB_F32_COMPONENT_RANGES_FROM_SRGB,
            #[cfg(feature = "colorspaces")]
            ColorSpace::Oklab => Self::OKLAB_F32_COMPONENT_RANGES_FROM_SRGB,
        }
    }

    #[must_use]
    pub fn default_binner_srgb_u8() -> impl Binner3<u8, 32> {
        UIntBinner
    }

    #[must_use]
    #[cfg(feature = "colorspaces")]
    pub fn default_binner_lab_f32() -> impl Binner3<f32, 32> {
        FloatBinner::new(Self::LAB_F32_COMPONENT_RANGES_FROM_SRGB)
    }

    #[must_use]
    #[cfg(feature = "colorspaces")]
    pub fn default_binner_oklab_f32() -> impl Binner3<f32, 32> {
        FloatBinner::new(Self::OKLAB_F32_COMPONENT_RANGES_FROM_SRGB)
    }
}

#[cfg(feature = "colorspaces")]
pub(crate) fn from_srgb<From, To>(color: From) -> To
where
    From: ColorComponents<u8, 3>,
    LinSrgb: IntoColor<To>,
{
    Srgb::from(cast::into_array(color))
        .into_linear()
        .into_color()
}

#[cfg(feature = "colorspaces")]
pub(crate) fn to_srgb<From, To>(color: From) -> To
where
    To: ColorComponents<u8, 3>,
    From: IntoColor<LinSrgb>,
{
    let srgb: Srgb<u8> = color.into_color().into_encoding();
    cast::from_array(srgb.into())
}

#[cfg(feature = "colorspaces")]
pub(crate) fn convert_color_space<FromColor, ToColor>(
    colors: ColorSlice<FromColor>,
    convert: impl Fn(FromColor) -> ToColor,
) -> Vec<ToColor>
where
    FromColor: Copy,
    ToColor: Copy,
{
    colors.as_slice().iter().copied().map(convert).collect()
}

#[cfg(all(feature = "colorspaces", feature = "threads"))]
pub(crate) fn convert_color_space_par<FromColor, ToColor>(
    colors: ColorSlice<FromColor>,
    convert: impl Fn(FromColor) -> ToColor + Send + Sync,
) -> Vec<ToColor>
where
    FromColor: Copy + Send + Sync,
    ToColor: Copy + Send,
{
    colors.as_slice().par_iter().copied().map(convert).collect()
}
