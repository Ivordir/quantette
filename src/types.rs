use crate::{ColorComponents, ColorCounts, ColorRemap, MAX_COLORS, MAX_PIXELS};

#[cfg(feature = "threads")]
use crate::ParallelColorRemap;

use std::{
    error::Error,
    fmt::{Debug, Display},
};

#[cfg(feature = "image")]
use image::RgbImage;
#[cfg(feature = "image")]
use palette::{cast::ComponentsAs, Srgb};

#[derive(Debug, Clone, Copy)]
pub struct AboveMaxLen<T>(pub T);

impl<T: Display> Display for AboveMaxLen<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "above the maximum length of {}", self.0)
    }
}

impl<T: Debug + Display> Error for AboveMaxLen<T> {}

#[derive(Debug)]
#[repr(transparent)]
pub struct ColorSlice<'a, Color>(&'a [Color]);

impl<'a, Color> Clone for ColorSlice<'a, Color> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, Color> Copy for ColorSlice<'a, Color> {}

impl<'a, Color> ColorSlice<'a, Color> {
    pub fn from_truncated(colors: &'a [Color]) -> Self {
        Self(&colors[..colors.len().min(MAX_PIXELS as usize)])
    }

    #[must_use]
    pub fn as_slice(&self) -> &'a [Color] {
        self.0
    }

    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub fn num_colors(&self) -> u32 {
        self.0.len() as u32
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<'a, Color> AsRef<[Color]> for ColorSlice<'a, Color> {
    fn as_ref(&self) -> &[Color] {
        self.as_slice()
    }
}

impl<'a, Color> From<ColorSlice<'a, Color>> for &'a [Color] {
    fn from(val: ColorSlice<'a, Color>) -> Self {
        val.as_slice()
    }
}

impl<'a, Color> TryFrom<&'a [Color]> for ColorSlice<'a, Color> {
    type Error = AboveMaxLen<u32>;

    fn try_from(value: &'a [Color]) -> Result<Self, Self::Error> {
        if value.len() <= MAX_PIXELS as usize {
            Ok(Self(value))
        } else {
            Err(AboveMaxLen(MAX_PIXELS))
        }
    }
}

#[cfg(feature = "image")]
impl<'a> TryFrom<&'a RgbImage> for ColorSlice<'a, Srgb<u8>> {
    type Error = AboveMaxLen<u32>;

    fn try_from(image: &'a RgbImage) -> Result<Self, Self::Error> {
        if image.pixels().len() <= MAX_PIXELS as usize {
            Ok(Self(image.components_as()))
        } else {
            Err(AboveMaxLen(MAX_PIXELS))
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct PaletteSize(u16);

impl PaletteSize {
    pub const MAX: Self = Self(MAX_COLORS);

    #[must_use]
    pub const fn into_inner(self) -> u16 {
        self.0
    }

    #[must_use]
    pub fn from_clamped(value: u16) -> Self {
        Self(u16::min(value, MAX_COLORS))
    }
}

impl Default for PaletteSize {
    fn default() -> Self {
        Self::MAX
    }
}

impl From<PaletteSize> for u16 {
    fn from(val: PaletteSize) -> Self {
        val.into_inner()
    }
}

impl From<u8> for PaletteSize {
    fn from(value: u8) -> Self {
        Self(value.into())
    }
}

impl TryFrom<u16> for PaletteSize {
    type Error = AboveMaxLen<u16>;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        if value <= MAX_COLORS {
            Ok(PaletteSize(value))
        } else {
            Err(AboveMaxLen(MAX_COLORS))
        }
    }
}

impl Display for PaletteSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.into_inner())
    }
}

#[derive(Debug, Clone)]
pub struct QuantizeOutput<Color> {
    pub palette: Vec<Color>,
    pub counts: Vec<u32>,
    pub indices: Vec<u8>,
}

impl<Color> QuantizeOutput<Color> {
    fn trivial<Component, const N: usize>(
        color_counts: &impl ColorCounts<Color, Component, N>,
        indices: Vec<u8>,
    ) -> Self
    where
        Color: ColorComponents<Component, N>,
    {
        QuantizeOutput {
            palette: color_counts.colors().to_vec(),
            counts: match color_counts.counts() {
                Some(counts) => counts.to_vec(),
                None => vec![1; color_counts.len()],
            },
            indices,
        }
    }

    pub(crate) fn trivial_palette<Component, const N: usize>(
        color_counts: &impl ColorCounts<Color, Component, N>,
    ) -> Self
    where
        Color: ColorComponents<Component, N>,
    {
        Self::trivial(color_counts, Vec::new())
    }

    pub(crate) fn trivial_quantize<Component, const N: usize>(
        color_counts: &(impl ColorCounts<Color, Component, N> + ColorRemap),
    ) -> Self
    where
        Color: ColorComponents<Component, N>,
    {
        debug_assert!(color_counts.num_colors() <= u32::from(MAX_COLORS));
        #[allow(clippy::cast_possible_truncation)]
        let indices = (0..color_counts.num_colors()).map(|i| i as u8).collect();
        Self::trivial(color_counts, color_counts.map_indices(indices))
    }

    #[cfg(feature = "threads")]
    pub(crate) fn trivial_quantize_par<Component, const N: usize>(
        color_counts: &(impl ColorCounts<Color, Component, N> + ParallelColorRemap),
    ) -> Self
    where
        Color: ColorComponents<Component, N>,
    {
        debug_assert!(color_counts.num_colors() <= u32::from(MAX_COLORS));
        #[allow(clippy::cast_possible_truncation)]
        let indices = (0..color_counts.num_colors()).map(|i| i as u8).collect();
        Self::trivial(color_counts, color_counts.map_indices_par(indices))
    }
}
