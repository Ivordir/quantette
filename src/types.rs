//! Contains various types needed across the crate.

use crate::{MAX_COLORS, MAX_PIXELS};
use std::{
    error::Error,
    fmt::{Debug, Display},
    ops::Deref,
};
#[cfg(feature = "image")]
use {
    image::RgbImage,
    palette::{cast::ComponentsAs, Srgb},
};

/// An error type for when the length of an input (e.g., `Vec` or slice)
/// is above the maximum supported value.
///
/// The inner value is the maximum supported value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AboveMaxLen<T>(pub T);

impl<T: Display> Display for AboveMaxLen<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "above the maximum length of {}", self.0)
    }
}

impl<T: Debug + Display> Error for AboveMaxLen<T> {}

/// A simple new type wrapper around `&'a [Color]` with the invariant that the length of the
/// inner slice must not be greater than [`MAX_PIXELS`].
///
/// # Examples
/// Use `try_into` or [`ColorSlice::from_truncated`] to create [`ColorSlice`]s.
///
/// From a raw color slice:
/// ```
/// # use quantette::{ColorSlice, AboveMaxLen};
/// # use palette::Srgb;
/// # fn main() -> Result<(), AboveMaxLen<u32>> {
/// let srgb = vec![Srgb::new(0, 0, 0)];
/// let colors: ColorSlice<_> = srgb.as_slice().try_into()?;
/// # Ok(())
/// # }
/// ```
///
/// From an image (needs the `image` feature to be enabled):
/// ```no_run
/// # use quantette::ColorSlice;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let img = image::open("some image")?.into_rgb8();
/// let colors = ColorSlice::try_from(&img)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct ColorSlice<'a, Color>(&'a [Color]);

impl<'a, Color> Clone for ColorSlice<'a, Color> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, Color> Copy for ColorSlice<'a, Color> {}

impl<'a, Color> ColorSlice<'a, Color> {
    /// Creates a [`ColorSlice`] without ensuring that its length
    /// is less than or equal to [`MAX_PIXELS`].
    #[allow(unused)]
    pub(crate) const fn new_unchecked(colors: &'a [Color]) -> Self {
        Self(colors)
    }

    /// Creates a new [`ColorSlice`] by truncating the input slice to a max length of [`MAX_PIXELS`].
    pub fn from_truncated(colors: &'a [Color]) -> Self {
        Self(&colors[..colors.len().min(MAX_PIXELS as usize)])
    }

    /// Returns the length of the slice as a `u32`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn num_colors(&self) -> u32 {
        self.0.len() as u32
    }
}

impl<'a, Color> AsRef<[Color]> for ColorSlice<'a, Color> {
    fn as_ref(&self) -> &[Color] {
        self
    }
}

impl<'a, Color> Deref for ColorSlice<'a, Color> {
    type Target = [Color];

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'a, Color> From<ColorSlice<'a, Color>> for &'a [Color] {
    fn from(val: ColorSlice<'a, Color>) -> Self {
        val.0
    }
}

impl<'a, Color> TryFrom<&'a [Color]> for ColorSlice<'a, Color> {
    type Error = AboveMaxLen<u32>;

    fn try_from(slice: &'a [Color]) -> Result<Self, Self::Error> {
        if slice.len() <= MAX_PIXELS as usize {
            Ok(Self(slice))
        } else {
            Err(AboveMaxLen(MAX_PIXELS))
        }
    }
}

#[cfg(feature = "image")]
impl<'a> TryFrom<&'a RgbImage> for ColorSlice<'a, Srgb<u8>> {
    type Error = AboveMaxLen<u32>;

    fn try_from(image: &'a RgbImage) -> Result<Self, Self::Error> {
        let pixels = image.pixels().len();
        if pixels <= MAX_PIXELS as usize {
            let buf = &image.as_raw()[..(pixels * 3)];
            Ok(Self(buf.components_as()))
        } else {
            Err(AboveMaxLen(MAX_PIXELS))
        }
    }
}

/// This type is used to specify the (maximum) number of colors to include in a palette.
///
/// This is a simple new type wrapper around `u16` with the invariant that it must be
/// less than or equal to [`MAX_COLORS`].
///
/// If a [`PaletteSize`] of `0` is provided to a quantization function,
/// an empty [`QuantizeOutput`] will be returned.
///
/// # Examples
/// Use `into` to create [`PaletteSize`]s from `u8`s.
/// For `u16`s, use `try_into` or [`PaletteSize::from_clamped`].
/// You can also use the [`PaletteSize::MAX`] constant.
///
/// From a `u8`:
/// ```
/// # use quantette::PaletteSize;
/// let size = PaletteSize::from(16);
/// let size: PaletteSize = 16.into();
/// ```
///
/// From a `u16`:
/// ```
/// # use quantette::{PaletteSize, AboveMaxLen};
/// # fn main() -> Result<(), AboveMaxLen<u16>> {
/// let size = PaletteSize::try_from(128u16)?;
/// let size: PaletteSize = 128u16.try_into()?;
/// let size = PaletteSize::from_clamped(1024);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct PaletteSize(u16);

impl PaletteSize {
    /// The maximum supported palette size (given by [`MAX_COLORS`]).
    pub const MAX: Self = Self(MAX_COLORS);

    /// Gets the inner `u16` value.
    #[must_use]
    pub const fn into_inner(self) -> u16 {
        self.0
    }

    /// Creates a [`PaletteSize`] directly from the given `u16`
    /// without ensuring that it is less than or equal to [`MAX_COLORS`].
    #[allow(unused)]
    pub(crate) const fn new_unchecked(value: u16) -> Self {
        Self(value)
    }

    /// Creates a [`PaletteSize`] by clamping the given `u16` to be less than or equal to [`MAX_COLORS`].
    #[must_use]
    pub const fn from_clamped(value: u16) -> Self {
        if value <= MAX_COLORS {
            Self(value)
        } else {
            Self(MAX_COLORS)
        }
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

/// The output struct returned by quantization functions.
///
/// It contains the color `palette` for the image, alongside `counts` which has
/// the number of pixels/samples assigned to each palette color.
/// Additionally, `indices` will contain a index into `palette` for each pixel,
/// but only if the quantization function computes an indexed palette
/// (e.g., [`wu::indexed_palette`](crate::wu::indexed_palette)).
/// Otherwise, `indices` will be empty (e.g., [`wu::palette`](crate::wu::palette)).
///
/// Note that all fields will be empty if a [`PaletteSize`] of `0` was provided to
/// the quantization function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizeOutput<Color> {
    /// The computed color palette that is representative of the colors in the image.
    ///
    /// The colors in the palette are not guaranteed to be unique.
    pub palette: Vec<Color>,
    /// The number of pixels or samples that were assigned to each color in `palette`.
    ///
    /// Each count is not guaranteed to be non-zero.
    pub counts: Vec<u32>,
    /// The remapped image, where each pixel is replaced with an index into `palette`.
    ///
    /// This will be empty if the quantization function does not compute an indexed palette.
    pub indices: Vec<u8>,
}

impl<Color> Default for QuantizeOutput<Color> {
    fn default() -> Self {
        Self {
            palette: Vec::new(),
            counts: Vec::new(),
            indices: Vec::new(),
        }
    }
}
