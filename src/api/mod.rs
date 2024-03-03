//! Contains the types and functions for the high level pipeline builder API.

pub(crate) mod colorspace;
mod image_pipeline;
mod palette_pipeline;
mod quantize_method;

pub use colorspace::ColorSpace;
pub use image_pipeline::ImagePipeline;
pub use palette_pipeline::PalettePipeline;
pub use quantize_method::*;

#[cfg(feature = "kmeans")]
use crate::{ColorComponents, ColorCounts};

/// Returns the number of samples to run based off the sampling factor.
#[cfg(feature = "kmeans")]
fn num_samples<Color, Component, const N: usize>(
    sampling_factor: f32,
    color_counts: &impl ColorCounts<Color, Component, N>,
) -> u32
where
    Color: ColorComponents<Component, N>,
{
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        (f64::from(sampling_factor) * f64::from(color_counts.num_colors())) as u32
    }
}
