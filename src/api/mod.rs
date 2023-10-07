mod image_pipeline;
mod palette_pipeline;

pub mod colorspace;

pub use colorspace::ColorSpace;
pub use image_pipeline::*;
pub use palette_pipeline::*;

use crate::{ColorAndFrequency, ColorComponents};

#[cfg(feature = "kmeans")]
use crate::{kmeans::Centroids, AboveMaxLen};

#[cfg(feature = "kmeans")]
#[derive(Debug, Clone)]
pub struct KmeansOptions<Color> {
    sampling_factor: f64,
    initial_centroids: Option<Centroids<Color>>,
    seed: u64,
    #[cfg(feature = "threads")]
    batch_size: u32,
}

#[cfg(feature = "kmeans")]
impl<Color> KmeansOptions<Color> {
    #[must_use]
    pub fn new(sampling_factor: f64) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Self {
            sampling_factor,
            initial_centroids: None,
            seed: 0,
            #[cfg(feature = "threads")]
            batch_size: 4096,
        }
    }

    #[must_use]
    pub fn initial_centroids(mut self, centroids: Centroids<Color>) -> Self {
        self.initial_centroids = Some(centroids);
        self
    }

    pub fn try_initial_centroids(
        mut self,
        centroids: Vec<Color>,
    ) -> Result<Self, AboveMaxLen<u16>> {
        self.initial_centroids = Some(centroids.try_into()?);
        Ok(self)
    }

    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    #[must_use]
    #[cfg(feature = "threads")]
    pub fn batch_size(mut self, batch_size: u32) -> Self {
        self.batch_size = batch_size;
        self
    }
}

#[derive(Debug, Clone)]
pub enum QuantizeMethod<Color> {
    Wu,
    #[cfg(feature = "kmeans")]
    Kmeans(KmeansOptions<Color>),
}

impl<Color> QuantizeMethod<Color> {
    /// A convenience method for `QuantizeMethod::Kmeans(KmeansOptions::new(sampling_factor))`
    #[cfg(feature = "kmeans")]
    #[must_use]
    pub fn kmeans(sampling_factor: f64) -> Self {
        Self::Kmeans(KmeansOptions::new(sampling_factor))
    }
}

#[cfg(feature = "kmeans")]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn num_samples<Color, Component, const N: usize>(
    sampling_factor: f64,
    color_counts: &impl ColorAndFrequency<Color, Component, N>,
) -> u32
where
    Color: ColorComponents<Component, N>,
{
    (sampling_factor * f64::from(color_counts.num_colors())) as u32
}
