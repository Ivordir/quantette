//! Contains the types and functions for the high level pipeline builder API.

pub(crate) mod colorspace;
mod image_pipeline;
mod palette_pipeline;

pub use colorspace::ColorSpace;
pub use image_pipeline::ImagePipeline;
pub use palette_pipeline::PalettePipeline;

#[cfg(feature = "kmeans")]
use crate::{kmeans::Centroids, ColorComponents, ColorCounts};

use std::marker::PhantomData;

/// A builder struct to specify the parameters for k-means.
///
/// # Examples
/// ```
/// # use quantette::KmeansOptions;
/// let options = KmeansOptions::new()
///     .sampling_factor(0.25)
///     .seed(42);
/// # let options: KmeansOptions<()> = options; // satisfy type inference
/// ```
#[cfg(feature = "kmeans")]
#[derive(Debug, Clone)]
pub struct KmeansOptions<Color> {
    /// The proportion of the image or unique colors to sample.
    sampling_factor: f32,
    /// The initial colors/centroids to use.
    initial_centroids: Option<Centroids<Color>>,
    /// The seed value for the random number generator.
    seed: u64,
    /// The batch size for minibatch k-means.
    #[allow(unused)]
    batch_size: u32,
}

#[cfg(feature = "kmeans")]
impl<Color> Default for KmeansOptions<Color> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "kmeans")]
impl<Color> KmeansOptions<Color> {
    /// Creates a new [`KmeansOptions`] with default values.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sampling_factor: 0.5,
            initial_centroids: None,
            seed: 0,
            batch_size: 4096,
        }
    }

    /// Sets the sampling factor which controls what percentage of the image/unique colors to sample.
    ///
    /// The default is `0.5`, that is, to sample half of the input.
    #[must_use]
    pub fn sampling_factor(mut self, sampling_factor: f32) -> Self {
        self.sampling_factor = sampling_factor;
        self
    }

    /// Sets the initial colors/centroids for the k-means algorithm.
    ///
    /// By default, these are computed through Wu's quantiztion algorithm
    /// (see the [`wu`](crate::wu) module).
    #[must_use]
    pub fn initial_centroids(mut self, centroids: Centroids<Color>) -> Self {
        self.initial_centroids = Some(centroids);
        self
    }

    /// Sets the seed value for the random number generator.
    ///
    /// The default seed is `0`.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the batch size for the parallel implementation of the k-means quantizer
    /// (minibatch k-means).
    ///
    /// Higher batch sizes should be faster with dimishing returns.
    /// Lower batch sizes are more accurate but slower to run.
    /// The batch is divided evenly among the number of threads.
    ///
    /// The default batch size is `4096`.
    #[must_use]
    #[cfg(feature = "threads")]
    pub fn batch_size(mut self, batch_size: u32) -> Self {
        self.batch_size = batch_size;
        self
    }
}

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

/// The set of supported color quantization methods.
///
/// If the `kmeans` feature is enabled, then support will be added for that method.
/// Otherwise, only Wu's color quantization method is supported.
///
/// See the descriptions on each enum variant for more information.
#[derive(Debug, Clone)]
pub enum QuantizeMethod<Color> {
    /// Wu's color quantizer (Greedy Orthogonal Bipartitioning).
    ///
    /// This method is quick and gives good or at least decent results.
    ///
    /// See the [`wu`](crate::wu) module for more details.
    Wu(PhantomData<Color>),
    /// Color quantization using k-means clustering.
    ///
    /// This method is slower than Wu's color quantizer but gives more accurate results.
    /// It is recommended to combine this method with either the [`ColorSpace::Oklab`] or [`ColorSpace::Lab`]
    /// settings, because `quantette` will, by default, deduplicate pixels to speed up k-means.
    /// This allows colorspace conversion to be much faster as well,
    /// since only each unique color needs to be converted instead of each pixel.
    ///
    /// See the [`kmeans`](crate::kmeans) module for more details.
    #[cfg(feature = "kmeans")]
    Kmeans(KmeansOptions<Color>),
}
