//! Contains the builder structs for the supported quantization methods.

/// A builder struct to specify the parameters for Wu's quantization method.
///
/// No options currently exist, but future options can be specified here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WuOptions {}

impl Default for WuOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl WuOptions {
    /// Creates a new [`WuOptions`] with default values.
    #[must_use]
    pub const fn new() -> Self {
        Self {}
    }
}

/// A builder struct to specify the parameters for k-means.
///
/// # Examples
/// ```
/// # use quantette::KmeansOptions;
/// let options = KmeansOptions::new()
///     .sampling_factor(0.25)
///     .seed(42);
/// ```
#[cfg(feature = "kmeans")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KmeansOptions {
    /// The proportion of the image or unique colors to sample.
    pub(crate) sampling_factor: f32,
    /// The seed value for the random number generator.
    pub(crate) seed: u64,
    /// The batch size for minibatch k-means.
    #[cfg(feature = "threads")]
    pub(crate) batch_size: u32,
}

#[cfg(feature = "kmeans")]
impl Default for KmeansOptions {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "kmeans")]
impl KmeansOptions {
    /// Creates a new [`KmeansOptions`] with default values.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sampling_factor: 0.5,
            seed: 0,
            #[cfg(feature = "threads")]
            batch_size: 4096,
        }
    }

    /// Sets the sampling factor which controls what percentage of the image/unique colors to sample.
    ///
    /// The default sampling factor is `0.5`, that is, to sample half of the input.
    ///
    /// Negative or `NAN` sampling factors will cause k-means to not be run.
    #[must_use]
    pub const fn sampling_factor(mut self, sampling_factor: f32) -> Self {
        self.sampling_factor = sampling_factor;
        self
    }

    /// Sets the seed value for the random number generator.
    ///
    /// The default seed is `0`.
    #[must_use]
    pub const fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the batch size for the parallel implementation of the k-means quantizer
    /// (minibatch k-means).
    ///
    /// Larger batch sizes are faster with dimishing returns.
    /// Smaller batch sizes are more accurate but slower to run.
    /// The batch is divided evenly among the number of threads.
    ///
    /// The default batch size is `4096`.
    #[must_use]
    #[cfg(feature = "threads")]
    pub const fn batch_size(mut self, batch_size: u32) -> Self {
        self.batch_size = batch_size;
        self
    }
}

/// The set of supported color quantization methods.
///
/// If the `kmeans` feature is enabled, then support will be added for that method.
/// Otherwise, only Wu's color quantization method is supported.
///
/// See the descriptions on each enum variant for more information.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizeMethod {
    /// Wu's color quantizer (Greedy Orthogonal Bipartitioning).
    ///
    /// This method is quick and gives good or at least decent results.
    ///
    /// See the [`wu`](crate::wu) module for more details.
    Wu(WuOptions),
    /// Color quantization using k-means clustering.
    ///
    /// This method is slower than Wu's color quantizer but gives more accurate results.
    /// It is recommended to combine this method with either the
    /// [`ColorSpace::Oklab`](crate::ColorSpace::Oklab) or [`ColorSpace::Lab`](crate::ColorSpace::Lab) settings,
    /// because `quantette` will, by default, deduplicate pixels to speed up k-means.
    /// This allows colorspace conversion to be much faster as well,
    /// since only each unique color needs to be converted instead of each pixel.
    ///
    /// See the [`kmeans`](crate::kmeans) module for more details.
    #[cfg(feature = "kmeans")]
    Kmeans(KmeansOptions),
}

impl QuantizeMethod {
    /// Creates a new [`QuantizeMethod::Wu`] with the default [`WuOptions`].
    #[must_use]
    pub const fn wu() -> Self {
        Self::Wu(WuOptions::new())
    }

    /// Creates a new [`QuantizeMethod::Kmeans`] with the default [`KmeansOptions`].
    #[must_use]
    #[cfg(feature = "kmeans")]
    pub const fn kmeans() -> Self {
        Self::Kmeans(KmeansOptions::new())
    }
}

impl From<WuOptions> for QuantizeMethod {
    fn from(options: WuOptions) -> Self {
        Self::Wu(options)
    }
}

#[cfg(feature = "kmeans")]
impl From<KmeansOptions> for QuantizeMethod {
    fn from(options: KmeansOptions) -> Self {
        Self::Kmeans(options)
    }
}
