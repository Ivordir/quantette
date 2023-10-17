//! A library for fast and high quality image quantization and palette generation.
//!
//! `quantette` can perform quantization in perceptually uniform color spaces like CIELAB and Oklab
//! for more accurate results.
//!
//! # Features
//! To reduce dependencies and compile times, `quanette` has several `cargo` features
//! that can be turned off or on:
//! - `pipelines`: exposes builder structs that serve as the high-level API (more details below).
//! - `colorspaces`: allows performing quantization in the CIELAB or Oklab color spaces via the high-level API.
//! - `kmeans`: adds an additional high quality quantization method that takes longer to run.
//! - `threads`: exposes parallel versions of most functions via [`rayon`].
//! - `image`: enables integration with the [`image`] crate.
//!
//! # High-Level API
//! To get started with the high-level API, see [`ImagePipeline`].
//! If you want a color palette instead of a quantized image, see [`PalettePipeline`] instead.
//! Both of these have examples in their documentation, but here is an additional example:
//! ```no_run
//! # use quantette::{ImagePipeline, AboveMaxLen, ColorSpace, QuantizeMethod, KmeansOptions};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let img = image::open("some image")?.into_rgb8();
//!
//! let pipeline = ImagePipeline::try_from(&img)?
//!     .palette_size(128.into()) // set the max number of colors in the palette
//!     .dither(false) // turn dithering off
//!     .colorspace(ColorSpace::Oklab) // use a more accurate color space
//!     .quantize_method(QuantizeMethod::Kmeans(KmeansOptions::new()));
//!
//! // Run the pipeline in parallel to get an RgbImage
//! let quantized = pipeline.quantized_rgbimage_par();
//! # Ok(())
//! # }
//! ```
//!
//! Note that some of the options and functions above require certain features to be enabled.

#![deny(unsafe_code, unsafe_op_in_unsafe_fn)]
#![warn(
    clippy::pedantic,
    clippy::cargo,
    clippy::use_debug,
    clippy::dbg_macro,
    clippy::todo,
    clippy::unimplemented,
    clippy::unwrap_used,
    clippy::unwrap_in_result,
    clippy::expect_used,
    clippy::unneeded_field_pattern,
    clippy::rest_pat_in_fully_bound_structs,
    clippy::unnecessary_self_imports,
    clippy::str_to_string,
    clippy::string_to_string,
    clippy::string_slice,
    missing_docs,
    clippy::missing_docs_in_private_items,
    rustdoc::all,
    clippy::float_cmp_const,
    clippy::lossy_float_literal
)]
#![allow(
    clippy::doc_markdown,
    clippy::module_name_repetitions,
    clippy::many_single_char_names,
    clippy::missing_panics_doc,
    clippy::unreadable_literal,
    clippy::wildcard_imports
)]

mod color_counts;
mod dither;
mod traits;
mod types;

#[cfg(feature = "pipelines")]
mod api;

pub mod wu;

#[cfg(feature = "kmeans")]
pub mod kmeans;

pub use color_counts::*;
pub use dither::FloydSteinberg;
pub use traits::*;
pub use types::*;

#[cfg(feature = "pipelines")]
pub use api::*;

use wu::{Binner3, FloatBinner, UIntBinner};

/// The maximum supported image size in number of pixels is `u32::MAX`.
pub const MAX_PIXELS: u32 = u32::MAX;

/// The maximum supported number of palette colors is `256`.
pub const MAX_COLORS: u16 = u8::MAX as u16 + 1;

/// `MAX_COLORS` as a `usize` for array and `Vec` lengths.
pub(crate) const MAX_K: usize = MAX_COLORS as usize;
