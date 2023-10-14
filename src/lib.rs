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
    // missing_docs,
    // clippy::missing_docs_in_private_items,
    rustdoc::all,
    clippy::float_cmp_const,
    clippy::lossy_float_literal
)]
#![allow(
    clippy::module_name_repetitions,
    clippy::many_single_char_names,
    clippy::missing_panics_doc,
    clippy::unreadable_literal,
    clippy::wildcard_imports
)]
// temporary
#![allow(clippy::missing_errors_doc)]

mod color_counts;
mod traits;
mod types;

#[cfg(feature = "pipelines")]
mod api;

pub mod dither;
pub mod wu;

#[cfg(feature = "kmeans")]
pub mod kmeans;

pub use color_counts::{ColorCounts, ColorRemap, IndexedColorCounts, UniqueColorCounts};
pub use traits::*;
pub use types::*;

#[cfg(feature = "pipelines")]
pub use api::*;
#[cfg(feature = "threads")]
pub use color_counts::ParallelColorRemap;

use wu::{Binner3, FloatBinner, UIntBinner};

/// The maximum supported image size in number of pixels is `u32::MAX`.
pub const MAX_PIXELS: u32 = u32::MAX;

/// The maximum supported number of palette colors is 256.
pub const MAX_COLORS: u16 = u8::MAX as u16 + 1;

/// `MAX_COLORS` as a `usize` for array/Vec lengths.
pub(crate) const MAX_K: usize = MAX_COLORS as usize;
