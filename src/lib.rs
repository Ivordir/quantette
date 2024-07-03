//! A library for fast and high quality image quantization and palette generation.
//!
//! `quantette` can perform quantization in perceptually uniform color spaces like CIELAB and Oklab
//! for more accurate results.
//!
//! # Features
//! To reduce dependencies and compile times, `quantette` has several `cargo` features
//! that can be turned off or on:
//! - `pipelines`: exposes builder structs that serve as the high-level API (more details below).
//! - `colorspaces`: allows performing quantization in the CIELAB or Oklab color spaces via the high-level API.
//! - `kmeans`: adds an additional high quality quantization method that takes longer to run.
//! - `threads`: exposes parallel versions of most functions via [`rayon`].
//! - `image`: enables integration with the [`image`] crate.
//!
//! By default, all features are enabled.
//!
//! # High-Level API
//! To get started with the high-level API, see [`ImagePipeline`].
//! If you want a color palette instead of a quantized image, see [`PalettePipeline`] instead.
//! Both of these have examples in their documentation, but here is an additional example:
//! ```no_run
//! # use quantette::{ImagePipeline, AboveMaxLen, ColorSpace, QuantizeMethod};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let img = image::open("some image")?.into_rgb8();
//!
//! let quantized = ImagePipeline::try_from(&img)?
//!     .palette_size(128) // set the max number of colors in the palette
//!     .dither(false) // turn dithering off
//!     .colorspace(ColorSpace::Oklab) // use a more accurate color space
//!     .quantize_method(QuantizeMethod::kmeans()) // use a more accurate quantization algorithm
//!     .quantized_rgbimage_par(); // run the pipeline in parallel to get a [`RgbImage`]
//! # Ok(())
//! # }
//! ```
//!
//! Note that some of the options and functions above require certain features to be enabled.
//!
//! All of the color types present in the public API for this crate
//! (like [`Srgb`](palette::Srgb) or [`Oklab`](palette::Oklab)) are from the [`palette`] crate.
//! You can check it out for more information. For example, its documentation
//! should provide you everything you need to know to [cast](palette::cast)
//! a `Vec<Srgb<u8>>` into a `Vec<[u8; 3]>`.

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
    clippy::unreadable_literal
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

// Re-export third party crates with types present in our public API
#[cfg(feature = "image")]
pub use image;
#[cfg(any(feature = "pipelines", feature = "image"))]
pub use palette;

/// The maximum supported image size in number of pixels is `u32::MAX`.
pub const MAX_PIXELS: u32 = u32::MAX;

/// The maximum supported number of palette colors is `256`.
pub const MAX_COLORS: u16 = u8::MAX as u16 + 1;

#[cfg(test)]
mod tests {
    use palette::{cast::ComponentsInto, Srgb};

    #[rustfmt::skip]
    pub fn test_data_256() -> Vec<Srgb<u8>> {
        vec![
            225, 173, 56, 117, 245, 55, 13, 86, 86, 74, 28, 233, 43, 1, 65, 174, 92, 158, 167, 92, 86, 68, 33, 246, 170, 140, 121, 207, 120, 147, 120, 70, 173, 153, 247, 69, 238, 26, 203, 224, 234, 37, 58, 138, 40, 107, 124, 149, 134, 22, 68, 211, 246, 164, 193, 60, 109, 0, 92, 174, 107, 129, 157, 122, 70, 195, 159, 138, 134, 157, 164, 90, 17, 40, 44, 47, 129, 100, 92, 230, 49, 157, 9, 216, 189, 162, 111, 155, 134, 74, 139, 60, 80, 7, 144, 66, 122, 5, 69, 227, 111, 252, 110, 59, 49, 70, 106, 43, 61, 249, 80, 24, 91, 39, 1, 212, 222, 110, 148, 110, 7, 157, 8, 253, 94, 57, 153, 228, 71, 0, 185, 151, 140, 179, 254, 60, 101, 7, 162, 214, 43, 15, 161, 235, 49, 176, 66, 102, 61, 250, 50, 219, 109, 91, 246, 238, 188, 33, 107, 72, 85, 16, 40, 44, 249, 62, 181, 234, 199, 90, 149, 75, 20, 72, 34, 227, 227, 29, 205, 228, 106, 176, 21, 202, 223, 73, 51, 39, 173, 118, 163, 159, 26, 80, 51, 140, 189, 93, 113, 25, 79, 149, 238, 65, 34, 43, 22, 232, 225, 18, 126, 7, 175, 125, 54, 27, 39, 12, 180, 93, 173, 176, 52, 160, 7, 187, 161, 220, 236, 127, 132, 7, 19, 36, 188, 121, 113, 214, 20, 6, 140, 52, 14, 93, 123, 148, 42, 254, 9, 159, 208, 118, 36, 245, 173, 70, 88, 192, 106, 209, 24, 72, 189, 8, 231, 199, 228, 48, 200, 209, 231, 18, 105, 5, 220, 19, 118, 47, 78, 229, 201, 210, 17, 20, 74, 203, 158, 70, 77, 14, 57, 119, 195, 158, 121, 71, 12, 209, 228, 216, 122, 144, 212, 204, 65, 210, 56, 190, 225, 111, 149, 172, 124, 200, 63, 123, 134, 9, 38, 0, 57, 54, 140, 78, 203, 76, 42, 133, 202, 8, 141, 107, 10, 144, 80, 220, 183, 236, 129, 44, 251, 188, 109, 173, 174, 38, 190, 8, 250, 243, 134, 61, 150, 82, 19, 254, 74, 56, 8, 218, 78, 136, 44, 63, 103, 207, 166, 144, 20, 104, 21, 245, 14, 104, 128, 202, 229, 7, 181, 116, 124, 127, 154, 34, 100, 186, 94, 142, 201, 89, 227, 98, 36, 192, 72, 9, 221, 95, 237, 105, 205, 106, 126, 145, 246, 162, 12, 96, 137, 9, 117, 66, 77, 19, 251, 137, 70, 237, 77, 36, 175, 127, 63, 165, 76, 88, 122, 118, 92, 190, 33, 81, 142, 175, 166, 9, 194, 198, 99, 3, 118, 12, 68, 169, 247, 151, 212, 87, 21, 107, 92, 200, 200, 9, 36, 212, 3, 38, 166, 163, 154, 123, 179, 72, 60, 253, 96, 235, 126, 186, 218, 132, 216, 87, 26, 43, 247, 233, 192, 54, 153, 192, 215, 59, 17, 68, 87, 103, 197, 11, 93, 179, 168, 12, 212, 10, 188, 241, 26, 94, 229, 240, 170, 133, 44, 88, 16, 121, 22, 251, 184, 87, 209, 97, 210, 190, 212, 61, 84, 233, 134, 230, 139, 226, 30, 145, 183, 117, 35, 70, 118, 197, 55, 173, 8, 174, 192, 216, 48, 10, 58, 189, 81, 80, 89, 79, 228, 42, 159, 202, 86, 183, 37, 200, 86, 153, 108, 79, 135, 51, 167, 149, 185, 145, 36, 146, 95, 244, 130, 135, 106, 89, 99, 65, 125, 224, 236, 41, 105, 226, 126, 164, 128, 223, 248, 49, 172, 107, 90, 66, 79, 215, 230, 51, 234, 109, 128, 43, 90, 135, 36, 121, 244, 51, 240, 145, 145, 185, 19, 115, 9, 72, 218, 119, 81, 42, 22, 15, 98, 132, 42, 211, 190, 115, 131, 198, 221, 189, 196, 92, 155, 23, 163, 0, 244, 188, 217, 64, 248, 141, 203, 226, 195, 6, 230, 51, 191, 12, 198, 181, 138, 33, 60, 11, 165, 191, 184, 193, 14, 86, 216, 167, 73, 9, 49, 169, 237, 16, 70, 196, 121, 252, 39, 71, 69, 156, 83, 174, 139, 175, 82, 9, 248, 25, 208, 201, 190, 86, 157, 214, 170, 246, 211, 168, 166, 227, 126, 89, 153, 204, 118, 220, 223, 228, 76, 104, 47, 53, 77, 45, 79, 91, 241, 178, 31, 45, 156, 244, 23, 120, 163, 61, 76, 68, 193, 144, 249, 110, 167, 103, 215, 201, 66, 143, 101, 237, 82, 173, 171, 67, 248, 126, 53, 223, 75, 186, 86, 196, 227, 129, 210, 56, 13, 18, 69, 13, 124, 162, 178, 141, 108, 174, 153, 212, 166, 14, 173, 112
        ].components_into()
    }

    #[rustfmt::skip]
    pub fn test_data_1024() -> Vec<Srgb<u8>> {
        vec![
            203, 78, 237, 188, 86, 167, 252, 122, 82, 240, 130, 219, 75, 67, 121, 181, 79, 219, 88, 107, 145, 79, 130, 210, 87, 48, 246, 46, 43, 55, 52, 41, 174, 15, 102, 21, 65, 89, 218, 133, 201, 211, 177, 103, 81, 79, 64, 132, 48, 195, 77, 246, 173, 213, 153, 8, 208, 15, 138, 57, 209, 221, 248, 187, 96, 163, 149, 36, 23, 178, 247, 196, 104, 204, 229, 226, 50, 240, 229, 15, 115, 65, 217, 133, 217, 2, 30, 108, 244, 94, 74, 211, 229, 16, 126, 1, 144, 243, 90, 172, 66, 24, 0, 148, 114, 118, 177, 201, 247, 214, 157, 91, 18, 165, 197, 142, 234, 44, 125, 197, 19, 174, 42, 131, 57, 67, 219, 172, 205, 24, 166, 248, 246, 231, 67, 24, 54, 163, 11, 156, 9, 26, 117, 119, 224, 240, 62, 227, 21, 142, 180, 99, 181, 225, 195, 157, 121, 235, 210, 227, 55, 26, 34, 217, 239, 134, 93, 240, 72, 130, 35, 7, 128, 169, 114, 129, 195, 162, 235, 34, 128, 161, 123, 200, 136, 127, 177, 50, 248, 184, 158, 225, 232, 41, 121, 8, 48, 171, 31, 154, 168, 69, 253, 115, 123, 172, 116, 95, 67, 66, 253, 129, 239, 142, 96, 8, 39, 63, 217, 108, 177, 88, 40, 45, 226, 199, 232, 198, 73, 56, 173, 112, 182, 84, 186, 35, 129, 49, 191, 181, 32, 207, 64, 115, 91, 242, 16, 77, 219, 28, 164, 77, 95, 101, 148, 125, 197, 220, 1, 240, 244, 46, 37, 71, 175, 177, 125, 143, 78, 171, 36, 59, 169, 78, 132, 254, 162, 107, 28, 146, 23, 188, 84, 251, 11, 189, 50, 192, 24, 82, 43, 84, 69, 115, 26, 150, 235, 27, 192, 56, 246, 151, 19, 106, 135, 158, 115, 160, 76, 0, 159, 115, 109, 128, 145, 133, 61, 152, 4, 10, 198, 87, 134, 47, 151, 244, 187, 67, 207, 56, 238, 63, 140, 126, 28, 78, 217, 33, 207, 16, 202, 145, 153, 23, 37, 176, 234, 86, 156, 161, 42, 49, 145, 47, 63, 159, 81, 64, 200, 193, 59, 36, 133, 36, 110, 154, 8, 151, 20, 156, 234, 213, 233, 115, 191, 136, 53, 52, 11, 89, 21, 236, 77, 164, 207, 194, 202, 0, 223, 201, 7, 27, 65, 246, 152, 239, 138, 144, 117, 226, 85, 197, 190, 167, 139, 73, 236, 207, 110, 240, 180, 223, 26, 194, 130, 142, 75, 60, 133, 118, 209, 49, 57, 170, 249, 170, 108, 66, 35, 225, 29, 182, 127, 227, 130, 160, 120, 97, 77, 60, 22, 255, 250, 32, 3, 20, 86, 39, 176, 149, 12, 167, 72, 59, 214, 201, 207, 244, 241, 30, 104, 165, 15, 206, 202, 12, 255, 56, 196, 211, 38, 106, 162, 184, 217, 70, 7, 254, 52, 79, 27, 124, 2, 5, 126, 92, 213, 29, 131, 39, 170, 127, 73, 177, 28, 46, 71, 151, 205, 148, 10, 81, 114, 144, 164, 123, 69, 109, 14, 106, 181, 251, 79, 238, 209, 194, 28, 38, 85, 55, 198, 193, 133, 222, 15, 234, 44, 135, 190, 222, 74, 180, 190, 83, 163, 248, 136, 186, 58, 130, 200, 210, 55, 93, 246, 15, 110, 130, 6, 215, 193, 90, 6, 32, 69, 21, 156, 32, 217, 50, 79, 63, 37, 43, 246, 147, 4, 74, 181, 147, 45, 109, 5, 176, 173, 96, 139, 147, 229, 193, 62, 197, 244, 166, 13, 167, 64, 8, 179, 77, 235, 216, 246, 22, 194, 142, 16, 64, 81, 144, 203, 107, 21, 185, 221, 14, 248, 35, 111, 190, 22, 112, 189, 146, 41, 17, 198, 6, 36, 253, 71, 128, 231, 100, 115, 190, 14, 197, 172, 78, 39, 139, 202, 251, 78, 111, 234, 205, 104, 31, 248, 137, 199, 247, 67, 89, 173, 9, 115, 160, 185, 197, 78, 30, 38, 234, 152, 160, 117, 232, 32, 216, 40, 157, 169, 13, 179, 217, 88, 137, 37, 164, 51, 221, 92, 221, 58, 200, 222, 86, 121, 84, 232, 65, 227, 170, 193, 114, 132, 85, 144, 61, 11, 152, 109, 212, 169, 191, 42, 76, 233, 168, 76, 233, 178, 180, 167, 105, 72, 255, 196, 192, 26, 193, 94, 209, 148, 108, 0, 165, 172, 121, 247, 28, 141, 169, 192, 49, 137, 245, 166, 226, 22, 126, 194, 133, 100, 185, 159, 44, 209, 250, 174, 49, 115, 190, 245, 99, 8, 53, 51, 57, 95, 128, 172, 30, 254, 157, 16, 208, 46, 13, 166, 161, 105, 145, 163, 233, 219, 41, 216, 82, 50, 202, 120, 61, 45, 123, 105, 62, 227, 74, 173, 157, 237, 200, 109, 100, 154, 48, 181, 39, 217, 31, 113, 83, 198, 17, 245, 203, 219, 52, 199, 126, 117, 38, 84, 201, 55, 211, 87, 37, 19, 225, 164, 98, 151, 68, 95, 245, 183, 170, 119, 229, 205, 212, 108, 45, 0, 236, 231, 17, 239, 216, 149, 186, 59, 81, 104, 142, 79, 69, 182, 167, 93, 90, 81, 225, 174, 99, 187, 253, 143, 239, 184, 44, 246, 147, 195, 149, 179, 254, 43, 87, 122, 44, 193, 222, 206, 33, 59, 18, 136, 26, 137, 193, 184, 146, 2, 46, 165, 199, 27, 148, 44, 75, 22, 134, 127, 133, 75, 178, 23, 104, 240, 222, 19, 191, 170, 75, 53, 23, 133, 89, 150, 42, 207, 186, 134, 136, 60, 76, 108, 249, 164, 232, 171, 64, 68, 235, 235, 205, 168, 28, 92, 248, 211, 191, 116, 241, 53, 251, 0, 152, 199, 144, 58, 133, 107, 185, 144, 209, 35, 97, 250, 212, 164, 149, 6, 108, 151, 97, 131, 252, 55, 175, 157, 210, 173, 216, 211, 74, 209, 23, 31, 233, 172, 195, 167, 98, 137, 132, 42, 49, 53, 59, 220, 7, 120, 81, 102, 195, 45, 49, 97, 177, 110, 234, 156, 96, 224, 54, 138, 221, 103, 238, 36, 241, 254, 6, 61, 217, 245, 244, 133, 146, 20, 71, 108, 241, 150, 166, 240, 127, 21, 38, 170, 70, 225, 153, 180, 209, 225, 166, 39, 166, 40, 206, 195, 64, 202, 184, 205, 251, 153, 210, 26, 55, 232, 165, 215, 130, 100, 190, 91, 191, 75, 99, 104, 91, 176, 9, 209, 101, 1, 179, 57, 28, 60, 57, 45, 54, 71, 136, 225, 252, 170, 55, 203, 45, 243, 165, 169, 57, 15, 71, 50, 239, 101, 211, 161, 199, 157, 14, 113, 46, 50, 78, 240, 145, 175, 0, 9, 228, 181, 209, 199, 50, 143, 119, 143, 94, 26, 117, 118, 203, 244, 136, 138, 43, 54, 60, 221, 72, 237, 131, 240, 82, 230, 215, 232, 114, 79, 164, 4, 201, 39, 102, 237, 228, 2, 22, 154, 80, 160, 229, 46, 208, 96, 255, 227, 29, 94, 163, 122, 195, 93, 83, 78, 53, 116, 80, 88, 247, 151, 197, 102, 141, 119, 232, 78, 228, 104, 173, 5, 96, 194, 93, 152, 155, 226, 97, 63, 172, 60, 81, 13, 253, 164, 14, 225, 230, 21, 77, 28, 36, 162, 82, 230, 153, 98, 235, 254, 42, 131, 139, 127, 113, 40, 107, 225, 0, 195, 186, 150, 82, 218, 123, 233, 57, 87, 33, 218, 193, 236, 103, 167, 246, 52, 57, 237, 4, 242, 114, 15, 128, 35, 54, 170, 49, 105, 2, 197, 32, 54, 229, 61, 187, 180, 125, 135, 236, 182, 11, 196, 239, 145, 132, 204, 158, 175, 87, 200, 42, 201, 136, 170, 171, 140, 176, 52, 29, 211, 234, 212, 222, 110, 137, 62, 47, 43, 196, 158, 88, 73, 97, 52, 43, 230, 21, 93, 242, 104, 164, 134, 220, 160, 200, 104, 247, 53, 198, 32, 115, 123, 56, 150, 195, 1, 230, 75, 253, 7, 180, 223, 76, 157, 223, 7, 125, 16, 63, 86, 162, 176, 5, 36, 28, 11, 107, 11, 48, 77, 195, 41, 172, 101, 78, 224, 210, 117, 79, 246, 82, 6, 235, 187, 99, 88, 202, 5, 2, 247, 192, 9, 227, 63, 237, 224, 52, 160, 80, 174, 41, 219, 165, 254, 28, 116, 16, 73, 44, 224, 26, 90, 131, 164, 151, 18, 83, 185, 169, 87, 188, 182, 160, 154, 24, 198, 243, 15, 80, 149, 112, 160, 52, 47, 94, 255, 93, 112, 119, 199, 55, 206, 237, 81, 216, 246, 73, 90, 108, 72, 135, 212, 144, 170, 108, 111, 154, 115, 253, 23, 35, 55, 21, 99, 44, 112, 106, 197, 241, 217, 76, 121, 161, 255, 98, 178, 75, 169, 61, 57, 207, 11, 152, 253, 192, 242, 186, 221, 161, 120, 36, 78, 224, 58, 193, 46, 136, 228, 113, 209, 21, 113, 113, 255, 71, 65, 122, 126, 35, 69, 98, 29, 186, 182, 123, 19, 143, 147, 81, 232, 244, 66, 1, 90, 15, 183, 174, 167, 52, 88, 42, 178, 137, 92, 50, 101, 29, 79, 29, 145, 212, 51, 107, 248, 133, 27, 245, 66, 126, 218, 229, 144, 12, 237, 6, 63, 183, 138, 78, 6, 190, 54, 253, 69, 124, 10, 45, 85, 4, 106, 16, 238, 24, 205, 198, 189, 144, 31, 135, 194, 169, 143, 126, 192, 187, 176, 242, 159, 74, 28, 161, 5, 199, 176, 37, 98, 167, 229, 93, 72, 202, 125, 26, 249, 203, 162, 36, 252, 10, 109, 189, 127, 50, 196, 71, 104, 249, 210, 104, 30, 170, 78, 10, 82, 135, 154, 14, 67, 220, 30, 174, 3, 132, 255, 67, 209, 31, 184, 115, 166, 61, 86, 19, 48, 242, 117, 239, 110, 77, 216, 131, 137, 118, 221, 95, 62, 181, 191, 67, 225, 57, 2, 154, 112, 115, 87, 69, 82, 121, 45, 219, 5, 112, 9, 221, 30, 141, 105, 162, 107, 114, 244, 80, 26, 153, 241, 138, 140, 144, 54, 70, 184, 120, 219, 184, 11, 20, 222, 89, 243, 194, 9, 245, 86, 147, 81, 12, 127, 90, 119, 9, 244, 193, 75, 94, 150, 35, 16, 186, 198, 17, 97, 213, 69, 134, 154, 139, 184, 139, 144, 17, 20, 28, 90, 135, 216, 201, 211, 190, 66, 77, 29, 118, 232, 194, 175, 22, 45, 7, 156, 183, 158, 2, 43, 144, 147, 125, 121, 162, 208, 133, 141, 19, 10, 97, 148, 3, 50, 61, 252, 216, 174, 126, 42, 127, 244, 225, 94, 242, 135, 26, 78, 49, 51, 192, 178, 143, 103, 251, 192, 236, 26, 119, 223, 71, 226, 74, 153, 106, 163, 108, 11, 59, 29, 172, 14, 20, 184, 14, 192, 8, 167, 239, 252, 190, 32, 210, 122, 5, 203, 123, 65, 20, 4, 186, 91, 197, 207, 215, 27, 154, 204, 59, 213, 51, 159, 254, 45, 18, 221, 174, 88, 224, 241, 40, 243, 32, 221, 163, 240, 27, 40, 203, 101, 240, 124, 249, 95, 73, 226, 166, 174, 25, 24, 190, 7, 141, 39, 136, 38, 39, 233, 42, 173, 142, 64, 29, 73, 197, 84, 227, 239, 118, 252, 195, 80, 41, 224, 170, 108, 193, 185, 80, 226, 230, 235, 128, 66, 119, 37, 35, 100, 72, 65, 42, 49, 131, 164, 140, 24, 197, 216, 157, 55, 57, 200, 45, 91, 208, 138, 162, 157, 104, 100, 199, 222, 92, 129, 63, 207, 60, 178, 135, 97, 220, 108, 120, 60, 121, 54, 160, 225, 150, 14, 106, 89, 116, 13, 90, 29, 216, 39, 239, 168, 213, 96, 66, 197, 248, 140, 253, 109, 226, 103, 248, 166, 206, 70, 245, 255, 213, 8, 104, 68, 217, 142, 185, 101, 18, 187, 50, 35, 101, 253, 169, 66, 112, 195, 27, 124, 192, 208, 14, 245, 131, 46, 140, 169, 65, 107, 248, 190, 0, 184, 216, 52, 144, 74, 91, 7, 67, 178, 210, 176, 248, 214, 229, 162, 172, 62, 178, 132, 14, 224, 101, 171, 47, 41, 153, 91, 89, 196, 64, 56, 68, 38, 70, 221, 23, 27, 72, 216, 31, 161, 166, 221, 168, 12, 89, 151, 70, 72, 244, 154, 114, 238, 250, 56, 41, 32, 52, 11, 129, 246, 170, 99, 32, 55, 52, 225, 0, 157, 40, 165, 26, 229, 39, 171, 64, 91, 26, 10, 154, 185, 99, 18, 91, 158, 188, 246, 198, 122, 40, 53, 141, 160, 250, 223, 170, 130, 234, 202, 197, 101, 154, 153, 237, 52, 65, 113, 205, 18, 202, 66, 5, 173, 31, 73, 254, 106, 49, 237, 227, 71, 123, 116, 251, 5, 125, 55, 122, 43, 243, 92, 57, 138, 19, 224, 53, 43, 154, 237, 204, 139, 227, 77, 196, 255, 217, 27, 21, 158, 150, 125, 2, 255, 121, 219, 137, 68, 80, 73, 131, 44, 136, 78, 81, 87, 107, 210, 6, 106, 114, 241, 212, 189, 254, 254, 43, 106, 123, 118, 26, 32, 191, 186, 126, 250, 200, 161, 29, 134, 6, 77, 143, 155, 8, 204, 251, 236, 149, 87, 39, 145, 242, 240, 109, 91, 34, 20, 182, 180, 227, 43, 14, 5, 95, 88, 253, 96, 168, 202, 186, 60, 50, 29, 227, 249, 153, 255, 32, 11, 6, 74, 22, 227, 106, 20, 94, 250, 133, 199, 142, 254, 242, 191, 59, 64, 203, 227, 167, 217, 8, 131, 169, 22, 118, 5, 109, 212, 190, 2, 211, 162, 57, 44, 170, 174, 7, 221, 68, 74, 102, 232, 202, 246, 123, 12, 28, 92, 217, 143, 230, 158, 168, 175, 139, 28, 65, 34, 67, 142, 54, 238, 23, 28, 73, 81, 98, 20, 214, 229, 112, 30, 194, 5, 222, 92, 36, 59, 252, 0, 87, 86, 38, 70, 174, 200, 6, 63, 106, 222, 64, 141, 20, 78, 248, 230, 248, 211, 90, 59, 81, 144, 223, 232, 93, 251, 13, 44, 147, 147, 159, 167, 242, 87, 174, 148, 255, 140, 14, 236, 109, 44, 219, 25, 81, 224, 111, 121, 44, 149, 100, 53, 129, 116, 139, 188, 223, 53, 131, 252, 49, 82, 18, 140, 51, 150, 249, 227, 173, 67, 115, 89, 248, 136, 107, 1, 123, 251, 216, 149, 252, 22, 64, 52, 6, 124, 29, 58, 216, 71, 20, 225, 99, 82, 18, 248, 160, 5, 12, 115, 173, 183, 182, 79, 101, 105, 146, 22, 35, 253, 234, 212, 253, 249, 191, 12, 114, 118, 156, 50, 248, 253, 119, 99, 183, 9, 144, 94, 239, 214, 12, 155, 180, 82, 62, 115, 161, 25, 36, 4, 114, 85, 183, 205, 242, 231, 224, 222, 218, 240, 239, 31, 219, 127, 248, 172, 172, 141, 77, 143, 21, 208, 65, 199, 10, 47, 67, 180, 160, 12, 154, 236, 188, 30, 78, 152, 110, 84, 124, 219, 200, 16, 233, 175, 0, 225, 144, 63, 199, 173, 81, 180, 86, 79, 228, 170, 187, 255, 75, 218, 173, 202, 9, 53, 15, 29, 255, 155, 92, 174, 128, 34, 145, 16, 163, 205, 244, 18, 181, 172, 5, 173, 127, 196, 31, 177, 228, 106, 12, 13, 38, 49, 219, 144, 0, 118, 233, 176, 172, 122, 192, 169, 16, 19, 124, 72, 153, 133, 105, 245, 80, 233, 219, 130, 242, 181, 187, 247, 163, 59, 234, 168, 159, 59, 180, 183, 234, 61, 236, 173, 100, 209, 217, 245, 171, 174, 3, 53, 47, 244, 184, 175, 106, 22, 78, 233, 186, 170, 93, 61, 146, 62, 47, 245, 197, 17, 53, 56, 32, 65, 174, 191, 197, 69, 199, 35, 88, 14, 134, 228, 142, 69, 171, 120, 163, 113, 92, 131, 213, 151, 175, 79, 30, 111, 246, 242, 57, 23, 179, 7, 236, 33, 191, 181, 146, 52, 218, 165, 118, 122, 61, 53, 190, 62, 110, 155, 95, 134, 202, 222, 227, 246, 200, 130, 245, 102, 223, 125, 248, 228, 147, 88, 80, 113, 30, 9, 55, 122, 207, 53, 215, 173, 0, 147, 4, 70, 71, 216, 81, 224, 155, 144, 144, 173, 242, 28, 168, 244, 23, 227, 178, 174, 214, 118, 24, 214, 167, 229, 188, 108, 169, 248, 138, 76, 217, 197, 55, 90, 100, 127, 193, 3, 174, 69, 3, 179, 237, 14, 128, 152, 166, 127, 199, 3, 54, 135, 3, 255, 145, 226, 27, 110, 24, 92, 235, 243, 158, 72, 80, 238, 92, 72, 113, 64, 95, 232, 231, 237, 114, 218, 196, 250, 100, 192, 151, 123, 143, 110, 231, 144, 217, 75, 49, 93, 72, 95, 249, 90, 1, 224, 123, 202, 220, 6, 93, 47, 219, 12, 23, 166, 135, 192, 0, 220, 255, 185, 231, 82, 38, 241, 32, 99, 208, 73, 102, 116, 64, 228, 245, 222, 233, 199, 80, 98, 77, 129, 130, 178, 20, 78, 199, 237, 187, 121, 150, 139, 101, 120, 90, 58, 195, 167, 153, 168, 214, 229, 224, 142, 172, 160, 210, 89, 167, 179, 226, 35, 57, 213, 100, 137, 155, 155, 192, 237, 232, 38, 149, 201, 37, 123, 36, 253, 159, 187, 149, 26, 153, 240, 73, 255, 147, 81, 99, 252, 23, 143, 88, 57, 76, 191, 0, 123, 217, 178, 137, 127, 33, 241, 223, 38, 164, 179, 146, 219, 242, 181, 54, 77, 5, 166, 25, 237, 82, 65, 209, 254, 101, 59, 83, 24, 4, 59, 105, 187, 182, 151, 132, 199, 74, 190, 174, 41, 160, 184, 217, 88, 89, 169, 172, 32, 97, 220, 95, 131, 89, 46, 192, 108, 141, 70, 125, 79, 209, 147, 164, 19, 210, 112, 134, 221, 219, 201, 224, 126, 201, 205, 110, 4, 94, 48, 234, 96, 2, 9, 215, 153, 223, 168, 126, 130, 120, 112, 54, 133, 131, 206, 197, 50, 27, 58, 41, 39, 202, 241, 112, 157, 4, 255, 43, 212, 103, 163, 212, 82, 101, 108, 95, 25, 210, 242, 106, 182, 38, 9, 23, 105, 97, 202, 104, 229, 76, 241, 249, 177, 97, 208, 213, 103, 177, 244, 135, 214, 240, 229, 80, 101, 29, 112, 153, 35, 141, 253, 155, 60, 188, 183, 234, 66, 86, 20, 238, 16, 67, 136, 18, 107, 178, 80, 233, 41, 206, 177, 196, 155, 158, 206, 203, 88, 82, 230, 250, 23, 189, 225, 167, 91, 164, 159, 237, 63, 27, 60, 154, 163, 77, 227, 43, 115, 104, 241, 241, 74, 161, 36, 214, 189, 60, 248, 82, 132, 118, 76, 154, 106, 246, 244, 160, 218, 237, 196, 44, 17, 48, 64, 247, 117, 218, 188, 96, 95, 131, 14, 32, 194, 143, 237, 222, 79, 51, 14
        ].components_into()
    }
}
