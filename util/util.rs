#![allow(dead_code)]

use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
};

use image::RgbImage;
use palette::{IntoColor, Oklab};
use quantette::IndexedColorCounts;

pub fn load_images(images: &[PathBuf]) -> Vec<(String, RgbImage)> {
    images
        .iter()
        .map(|path| {
            image::open(path).map(|image| {
                (
                    path.file_name().unwrap().to_owned().into_string().unwrap(),
                    image.into_rgb8(),
                )
            })
        })
        .collect::<Result<_, _>>()
        .unwrap()
}

pub fn load_image_dir(dir: impl AsRef<Path>) -> Vec<(String, RgbImage)> {
    let mut paths = std::fs::read_dir(dir)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .iter()
        .map(std::fs::DirEntry::path)
        .collect::<Vec<_>>();

    paths.sort();

    load_images(&paths)
}

fn generate_resolutions(images: Vec<(String, RgbImage)>) -> Vec<(String, RgbImage)> {
    images
        .into_iter()
        .flat_map(|(path, image)| {
            let mut images = Vec::new();
            let (width, height) = image.dimensions();
            let pixels = width * height;
            let mut w = 240;
            let mut h = 160;
            let mut next_width = w * 2;
            let mut next_height = h * 2;
            while next_width * next_height < pixels {
                let image = image::imageops::thumbnail(&image, w, h);
                images.push((format!("{path}@{w}x{h}"), image));
                w = next_width;
                h = next_height;
                next_width *= 2;
                next_height *= 2;
            }
            images.push((format!("{path}@original"), image));
            images
        })
        .collect()
}

fn to_oklab_counts(
    images: &[(String, RgbImage)],
) -> Vec<(String, IndexedColorCounts<Oklab, f32, 3>)> {
    images
        .iter()
        .map(|(path, image)| {
            (
                path.clone(),
                IndexedColorCounts::try_from_rgbimage_par(image, |srgb| {
                    srgb.into_linear().into_color()
                })
                .unwrap(),
            )
        })
        .collect()
}

pub const CQ100_DIR: &str = "img/CQ100/img";
pub const UNSPLASH_DIR: &str = "img/unsplash/img";

pub fn load_image_dir_relative_to_root(dir: impl AsRef<Path>) -> Vec<(String, RgbImage)> {
    // assume current exe path is something like: target/build/deps/current_exe
    let exe = std::env::current_exe().unwrap();
    let root = exe
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .and_then(Path::parent)
        .unwrap();

    load_image_dir(root.join(dir.as_ref()))
}

static BENCHMARK_IMAGES: OnceLock<Vec<(String, RgbImage)>> = OnceLock::new();

pub fn benchmark_images() -> &'static [(String, RgbImage)] {
    BENCHMARK_IMAGES
        .get_or_init(|| generate_resolutions(load_image_dir_relative_to_root(UNSPLASH_DIR)))
}

static BENCHMARK_COUNTS: OnceLock<Vec<(String, IndexedColorCounts<Oklab, f32, 3>)>> =
    OnceLock::new();

pub fn benchmark_counts() -> &'static [(String, IndexedColorCounts<Oklab, f32, 3>)] {
    BENCHMARK_COUNTS.get_or_init(|| to_oklab_counts(benchmark_images()))
}
