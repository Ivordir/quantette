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
        .expect("loaded each image")
}

pub fn load_image_dir(dir: impl AsRef<Path>) -> Vec<(String, RgbImage)> {
    let mut paths = std::fs::read_dir(dir)
        .expect("read img directory")
        .collect::<Result<Vec<_>, _>>()
        .expect("read each file")
        .iter()
        .map(std::fs::DirEntry::path)
        .collect::<Vec<_>>();

    paths.sort();

    load_images(&paths)
}

pub fn to_oklab_counts(
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

static UNSPLASH_IMAGES: OnceLock<Vec<(String, RgbImage)>> = OnceLock::new();

pub fn load_unsplash_images() -> Vec<(String, RgbImage)> {
    load_image_dir_relative_to_root(UNSPLASH_DIR)
}

pub fn unsplash_images() -> &'static [(String, RgbImage)] {
    UNSPLASH_IMAGES.get_or_init(load_unsplash_images)
}

static CQ100_IMAGES: OnceLock<Vec<(String, RgbImage)>> = OnceLock::new();

pub fn load_cq100_images() -> Vec<(String, RgbImage)> {
    load_image_dir_relative_to_root(CQ100_DIR)
}

pub fn cq100_images() -> &'static [(String, RgbImage)] {
    CQ100_IMAGES.get_or_init(load_cq100_images)
}
