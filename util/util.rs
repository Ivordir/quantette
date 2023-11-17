#![allow(dead_code)]

use std::{
    cmp::Reverse,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use image::RgbImage;
use palette::{IntoColor, Oklab};
use quantette::IndexedColorCounts;

pub fn load_images<P>(images: &[P]) -> Vec<RgbImage>
where
    P: AsRef<Path>,
{
    images
        .iter()
        .map(|path| image::open(path).unwrap().into_rgb8())
        .collect()
}

pub fn load_image_dir(dir: impl AsRef<Path>) -> Vec<(PathBuf, RgbImage)> {
    let mut paths = fs::read_dir(dir.as_ref())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();

    paths.sort_unstable();

    let images = load_images(&paths);

    paths.into_iter().zip(images).collect()
}

fn root_dir() -> PathBuf {
    // assume current exe path is something like: target/profile/dir/current_exe
    std::env::current_exe()
        .unwrap()
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .and_then(Path::parent)
        .unwrap()
        .into()
}

pub fn load_image_dir_relative_to_root(dir: impl AsRef<Path>) -> Vec<(PathBuf, RgbImage)> {
    let mut root = root_dir();
    root.push(dir.as_ref());
    load_image_dir(root)
}

static BENCHMARK_IMAGES: OnceLock<Vec<(String, RgbImage)>> = OnceLock::new();

pub fn benchmark_images() -> &'static [(String, RgbImage)] {
    BENCHMARK_IMAGES.get_or_init(|| {
        let images = {
            let mut path = root_dir();
            path.push("img");
            path.push("unsplash");
            path.push("img");
            path
        };

        let mut images = fs::read_dir(images)
            .unwrap()
            .map(|dir| {
                let dir = dir.unwrap().path();
                let resolution = dir.file_stem().and_then(OsStr::to_str).unwrap();
                let resolution = if let Some((width, height)) = resolution.split_once('x') {
                    (width.parse().unwrap(), height.parse().unwrap())
                } else {
                    (u32::MAX, u32::MAX)
                };
                (resolution, dir)
            })
            .collect::<Vec<_>>();

        images.sort_unstable_by_key(|&(res, _)| Reverse(res));

        images
            .into_iter()
            .flat_map(|(_, dir)| load_image_dir(dir))
            .map(|(path, img)| {
                let name = path
                    .components()
                    .rev()
                    .take(2)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<PathBuf>()
                    .display()
                    .to_string();

                (name, img)
            })
            .collect()
    })
}

fn to_oklab_counts(
    images: &[(String, RgbImage)],
) -> Vec<(String, IndexedColorCounts<Oklab, f32, 3>)> {
    images
        .iter()
        .map(|(path, image)| {
            let counts = IndexedColorCounts::try_from_rgbimage_par(image, |srgb| {
                srgb.into_linear().into_color()
            })
            .unwrap();

            (path.clone(), counts)
        })
        .collect()
}

static BENCHMARK_COUNTS: OnceLock<Vec<(String, IndexedColorCounts<Oklab, f32, 3>)>> =
    OnceLock::new();

pub fn benchmark_counts() -> &'static [(String, IndexedColorCounts<Oklab, f32, 3>)] {
    BENCHMARK_COUNTS.get_or_init(|| to_oklab_counts(benchmark_images()))
}
