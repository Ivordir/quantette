#![deny(unsafe_code, unsafe_op_in_unsafe_fn)]
#![warn(
    clippy::use_debug,
    clippy::dbg_macro,
    clippy::todo,
    clippy::unimplemented,
    clippy::unneeded_field_pattern,
    clippy::rest_pat_in_fully_bound_structs,
    clippy::unnecessary_self_imports,
    clippy::str_to_string,
    clippy::string_to_string,
    clippy::string_slice
)]

use std::{
    ffi::OsStr,
    fmt::{self, Display},
    path::PathBuf,
};

use clap::{Args, Parser, Subcommand, ValueEnum};
use image::{buffer::ConvertBuffer, RgbImage, RgbaImage};
use palette::{IntoColor, Lab, LinSrgb, Oklab, Srgb};
use quantette::{
    kmeans, wu, ColorComponents, ColorCounts, ColorSpace, FloydSteinberg, IndexedColorCounts,
    PaletteSize,
};
use rayon::prelude::*;
use rgb::{FromSlice, RGB8, RGBA};

#[path = "../util/util.rs"]
mod util;

/// Set of algorithm choices to create a palette
#[derive(Debug, Copy, Clone, ValueEnum)]
enum Algorithm {
    Minibatch,
    Online,
    Wu,
    Neuquant,
    Exoquant,
    Imagequant,
}

impl Display for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Algorithm::Minibatch => "minibatch",
                Algorithm::Online => "online",
                Algorithm::Wu => "wu",
                Algorithm::Neuquant => "neuquant",
                Algorithm::Imagequant => "imagequant",
                Algorithm::Exoquant => "exoquant",
            }
        )
    }
}

#[derive(Debug, Copy, Clone, ValueEnum)]
enum CliColorSpace {
    Srgb,
    Lab,
    Oklab,
}

impl Display for CliColorSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                CliColorSpace::Srgb => "srgb",
                CliColorSpace::Lab => "lab",
                CliColorSpace::Oklab => "oklab",
            }
        )
    }
}

impl From<CliColorSpace> for ColorSpace {
    fn from(value: CliColorSpace) -> Self {
        match value {
            CliColorSpace::Srgb => ColorSpace::Srgb,
            CliColorSpace::Lab => ColorSpace::Lab,
            CliColorSpace::Oklab => ColorSpace::Oklab,
        }
    }
}

#[derive(Args)]
struct Report {
    #[arg(short, long, default_value_t = Algorithm::Wu)]
    algo: Algorithm,

    #[arg(short, long, default_value_t = CliColorSpace::Srgb)]
    colorspace: CliColorSpace,

    #[arg(short, long, default_value = "16,64,256", value_delimiter = ',', value_parser = parse_palette_size)]
    k: Vec<PaletteSize>,

    #[arg(short = 'f', long, default_value_t = 0.5)]
    sampling_factor: f64,

    #[arg(long, default_value_t = 4096)]
    batch_size: u32,

    #[arg(long, default_value_t = 0)]
    seed: u64,

    #[arg(long, default_value_t = 1)]
    sample_frac: u8,

    #[arg(long)]
    kmeans_optimize: bool,

    #[arg(long)]
    dither: bool,

    #[arg(long, default_value_t = FloydSteinberg::DEFAULT_ERROR_DIFFUSION)]
    dither_error_diffusion: f32,

    images: Vec<PathBuf>,
}

impl Report {
    fn num_samples<Color, Component, const N: usize>(
        &self,
        color_counts: &impl ColorCounts<Color, Component, N>,
    ) -> u32
    where
        Color: ColorComponents<Component, N>,
    {
        (self.sampling_factor * f64::from(color_counts.num_colors())) as u32
    }
}

#[derive(Subcommand)]
enum Command {
    Report(Report),
    Compare { image_a: PathBuf, image_b: PathBuf },
}

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

fn parse_palette_size(s: &str) -> Result<PaletteSize, String> {
    let value: u16 = s.parse().map_err(|e| format!("{e}"))?;
    value.try_into().map_err(|e| format!("{e}"))
}

const COL_WIDTH: usize = 10;
const NUM_DECIMALS: usize = 4;

fn main() {
    let Cli { command } = Cli::parse();

    match command {
        Command::Report(options) => report(options),
        Command::Compare { image_a, image_b } => {
            let ds = dssim::new();
            let a = image::open(image_a).unwrap().into_rgb8();
            let b = image::open(image_b).unwrap().into_rgb8();
            let a = ds
                .create_image_rgb(a.as_rgb(), a.width() as usize, a.height() as usize)
                .unwrap();

            let b = ds
                .create_image_rgb(b.as_rgb(), b.width() as usize, b.height() as usize)
                .unwrap();

            println!("{}", f64::from(ds.compare(&a, b).0))
        }
    }
}

fn report(options: Report) {
    let images = if options.images.is_empty() {
        util::load_image_dir_relative_to_root(
            ["img", "CQ100", "img"].into_iter().collect::<PathBuf>(),
        )
        .into_iter()
        .map(|(path, img)| {
            let name = path.file_stem().and_then(OsStr::to_str).unwrap().to_owned();
            (name, img)
        })
        .collect::<Vec<_>>()
    } else {
        options
            .images
            .iter()
            .map(|path| path.display().to_string())
            .zip(util::load_images(&options.images))
            .collect()
    };

    // use char count as supplement for grapheme count
    let max_name_len = images
        .iter()
        .map(|(name, _)| name.chars().count())
        .max()
        .unwrap_or(0);

    println!(
        "{:max_name_len$} {}",
        "image",
        options
            .k
            .iter()
            .map(|k| format!(
                "{:>1$} {2}",
                k.into_inner(),
                COL_WIDTH - NUM_DECIMALS - 1,
                str::repeat(" ", NUM_DECIMALS)
            ))
            .collect::<Vec<_>>()
            .join(" "),
    );

    fn each_image<F1, F2>(
        options: &Report,
        images: Vec<(String, RgbImage)>,
        name_len: usize,
        mut f1: F1,
    ) where
        F1: FnMut(RgbImage) -> F2,
        F2: FnMut(PaletteSize) -> Vec<RGB8>,
    {
        let ds = dssim::new();
        for (path, image) in images {
            let width = image.width() as usize;
            let height = image.height() as usize;

            let original = ds.create_image_rgb(image.as_rgb(), width, height).unwrap();

            let mut f2 = f1(image);
            let ssim_by_k = options
                .k
                .iter()
                .map(|&k| {
                    let quantized = ds.create_image_rgb(&f2(k), width, height).unwrap();
                    let ssim = 100.0 * f64::from(ds.compare(&original, quantized).0);
                    format!("{ssim:>COL_WIDTH$.NUM_DECIMALS$}")
                })
                .collect::<Vec<_>>()
                .join(" ");

            println!("{path:name_len$} {ssim_by_k}");
        }
    }

    fn each_image_color_counts<Color, Component, const N: usize>(
        options: &Report,
        images: Vec<(String, RgbImage)>,
        name_len: usize,
        convert_to: impl Fn(Srgb<u8>) -> Color + Sync,
        convert_from: impl Fn(Color) -> Srgb<u8> + Copy,
        f: impl Fn(&IndexedColorCounts<Color, Component, N>, PaletteSize) -> (Vec<Color>, Vec<u8>)
            + Copy,
    ) where
        Color: ColorComponents<Component, N> + Send + Sync,
        Component: Copy + Into<f32> + 'static,
    {
        each_image(options, images, name_len, |image| {
            let color_counts =
                IndexedColorCounts::try_from_rgbimage_par(&image, &convert_to).unwrap();

            move |k| {
                let (colors, mut indices) = f(&color_counts, k);
                if options.dither {
                    FloydSteinberg::with_error_diffusion(options.dither_error_diffusion)
                        .unwrap()
                        .dither_indexed(
                            &colors,
                            &mut indices,
                            color_counts.colors(),
                            color_counts.indices(),
                            image.width(),
                            image.height(),
                        )
                }
                let colors = colors.into_iter().map(convert_from).collect::<Vec<_>>();
                indices
                    .into_par_iter()
                    .map(|i| colors[usize::from(i)].into_components().into())
                    .collect()
            }
        });
    }

    fn each_image_color_counts_convert<Color, Component, const N: usize>(
        options: &Report,
        images: Vec<(String, RgbImage)>,
        name_len: usize,
        f: impl Fn(&IndexedColorCounts<Color, Component, N>, PaletteSize) -> (Vec<Color>, Vec<u8>)
            + Copy,
    ) where
        Color: ColorComponents<Component, N> + Send + Sync,
        LinSrgb: IntoColor<Color>,
        Color: IntoColor<LinSrgb>,
        Component: Copy + Into<f32> + 'static,
    {
        each_image_color_counts(
            options,
            images,
            name_len,
            |srgb| srgb.into_linear().into_color(),
            |color| color.into_color().into_encoding(),
            f,
        )
    }

    match (options.algo, options.colorspace.into()) {
        (Algorithm::Minibatch, ColorSpace::Srgb) => {
            each_image_color_counts::<Srgb<u8>, _, 3>(
                &options,
                images,
                max_name_len,
                |srgb| srgb,
                |srgb| srgb,
                |color_counts, k| {
                    let res = kmeans::indexed_palette_par::<_, _, 3>(
                        color_counts,
                        options.num_samples(color_counts),
                        options.batch_size,
                        wu::palette_par(color_counts, k, &ColorSpace::default_binner_srgb_u8())
                            .palette
                            .try_into()
                            .unwrap(),
                        options.seed,
                    );
                    (res.palette, res.indices)
                },
            );
        }
        (Algorithm::Minibatch, ColorSpace::Lab) => {
            each_image_color_counts_convert::<Lab, _, 3>(
                &options,
                images,
                max_name_len,
                |color_counts, k| {
                    let res = kmeans::indexed_palette_par::<_, _, 3>(
                        color_counts,
                        options.num_samples(color_counts),
                        options.batch_size,
                        wu::palette_par(color_counts, k, &ColorSpace::default_binner_lab_f32())
                            .palette
                            .try_into()
                            .unwrap(),
                        options.seed,
                    );
                    (res.palette, res.indices)
                },
            );
        }
        (Algorithm::Minibatch, ColorSpace::Oklab) => {
            each_image_color_counts_convert::<Oklab, _, 3>(
                &options,
                images,
                max_name_len,
                |color_counts, k| {
                    let res = kmeans::indexed_palette_par::<_, _, 3>(
                        color_counts,
                        options.num_samples(color_counts),
                        options.batch_size,
                        wu::palette_par(color_counts, k, &ColorSpace::default_binner_oklab_f32())
                            .palette
                            .try_into()
                            .unwrap(),
                        options.seed,
                    );
                    (res.palette, res.indices)
                },
            );
        }
        (Algorithm::Online, ColorSpace::Srgb) => {
            each_image_color_counts::<Srgb<u8>, _, 3>(
                &options,
                images,
                max_name_len,
                |srgb| srgb,
                |srgb| srgb,
                |color_counts, k| {
                    let res = kmeans::indexed_palette::<_, _, 3>(
                        color_counts,
                        options.num_samples(color_counts),
                        wu::palette(color_counts, k, &ColorSpace::default_binner_srgb_u8())
                            .palette
                            .try_into()
                            .unwrap(),
                        options.seed,
                    );
                    (res.palette, res.indices)
                },
            );
        }
        (Algorithm::Online, ColorSpace::Lab) => {
            each_image_color_counts_convert::<Lab, _, 3>(
                &options,
                images,
                max_name_len,
                |color_counts, k| {
                    let res = kmeans::indexed_palette::<_, _, 3>(
                        color_counts,
                        options.num_samples(color_counts),
                        wu::palette(color_counts, k, &ColorSpace::default_binner_lab_f32())
                            .palette
                            .try_into()
                            .unwrap(),
                        options.seed,
                    );
                    (res.palette, res.indices)
                },
            );
        }
        (Algorithm::Online, ColorSpace::Oklab) => {
            each_image_color_counts_convert::<Oklab, _, 3>(
                &options,
                images,
                max_name_len,
                |color_counts, k| {
                    let res = kmeans::indexed_palette::<_, _, 3>(
                        color_counts,
                        options.num_samples(color_counts),
                        wu::palette(color_counts, k, &ColorSpace::default_binner_oklab_f32())
                            .palette
                            .try_into()
                            .unwrap(),
                        options.seed,
                    );
                    (res.palette, res.indices)
                },
            );
        }
        (Algorithm::Wu, ColorSpace::Srgb) => {
            each_image_color_counts::<Srgb<u8>, _, 3>(
                &options,
                images,
                max_name_len,
                |srgb| srgb,
                |srgb| srgb,
                |image, k| {
                    let res = wu::indexed_palette(image, k, &ColorSpace::default_binner_srgb_u8());
                    (res.palette, res.indices)
                },
            );
        }
        (Algorithm::Wu, ColorSpace::Lab) => {
            each_image_color_counts_convert::<Lab, _, 3>(
                &options,
                images,
                max_name_len,
                |color_counts, k| {
                    let res =
                        wu::indexed_palette(color_counts, k, &ColorSpace::default_binner_lab_f32());
                    (res.palette, res.indices)
                },
            );
        }
        (Algorithm::Wu, ColorSpace::Oklab) => {
            each_image_color_counts_convert::<Oklab, _, 3>(
                &options,
                images,
                max_name_len,
                |color_counts, k| {
                    let re = wu::indexed_palette(
                        color_counts,
                        k,
                        &ColorSpace::default_binner_oklab_f32(),
                    );
                    (re.palette, re.indices)
                },
            );
        }
        (Algorithm::Neuquant, ColorSpace::Srgb) => {
            each_image(&options, images, max_name_len, |image| {
                let image: RgbaImage = image.convert();

                move |k| {
                    let nq = color_quant::NeuQuant::new(
                        options.sample_frac.into(),
                        k.into_inner().into(),
                        &image,
                    );

                    let colors = nq
                        .color_map_rgba()
                        .as_rgba()
                        .iter()
                        .map(RGBA::rgb)
                        .collect::<Vec<_>>();

                    image
                        .chunks_exact(4)
                        .map(|pix| colors[nq.index_of(pix)])
                        .collect()
                }
            });
        }
        (Algorithm::Imagequant, ColorSpace::Srgb) => {
            each_image(&options, images, max_name_len, |image| {
                let image_rgba: RgbaImage = image.convert();

                move |k| {
                    let mut libq = imagequant::new();

                    let mut img = libq
                        .new_image(
                            image_rgba.as_rgba(),
                            image.width() as usize,
                            image.height() as usize,
                            0.0,
                        )
                        .unwrap();

                    libq.set_max_colors(k.into_inner().into()).unwrap();

                    let mut quantized = libq.quantize(&mut img).unwrap();
                    if !options.dither {
                        quantized.set_dithering_level(0.0).unwrap()
                    };
                    let (colors, indices) = quantized.remapped(&mut img).unwrap();

                    indices
                        .into_iter()
                        .map(|i| colors[usize::from(i)].rgb())
                        .collect()
                }
            });
        }
        (Algorithm::Exoquant, ColorSpace::Srgb) => {
            use exoquant::{convert_to_indexed, ditherer, optimizer, Color};

            each_image(&options, images, max_name_len, |image| {
                let pixels = image
                    .pixels()
                    .map(|p| Color::new(p.0[0], p.0[1], p.0[2], u8::MAX))
                    .collect::<Vec<_>>();

                move |k| {
                    let k = k.into_inner().into();
                    let (colors, indices) = match (options.kmeans_optimize, options.dither) {
                        (true, true) => convert_to_indexed(
                            &pixels,
                            image.width() as usize,
                            k,
                            &optimizer::KMeans,
                            &ditherer::FloydSteinberg::new(),
                        ),
                        (true, false) => convert_to_indexed(
                            &pixels,
                            image.width() as usize,
                            k,
                            &optimizer::KMeans,
                            &ditherer::None,
                        ),
                        (false, true) => convert_to_indexed(
                            &pixels,
                            image.width() as usize,
                            k,
                            &optimizer::None,
                            &ditherer::FloydSteinberg::new(),
                        ),
                        (false, false) => convert_to_indexed(
                            &pixels,
                            image.width() as usize,
                            k,
                            &optimizer::None,
                            &ditherer::None,
                        ),
                    };

                    indices
                        .into_iter()
                        .map(|i| {
                            let color = colors[usize::from(i)];
                            RGB8::new(color.r, color.g, color.b)
                        })
                        .collect()
                }
            });
        }
        (algo, _) => {
            panic!(
                "{algo} does not support the {} color space",
                options.colorspace
            )
        }
    }
}
