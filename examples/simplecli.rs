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

use std::{fmt::Display, path::PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
use image::RgbImage;
use palette::{cast::IntoComponents, Srgb};
use quantette::{
    ColorSpace, FloydSteinberg, ImagePipeline, KmeansOptions, PaletteSize, QuantizeMethod,
};
use rayon::prelude::*;
use rgb::FromSlice;

#[derive(Copy, Clone, ValueEnum)]
enum CliColorSpace {
    Oklab,
    Lab,
    Srgb,
}

impl From<CliColorSpace> for ColorSpace {
    fn from(value: CliColorSpace) -> Self {
        match value {
            CliColorSpace::Oklab => ColorSpace::Oklab,
            CliColorSpace::Lab => ColorSpace::Lab,
            CliColorSpace::Srgb => ColorSpace::Srgb,
        }
    }
}

impl Display for CliColorSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                CliColorSpace::Oklab => "oklab",
                CliColorSpace::Lab => "lab",
                CliColorSpace::Srgb => "srgb",
            }
        )
    }
}

#[derive(Subcommand)]
enum Quantizer {
    Quantette {
        #[arg(long, default_value_t = CliColorSpace::Srgb)]
        colorspace: CliColorSpace,

        #[arg(long)]
        kmeans: bool,

        #[arg(long)]
        dither: bool,

        #[arg(long, default_value_t = FloydSteinberg::DEFAULT_ERROR_DIFFUSION)]
        dither_error_diffusion: f32,

        #[arg(long, default_value_t = 0.5)]
        sampling_factor: f32,

        #[arg(long, default_value_t = 4096)]
        batch_size: u32,

        #[arg(long, default_value_t = 0)]
        seed: u64,

        #[arg(short, long, default_value_t = 0)]
        threads: u8,
    },
    Neuquant {
        #[arg(long, default_value_t = 1)]
        sample_frac: u8,
    },
    Imagequant {
        #[arg(short, long)]
        quality: Option<u8>,

        #[arg(long, default_value_t = 1.0)]
        dither_level: f32,

        #[arg(short, long, default_value_t = 0)]
        threads: u8,
    },
    Exoquant {
        #[arg(long)]
        kmeans: bool,

        #[arg(long)]
        dither: bool,
    },
}

#[derive(Parser)]
pub struct Options {
    #[arg(short, long, default_value_t = PaletteSize::default(), value_parser = parse_palette_size)]
    k: PaletteSize,

    #[arg(long)]
    verbose: bool,

    input: PathBuf,

    #[arg(short, long)]
    output: Option<PathBuf>,

    #[command(subcommand)]
    quantizer: Quantizer,
}

fn parse_palette_size(s: &str) -> Result<PaletteSize, String> {
    let value: u16 = s.parse().map_err(|e| format!("{e}"))?;
    value.try_into().map_err(|e| format!("{e}"))
}

fn main() {
    let Options { quantizer, k, verbose, input, output } = Options::parse();

    macro_rules! log {
        ($name: literal, $val: expr) => {
            if verbose {
                let time = std::time::Instant::now();
                let value = $val;
                println!("{} took {}ms", $name, time.elapsed().as_millis());
                value
            } else {
                $val
            }
        };
    }

    let image = log!("read image", image::open(input).unwrap());

    match quantizer {
        Quantizer::Quantette {
            colorspace,
            dither,
            dither_error_diffusion,
            kmeans,
            sampling_factor,
            batch_size,
            seed,
            threads,
        } => {
            let image = image.into_rgb8();

            let method = if kmeans {
                QuantizeMethod::Kmeans(
                    KmeansOptions::new()
                        .sampling_factor(sampling_factor)
                        .batch_size(batch_size)
                        .seed(seed),
                )
            } else {
                QuantizeMethod::wu()
            };

            let colorspace = colorspace.into();
            let pipeline = ImagePipeline::try_from(&image)
                .unwrap()
                .quantize_method(method)
                .colorspace(colorspace)
                .dither(dither)
                .dither_error_diffusion(dither_error_diffusion)
                .palette_size(k);

            if let Some(output) = output {
                let image = log!(
                    "quantization and remapping",
                    match threads {
                        0 => pipeline.quantized_rgbimage_par(),
                        1 => pipeline.quantized_rgbimage(),
                        t => {
                            let pool = rayon::ThreadPoolBuilder::new()
                                .num_threads(t.into())
                                .build()
                                .unwrap();

                            pool.install(|| pipeline.quantized_rgbimage_par())
                        }
                    }
                );
                log!("write image", image.save(output).unwrap())
            } else {
                let colors = log!(
                    "quantization",
                    match threads {
                        0 => pipeline.palette_par(),
                        1 => pipeline.palette(),
                        t => {
                            let pool = rayon::ThreadPoolBuilder::new()
                                .num_threads(t.into())
                                .build()
                                .unwrap();

                            pool.install(|| pipeline.palette_par())
                        }
                    }
                );
                print_palette(colors)
            }
        }
        Quantizer::Neuquant { sample_frac } => {
            let image = image.into_rgba8();

            let nq = log!(
                "quantization",
                color_quant::NeuQuant::new(sample_frac.into(), k.into_inner().into(), &image)
            );

            let colors = nq
                .color_map_rgba()
                .chunks_exact(4)
                .map(|c| Srgb::new(c[0], c[1], c[2]))
                .collect();

            if let Some(output) = output {
                let indices = log!(
                    "remapping",
                    image
                        .chunks_exact(4)
                        .map(|pix| nq.index_of(pix) as u8)
                        .collect()
                );

                let image = indexed_image(image.dimensions(), colors, indices);
                log!("write image", image.save(output).unwrap())
            } else {
                print_palette(colors)
            }
        }
        Quantizer::Imagequant { quality, dither_level, threads } => {
            let image = image.into_rgba8();

            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads.into())
                .build()
                .unwrap();

            pool.install(|| {
                let mut libq = imagequant::new();
                let mut img = libq
                    .new_image(
                        image.as_rgba(),
                        image.width() as usize,
                        image.height() as usize,
                        0.0,
                    )
                    .unwrap();

                if let Some(quality) = quality {
                    libq.set_quality(0, quality).unwrap();
                } else {
                    libq.set_max_colors(k.into_inner().into()).unwrap();
                }

                let mut quantized = log!("quantization", libq.quantize(&mut img).unwrap());

                if let Some(output) = output {
                    let (colors, indices) = log!("remapping", {
                        quantized.set_dithering_level(dither_level).unwrap();
                        quantized.remapped(&mut img).unwrap()
                    });

                    let colors = colors
                        .into_par_iter()
                        .map(|c| Srgb::new(c.r, c.g, c.b))
                        .collect();

                    let image = indexed_image(image.dimensions(), colors, indices);
                    log!("write image", image.save(output).unwrap())
                } else {
                    let colors = quantized
                        .palette()
                        .iter()
                        .map(|c| Srgb::new(c.r, c.g, c.b))
                        .collect();

                    print_palette(colors)
                }
            })
        }
        Quantizer::Exoquant { kmeans, dither } => {
            let image = image.into_rgba8();

            let pixels = image
                .pixels()
                .map(|p| exoquant::Color::new(p.0[0], p.0[1], p.0[2], p.0[3]))
                .collect::<Vec<_>>();

            let width = image.width() as usize;

            let k = k.into_inner().into();

            if let Some(output) = output {
                let (colors, indices) = log!(
                    "quantization and remapping",
                    match (kmeans, dither) {
                        (true, true) => exoquant::convert_to_indexed(
                            &pixels,
                            width,
                            k,
                            &exoquant::optimizer::KMeans,
                            &exoquant::ditherer::FloydSteinberg::new(),
                        ),
                        (true, false) => exoquant::convert_to_indexed(
                            &pixels,
                            width,
                            k,
                            &exoquant::optimizer::KMeans,
                            &exoquant::ditherer::None,
                        ),
                        (false, true) => exoquant::convert_to_indexed(
                            &pixels,
                            width,
                            k,
                            &exoquant::optimizer::None,
                            &exoquant::ditherer::FloydSteinberg::new(),
                        ),
                        (false, false) => exoquant::convert_to_indexed(
                            &pixels,
                            width,
                            k,
                            &exoquant::optimizer::None,
                            &exoquant::ditherer::None,
                        ),
                    }
                );

                let colors = colors
                    .into_iter()
                    .map(|c| Srgb::new(c.r, c.g, c.b))
                    .collect();

                let image = indexed_image(image.dimensions(), colors, indices);
                log!("write image", image.save(output).unwrap())
            } else {
                let colors = log!(
                    "quantization",
                    if kmeans {
                        exoquant::generate_palette(
                            &pixels.into_iter().collect(),
                            &exoquant::SimpleColorSpace::default(),
                            &exoquant::optimizer::KMeans,
                            k,
                        )
                    } else {
                        exoquant::generate_palette(
                            &pixels.into_iter().collect(),
                            &exoquant::SimpleColorSpace::default(),
                            &exoquant::optimizer::None,
                            k,
                        )
                    }
                );

                let colors = colors
                    .into_iter()
                    .map(|c| Srgb::new(c.r, c.g, c.b))
                    .collect();

                print_palette(colors)
            }
        }
    }
}

fn indexed_image(
    (width, height): (u32, u32),
    palette: Vec<Srgb<u8>>,
    indices: Vec<u8>,
) -> RgbImage {
    let palette = palette.as_slice(); // faster for some reason
    let buf = indices
        .par_iter()
        .map(|&i| palette[usize::from(i)])
        .collect::<Vec<_>>()
        .into_components();

    #[allow(clippy::unwrap_used)]
    {
        // indices.len() will be equal to width * height,
        // so buf should be large enough by nature of its construction
        RgbImage::from_vec(width, height, buf).unwrap()
    }
}

fn print_palette(palette: Vec<Srgb<u8>>) {
    println!(
        "{}",
        palette
            .into_iter()
            .map(|color| format!("{color:X}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
}
