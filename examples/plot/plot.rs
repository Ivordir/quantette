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
    fmt::Display,
    io::{BufWriter, Write},
    path::PathBuf,
};

use clap::{Parser, ValueEnum};
use palette::{FromColor, IntoColor, LinSrgb, Oklab, Srgb};
use quantette::{
    kmeans::{self, Centroids},
    wu, ColorSpace, PaletteSize, UniqueColorCounts,
};

#[derive(Clone, Copy, ValueEnum)]
enum Algorithm {
    Wu,
    Online,
    Minibatch,
}

impl Display for Algorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Algorithm::Wu => "wu",
                Algorithm::Online => "online",
                Algorithm::Minibatch => "minibatch",
            }
        )
    }
}

#[derive(Parser)]
struct Options {
    image: PathBuf,

    #[arg(short, long, default_value_t = 16.into(), value_parser = parse_palette_size)]
    k: PaletteSize,

    #[arg(long, default_value_t = Algorithm::Wu)]
    algo: Algorithm,

    #[arg(long, default_value_t = 0.5)]
    sampling_factor: f64,

    #[arg(long, default_value_t = 4096)]
    batch_size: u32,

    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn parse_palette_size(s: &str) -> Result<PaletteSize, String> {
    let value: u16 = s.parse().map_err(|e| format!("{e}"))?;
    value.try_into().map_err(|e| format!("{e}"))
}

fn main() {
    let Options {
        image,
        k,
        algo,
        sampling_factor,
        batch_size,
        seed,
    } = Options::parse();

    let image = image::open(image).unwrap().into_rgb8();

    let color_counts = UniqueColorCounts::<Oklab, _, 3>::try_from_rgbimage_par(&image, |srgb| {
        srgb.into_linear().into_color()
    })
    .unwrap();

    let result = wu::palette_par(&color_counts, k, &ColorSpace::default_binner_oklab_f32());

    let result = match algo {
        Algorithm::Wu => result,
        Algorithm::Online => {
            let initial_centroids = Centroids::try_from(result.palette).unwrap();
            let num_samples = (f64::from(color_counts.num_colors()) * sampling_factor) as u32;
            kmeans::palette::<_, _, 3>(&color_counts, num_samples, initial_centroids, seed)
        }
        Algorithm::Minibatch => {
            let initial_centroids = Centroids::try_from(result.palette).unwrap();
            let num_samples = (f64::from(color_counts.num_colors()) * sampling_factor) as u32;
            kmeans::palette_par::<_, _, 3>(
                &color_counts,
                num_samples,
                batch_size,
                initial_centroids,
                seed,
            )
        }
    };

    let mut out = BufWriter::new(std::io::stdout());

    let _ = writeln!(out, "#Colors");
    let _ = writeln!(out, "a b l n color");
    for (&oklab, count) in color_counts.colors().iter().zip(color_counts.counts()) {
        let srgb: Srgb<u8> = LinSrgb::from_color(oklab).into_encoding();
        let _ = writeln!(
            out,
            "{} {} {} {} 0x{:X}",
            oklab.a, oklab.b, oklab.l, count, srgb
        );
    }

    let _ = writeln!(out);
    let _ = writeln!(out);

    let _ = writeln!(out, "#Centroids");
    let _ = writeln!(out, "a b l n color");
    for (centroid, count) in result.palette.into_iter().zip(result.counts) {
        let srgb: Srgb<u8> = LinSrgb::from_color(centroid).into_encoding();
        let _ = writeln!(
            out,
            "{} {} {} {} 0x{:X}",
            centroid.a, centroid.b, centroid.l, count, srgb
        );
    }
}
