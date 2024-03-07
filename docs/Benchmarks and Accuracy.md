# Benchmarks and Accuracy

Below are some comparisons between `quantette` and some other libraries.
Each library has different situations and/or options that can make then perform better or worse,
but these limited comparisons should give a rough point of reference.
Note that `quantette` currently doesn't support alpha channel/component, while the other libraries do.

# Setup

The settings and libraries being compared are:

- `Wu - sRGB`: `quantette`'s Wu's quantizer implementation running in the sRGB color space.
  The default number of bins was used (`32`).
- `K-means - Oklab`: `quantette`'s k-means quantizer running in the Oklab color space.
  The default settings (sampling factor of `0.5` and batch size of `4096`) were used.
- `imagequant` version `4.3.0` run with the default library options (quality of `100`).
- `color_quant` version `1.1.0` run with a `sample_frac` of `10`.
- `exoquant` version `0.2.0` run without k-means optimization, since it would otherwise take too long.

The "Time" tables below provide the total time, in milliseconds, for quantization and remapping
as reported by the `cli` binary with the `--verbose` flag.
The binary in question can be found in `examples/`. 30 trials were run and averaged for each data point.
The `Wu - Srgb`, `K-means - Oklab`, and `imagequant` columns used 4 threads,
while `color_quant` and `exoquant` only support single-threaded execution.
So, multiply or divide by 4 as you see fit.

The "Accuracy" tables list the DSSIM values as reported by the `accuracy` binary found in `examples/`
using the [`dssim`](https://crates.io/crates/dssim) crate.
Note that `exoquant` results are not deterministic, since it uses `rand::random()`.

All results below are for 256 colors, and no `rustc` flags were used (e.g., no `avx`).
The images used can be found in `img/unsplash/img/Original`, and they are all roughly 6000x4000 in resolution.

# Without Dithering

## Time

| Image            | Wu - sRGB | K-means - Oklab | imagequant | color_quant | exoquant |
| ---------------- | ---------:| ---------------:| ----------:| -----------:| --------:|
| Akihabara.jpg    | 37        | 276             | 1514       | 3571        | 8697     |
| Boothbay.jpg     | 52        | 265             | 1107       | 4574        | 7839     |
| Hokkaido.jpg     | 41        | 213             | 785        | 3351        | 5789     |
| Jewel Changi.jpg | 40        | 169             | 660        | 2942        | 4873     |
| Louvre.jpg       | 40        | 188             | 731        | 3786        | 5544     |

## Accuracy/DSSIM

| Image            | Wu - sRGB | K-means - Oklab | imagequant | color_quant | exoquant |
| ---------------- | ---------:| ---------------:| ----------:| -----------:| --------:|
| Akihabara.jpg    | 0.007623  | 0.003881        | 0.004322   | 0.007492    | 0.005798 |
| Boothbay.jpg     | 0.004376  | 0.002269        | 0.002425   | 0.005851    | 0.003451 |
| Hokkaido.jpg     | 0.003395  | 0.001572        | 0.001728   | 0.003251    | 0.003808 |
| Jewel Changi.jpg | 0.001597  | 0.000750        | 0.000769   | 0.001567    | 0.001028 |
| Louvre.jpg       | 0.003057  | 0.001434        | 0.001561   | 0.003485    | 0.002444 |

# With Dithering

`color_quant` does not have dithering, so it is not included in the tables below.

## Time

| Image            | Wu - sRGB | K-means - Oklab | imagequant | exoquant |
| ---------------- | ---------:| ---------------:| ----------:| --------:|
| Akihabara.jpg    | 255       | 482             | 1831       | 12541    |
| Boothbay.jpg     | 367       | 556             | 1578       | 14068    |
| Hokkaido.jpg     | 296       | 473             | 1183       | 11535    |
| Jewel Changi.jpg | 266       | 391             | 832        | 9811     |
| Louvre.jpg       | 294       | 431             | 990        | 10935    |

## Accuracy/DSSIM

| Image            | Wu - sRGB | K-means - Oklab | imagequant | exoquant |
| ---------------- | ---------:| ---------------:| ----------:| --------:|
| Akihabara.jpg    | 0.003858  | 0.001981        | 0.002221   | 0.002800 |
| Boothbay.jpg     | 0.002687  | 0.001259        | 0.001408   | 0.001806 |
| Hokkaido.jpg     | 0.001860  | 0.000953        | 0.001203   | 0.001959 |
| Jewel Changi.jpg | 0.000920  | 0.000521        | 0.000589   | 0.001035 |
| Louvre.jpg       | 0.001470  | 0.000896        | 0.001077   | 0.001761 |
