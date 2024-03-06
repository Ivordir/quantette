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
- `imagequant` version `4.2.2` run with the default library options (quality of `100`).
- `color_quant` version `1.1.0` run with a `sample_frac` of `10`.
- `exoquant` version `0.2.0` run without k-means optimization, since it would otherwise take too long.

The "Time" tables below provide the total time, in milliseconds, for quantization and remapping
as reported by the `simplecli` binary with the `--verbose` flag.
The binary in question can be found in `examples/`. 30 trials were run and averaged for each data point.
The `Wu - Srgb`, `K-means - Oklab`, and `imagequant` columns used 4 threads,
while `color_quant` and `exoquant` only support single-threaded execution.
So, multiply or divide by 4 as you see fit.

The "Accuracy" tables list the DSSIM values as reported by the `accuracy` binary found in `examples/`.
Note that `exoquant` results are not deterministic, since it uses `rand::random()`.

All results below are for 256 colors, and no `rustc` flags were used (e.g., no `avx`).

# Without Dither

## Time

| Image                  | Width | Height | Wu - Srgb | K-means - Oklab | imagequant | color_quant | exoquant |
| ---------------------- | ----- | ------ | --------- | --------------- | ---------- | ----------- | -------- |
| Akihabara.jpg          | 5663  | 3769   | 36        | 266             | 1492       | 3477        | 8672     |
| Boothbay.jpg           | 6720  | 4480   | 49        | 261             | 1106       | 4514        | 7738     |
| Hokkaido.jpg           | 6000  | 4000   | 39        | 205             | 776        | 3321        | 5844     |
| Jewel Changi.jpg       | 6000  | 4000   | 41        | 166             | 652        | 2932        | 4915     |
| Louvre.jpg             | 6056  | 4000   | 41        | 182             | 723        | 3701        | 5525     |

## Accuracy/DSSIM

| Image                  | Width | Height | Wu - Srgb  | K-means - Oklab | imagequant | color_quant | exoquant   |
| ---------------------- | ----- | ------ | ---------- | --------------- | ---------- | ----------- | ---------- |
| Akihabara.jpg          | 5663  | 3769   | 0.00762276 | 0.00388072      | 0.00432195 | 0.00749248  | 0.00583372 |
| Boothbay.jpg           | 6720  | 4480   | 0.00437625 | 0.00226944      | 0.00242491 | 0.00585143  | 0.00345061 |
| Hokkaido.jpg           | 6000  | 4000   | 0.00339461 | 0.00157206      | 0.00172781 | 0.00325078  | 0.00382358 |
| Jewel Changi.jpg       | 6000  | 4000   | 0.00159665 | 0.00074988      | 0.00076888 | 0.00156673  | 0.00102799 |
| Louvre.jpg             | 6056  | 4000   | 0.00305672 | 0.00143401      | 0.00156126 | 0.00348470  | 0.00244134 |

# With Dither

`color_quant` does not have dithering, so it is not included in the tables below.

## Time

| Image                  | Width | Height | Wu - Srgb | K-means - Oklab | imagequant | exoquant |
| ---------------------- | ----- | ------ | --------- | --------------- | ---------- | -------- |
| Akihabara.jpg          | 5663  | 3769   | 0         | 0               | 0          | 0        |
| Boothbay.jpg           | 6720  | 4480   | 0         | 0               | 0          | 0        |
| Hokkaido.jpg           | 6000  | 4000   | 0         | 0               | 0          | 0        |
| Jewel Changi.jpg       | 6000  | 4000   | 0         | 0               | 0          | 0        |
| Louvre.jpg             | 6056  | 4000   | 0         | 0               | 0          | 0        |

## Accuracy/DSSIM

The results below are DSSIM values as reported by the `accuracy` binary found in `examples/`.
Note that `exoquant` results are not deterministic, since it uses `rand::random()`.

| Image                  | Width | Height | Wu - Srgb  | K-means - Oklab | imagequant | exoquant   |
| ---------------------- | ----- | ------ | ---------- | --------------- | ---------- | ---------- |
| Akihabara.jpg          | 5663  | 3769   | 0. | 0.      | 0. | 0. |
| Boothbay.jpg           | 6720  | 4480   | 0. | 0.      | 0. | 0. |
| Hokkaido.jpg           | 6000  | 4000   | 0. | 0.      | 0. | 0. |
| Jewel Changi.jpg       | 6000  | 4000   | 0. | 0.      | 0. | 0. |
| Louvre.jpg             | 6056  | 4000   | 0. | 0.      | 0. | 0. |
