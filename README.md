# quantette

[![Crate](https://badgen.net/crates/v/quantette)](https://crates.io/crates/quantette)
[![Docs](https://docs.rs/quantette/badge.svg)](https://docs.rs/quantette)

`quantette` is a library for fast and high quality image quantization and palette generation.
It supports the sRGB color space for fast color quantization but also the CIELAB and Oklab
color spaces for more accurate quantization. Similarly, `quantette`'s k-means color quantizer
gives high quality results while the included Wu color quantizer gives fast but still quite good results.

In some critical locations, `quantette` makes use of SIMD
(via the [`wide`](https://crates.io/crates/wide) crate).
Consider enabling the `avx` or `avx2`
[target features](https://doc.rust-lang.org/reference/conditional-compilation.html#target_feature)
for a noticeable speed up if your target architecture supports these features.
If the `threads` cargo feature is enabled, multi-threaded versions of most functions
become available for even greater speed up.

# Examples

Below are some examples of `quantette` in action.
The dissimilarity between the each image and the original is reported in the tables below
using [`dssim`](https://crates.io/crates/dssim) (lower numbers are better).
Each table starts with output from GIMP as a comparison.

Each output image was created like so:
- The GIMP output was creating using `Image > Mode > Indexed` with GIMP version `2.10.34`.
- The `Wu - sRGB` output was creating using `quantette`'s fastest quantization method.
  The default number of bins was used (`32`).
- The `K-means - Oklab` output was creating using `quantette`'s most accurate quantization method.
  A sampling factor of `0.5` and a batch size of `4096` was used.

All output images are undithered to better highlight differences.

## Original Image

![Calaveras](img/CQ100/img/calaveras.png)

## 16 Colors

| Method          | DSSIM      | Result                        |
| --------------- | ---------- | ----------------------------- |
| Gimp            | 0.06368717 | ![](docs/gimp_16.png)         |
| Wu - sRGB       | 0.04014392 | ![](docs/wu_srgb_16.png)      |
| K-means - Oklab | 0.02632949 | ![](docs/kmeans_oklab_16.png) |

## 64 Colors

| Method          | DSSIM      | Result                        |
| --------------- | ---------- | ----------------------------- |
| Gimp            | 0.01730340 | ![](docs/gimp_64.png)         |
| Wu - sRGB       | 0.01256557 | ![](docs/wu_srgb_64.png)      |
| K-means - Oklab | 0.00638550 | ![](docs/kmeans_oklab_64.png) |

## 256 Colors

| Method          | DSSIM      | Result                         |
| --------------- | ---------- | ------------------------------ |
| Gimp            | 0.00488789 | ![](docs/gimp_256.png)         |
| Wu - sRGB       | 0.00330477 | ![](docs/wu_srgb_256.png)      |
| K-means - Oklab | 0.00160596 | ![](docs/kmeans_oklab_256.png) |

# License

`quantette` is licensed under either
- the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) (see [LICENSE-APACHE](LICENSE-APACHE))
- the [MIT](http://opensource.org/licenses/MIT) license (see [LICENSE-MIT](LICENSE-MIT))

at your option.
