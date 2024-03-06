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

# Original Image

![Calaveras](../img/CQ100/img/calaveras.png)

# Without Dithering

## 16 Colors

| Method          | DSSIM      | Result                       |
| --------------- | ---------- | ---------------------------- |
| Gimp            | 0.06368717 | ![](img/gimp_16.png)         |
| Wu - sRGB       | 0.04014392 | ![](img/wu_srgb_16.png)      |
| K-means - Oklab | 0.02632949 | ![](img/kmeans_oklab_16.png) |

## 64 Colors

| Method          | DSSIM      | Result                       |
| --------------- | ---------- | ---------------------------- |
| Gimp            | 0.01730340 | ![](img/gimp_64.png)         |
| Wu - sRGB       | 0.01256557 | ![](img/wu_srgb_64.png)      |
| K-means - Oklab | 0.00638550 | ![](img/kmeans_oklab_64.png) |

## 256 Colors

| Method          | DSSIM      | Result                        |
| --------------- | ---------- | ----------------------------- |
| Gimp            | 0.00488789 | ![](img/gimp_256.png)         |
| Wu - sRGB       | 0.00330477 | ![](img/wu_srgb_256.png)      |
| K-means - Oklab | 0.00160596 | ![](img/kmeans_oklab_256.png) |

# With Dithering

## 16 Colors

| Method          | DSSIM      | Result                              |
| --------------- | ---------- | ----------------------------------- |
| Gimp            | 0. | ![](img/gimp_16_dither.png)         |
| Wu - sRGB       | 0. | ![](img/wu_srgb_16_dither.png)      |
| K-means - Oklab | 0. | ![](img/kmeans_oklab_16_dither.png) |

## 64 Colors

| Method          | DSSIM      | Result                              |
| --------------- | ---------- | ----------------------------------- |
| Gimp            | 0. | ![](img/gimp_64_dither.png)         |
| Wu - sRGB       | 0. | ![](img/wu_srgb_64_dither.png)      |
| K-means - Oklab | 0. | ![](img/kmeans_oklab_64_dither.png) |

## 256 Colors

| Method          | DSSIM      | Result                               |
| --------------- | ---------- | ------------------------------------ |
| Gimp            | 0. | ![](img/gimp_256_dither.png)         |
| Wu - sRGB       | 0. | ![](img/wu_srgb_256_dither.png)      |
| K-means - Oklab | 0. | ![](img/kmeans_oklab_256_dither.png) |
